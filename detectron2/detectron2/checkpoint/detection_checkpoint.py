# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
from urllib.parse import parse_qs, urlparse
import torch
import torch.nn as nn
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager

from .c2_model_loading import align_and_update_state_dicts
from typing import Any, cast, Dict, IO, Iterable, List, NamedTuple, Optional, Tuple

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)
    
class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.path_manager = PathManager
        self._parsed_url_during_load = None

    def load(self, path, *args, **kwargs):
        assert self._parsed_url_during_load is None
        need_sync = False
        logger = logging.getLogger(__name__)
        logger.info("[DetectionCheckpointer] Loading from {} ...".format(path))

        if path and isinstance(self.model, DistributedDataParallel):
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable

        if path:
            parsed_url = urlparse(path)
            self._parsed_url_during_load = parsed_url
            path = parsed_url._replace(query="").geturl()  # remove query from filename
            path = self.path_manager.get_local_path(path)
        ret = super().load(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            self.model._sync_params_and_buffers()
        self._parsed_url_during_load = None  # reset to None
        return ret

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return dict({"is_detector": not ("resizer" in filename)}, **data)
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True,
                        "is_detector": not ("resizer" in filename)}
        elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            return {"model": model_state, "__author__": "pycls", 
                    "matching_heuristics": True, "is_detector": not ("resizer" in filename)}

        loaded = self._torch_load(filename)
        if "model" not in loaded:    loaded = {"model": loaded}
        assert self._parsed_url_during_load is not None, "`_load_file` must be called inside `load`"
        parsed_url = self._parsed_url_during_load
        queries = parse_qs(parsed_url.query)
        if queries.pop("matching_heuristics", "False") == ["True"]:
            loaded["matching_heuristics"] = True
        if len(queries) > 0:
            raise ValueError(
                f"Unsupported query remaining: f{queries}, orginal filename: {parsed_url.geturl()}"
            )
        return dict({"is_detector": not ("resizer" in filename)}, **loaded)

    def _torch_load(self, f):
        return super()._load_file(f)

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                is_detector=checkpoint.get("is_detector")
            )
        # for non-caffe2 models, use standard ways to load it
        incompatible = self._custom_load_model(checkpoint, is_detector=checkpoint.get("is_detector"))

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:    incompatible.missing_keys.remove(k)
                except ValueError:    pass
        for k in incompatible.unexpected_keys[:]:
            # Ignore unexpected keys about cell anchors. They exist in old checkpoints
            # but now they are non-persistent buffers and will not be in new checkpoints.
            if "anchor_generator.cell_anchors" in k:
                incompatible.unexpected_keys.remove(k)
        return incompatible

    def _custom_load_model(self, checkpoint: Any, is_detector=True) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # workaround https://github.com/pytorch/pytorch/issues/24139
        incorrect_shapes = []
        model_state_dict = self.model.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                # Allow mismatch for uninitialized parameters
                if TORCH_VERSION >= (1, 8) and isinstance(
                    model_param, nn.parameter.UninitializedParameter
                ):
                    continue
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    has_observer_base_classes = (
                        TORCH_VERSION >= (1, 8)
                        and hasattr(quantization, "ObserverBase")
                        and hasattr(quantization, "FakeQuantizeBase")
                    )
                    if has_observer_base_classes:
                        # Handle the special case of quantization per channel observers,
                        # where buffer shape mismatches are expected.
                        def _get_module_for_key(
                            model: torch.nn.Module, key: str
                        ) -> torch.nn.Module:
                            # foo.bar.param_or_buffer_name -> [foo, bar]
                            key_parts = key.split(".")[:-1]
                            cur_module = model
                            for key_part in key_parts:
                                cur_module = getattr(cur_module, key_part)
                            return cur_module

                        cls_to_skip = (
                            ObserverBase,
                            FakeQuantizeBase,
                        )
                        target_module = _get_module_for_key(self.model, k)
                        if isinstance(target_module, cls_to_skip):
                            # Do not remove modules with expected shape mismatches
                            # them from the state_dict loading. They have special logic
                            # in _load_from_state_dict to handle the mismatches.
                            continue

                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        
        if is_detector:
            incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        else:
            if self.model.resizer is not None:
                new_stat = {}
                for key, value in checkpoint_state_dict.items():
                    new_stat[key.replace('resizer.', '')] = value
                incompatible = self.model.resizer.load_state_dict(new_stat, strict=False)
            
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

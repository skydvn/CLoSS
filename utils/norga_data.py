"""
Data access for the NoRGa learner.

CIL uses the PILOT / CaRE ``DataManager`` API: ``get_dataset(indices, source, mode)`` with
class indices, and ``get_task_size(task)``. Labels are already remapped to the class
order, so task t owns a contiguous class range.

DIL needs one dataset per domain. ``dil_task_dataset`` is the single place to adapt if
``utils/dil_data_manager.py`` exposes per-domain datasets under another name or signature.
"""
import numpy as np
import torch

# Tried in order: method(task, source=..., mode=...) -> torch Dataset for that domain.
DIL_METHOD_CANDIDATES = ("get_task_dataset", "get_domain_dataset")


def cil_dataset(dm, start, end, source, mode):
    """Dataset of classes [start, end) from a PILOT/CaRE DataManager."""
    return dm.get_dataset(np.arange(start, end), source=source, mode=mode)


def dil_task_dataset(dm, task, source, mode):
    """Dataset of one domain (task) from the DIL data manager.

    source: "train" or "test" split; mode: "train" (augmenting) or "test" transforms.
    """
    for name in DIL_METHOD_CANDIDATES:
        fn = getattr(dm, name, None)
        if callable(fn):
            return fn(task, source=source, mode=mode)
    public = sorted(n for n in dir(dm) if not n.startswith("_"))
    raise NotImplementedError(
        "NoRGa needs one dataset per domain in DIL. Edit utils/norga_data.py::dil_task_dataset "
        f"so it returns the dataset of domain `task` from {type(dm).__name__}. "
        f"Public attributes of that object: {public}")


def unpack_batch(batch):
    """Return (images, labels) from (x, y), (idx, x, y), (idx, x, y, ...) or a dict."""
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("x"))
        y = batch.get("label", batch.get("y"))
    elif len(batch) == 2:
        x, y = batch
    else:
        x, y = batch[1], batch[2]
    return x, torch.as_tensor(y).long()

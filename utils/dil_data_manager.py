"""
Domain-incremental (DIL) data for MoSS: every task shares one label space Y = {0, ..., C-1} with
consistent semantics, and tasks differ in their input distribution. This complements the
class-incremental DataManager of the original repo.

Config keys (all optional unless noted):
  dataset            "domainnet" | "imagenetr_dil" | "folder_dil" | "synthetic_subset"   (required)
  data_path          dataset root (defaults below are relative to utils.data.data_root)
  domains            task order; each entry is a domain name or a list of names merged into one task
  shuffle_domains    permute the task order with the run seed (default: false)
  test_frac          held-out fraction for layouts without an explicit test split (default: 0.2)

Layouts:
  domainnet        <root>/<domain>_train.txt, <root>/<domain>_test.txt  (official split files; lines
                   "<domain>/<class>/<file> <label>", paths relative to <root>)
  imagenetr_dil    <root>/{train,test}/<wnid>/<rendition>_<n>.jpg  (rendition parsed from the file name)
  folder_dil       <root>/<domain>/{train,test}/<class>/*  or  <root>/<domain>/<class>/*  (seeded split)
  synthetic_subset generated in memory; see SyntheticSubsetDIL
"""
import logging
import os
import re
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data import build_transform, data_root
from utils.data_manager import DummyDataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


class FeatureDataset(Dataset):
    """Pre-computed vectors; yields (idx, x, y) like DummyDataset."""

    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = np.asarray(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return idx, self.x[idx], int(self.y[idx])


# ----------------------------------------------------------------------------------------------
# Sources: each exposes .domains (OrderedDict name -> dict(train=(X, Y), test=(X, Y))), .class_names,
# .use_path, .use_features, and transforms.
# ----------------------------------------------------------------------------------------------
class _ImageSource:
    use_path = True
    use_features = False

    def __init__(self, args):
        self.args = args
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []


class DomainNetDIL(_ImageSource):
    DEFAULT_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def load(self, seed):
        root = self.args.get("data_path", os.path.join(data_root, "domainnet"))
        names = _flatten(self.args.get("domains", self.DEFAULT_DOMAINS))
        self.domains = OrderedDict()
        labels_seen = {}
        for d in names:
            split = {}
            for part in ("train", "test"):
                path = os.path.join(root, f"{d}_{part}.txt")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"DomainNet split file not found: {path}")
                xs, ys = [], []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rel, lab = line.rsplit(" ", 1)
                        xs.append(os.path.join(root, rel))
                        ys.append(int(lab))
                        labels_seen.setdefault(int(lab), rel.split("/")[1])
                split[part] = (np.array(xs), np.array(ys))
            self.domains[d] = split
        self.class_names = [labels_seen[i] for i in sorted(labels_seen)]


class ImageNetRDIL(_ImageSource):
    """ImageNet-R as DIL: tasks are rendition types parsed from file names such as 'art_12.jpg'."""

    _pat = re.compile(r"^(?P<dom>[A-Za-z]+)_\d+\.[A-Za-z]+$")

    def load(self, seed):
        root = self.args.get("data_path", os.path.join(data_root, "imagenet-r"))
        per_split = {}
        classes = None
        for part in ("train", "test"):
            pdir = os.path.join(root, part)
            cls = sorted(d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d)))
            classes = cls if classes is None else classes
            if cls != classes:
                raise ValueError("ImageNet-R train/test class folders differ.")
            recs = []
            for ci, c in enumerate(cls):
                for fn in sorted(os.listdir(os.path.join(pdir, c))):
                    m = self._pat.match(fn)
                    if m is None:
                        raise ValueError(f"Cannot parse a rendition prefix from '{fn}'. "
                                         "Use dataset 'folder_dil' with an explicit domain layout instead.")
                    recs.append((m.group("dom"), os.path.join(pdir, c, fn), ci))
            per_split[part] = recs
        found = sorted({r[0] for r in per_split["train"]})
        names = _flatten(self.args.get("domains", found))
        self.domains = OrderedDict()
        for d in names:
            split = {}
            for part in ("train", "test"):
                sel = [r for r in per_split[part] if r[0] == d]
                split[part] = (np.array([r[1] for r in sel]), np.array([r[2] for r in sel]))
            self.domains[d] = split
        self.class_names = classes


class FolderDIL(_ImageSource):
    def load(self, seed):
        root = self.args["data_path"]
        names = self.args.get("domains") or sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        names = _flatten(names)
        test_frac = float(self.args.get("test_frac", 0.2))

        def class_dirs(path):
            return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))

        per_domain_classes = {}
        for d in names:
            base = os.path.join(root, d)
            sub = os.path.join(base, "train") if os.path.isdir(os.path.join(base, "train")) else base
            per_domain_classes[d] = set(class_dirs(sub))
        classes = sorted(set.intersection(*per_domain_classes.values()))
        union = set.union(*per_domain_classes.values())
        if len(classes) < len(union):
            logging.warning(f"[DIL] {len(union) - len(classes)} classes are missing from some domains; "
                            f"keeping the {len(classes)} classes shared by all domains.")
        cidx = {c: i for i, c in enumerate(classes)}

        def list_imgs(path):
            xs, ys = [], []
            for c in classes:
                cdir = os.path.join(path, c)
                if not os.path.isdir(cdir):
                    continue
                for fn in sorted(os.listdir(cdir)):
                    if fn.endswith(IMG_EXT):
                        xs.append(os.path.join(cdir, fn))
                        ys.append(cidx[c])
            return np.array(xs), np.array(ys)

        rng = np.random.default_rng(seed)
        self.domains = OrderedDict()
        for d in names:
            base = os.path.join(root, d)
            if os.path.isdir(os.path.join(base, "train")) and os.path.isdir(os.path.join(base, "test")):
                self.domains[d] = {"train": list_imgs(os.path.join(base, "train")),
                                   "test": list_imgs(os.path.join(base, "test"))}
            else:
                xs, ys = list_imgs(base)
                perm = rng.permutation(len(ys))
                n_test = int(round(len(ys) * test_frac))
                te, tr = perm[:n_test], perm[n_test:]
                self.domains[d] = {"train": (xs[tr], ys[tr]), "test": (xs[te], ys[te])}
        self.class_names = classes


class SyntheticSubsetDIL:
    """
    Controlled benchmark with known subset-shared predictive factors (CPU friendly).

    The input is F blocks of `factor_dim` dims. Each factor f has class prototypes mu[f, c]. Task s
    activates a subset S_s of factors: blocks in S_s carry mu[f, y] + noise; the remaining blocks
    carry a task-specific, class-independent nuisance offset + noise. Tasks therefore share
    predictive structure only through overlapping factor subsets. The default schedule mixes new
    combinations of familiar factors ({0,2}) with a genuinely new factor ({3}).
    """
    use_path = False
    use_features = True
    train_trsf, test_trsf, common_trsf = [], [], []

    def __init__(self, args):
        self.args = args

    def load(self, seed):
        a = self.args
        C = int(a.get("syn_num_classes", 10))
        Fn = int(a.get("syn_num_factors", 4))
        p = int(a.get("syn_factor_dim", 16))
        schedule = a.get("syn_task_factors", [[0, 1], [1, 2], [0, 2], [3], [1, 3]])
        n_tr = int(a.get("syn_train_per_class", 100))
        n_te = int(a.get("syn_test_per_class", 50))
        noise = float(a.get("syn_noise", 1.0))
        proto_scale = float(a.get("syn_proto_scale", 1.5))
        nuis_scale = float(a.get("syn_nuisance_scale", 3.0))
        rng = np.random.default_rng(int(a.get("syn_seed", seed)))
        mu = rng.normal(0.0, proto_scale, size=(Fn, C, p))
        self.domains = OrderedDict()
        self.task_factors = [list(map(int, s)) for s in schedule]
        for t, S in enumerate(self.task_factors):
            nuis = rng.normal(0.0, nuis_scale, size=(Fn, p))
            split = {}
            for part, n in (("train", n_tr), ("test", n_te)):
                y = np.repeat(np.arange(C), n)
                x = rng.normal(0.0, noise, size=(len(y), Fn, p))
                for f in range(Fn):
                    x[:, f, :] += mu[f, y] if f in S else nuis[f]
                split[part] = (x.reshape(len(y), Fn * p).astype(np.float32), y)
            self.domains[f"task{t}_f{'-'.join(map(str, S))}"] = split
        self.class_names = [str(c) for c in range(C)]
        a.setdefault("feature_dim", Fn * p)


def _flatten(domains):
    """Entries may be names or lists of names (merged into one task)."""
    return [d if isinstance(d, str) else "+".join(d) for d in domains]


_SOURCES = {
    "domainnet": DomainNetDIL,
    "imagenetr_dil": ImageNetRDIL,
    "folder_dil": FolderDIL,
    "synthetic_subset": SyntheticSubsetDIL,
}


class DILDataManager:
    def __init__(self, dataset_name, seed, args):
        name = dataset_name.lower()
        if name not in _SOURCES:
            raise NotImplementedError(f"Unknown DIL dataset {dataset_name}. Options: {list(_SOURCES)}")
        self.args = args
        src = _SOURCES[name](args)
        groups = args.get("domains")
        if groups is not None and name != "synthetic_subset":
            # Merge grouped domains: load every member, then concatenate per group.
            flat_members = [m for g in groups for m in ([g] if isinstance(g, str) else g)]
            args_members = dict(args, domains=flat_members)
            src.args = args_members
            src.load(seed)
            merged = OrderedDict()
            for g in groups:
                members = [g] if isinstance(g, str) else list(g)
                merged["+".join(members)] = {
                    part: (np.concatenate([src.domains[m][part][0] for m in members]),
                           np.concatenate([src.domains[m][part][1] for m in members]))
                    for part in ("train", "test")
                }
            src.domains = merged
        else:
            src.load(seed)

        names = list(src.domains.keys())
        empty = [n for n in names if len(src.domains[n]["train"][1]) == 0 or len(src.domains[n]["test"][1]) == 0]
        if empty:
            raise ValueError(f"[DIL] No train or test samples for task(s) {empty}. "
                             "Check the 'domains' entries against the dataset on disk.")
        if args.get("shuffle_domains", False):
            names = [names[i] for i in np.random.default_rng(seed).permutation(len(names))]
        self.domain_names = names
        self._domains = src.domains
        self.class_names = src.class_names
        self.use_path = src.use_path
        self.use_features = src.use_features
        self._train_trsf = src.train_trsf
        self._test_trsf = src.test_trsf
        self._common_trsf = src.common_trsf
        self.task_factors = getattr(src, "task_factors", None)
        logging.info(f"[DIL] {len(names)} tasks, {self.nb_classes} classes. Order: {names}")
        for n in names:
            logging.info(f"[DIL]   {n}: {len(self._domains[n]['train'][1])} train / "
                         f"{len(self._domains[n]['test'][1])} test")

    @property
    def nb_tasks(self):
        return len(self.domain_names)

    @property
    def nb_classes(self):
        return len(self.class_names)

    def get_task_size(self, task):
        return self.nb_classes

    def get_task_dataset(self, task, source, mode):
        x, y = self._domains[self.domain_names[task]][source]
        if self.use_features:
            return FeatureDataset(x, y)
        from torchvision import transforms
        trsf = self._train_trsf if mode == "train" else self._test_trsf
        return DummyDataset(x, y, transforms.Compose([*trsf, *self._common_trsf]), self.use_path)

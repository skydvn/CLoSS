"""
Bounded historical memory for MoSS (Sec. 3.1 and "Updating the bounded memory" in Sec. 3.5).

Records are (u, y, s): frozen-backbone feature, label, task index. Two independent reservoirs
(fitting / validation) with fixed capacities B_tr + B_val = B. Each current example is presented
exactly once to the reservoir matching its original fit/val assignment; records never move
between reservoirs.
"""
import numpy as np
import torch


class Reservoir:
    """Algorithm-R reservoir: the first B records are stored; record number n > B is accepted with
    probability B / n and replaces a uniformly chosen stored record. n counts every record ever
    presented to this reservoir, across tasks."""

    def __init__(self, capacity, seed=0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.u = None
        self.y = None
        self.s = None
        self.size = 0
        self.n_seen = 0

    def __len__(self):
        return self.size

    def _alloc(self, dim):
        self.u = torch.zeros(self.capacity, dim, dtype=torch.float32)
        self.y = torch.full((self.capacity,), -1, dtype=torch.long)
        self.s = torch.full((self.capacity,), -1, dtype=torch.long)

    def add(self, u, y, task_id):
        u = u.detach().float().cpu()
        y = torch.as_tensor(y).long().cpu()
        n = len(y)
        if self.capacity <= 0:
            self.n_seen += n
            return
        if self.u is None:
            self._alloc(u.shape[1])
        for i in range(n):
            self.n_seen += 1
            if self.size < self.capacity:
                j = self.size
                self.size += 1
            else:
                j = int(self.rng.integers(0, self.n_seen))  # uniform on {0, ..., n-1}
                if j >= self.capacity:
                    continue  # rejected with probability 1 - B / n
            self.u[j] = u[i]
            self.y[j] = y[i]
            self.s[j] = int(task_id)

    def data(self):
        if self.size == 0:
            return None, None, None
        return self.u[: self.size], self.y[: self.size], self.s[: self.size]

    def tasks(self):
        if self.size == 0:
            return []
        return sorted(torch.unique(self.s[: self.size]).tolist())

    def indices_by_task(self):
        _, _, s = self.data()
        if s is None:
            return {}
        return {int(t): torch.nonzero(s == t, as_tuple=True)[0] for t in torch.unique(s).tolist()}

    def indices_by_cell(self):
        """(task, class) -> indices of stored records."""
        _, y, s = self.data()
        if s is None:
            return {}
        out = {}
        key = s * (int(y.max()) + 1) + y
        for k in torch.unique(key).tolist():
            idx = torch.nonzero(key == k, as_tuple=True)[0]
            out[(int(s[idx[0]]), int(y[idx[0]]))] = idx
        return out


class MoSSMemory:
    def __init__(self, total_size, val_frac=0.2, seed=0):
        b_val = int(round(total_size * val_frac))
        b_tr = int(total_size) - b_val
        self.fit = Reservoir(b_tr, seed=seed)
        self.val = Reservoir(b_val, seed=seed + 1)
        self._replay_cache = None

    def __len__(self):
        return len(self.fit) + len(self.val)

    def update(self, u_fit, y_fit, u_val, y_val, task_id):
        self.fit.add(u_fit, y_fit, task_id)
        self.val.add(u_val, y_val, task_id)
        self._replay_cache = None

    # ------------------------------------------------------------------ replay (Sec. 3.5, E_rep)
    def _build_replay_cache(self):
        groups = self.fit.indices_by_task()
        if not groups:
            self._replay_cache = None
            return
        tasks = sorted(groups)
        order = torch.cat([groups[t] for t in tasks])
        lens = torch.tensor([len(groups[t]) for t in tasks], dtype=torch.long)
        starts = torch.cumsum(lens, 0) - lens
        self._replay_cache = (order, lens, starts)

    def sample_replay_indices(self, n, generator):
        """Pick a represented task uniformly, then a record uniformly within that task, so tasks
        with larger stored subsets do not automatically receive more replay weight."""
        if len(self.fit) == 0 or n <= 0:
            return None
        if self._replay_cache is None:
            self._build_replay_cache()
        order, lens, starts = self._replay_cache
        t_choice = torch.randint(0, len(lens), (n,), generator=generator)
        offs = (torch.rand(n, generator=generator) * lens[t_choice]).long()
        offs = torch.minimum(offs, lens[t_choice] - 1)
        return order[starts[t_choice] + offs]

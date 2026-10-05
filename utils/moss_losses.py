"""
Loss primitives for MoSS.

- mmd2:            D(P, Q) = MMD^2 with a Gaussian kernel (Sec. 3.3). The paper's estimator is the
                   V-statistic over empirical laws "using all pairs including diagonal terms"
                   (estimator="v"). estimator="u" (unbiased, diagonal removed) is offered as an option
                   because the V-statistic is biased upward by O(1/n), which penalizes small cells.
- cross_cov_div:   normalized cross-covariance between centered expert outputs (Eqs. 17-18). This is
                   linear CKA between the two experts' outputs on a common minibatch.
- kd_kl:           KL(sg[p_teacher] || p_student) for prediction retention (Eq. 27).
"""
import torch
import torch.nn.functional as F


def pairwise_sq_dists(x, y):
    # ||x||^2 + ||y||^2 - 2 x.y is differentiable everywhere (unlike cdist at zero distance).
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    d = x2 + y2.transpose(-1, -2) - 2.0 * x @ y.transpose(-1, -2)
    return d.clamp_min(0.0)


def gaussian_kernel(x, y, sigma):
    return torch.exp(-pairwise_sq_dists(x, y) / (2.0 * sigma * sigma))


def mmd2(x, y, sigma, estimator="v"):
    """Squared MMD between the empirical laws of x (n, r) and y (m, r)."""
    kxx = gaussian_kernel(x, x, sigma)
    kyy = gaussian_kernel(y, y, sigma)
    kxy = gaussian_kernel(x, y, sigma)
    if estimator == "v":
        return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
    if estimator == "u":
        n, m = x.shape[0], y.shape[0]
        if n < 2 or m < 2:
            raise ValueError("Unbiased MMD needs at least two samples per set.")
        sxx = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        syy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
        return sxx + syy - 2.0 * kxy.mean()
    raise ValueError(f"Unknown MMD estimator: {estimator}")


def cross_cov_div(expert_out, pairs, eps=1e-8):
    """
    expert_out: (N, M, r) outputs of every expert on a common minibatch.
    pairs: list of unordered (m, n) pairs (fixed w.r.t. differentiation).
    Returns L_div = mean over pairs of ||H~_m^T H~_n||_F^2 / (||H~_m^T H~_m||_F ||H~_n^T H~_n||_F + eps).
    """
    if len(pairs) == 0:
        return expert_out.new_zeros(())
    h = expert_out - expert_out.mean(dim=0, keepdim=True)  # (I - 11^T/N) H_m for every m
    vals = []
    for m, n in pairs:
        hm, hn = h[:, m, :], h[:, n, :]
        num = (hm.t() @ hn).pow(2).sum()
        den = torch.linalg.matrix_norm(hm.t() @ hm) * torch.linalg.matrix_norm(hn.t() @ hn) + eps
        vals.append(num / den)
    return torch.stack(vals).sum() / max(1, len(pairs))


def kd_kl(student_logits, teacher_logits):
    """KL(sg[p_teacher] || p_student), averaged over the batch."""
    t = F.softmax(teacher_logits.detach(), dim=-1)
    log_s = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(log_s, t, reduction="batchmean")

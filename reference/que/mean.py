import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch
import numpy as np
import numpy.linalg as linalg
import sklearn.decomposition as decom
from scipy.stats import ortho_group
import scipy.stats as st
import scipy as sp
import random
import utils
import os.path as osp
import baselines
import time

import pdb

device = utils.device
NOISE_INN_THRESH = 0.1
DEBUG = False

"""
Compute QUE scoring matrix U.
"""


def compute_m(X, lamb, noise_vecs=None):
    X_cov = lamb * cov(X)
    # torch svd has bug. U and V not equal up to sign or permutation, for non-duplicate entries.
    # U, D, Vt = (lamb*X_cov).svd()

    U, D, Vt = linalg.svd(X_cov.cpu().numpy())
    U = torch.from_numpy(U.astype("float64")).to(device)
    # torch can't take exponential on int64 types.
    D_exp = torch.from_numpy(np.exp(D.astype("float64"))).to(device).diag()

    # projection of noise onto the singular vecs.
    if noise_vecs is not None:
        n_noise = noise_vecs.size(0)
        print(utils.inner_mx(noise_vecs, U)[:, : int(1.5 * n_noise)])

    m = torch.mm(U, D_exp)
    m = torch.mm(m, U.t())

    assert m.max().item() < float("Inf")
    m_tr = m.diag().sum()
    m = m / m_tr

    return m.to(torch.float32)


def compute_m0(X, lamb, noise_vecs=None):
    X_cov = lamb * cov(X)
    u, v, w = sp.linalg.svd(X_cov.cpu().numpy())
    # pdb.set_trace()
    m = torch.from_numpy(sp.linalg.expm(lamb * X_cov.cpu().numpy() / v[0])).to(
        utils.device
    )
    m_tr = m.diag().sum()
    m = m / m_tr
    return m


"""
Modifies in-place
(More complex choosing directions.)
@deprecated
"""


def corrupt_random_sample_dep(X, n_dir):
    prev_dir_l = []
    n_points = X.size(0)
    n_cor = max(1, int(n_points * cor_portion))
    cor_idx = torch.zeros(n_dir, n_cor, dtype=torch.int64, device=X.device)

    for i in range(n_dir):
        cor_idx[i] = corrupt1d(X, prev_dir_l).view(-1)

    idx = torch.zeros(n_dir, n_points, device=X.device)
    src = torch.ones(1, n_cor, device=X.device).expand(n_dir, -1)

    idx.scatter_add_(1, cor_idx, src)
    idx = idx.sum(0)
    cor_idx = torch.LongTensor(range(n_points))[idx.view(-1) > 0].to(X.device)

    return cor_idx


"""
Modifies in-place.
Need to re-center X again after this function call.
"""


def corrupt(feat_dim, n_dir, cor_portion, opt):
    prev_dir_l = []
    # noise_norm = opt.norm_scale*np.sqrt(feat_dim)
    # noise_m = torch.from_numpy(ortho_group.rvs(dim=feat_dim).astype(np.float32)).to(device)
    # chunk_sz = n_cor // n_dir
    # cor_idx = torch.LongTensor(list(range(n_cor))).to(utils.device).unsqueeze(-1)

    noise_idx = 0
    # generate n_dir number of norms, sample in interval [kp, sqrt(d)]
    # for testing, to achieve high acc for tau0 & tau1: noise_norms = np.random.normal( np.sqrt(feat_dim), 1. , (int(np.ceil(n_c

    # min number of samples per noise dir
    n_noise_min = 520
    end = 0
    noise_vecs_l = []
    chunk_sz = (feat_dim - 1) // n_dir
    for i in range(n_dir):
        cur_n = int(n_noise_min * 1.1**i)
        cur_noise_vecs = 0.1 * torch.randn(cur_n, feat_dim).to(utils.device)

        cur_noise_vecs[:, i * chunk_sz] += np.sqrt(
            n_dir / np.clip(cor_portion, 0.01, None)
        )
        # noise_vecs[start:end, noise_idx] += 1./np.clip(cor_portion, 0.01, None)

        cur_noise_vecs[cur_n // 2 :] *= -1
        ###corrupt1d(X, prev_dir_l, cor_idx[start:end], noise_vecs[start:end])
        noise_vecs_l.append(cur_noise_vecs)

    # noise_vecs = 0.1 *torch.randn(n_cor, feat_dim, device=X.device)
    noise_vecs = torch.cat(noise_vecs_l, dim=0)
    cor_idx = torch.LongTensor(list(range(len(noise_vecs)))).to(utils.device)
    n = int(len(noise_vecs) / (cor_portion / (1 - cor_portion)))
    X = generate_sample(n, feat_dim)
    X = torch.cat((noise_vecs, X), dim=0)

    if len(X) < feat_dim:
        print("Warning: number of samples smaller than feature dim!")
    opt.true_mean = torch.zeros(1, feat_dim, device=utils.device)
    """
    idx = torch.zeros(n_dir, n_points, device=X.device)
    src = torch.ones(1, n_cor, device=X.device).expand(n_dir, -1)
    
    idx.scatter_add_(1, cor_idx, src)
    idx = idx.sum(0)
    cor_idx = torch.LongTensor(range(n_points))[idx.view(-1)>0].to(X.device)
    """
    return X, cor_idx, noise_vecs


"""
Returns:
-noise: (1, feat_dim) noise vec
@deprecated
"""


def create_noise_dep(X, prev_dir_l):
    # with high prob, randomly generated vecs are orthogonal
    no_noise = True
    feat_dim = X.size(1)
    while no_noise:
        noise = torch.randn(1, feat_dim, device=X.device)
        too_close = False
        for d in prev_dir_l:
            if (d * noise).sum().item() > NOISE_INN_THRESH:
                too_close = True
                break
        if not too_close:
            no_noise = False
    return noise


"""
Modifies in-place.
Input:
-X: tensor
-n_dir: number of directions
-n_cor: number of points to be corrupted
-noise: 2D vec, (1, feat_dim)
"""


def corrupt1d(X, prev_dir_l, cor_idx, noise):
    n_points, feat_dim = X.size()

    prev_dir_l.append(noise)
    # add to a portion of samples
    n_cor = cor_idx.size(0)

    ##print('indices corrupted {}'.format(cor_idx.view(-1)))
    # create index vec
    # e.g. [[1,1,...],[4,4,...]]
    idx = cor_idx.expand(n_cor, feat_dim)  ##

    # add noise in opposing directions
    noise2dir = True
    if noise2dir and n_cor > 1:
        noise_neg = -noise
        len0 = n_cor // 2
        len1 = n_cor - len0
        # X.scatter_(0, idx[:len0], noise.expand(len0, -1))
        # X.scatter_(0, idx[len0:], noise_neg.expand(len1, -1))
        X.scatter_(0, idx[:len0], noise[:len0])
        X.scatter_(0, idx[len0:], noise_neg[len0:])
    else:
        X.scatter_(0, idx, noise.expand(n_cor, -1))
        X.scatter_(0, idx[len0:], noise)

    return cor_idx


"""
Create data samples
"""


def generate_sample(n, feat_dim):
    # create sample with mean 0 and variance 1
    X = torch.randn(n, feat_dim, device=device)
    # X = X/(X**2).sum(-1, keepdim=True)
    return X


"""
Compute top cov dir. To compute \tau_old
Returns:
-2D array, of shape (1, n_feat)
"""


def top_dir(X, opt, noise_vecs=None):
    X = X - X.mean(dim=0, keepdim=True)
    X_cov = cov(X)
    if False:
        u, d, v_t = linalg.svd(X_cov.cpu().numpy())
        # pdb.set_trace()
        u = u[: opt.n_top_dir]
    else:
        # convert to numpy tensor.
        sv = decom.TruncatedSVD(opt.n_top_dir)
        sv.fit(X.cpu().numpy())
        u = sv.components_

    if noise_vecs is not None:
        print("inner of noise with top cov dirs")
        n_noise = noise_vecs.size(0)
        sv1 = decom.TruncatedSVD(n_noise)
        sv1.fit(X.cpu().numpy())
        u1 = torch.from_numpy(sv1.components_).to(device)
        print(utils.inner_mx(noise_vecs, u1)[:, : int(1.5 * n_noise)])

    # U, D, V = svd(X, k=1)
    return torch.from_numpy(u).to(device)


"""
Input:
-X: shape (n_sample, n_feat)
"""


def cov(X):
    X = X - X.mean(dim=0, keepdim=True)
    cov = torch.mm(X.t(), X) / X.size(0)
    return cov


"""
Compute accuracy.
Input:
-score: 1D tensor
-corrupt_idx: 1D tensor
Returns:
-percentage of highest-scoring points that are corrupt.
"""


def compute_acc(score, cor_idx):
    cor_idx = cor_idx.view(-1)
    n_idx = cor_idx.size(0)
    top_idx = torch.topk(score, k=n_idx)[1]  # k
    # (1,k)
    top_idx = top_idx.unsqueeze(0).expand(n_idx, -1)
    cor_idx = cor_idx.unsqueeze(-1).expand(-1, n_idx)

    return float(top_idx.eq(cor_idx).sum()) / n_idx


"""
Compute accuracy with select index.
Input:
-score: 1D tensor
-corrupt_idx: 1D tensor
Returns:
-percentage of highest-scoring points that are corrupt.
"""


def compute_acc_with_idx(select_idx, cor_idx, X, n_removed):
    cor_idx = cor_idx.view(-1)
    n_idx = cor_idx.size(0)
    all_idx = torch.zeros(X.size(0), device=device)
    ones = torch.ones(select_idx.size(0), device=device)
    all_idx.scatter_add_(dim=0, index=select_idx, src=ones)

    if device == "cuda":
        try:
            range_idx = torch.cuda.LongTensor(range(X.size(0)))
        except RuntimeError:
            print("Run time error!")
            pdb.set_trace()
    else:
        range_idx = torch.LongTensor(range(X.size(0)))
    # (1,k)
    drop_idx = range_idx[all_idx == 0]
    top_idx = drop_idx.unsqueeze(0).expand(n_idx, -1)

    """
    top_idx = torch.topk(score, k=n_idx)[1] #k
    #(1,k)
    top_idx = top_idx.unsqueeze(0).expand(n_idx, -1)
    """
    # (X.size(0), n_idx)
    cor_idx = cor_idx.unsqueeze(-1).expand(-1, n_removed)

    return float(top_idx.eq(cor_idx).sum()) / n_idx


def compute_tau1_fast(X, select_idx, opt, noise_vecs):
    X = torch.index_select(X, dim=0, index=select_idx)
    X_centered = X - X.mean(0, keepdim=True)

    if True:
        tau1 = utils.jl_chebyshev(X, opt.lamb)
    else:
        M = compute_m(X, opt.lamb, noise_vecs)
        X_m = torch.mm(X_centered, M)  # M should be symmetric, so not M.t()
        tau1 = (X_centered * X_m).sum(-1)

    return tau1


"""
Input:
-X: centered
-select_idx: idx to keep for this iter, 1D tensor.
Output:
-X: updated X
-tau1
"""


def compute_tau1(X, select_idx, opt, noise_vecs):
    X = torch.index_select(X, dim=0, index=select_idx)
    # input should already be centered!
    X_centered = X - X.mean(0, keepdim=True)
    M = compute_m(X, opt.lamb, noise_vecs)
    X_m = torch.mm(X_centered, M)  # M should be symmetric, so not M.t()
    tau1 = (X_centered * X_m).sum(-1)

    return tau1


"""
Input: already centered
"""


def compute_tau0(X, select_idx, opt, noise_vecs=None):
    X = torch.index_select(X, dim=0, index=select_idx)
    cov_dir = top_dir(X, opt, noise_vecs)
    # top dir can be > 1
    cov_dir = cov_dir.sum(dim=0, keepdim=True)
    tau0 = (torch.mm(cov_dir, X.t()) ** 2).squeeze()
    return tau0


def compute_tau2(X, select_idx, opt, noise_vecs=None):
    """
    compute tau2, v^tM^{-1}v
    """
    X = torch.index_select(X, dim=0, index=select_idx)
    M = cov(X).cpu().numpy()
    M_inv = torch.from_numpy(linalg.pinv(M)).to(utils.device)
    scores = (torch.mm(X, M_inv) * X).sum(-1)
    return scores


def train_rme(X, opt):
    """
    Evaluate robust mean estimation using various scorings on X, in terms
    of both error from true mean and time.
    Input:
    -X: input, already corrupted and centered.
    -n: number of samples
    """
    opt.norm_thresh = 1.2
    spectral_norm, _ = utils.dominant_eval_cov(X)
    initial_norm = spectral_norm
    X0 = X.clone()
    X1 = X
    # run in loop until spectral norm small
    counter0 = 0
    time0 = time.time()
    while spectral_norm > opt.norm_thresh:
        select_idx0, n_removed0, tau0 = get_select_idx(X0, compute_tau0, opt)
        X0 = X0[select_idx0]  # Perform
        spectral_norm, _ = utils.dominant_eval_cov(X0)
        counter0 += 1
    time0 = time.time() - time0
    print("time0 {} counter0 {}".format(time0, counter0))
    error0 = utils.l2_dist(X0.mean(dim=0), opt.true_mean)

    spectral_norm = initial_norm
    opt.remove_p = opt.p / 2
    opt.lamb = 1.0 / spectral_norm * opt.lamb_multiplier
    counter1 = 0
    time1 = time.time()
    while spectral_norm > opt.norm_thresh:
        select_idx1, n_removed1, tau1 = get_select_idx(X1, compute_tau1_fast, opt)
        X1 = X1[select_idx1]
        # find dominant eval.
        dom_eval, _ = utils.dominant_eval_cov(X1)
        opt.lamb = 1.0 / dom_eval * opt.lamb_multiplier
        spectral_norm = dom_eval
        counter1 += 1

    time1 = time.time() - time1
    print("time1 {} counter1 {}".format(time1, counter1))

    error1 = utils.l2_dist(X1.mean(dim=0), opt.true_mean)
    scores_l = [error1, error0, time1, time0]
    return scores_l


"""
Computes tau1 and tau0.
Note: after calling this for multiple iterations, use select_idx rather than the scores tau 
for determining which have been selected as outliers. Since tau's are scores for remaining points after outliers.
Returns:
-tau1 and tau0, select indices for each, and n_removed for each
"""


def compute_tau1_tau0(X, opt):
    use_dom_eval = True
    if use_dom_eval:
        # dynamically set lamb now
        # find dominant eval.
        dom_eval, _ = utils.dominant_eval_cov(X)
        opt.lamb = 1.0 / dom_eval * opt.lamb_multiplier
        lamb = opt.lamb

    # noise_vecs can be used for visualization.
    no_evec = True
    if no_evec:
        noise_vecs = None

    def get_select_idx(tau_method):
        if device == "cuda":
            select_idx = torch.cuda.LongTensor(list(range(X.size(0))))
        else:
            select_idx = torch.LongTensor(list(range(X.size(0))))
        n_removed = 0
        for _ in range(opt.n_iter):
            tau1 = tau_method(X, select_idx, opt, noise_vecs)
            # select idx to keep
            cur_select_idx = torch.topk(
                tau1, k=int(tau1.size(0) * (1 - opt.remove_p)), largest=False
            )[1]
            # note these points are indices of current iteration
            n_removed += select_idx.size(0) - cur_select_idx.size(0)
            select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0)
        return select_idx, n_removed, tau1

    if opt.fast_jl:
        select_idx1, n_removed1, tau1 = get_select_idx(compute_tau1_fast)
    else:
        select_idx1, n_removed1, tau1 = get_select_idx(compute_tau1)

    # acc1 = compute_acc_with_idx(select_idx, cor_idx, X, n_removed)
    if DEBUG:
        print("new acc1 {}".format(acc1))
        M = compute_m(X, opt.lamb, noise_vecs)
        X_centered = X - X.mean(0, keepdim=True)
        X_m = torch.mm(X_centered, M)  # M should be symmetric, so not M.t()
        tau1 = (X_centered * X_m).sum(-1)
        print("old acc1 {}".format(compute_acc(tau1, cor_idx)))
        pdb.set_trace()

    """
    if device == 'cuda':
        select_idx = torch.cuda.LongTensor(range(X.size(0)))
    else:
        select_idx = torch.LongTensor(range(X.size(0)))
    for _ in range(opt.n_iter):
        tau0 = compute_tau0(X, select_idx, opt)
        cur_select_idx = torch.topk(tau0, k=tau1.size(0)*(1-opt.remove_p), largest=False)[1]
        select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0)
    """
    select_idx0, n_removed0, tau0 = get_select_idx(compute_tau0)

    return tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0


def get_select_idx(X, tau_method, opt):
    if device == "cuda":
        select_idx = torch.cuda.LongTensor(list(range(X.size(0))))
    else:
        select_idx = torch.LongTensor(list(range(X.size(0))))
    n_removed = 0
    noise_vecs = None
    for _ in range(opt.n_iter):
        tau1 = tau_method(X, select_idx, opt, noise_vecs)
        # select idx to keep
        cur_select_idx = torch.topk(
            tau1, k=int(tau1.size(0) * (1 - opt.remove_p)), largest=False
        )[1]
        # note these points are indices of current iteration
        n_removed += select_idx.size(0) - cur_select_idx.size(0)
        select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0)
    return select_idx, n_removed, tau1

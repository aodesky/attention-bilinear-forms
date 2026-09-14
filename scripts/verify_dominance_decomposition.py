#!/usr/bin/env python3
"""Numerical verification of the dominance-decomposition proposition.

The proposition (mechanistic interpretability section of attention.tex)
states, for a symmetric matrix S != 0 with rank-paired spectrum lists
lambda_+, lambda_- and u = lambda_+ - lambda_-:

    d = 0
    <=> (lambda_+)_j >= (lambda_-)_j for all j
    <=> S = B + P with spec(B) symmetric about zero and P psd,

and that in the constructed decomposition B, P commute with S and
||P||^2 = c ||S||^2.

Checks (all should hold to machine precision):
  1. B (spectrum symmetric about zero) + P (psd), with generic
     NON-commuting eigenbases, always yields d = 0.
  2. For random spectra with pointwise dominance, the rank-paired
     decomposition reconstructs S, has P psd, has spec(B) symmetric
     about zero, commutes, and satisfies ||P||^2 = c ||S||^2.
"""

import numpy as np

rng = np.random.default_rng(0)
TOL = 1e-10


def paired_lists(lam):
    lp = np.sort(lam[lam > 1e-12])[::-1]
    ln = np.sort(-lam[lam < -1e-12])[::-1]
    m = max(lp.size, ln.size)
    return np.pad(lp, (0, m - lp.size)), np.pad(ln, (0, m - ln.size))


def d_weight(S):
    lp, ln = paired_lists(np.linalg.eigvalsh(S))
    u = lp - ln
    return (u[u < 0] ** 2).sum()


def check1(trials=300):
    worst = 0.0
    for _ in range(trials):
        N = rng.integers(4, 40)
        k = N // 2
        beta = rng.exponential(1, k)
        Bdiag = np.concatenate([beta, -beta, np.zeros(N - 2 * k)])
        Q1 = np.linalg.qr(rng.standard_normal((N, N)))[0]
        B = Q1 @ np.diag(Bdiag) @ Q1.T
        r = rng.integers(1, N + 1)
        M = rng.standard_normal((N, r))
        P = M @ M.T * rng.exponential(1)
        worst = max(worst, d_weight(B + P))
    return worst


def check2(trials=300):
    worst = 0.0
    for _ in range(trials):
        n = rng.integers(2, 20)
        ln_ = np.sort(rng.exponential(1, n))[::-1]
        lp_ = np.sort(ln_ + rng.exponential(0.7, n)
                      * (rng.random(n) < 0.7))[::-1]
        assert (lp_ >= ln_ - 1e-12).all()
        lam = np.concatenate([lp_, -ln_])
        Q = np.linalg.qr(rng.standard_normal((2 * n, 2 * n)))[0]
        S = Q @ np.diag(lam) @ Q.T
        # rank-paired decomposition from the proof
        Bd = np.concatenate([ln_, -ln_])
        Pd = np.concatenate([lp_ - ln_, np.zeros(n)])
        B = Q @ np.diag(Bd) @ Q.T
        P = Q @ np.diag(Pd) @ Q.T
        worst = max(worst, np.abs(B + P - S).max())
        worst = max(worst, -np.linalg.eigvalsh(P).min())
        bl = np.sort(np.linalg.eigvalsh(B))
        worst = max(worst, np.abs(bl + bl[::-1]).max())
        worst = max(worst, np.abs(B @ S - S @ B).max())
        worst = max(worst, np.abs(B @ P - P @ B).max())
        # ||P||^2 = c ||S||^2 with c the weight of S/||S||
        nS = np.sqrt((S ** 2).sum())
        lp2, ln2 = paired_lists(np.linalg.eigvalsh(S / nS))
        c = ((lp2 - ln2).clip(0) ** 2).sum()
        worst = max(worst, abs((P ** 2).sum() - c * nS ** 2))
    return worst


def check3(trials=2000):
    """b = 0 iff S semidefinite (the empty open edge of the simplex).

    For indefinite spectra, b = 2<lp, ln> >= 2*lp[0]*ln[0] > 0 since
    both sorted lists have their largest entry in the first slot;
    for semidefinite spectra, b = 0 trivially.
    Returns the worst violation of b >= 2*lp[0]*ln[0]."""
    worst = 0.0
    for _ in range(trials):
        n = rng.integers(2, 30)
        lam = rng.standard_normal(n)
        lam /= np.linalg.norm(lam)
        lp, ln = paired_lists(lam)
        b = 2 * lp @ ln
        if (lam > 1e-12).any() and (lam < -1e-12).any():
            worst = max(worst, 2 * lp[0] * ln[0] - b)
            assert b > 0
        else:
            assert b == 0
    return worst


def check4(trials=300, competitors=200):
    """Lexicographic uniqueness of the commuting decomposition.

    The canonical pair has ||B||^2 = 2 ||S_-||^2, and among commuting
    decompositions (diagonal beta with symmetric multiset, beta <= s
    slotwise) no competitor has smaller ||B||, nor equal ||B|| with
    smaller ||P||. Competitors are drawn as random permutations of the
    canonical symmetric multiset (a spot check, not an exhaustive
    search of all admissible multisets). Returns the worst violation
    (0 if none)."""
    worst = 0.0
    for _ in range(trials):
        n_pos = rng.integers(2, 6)
        n_neg = rng.integers(1, n_pos + 1)
        n_zero = rng.integers(0, 3)
        ln_ = np.sort(rng.exponential(1, n_neg))[::-1]
        lp_ = np.sort(np.pad(ln_, (0, n_pos - n_neg))
                      + rng.exponential(1, n_pos)
                      * (rng.random(n_pos) < 0.8))[::-1]
        s = np.concatenate([lp_, -ln_, np.zeros(n_zero)])
        beta_can = np.concatenate(
            [np.pad(ln_, (0, n_pos - n_neg)), -ln_, np.zeros(n_zero)])
        nB_can = (beta_can**2).sum()
        nP_can = ((s - beta_can)**2).sum()
        assert abs(nB_can - 2 * (ln_**2).sum()) < 1e-10
        for _ in range(competitors):
            beta = beta_can[rng.permutation(len(s))]
            srt = np.sort(beta)
            if (beta <= s + 1e-12).all() \
                    and np.abs(srt + srt[::-1]).max() < 1e-9:
                nB = (beta**2).sum()
                nP = ((s - beta)**2).sum()
                worst = max(worst, nB_can - nB)
                if abs(nB - nB_can) < 1e-9:
                    worst = max(worst, nP_can - nP)
    return worst


def main() -> None:
    w1 = check1()
    print(f"check 1 (balanced-spectrum + psd => d = 0): max d = {w1:.2e}")
    w2 = check2()
    print(f"check 2 (decomposition under dominance):    max err = {w2:.2e}")
    w3 = check3()
    print(f"check 3 (b = 0 iff semidefinite):           "
          f"max bound violation = {w3:.2e}")
    w4 = check4()
    print(f"check 4 (lexicographic uniqueness):         "
          f"max violation = {w4:.2e}")
    assert w1 < TOL and w2 < TOL and w3 < TOL and w4 < TOL
    print("all checks passed")


if __name__ == "__main__":
    main()

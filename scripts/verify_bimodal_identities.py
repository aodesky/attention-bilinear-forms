#!/usr/bin/env python3
"""Numerical verification of the identities for balanced bimodal spectra.

Checks, on the 144 gpt2 spectra (sorted, unit norm, 64 negative and 64
positive entries each) and on random data where appropriate:

  (1) ||l+||^2 + ||l-||^2 = 1 and ||l+||^2 = (1+J)/2
  (2) ||l+|| ||l-|| = sqrt(1-J^2)/2
  (3) ||l+ - l-||^2 = 1 - 2C,  C = <l+, l->
  (4) P = 1 - sqrt(1-2C)   (P = pairing score, balanced case)
  (5) sum_j (l+_j + i l-_j)^2 = J + 2iC
  (6) ||l - iota(l)||^2 = 2 ||l+ - l-||^2,  iota(l)_i = -l_{n+1-i}
  (7) iota is an isometry and preserves sortedness/balance
  (8) c^2 <= int x^4 dmu <= c for random measures on [-1,1] with
      int x^2 dmu = c
  (9) the model family mu_{j,k}: int x^2 = c, J = j, K = k
 (10) tr(Wk^T Wq) = <Wk, Wq>_F and m1(spectrum/||.||) = tr S/||S||_F
 (11) O = sum l/(1-l^2) equals the truncated odd-moment series, and
      E = sum l^2/(1-l^2) the truncated even-moment series
 (12) O = (R(l) - R(iota l))/2 and n + E = (R(l) + R(iota l))/2,
      R(l) = sum 1/(1-l_i); hence O odd and E even under iota
 (13) trace forms: O = tr(T(I-T^2)^{-1}), E = tr(T^2(I-T^2)^{-1}) for
      a symmetric matrix T with spectrum l
 (14) Gram form: tr(S^k) = tr((J Gamma / 2)^k), the nonzero spectrum
      of S equals the spectrum of J Gamma / 2 for surjective M, and
      O, E, R are traces of resolvents of H = J Gamma / (2 ||S||)
 (15) archetype energies: for unit-norm L = S + A,
      a = ||A||^2, b = 2||S||^2 <l+,l->, c = ||S||^2 ||u+||^2,
      d = ||S||^2 ||u-||^2 satisfy a+b+c+d = 1, with the four
      endpoint characterizations

Prints the maximum absolute error of each identity.
"""

from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "cluster_results"
rng = np.random.default_rng(0)


def halves(lam):
    """l+ and l- (both positive, sorted decreasing) of a balanced spectrum."""
    pos = np.sort(lam[lam > 0])[::-1]
    neg = np.sort(-lam[lam < 0])[::-1]
    assert pos.size == neg.size, "not balanced"
    return pos, neg


def main() -> None:
    X = np.load(RESULTS / "gpt2_sorted_spectra.npy")
    errs = {k: 0.0 for k in range(1, 16)}

    for lam in X:
        lp, ln = halves(lam)
        a, b = (lp**2).sum(), (ln**2).sum()
        J = a - b
        C = lp @ ln
        d2 = ((lp - ln) ** 2).sum()
        P = 1 - np.sqrt(d2)

        errs[1] = max(errs[1], abs(a + b - 1), abs(a - (1 + J) / 2))
        errs[2] = max(errs[2], abs(np.sqrt(a * b) - np.sqrt(1 - J**2) / 2))
        errs[3] = max(errs[3], abs(d2 - (1 - 2 * C)))
        errs[4] = max(errs[4], abs(P - (1 - np.sqrt(1 - 2 * C))))
        z = ((lp + 1j * ln) ** 2).sum()
        errs[5] = max(errs[5], abs(z - (J + 2j * C)))
        iota = -lam[::-1]
        errs[6] = max(errs[6], abs(((lam - iota) ** 2).sum() - 2 * d2))
        errs[7] = max(errs[7],
                      abs(np.linalg.norm(iota) - np.linalg.norm(lam)),
                      0.0 if np.all(np.diff(iota) >= 0) else 1.0)

        O = (lam / (1 - lam**2)).sum()
        E = (lam**2 / (1 - lam**2)).sum()
        odd_series = sum((lam**k).sum() for k in range(1, 200, 2))
        even_series = sum((lam**k).sum() for k in range(2, 200, 2))
        errs[11] = max(errs[11], abs(O - odd_series), abs(E - even_series))
        R = (1 / (1 - lam)).sum()
        Riota = (1 / (1 - iota)).sum()
        errs[12] = max(errs[12],
                       abs(O - (R - Riota) / 2),
                       abs(lam.size + E - (R + Riota) / 2),
                       abs((iota / (1 - iota**2)).sum() + O),
                       abs((iota**2 / (1 - iota**2)).sum() - E))

    # (8) random measures on [-1,1] with fixed second moment
    for _ in range(2000):
        x = rng.uniform(-1, 1, size=rng.integers(2, 50))
        w = rng.dirichlet(np.ones(x.size))
        c = float(w @ x**2)
        if not (0 < c < 1):
            continue
        m4 = float(w @ x**4)
        errs[8] = max(errs[8], max(0.0, c**2 - m4), max(0.0, m4 - c))

    # (9) model family mu_{j,k}
    for _ in range(2000):
        j = rng.uniform(-1, 1)
        k = rng.uniform(0, 1)
        c = rng.uniform(0.05, 0.95)
        # atoms: sqrt(c), -sqrt(c), 0, 1, -1 with the family's weights
        xs = np.array([np.sqrt(c), -np.sqrt(c), 0.0, 1.0, -1.0])
        ws = np.array([
            (1 - k) * (1 + j) / 2,
            (1 - k) * (1 - j) / 2,
            k * (1 - c),
            k * c * (1 + j) / 2,
            k * c * (1 - j) / 2,
        ])
        errs[9] = max(errs[9],
                      abs(ws.sum() - 1),
                      abs(ws @ xs**2 - c),
                      abs((ws @ (np.sign(xs) * xs**2)) / c - j),
                      abs((ws @ xs**4 - c**2) / (c * (1 - c)) - k))

    # (10) trace identity on random low-rank forms
    for _ in range(200):
        n, N = 16, 96
        Wk = rng.standard_normal((n, N))
        Wq = rng.standard_normal((n, N))
        L = Wk.T @ Wq
        S = 0.5 * (L + L.T)
        errs[10] = max(errs[10],
                       abs(np.trace(L) - (Wk * Wq).sum()),
                       abs(np.trace(S) - (Wk * Wq).sum()),
                       abs(np.linalg.eigvalsh(S).sum()
                           / np.linalg.norm(S) - np.trace(S)
                           / np.linalg.norm(S)))

    # (13) trace forms on random symmetric matrices with unit spectrum
    for _ in range(200):
        n = 16
        lam = rng.standard_normal(n)
        lam /= np.linalg.norm(lam)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        T = Q @ np.diag(lam) @ Q.T
        I = np.eye(n)
        inv = np.linalg.inv(I - T @ T)
        O = (lam / (1 - lam**2)).sum()
        E = (lam**2 / (1 - lam**2)).sum()
        errs[13] = max(errs[13],
                       abs(np.trace(T @ inv) - O),
                       abs(np.trace(T @ T @ inv) - E),
                       abs(np.trace(np.linalg.inv(I - T))
                           - (1 / (1 - lam)).sum()))

    # (14) Gram form on random low-rank symmetrized products
    for _ in range(100):
        n, N = 8, 48
        Wk = rng.standard_normal((n, N))
        Wq = rng.standard_normal((n, N))
        S = 0.5 * (Wk.T @ Wq + Wq.T @ Wk)
        M = np.vstack([Wk, Wq])
        G = M @ M.T
        J = np.block([[np.zeros((n, n)), np.eye(n)],
                      [np.eye(n), np.zeros((n, n))]])
        JG2 = J @ G / 2
        scale = np.linalg.norm(S)
        for k in range(1, 6):
            errs[14] = max(errs[14],
                           abs(np.trace(np.linalg.matrix_power(S / scale, k))
                               - np.trace(np.linalg.matrix_power(
                                   JG2 / scale, k))))
        lam_all = np.linalg.eigvalsh(S)
        nz = np.sort(lam_all[np.abs(lam_all)
                             > 1e-9 * np.abs(lam_all).max()])
        # J Gamma is similar to the symmetric Gamma^{1/2} J Gamma^{1/2}
        gev, gvec = np.linalg.eigh(G)
        Ghalf = gvec @ np.diag(np.sqrt(gev)) @ gvec.T
        spec_small = np.sort(np.linalg.eigvalsh(Ghalf @ J @ Ghalf / 2))
        errs[14] = max(errs[14], float(np.max(np.abs(nz - spec_small))))
        nrm = np.linalg.norm(S)
        lam = nz / nrm
        O = (lam / (1 - lam**2)).sum()
        E = (lam**2 / (1 - lam**2)).sum()
        R = (1 / (1 - lam)).sum()
        H = JG2 / nrm
        I2 = np.eye(2 * n)
        inv = np.linalg.inv(I2 - H @ H)
        errs[14] = max(errs[14],
                       abs(nrm - 0.5 * np.sqrt(np.trace((J @ G) @ (J @ G)))),
                       abs(np.trace(H @ inv) - O),
                       abs(np.trace(H @ H @ inv) - E),
                       abs(np.trace(np.linalg.inv(I2 - H)) - R))

    # (15) archetype energies on random forms and archetype endpoints
    def abcd(L):
        S = 0.5 * (L + L.T)
        A = 0.5 * (L - L.T)
        if np.linalg.norm(S) == 0:
            return (np.linalg.norm(A) ** 2, 0.0, 0.0, 0.0)
        ev = np.linalg.eigvalsh(S) / np.linalg.norm(S)
        pos = np.sort(ev[ev > 0])[::-1]
        neg = np.sort(-ev[ev < 0])[::-1]
        m = max(pos.size, neg.size)
        pos = np.pad(pos, (0, m - pos.size))
        neg = np.pad(neg, (0, m - neg.size))
        u = pos - neg
        s2 = np.linalg.norm(S) ** 2
        return (np.linalg.norm(A) ** 2, 2 * s2 * (pos @ neg),
                s2 * (u[u > 0] ** 2).sum(), s2 * (u[u < 0] ** 2).sum())

    for _ in range(200):
        n = 12
        L = rng.standard_normal((n, n))
        L /= np.linalg.norm(L)
        a, b, c, d = abcd(L)
        errs[15] = max(errs[15], abs(a + b + c + d - 1),
                       max(0.0, -min(a, b, c, d)))
    # endpoints: antisymmetric, paired symmetric, PSD, NSD
    Z = rng.standard_normal((12, 12))
    An = (Z - Z.T); An /= np.linalg.norm(An)
    G = rng.standard_normal((12, 6))
    P = G @ G.T; P /= np.linalg.norm(P)
    Q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    w = rng.standard_normal(6) ** 2 + 0.1
    B = Q @ np.diag(np.concatenate([w, -w])) @ Q.T
    B = 0.5 * (B + B.T); B /= np.linalg.norm(B)
    for L, idx in [(An, 0), (B, 1), (P, 2), (-P, 3)]:
        vals = abcd(L)
        errs[15] = max(errs[15], abs(vals[idx] - 1))

    names = {
        1: "energy split:  a+b=1,  a=(1+J)/2",
        2: "product:       ||l+||||l-|| = sqrt(1-J^2)/2",
        3: "cross term:    ||l+-l-||^2 = 1-2C",
        4: "pairing score: P = 1-sqrt(1-2C)",
        5: "complex form:  sum (l+_j + i l-_j)^2 = J+2iC",
        6: "involution:    ||l-iota(l)||^2 = 2||l+-l-||^2",
        7: "iota isometry, sorted",
        8: "bounds:        c^2 <= m4 <= c",
        9: "model family:  mass/second moment/J/K exact",
        10: "trace:         tr S = <Wk,Wq>_F,  m1 = trS/||S||_F",
        11: "O,E closed forms = moment series",
        12: "O,E = odd/even parts of R,  O odd / E even under iota",
        13: "O,E,R trace forms",
        14: "Gram form: spectra and O,E,R from J Gamma / 2",
        15: "archetype energies: a+b+c+d = 1 and endpoints",
    }
    for k in range(1, 16):
        print(f"({k:2d}) {names[k]:52s} max err = {errs[k]:.2e}")


if __name__ == "__main__":
    main()

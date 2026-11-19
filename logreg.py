"""
CCD vs RCD on L2-regularized logistic regression

    f(x) = (1/m) sum_i log(1 + exp(-b_i a_i^T x)) + (lam/2) ||x||^2,

with a_i ~ N(0, I_n), b_i = sign(a_i^T x_true), each b_i flipped w.p. 0.1.

Conventions (same as the quadratic code):
  * H = D - L - L^T, with L = -(strict lower part of H).
  * All rates are PER EPOCH (one epoch = n coordinate updates).
  * Coordinate step: x_j <- x_j - grad_j f(x) / L_j, with
    L_j = ||A[:, j]||^2 / (4m) + lam  (coordinate Lipschitz constant,
    since sigma'(t) <= 1/4). This is a coordinate GRADIENT step, not exact
    minimization: f is not quadratic, so there is no closed-form 1-D minimizer.

Two kinds of quantity:
  (1) Nonlinear runs: the actual algorithm on f. The map x -> x+ is NOT linear
      in e = x - x*, so the renormalization trick is not valid here. We track
      log ||x_l - x*|| directly; it bottoms out at machine precision (~1e-15).
  (2) Linearized rates at x*: near x*, e+ = P_j e + O(||e||^2) with
      P_j = I - u_j h_j^T / L_j, where h_j is row j of H* = Hessian at x*.
      These are the exact asymptotic rates of (1), and for them renormalization
      is valid again (linear maps), so the quadratic machinery is reused.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

STYLE = {
    "CCD":   dict(color="#0072B2", ls="-",  lw=1.3),
    "ERate": dict(color="#000000", ls="--", lw=1.0),
    "PRate": dict(color="#D55E00", ls="-",  lw=1.3),
}

# Deterministic quantities
def make_logreg(m, n, lam, rng, flip=0.1):
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    b = np.sign(A @ x_true)
    b[b == 0] = 1.0
    b[rng.random(m) < flip] *= -1
    return A, b


def loss(x, A, b, lam):
    return np.logaddexp(0.0, -b * (A @ x)).mean() + 0.5 * lam * x @ x


def grad(x, A, b, lam):
    m = A.shape[0]
    s = -b * expit(-b * (A @ x))            # d/dz_i of log(1 + exp(-b_i z_i))
    return A.T @ s / m + lam * x


def hessian(x, A, b, lam):
    m, n = A.shape
    p = expit(A @ x)
    w = p * (1.0 - p)                        # sigma(z)(1 - sigma(z)), b_i^2 = 1
    return (A.T * w) @ A / m + lam * np.identity(n)


def solve_newton(A, b, lam, tol=1e-14, max_iter=100):
    """x* to machine precision (needed as the reference for ||x - x*||)."""
    x = np.zeros(A.shape[1])
    for _ in range(max_iter):
        g = grad(x, A, b, lam)
        dx = np.linalg.solve(hessian(x, A, b, lam), g)
        x -= dx
        if np.linalg.norm(dx) <= tol * max(1.0, np.linalg.norm(x)):
            break
    return x


def coord_lipschitz(A, lam):
    m = A.shape[0]
    return (A ** 2).sum(axis=0) / (4 * m) + lam


# CCD and RCD runs
def _epoch(x, z, A, b, lam, Lc, order):
    """One epoch of coordinate gradient steps in the given index order.
    z = A @ x is maintained in O(m) per step."""
    m = A.shape[0]
    for j in order:
        s = -b * expit(-b * z)
        g_j = A[:, j] @ s / m + lam * x[j]
        delta = -g_j / Lc[j]
        x[j] += delta
        z += delta * A[:, j]


def run_ccd_logreg(epochs, x0, A, b, lam, x_star):
    """log(||x_l - x*|| / ||x_0 - x*||) for cyclic order 1..n, per epoch."""
    n = A.shape[1]
    Lc = coord_lipschitz(A, lam)
    x = np.array(x0, float)
    r0 = np.linalg.norm(x - x_star)
    errors = np.zeros(epochs + 1)
    order = np.arange(n)
    for ep in range(epochs):
        z = A @ x                            # recompute each epoch (no drift)
        _epoch(x, z, A, b, lam, Lc, order)
        errors[ep + 1] = np.log(np.linalg.norm(x - x_star) / r0)
    return errors


def run_rcd_logreg(epochs, x0, A, b, lam, x_star, n_runs, rng, probs=None):
    """Same, with n i.i.d. random coordinates per epoch; n_runs paths."""
    n = A.shape[1]
    Lc = coord_lipschitz(A, lam)
    errors = np.zeros((n_runs, epochs + 1))
    for row in range(n_runs):
        x = np.array(x0, float)
        r0 = np.linalg.norm(x - x_star)
        for ep in range(epochs):
            z = A @ x
            _epoch(x, z, A, b, lam, Lc, rng.choice(n, size=n, p=probs))
            errors[row, ep + 1] = np.log(np.linalg.norm(x - x_star) / r0)
    return errors, errors.mean(axis=0), errors.std(axis=0)


# Linearized Asymptotic Rate
def rate_ccd_steps(H, steps):
    """Per-epoch rate of cyclic coordinate steps e_j <- e_j - (He)_j / steps_j.
    Derivation (sweep j = 1..n, H = D - L - L^T, Lam = diag(steps)):
        Lam e+ - L e+ = (Lam - D) e + L^T e
        =>  e+ = (Lam - L)^{-1} (Lam - D + L^T) e.
    Reduces to C = (D - L)^{-1} L^T when steps = diag(H) (exact minimization)."""
    D = np.diag(np.diag(H))
    L = -np.tril(H, -1)
    Lam = np.diag(steps)
    M = np.linalg.solve(Lam - L, Lam - D + L.T)
    return -np.log(np.max(np.abs(np.linalg.eigvals(M))))


def erate_rcd_steps(H, steps):
    """Per-epoch expected rate, uniform sampling: -n log rho(I - Lam^{-1} H / n)."""
    n = H.shape[0]
    M = np.identity(n) - (H / steps[:, None]) / n
    return -n * np.log(np.max(np.abs(np.linalg.eigvals(M))))


def run_prcd_steps(epochs, H, steps, n_runs, rng, probs=None):
    """Pathwise rate of the LINEARIZED random iteration at x* (renormalized,
    so it can run indefinitely without hitting machine precision)."""
    n = H.shape[0]
    errors = np.zeros((n_runs, epochs + 1))
    for row in range(n_runs):
        e = rng.standard_normal(n)
        e /= np.linalg.norm(e)
        for ep in range(epochs):
            g = H @ e
            for i in rng.choice(n, size=n, p=probs):
                delta = -g[i] / steps[i]
                e[i] += delta
                g += delta * H[:, i]
            nrm = np.linalg.norm(e)
            e /= nrm
            errors[row, ep + 1] = errors[row, ep] + np.log(nrm)
    return errors


def second_half_slope(err):
    """-slope over the second half (burn-in discarded); works row-wise."""
    err = np.atleast_2d(err)
    E = err.shape[1] - 1
    h = E // 2
    return -(err[:, -1] - err[:, h]) / (E - h)


# Running Experiment
def window_slope(err, hi=np.log(1e-3), lo=np.log(1e-11)):
    """-slope of log-error fitted where it lies in (lo, hi): past the transient,
    before the machine-precision floor. Returns nan if the window is too short."""
    idx = np.where((err < hi) & (err > lo))[0]
    if len(idx) < 3:
        return np.nan
    return -np.polyfit(idx, err[idx], 1)[0]


def plot_algo_logreg(epochs, x0, A, b, lam, n_runs, rng, floor=14 * np.log(10)):
    """
    Nonlinear CCD / RCD runs on regularized logistic regression, with the
    linearized asymptotic slopes at x* drawn as dashed reference lines.

    floor: plot depth in nats; ~14 ln 10 because ||x - x*|| bottoms out near
           1e-15 relative (no renormalization possible for a nonlinear map).
    style: your STYLE dict (keys "CCD", "PRate", "ERate"); None -> defaults.
    """

    x_star = solve_newton(A, b, lam)
    H = hessian(x_star, A, b, lam)
    Lc = coord_lipschitz(A, lam)

    ccd_rate = rate_ccd_steps(H, Lc)                    
    rcd_erate = erate_rcd_steps(H, Lc)                  
    lin = run_prcd_steps(max(epochs, 2000), H, Lc, n_runs, rng)
    prate = second_half_slope(lin)                      

    x_coor = np.arange(epochs + 1)
    ccd_errors = run_ccd_logreg(epochs, x0, A, b, lam, x_star)
    rcd_errors, mean_rcd, std_rcd = run_rcd_logreg(epochs, x0, A, b, lam,
                                                   x_star, n_runs, rng)
    ccd_fit = window_slope(ccd_errors)
    rcd_fit = np.array([window_slope(r) for r in rcd_errors])

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x_coor, ccd_errors, **STYLE["CCD"],
            label=f"CCD (lin. rate {ccd_rate:.3g}, fit {ccd_fit:.3g})")
    ax.plot(x_coor, -ccd_rate * x_coor, color=STYLE["CCD"]["color"],
            lw=0.8, ls=":")

    c1 = STYLE["PRate"]["color"]
    ax.plot(x_coor, rcd_errors.T, color=c1, lw=0.3, alpha=0.3)
    ax.fill_between(x_coor, mean_rcd - std_rcd, mean_rcd + std_rcd,
                    color=c1, alpha=0.25, lw=0)
    ax.plot(x_coor, mean_rcd, **STYLE["PRate"],
            label=(f"RCD, mean ± std of {n_runs} runs\n"
                   f"(lin. PRate {prate.mean():.3g} ± "
                   f"{prate.std(ddof=1) / np.sqrt(n_runs):.2g}, "
                   f"fit {np.nanmean(rcd_fit):.3g})"))
    ax.plot(x_coor, -rcd_erate * x_coor, **STYLE["ERate"],
            label=f"RCD lin. ERate {rcd_erate:.3g}")

    slowest = min(ccd_rate, rcd_erate)
    ax.set_xlim(0, min(epochs, 1.1 * floor / slowest))
    ax.set_ylim(-floor, 1)
    ax.set_xlabel(r"epoch $\ell$ ($n$ coordinate updates)")
    ax.set_ylabel(r"$\log\frac{\|x_\ell - x^\star\|}{\|x_0 - x^\star\|}$")
    ax.legend(loc="upper right", fontsize=6)

    return fig, ax

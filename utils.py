import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

LN10 = np.log(10)
STYLE = {
    "CCD":   dict(color="#0072B2", ls="-",  lw=1.3),
    "ERate": dict(color="#000000", ls="--", lw=1.0),
    "PRate": dict(color="#D55E00", ls="-",  lw=1.3),
}

def set_paper_style(fontsize=8, usetex=False):
    """
    Matplotlib defaults for paper figures.
    """
    mpl.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "legend.frameon": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",         
        "text.usetex": usetex,          

        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,

        "figure.constrained_layout.use": True, 
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,              
        "ps.fonttype": 42,
        "figure.dpi": 150,               
        "savefig.dpi": 600,             
    })

# Deterministic quantities
def ccd_matrix(n, c):
    """
    Create the motivation square matrix with 1 on the diagonal and c otherwise.

    params:
        n: dimension
        c: off-diagonal value
    """

    I = np.identity(n)
    L = -np.tril(c * np.ones((n, n)), k=-1) 
    return np.linalg.inv(I-L) @ L.T

def rate_ccd(n, c): 
    """
    Compute rate for Cyclic Coordinate Descent (CCD) using the spectral radius.

    params:
        n: dimension
        c: off-diagonal value
    """

    return -np.log(np.max(np.abs(np.linalg.eigvals(ccd_matrix(n, c)))))

def erate_rcd(n, c):
    """
    Compute rate expectedly for Randomized Coordinate Descent (RCD) using the closed-form solution and the eigenvalue from the motivation matrix.
    
    params:
        n: dimension
        c: off-diagonal value
    """
    
    return -n * np.log(1 - (1 / n) * (1 - c))

# Running experiment
def run_ccd(epochs, n, c, x0):
    """
    Run CCD algorithms 
    """
    C = ccd_matrix(n, c)
    x = x0 / np.linalg.norm(x0)
    errors = np.zeros(epochs + 1)

    for epoch in range(epochs):
        x = C @ x
        norm = np.linalg.norm(x)
        x /= norm
        errors[epoch + 1] = errors[epoch] + np.log(norm) 

    return errors

def run_prcd(epochs, n, c, x0, n_runs, rng): 
    """
    Run each path rate for RCD. Normalized the initial vector

    params:
        epochs: number of epochs
        n: dimension
        c: off-diagonal value
        x0: initial vector
        n_runs: number of runs for path rate
        rng: random number generator from np.random.default_rng(seed)

    return:
        mean_prcd: mean for each path rate for RCD
        std_prcd: standard deviation for each path rate for RCD
    """

    errors = np.zeros((n_runs, epochs + 1))

    for row in range(n_runs):
        x = x0 / np.linalg.norm(x0)             
        for epoch in range(epochs):
            S = x.sum()
            for i in rng.integers(0, n, size=n):  
                new = -c * (S - x[i])
                S += new - x[i]
                x[i] = new
            norm = np.linalg.norm(x)
            x /= norm
            errors[row, epoch + 1] = errors[row, epoch] + np.log(norm)

    mean_prcd = errors.mean(axis=0)
    std_prcd = errors.std(axis=0)

    return errors, mean_prcd, std_prcd


def plot_algo(epochs, n, c, x0, n_runs, rng, floor=40 * LN10):
    """
    Plot different algorithms such as CCD, RCD, and ERCD. For RCD, plot both the spread for each path rate and the mean rate for n_runs.
    The plot shows the convergence rate for every algorithms and the error for path rate.

    params:
        epochs: number of epochs
        n: dimension
        c: off-diagonal value
        x0: initial vector
        n_runs: number of runs for path rate
        rng: random number generator from np.random.default_rng(seed)
        floor: threshold to set xlim in the plot (default to 40)

    return:
        fig, ax
    """
    ccd_rate = rate_ccd(n, c)
    rcd_erate = erate_rcd(n, c)
    x_coor = np.arange(epochs + 1)

    ccd_errors = run_ccd(epochs, n, c, x0)
    rcd_errors = -rcd_erate * x_coor
    prcd_errors, mean_prcd, std_prcd = run_prcd(epochs, n, c, x0, n_runs, rng)

    half = epochs // 2
    prcd_rates = -(prcd_errors[:, -1] - prcd_errors[:, half]) / (epochs - half)
    prcd_rate = np.mean(prcd_rates)
    prcd_se = prcd_rates.std(ddof=1) / np.sqrt(n_runs)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x_coor, ccd_errors, **STYLE["CCD"], label=f"CCD (Rate = {ccd_rate:.3f})")
    ax.plot(x_coor, rcd_errors, **STYLE["ERate"], label=f"Exact slope (ERate = {rcd_erate:.3f})")

    ax.plot(x_coor, prcd_errors.T, color=STYLE["PRate"]["color"], lw=0.3, alpha=0.3)
    ax.fill_between(x_coor, mean_prcd - std_prcd, mean_prcd + std_prcd, color=STYLE["PRate"]["color"], alpha=0.25, lw=0)
    ax.plot(x_coor, mean_prcd, **STYLE["PRate"], label=f"RCD, mean ± std of {n_runs} runs\n(PRate = {prcd_rate:.3f} ± {prcd_se:.3f})")

    slowest = min(ccd_rate, rcd_erate, prcd_rate)
    ax.set_xlim(0, min(epochs, 1.1 * floor / slowest))
    ax.set_ylim(-floor, 1)
    ax.set_xlabel(r"epoch $\ell$ ($n$ coordinate updates)")
    ax.set_ylabel(r"$\log\|x_\ell\| / \|x_0\|$")
    ax.legend(loc="upper right")
    return fig, ax


def save_algo(epochs, n_vals, c_vals, n_runs, rng, fixed=None):
    if fixed == "n":
        for n_val in n_vals:
            x0 = rng.random(n_val)

            ccd_list = []
            ercd_list = []
            prcd_list = []
            for c_val in c_vals:
                ccd_rate = rate_ccd(n_val, c_val)
                rcd_rate = erate_rcd(n_val, c_val)
                prcd_errors, _, _ = run_prcd(epochs, n_val, c_val, x0, n_runs, rng)

                half = epochs // 2
                prcd_rates = -(prcd_errors[:, -1] - prcd_errors[:, half]) / (epochs - half)
                prcd_rate = np.mean(prcd_rates)

                ccd_list.append(round(-ccd_rate, 3))
                ercd_list.append(round(-rcd_rate, 3))
                prcd_list.append(round(-prcd_rate, 3))

            df = pd.DataFrame({
                    "c": c_vals.round(3),
                    "CCD": ccd_list,
                    "ERCD": ercd_list,
                    "PRCD": prcd_list,
                })
            
            file_name = f"fixed_n_{n_val}.csv"
            file_path = os.path.join("data", file_name)
            df.to_csv(file_path, mode="w", index=False, header=True)
        
    elif fixed == "c":        
        for c_val in c_vals:
            ccd_list = []
            ercd_list = []
            prcd_list = []
            for n_val in n_vals:
                x0 = rng.random(n_val)

                ccd_rate = rate_ccd(n_val, c_val)
                rcd_rate = erate_rcd(n_val, c_val)
                prcd_errors, _, _ = run_prcd(epochs, n_val, c_val, x0, n_runs, rng)

                half = epochs // 2
                prcd_rates = -(prcd_errors[:, -1] - prcd_errors[:, half]) / (epochs - half)
                prcd_rate = np.mean(prcd_rates)

                ccd_list.append(round(-ccd_rate, 3))
                ercd_list.append(round(-rcd_rate, 3))
                prcd_list.append(round(-prcd_rate, 3))

            df = pd.DataFrame({
                    "n": n_vals,
                    "CCD": ccd_list,
                    "ERCD": ercd_list,
                    "PRCD": prcd_list,
                })

            file_name = f"fixed_c_{round(c_val, 3)}.csv"
            file_path = os.path.join("data", file_name)
            df.to_csv(file_path, mode="w", index=False, header=True)



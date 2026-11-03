import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def initialize_mat(n, c):
    I = np.identity(n)
    L = -np.tril(c * np.ones((n, n)), k=-1) 
    A_c = np.identity(n)- L - L.T
    S = np.diag([1/(n) for i in range(n)])
    R = np.linalg.matrix_power(I - S @ np.linalg.inv(I) @ A_c, n) 
    C = np.linalg.inv(I-L) @ L.T

    return R, C

def build_coordinate_matrices(n, c):
    mats = []
    for i in range(n):
        M = np.eye(n)
        M[i, :] = -c  
        M[i, i] = 0.0  
        mats.append(M)
    return mats

def run_ccd(epochs, n, c, verbose=False): 
    _, C = initialize_mat(n, c)
    
    x0 = np.ones((n,1))
    x = x0.copy()
    x_star = np.zeros((n,1))
    ccd_errors = []
    
    for l in range(epochs):
        x = C @ x
        norm = np.linalg.norm(x - x_star)
        if norm < 1e-50:
            if verbose:
                print(f"CCD converges at epoch {l}")
            break
        ratio = norm / np.linalg.norm(x0 - x_star)
        ccd_errors.append(np.log(ratio))
        
    return ccd_errors 

def run_rcd(epochs, n, c):
    rcd_errors = [epoch * n * np.log(1 - (1 / n) * (1 - c)) for epoch in range(epochs)]
    return np.array(rcd_errors)

def run_prcd(epochs, n, c, verbose=False): 
    mats = build_coordinate_matrices(n, c)
    x0 = np.ones((n, 1))
    x = x0.copy()
    x_star = np.zeros((n, 1))
    prcd_errors = []
    
    for l in range(epochs):
        for i in range(n): 
            i = random.randint(0, n - 1)
            x = mats[i] @ x
        norm = np.linalg.norm(x - x_star)
        if norm < 1e-50:
            if verbose:
                print(f"PRCD converges at iteration {l}")
            break
        else:
            ratio = norm / np.linalg.norm(x0 - x_star)
            prcd_errors.append(np.log(ratio))
            
    return prcd_errors

def multiple_prcd(epochs, n, c, n_runs):
    prcd_errors = [run_prcd(epochs, n, c) for _ in range(n_runs)]
    min_len = min(len(errors) for errors in prcd_errors)

    mean_prcd= np.mean([errors[:min_len] for errors in prcd_errors], axis=0)
    std_prcd = np.std([errors[:min_len] for errors in prcd_errors], axis=0)
    return prcd_errors, mean_prcd, std_prcd


def compute_slope(ccd_errors, rcd_errors, prcd_errors):
    ccd_slope, _ = np.polyfit(range(len(ccd_errors)), ccd_errors, 1)
    ercd_slope, _ = np.polyfit(range(len(rcd_errors)), rcd_errors, 1)
    prcd_slope, _ = np.polyfit(range(len(prcd_errors)), prcd_errors, 1)

    return ccd_slope, ercd_slope, prcd_slope

def plot_algo(epochs, n, c, n_runs, verbose=False):
    R, C = initialize_mat(n, c)
    
    ccd_errors, rcd_errors = run_ccd(epochs, n, c), run_rcd(epochs, n, c)
    prcd_errors, mean_prcd, std_prcd = multiple_prcd(epochs, n, c, n_runs)

    fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)
    ccd = ax.plot(np.arange(len(ccd_errors)), ccd_errors, label="Rate(CCD)", linewidth=3)
    rcd = ax.plot(np.arange(len(ccd_errors)), rcd_errors[:len(ccd_errors)], label="ERate(RCD)", linewidth=3)

    for i, errors in enumerate(prcd_errors):
        ax.plot(np.arange(len(errors)), errors, color="black", alpha=0.25, linestyle='dotted',
            label="PRate(RCD)" if i == 0 else None)

    ax.plot(np.arange(len(mean_prcd)), mean_prcd, color="green", label="Mean PRate(RCD)", linewidth=1)
    ax.fill_between(np.arange(len(mean_prcd)), mean_prcd - std_prcd, mean_prcd + std_prcd, color="blue", 
                    label="Std PRate(RCD)", alpha=0.5)

    rho_ccd = np.max(np.abs(np.linalg.eigvals(C)))
    rho_ercd = np.max(np.abs(np.linalg.eigvals(R)))
    rate_ccd = -np.log(rho_ccd)
    rate_ercd = -np.log(rho_ercd)

    ccd_slope, ercd_slope, prcd_slope = compute_slope(ccd_errors, rcd_errors, mean_prcd)

    if verbose:
        print(f"Rate(CCD): {rate_ccd:.4f} | ERate(RCD): {rate_ercd:.4f} (From Spectral Radius)")
        print(f"Slope for Rate(CCD): {ccd_slope:.4f} | ERate(RCD): {ercd_slope:.4f} | PRate(RCD):{prcd_slope:.4f}")
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Error Log Ratio")
    ax.set_title(f"Algorithms Convergence Rate for n = {n} and c = {c}")
    ax.legend()

    return prcd_errors


def save_algo(epochs, n_vals, c_vals, n_runs, fixed=None):
    if fixed == "n":
        for n_val in n_vals:
            ccd_list = []
            ercd_list = []
            prcd_list = []
            for c_val in c_vals:
                ccd_errors, rcd_errors = run_ccd(epochs, n_val, c_val), run_rcd(epochs, n_val, c_val)
                _, mean_prcd, _ = multiple_prcd(epochs, n_val, c_val, n_runs)
                ccd_slope, ercd_slope, prcd_slope = compute_slope(ccd_errors, rcd_errors, mean_prcd)

                ccd_list.append(round(-ccd_slope, 3))
                ercd_list.append(round(-ercd_slope, 3))
                prcd_list.append(round(-prcd_slope, 3))

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
                ccd_errors, rcd_errors = run_ccd(epochs, n_val, c_val), run_rcd(epochs, n_val, c_val)
                _, mean_prcd, _ = multiple_prcd(epochs, n_val, c_val, n_runs)
                ccd_slope, ercd_slope, prcd_slope = compute_slope(ccd_errors, rcd_errors, mean_prcd)

                ccd_list.append(round(-ccd_slope, 3))
                ercd_list.append(round(-ercd_slope, 3))
                prcd_list.append(round(-prcd_slope, 3))

            df = pd.DataFrame({
                    "n": n_vals,
                    "CCD": ccd_list,
                    "ERCD": ercd_list,
                    "PRCD": prcd_list,
                })

            file_name = f"fixed_c_{round(c_val, 3)}.csv"
            file_path = os.path.join("data", file_name)
            df.to_csv(file_path, mode="w", index=False, header=True)



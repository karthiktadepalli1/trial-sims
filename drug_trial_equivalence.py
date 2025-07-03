"""
Drug Trial Equivalence Simulation

Simulates the trial-slot vs. discovery-draw equivalence constant k for Beta-distributed candidate probabilities.

NOTE: Instead of directly specifying alpha and beta, you can now specify the desired mean success rate among the top N out of m candidates (mean_top_N). The script will back out the implied alpha and beta for the Beta distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import sys
import matplotlib.ticker as mticker
from scipy.optimize import minimize

# ---- PARAMETERS: Set these at the top ----
mean_top_N = 0.46    # Desired mean success rate among top N out of m candidates (e.g., 0.4 for 40%)
N = 100             # Number of clinical-trial slots (>=1)
m = 1000           # Number of candidates (>N)
runs = 100          # Monte-Carlo replications per point (>=1)
seed = 42           # RNG seed (optional, set to None for random)
outfile = "trials.png"       # Set to a filename to save figure, or None to display
c_min = 1.5         # Minimum c value (must be >1)
c_max = 5.01        # Maximum c value
c_step = 0.5        # Step size for c values
g_fixed_for_c = 5   # Increase in N for numerator (default 5)
g_min = 1           # Minimum g value for g sweep
g_max = 10          # Maximum g value for g sweep
g_step = 1          # Step size for g values
c_fixed_for_g = 2   # Value of c to use when sweeping g

# ------------------------------------------

def estimate_top_N_mean(alpha, beta_param, N, m, runs, rng):
    """
    Estimate the mean of the top N out of m Beta(alpha, beta_param) draws.
    """
    means = np.empty(runs)
    for i in range(runs):
        draws = rng.beta(alpha, beta_param, size=m)
        means[i] = np.mean(np.partition(draws, -N)[-N:])
    return np.mean(means)

def find_beta_params_for_top_N_mean(target_mean, N, m, runs=100, seed=42):
    """
    Find beta (with alpha fixed at 1) such that the mean of the top N out of m Beta(1, beta) draws is close to target_mean.
    Returns (alpha, beta)
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0
    def objective(beta_param):
        if beta_param <= 0:
            return 1e6
        est_mean = estimate_top_N_mean(alpha, beta_param, N, m, runs, rng)
        return (est_mean - target_mean) ** 2
    # Initial guess: mean = alpha/(alpha+beta), so try to match target_mean for the mean
    mean_guess = target_mean
    beta0 = (1 - mean_guess) / mean_guess
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(objective, bounds=(1e-3, 1e3), method='bounded')
    if not res.success:
        raise RuntimeError(f"Could not find suitable beta: {res.message}")
    return alpha, res.x

def draw_candidates(alpha, beta_param, m, rng):
    """
    Draw m i.i.d. candidate probabilities from Beta(alpha, beta).

    Parameters
    ----------
    alpha : float
        Beta distribution alpha parameter (>0).
    beta_param : float
        Beta distribution beta parameter (>0).
    m : int
        Number of candidates to draw (>=1).
    rng : np.random.Generator
        Numpy random generator.

    Returns
    -------
    draws : np.ndarray
        Array of shape (m,) of candidate probabilities.
    """
    if alpha <= 0 or beta_param <= 0:
        raise ValueError("Beta parameters must be positive.")
    if m < 1:
        raise ValueError("m must be >= 1.")
    return rng.beta(alpha, beta_param, size=m)


def social_value(draws, N):
    """
    Compute the sum of the N largest values in draws.

    Parameters
    ----------
    draws : np.ndarray
        Array of candidate probabilities.
    N : int
        Number of slots (>=1).

    Returns
    -------
    value : float
        Sum of the N largest values in draws.
    """
    if N < 1:
        raise ValueError("N must be >= 1.")
    if len(draws) < N:
        raise ValueError("Number of draws must be >= N.")
    return np.sum(np.partition(draws, -N)[-N:])


def estimate_k(alpha, beta_param, m, N, runs, rng, c, g):
    """
    Estimate the equivalence constant k by Monte Carlo simulation.
    Numerator: W_{m,N+g} - W_{m,N}
    Denominator: W_{c*m,N} - W_{m,N}
    """
    if m <= N:
        raise ValueError("m must be > N.")
    if runs < 1:
        raise ValueError("runs must be >= 1.")
    if g < 1:
        raise ValueError("g must be >= 1.")
    m_c = int(np.ceil(c * m))
    if m_c <= m:
        raise ValueError("m_c must be > m.")
    k_vals = np.empty(runs)
    for i in range(runs):
        draws = draw_candidates(alpha, beta_param, m_c, rng)
        W_m_N = social_value(draws[:m], N)
        W_m_Npg = social_value(draws[:m], N+g)
        W_mc_N = social_value(draws, N)
        num = W_m_Npg - W_m_N
        denom = W_mc_N - W_m_N
        if abs(denom) < 1e-10:
            k_vals[i] = np.nan
        else:
            k_vals[i] = num / denom
    k_mean = np.nanmean(k_vals)
    k_se = np.nanstd(k_vals, ddof=1) / np.sqrt(runs)
    return k_mean, k_se


def plot_k_vs_c(alpha, beta_param, m, N, runs, seed, c_min, c_max, c_step, g_fixed, outfile):
    c_vals = np.arange(c_min, c_max, c_step)
    rng = np.random.default_rng(seed)
    k_means = []
    k_ses = []
    for c in c_vals:
        k_mean, k_se = estimate_k(alpha, beta_param, m, N, runs, rng, c, g_fixed)
        k_means.append(k_mean)
        k_ses.append(k_se)
    plt.figure(figsize=(8, 5))
    plt.errorbar(c_vals, k_means, yerr=k_ses, fmt='o-', capsize=4)
    plt.axhline(1, color='gray', linestyle='--')
    plt.xlabel('Expansion factor for number of candidates')
    plt.ylabel('Ratio of social values from extra trials vs extra candidates')
    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(f"Additional candidates vs +{g_fixed} trials")
    plt.tight_layout()
    param_text = (f"candidates drawn from Beta(α={alpha}, β={beta_param}). baseline trials={N}, "
                  f"baseline candidates={m}, additional trials={g_fixed_for_c}")
    plt.figtext(0.5, 0.01, param_text, wrap=True, horizontalalignment='center', fontsize=7)
    if outfile:
        plt.savefig(outfile.replace('.png', '_k_vs_c.png'))
        print(f"Saved plot to {outfile.replace('.png', '_k_vs_c.png')}")
    else:
        plt.show()


def plot_k_vs_g(alpha, beta_param, m, N, runs, seed, c_fixed, g_min, g_max, g_step, outfile):
    g_vals = np.arange(g_min, g_max + 1, g_step)
    rng = np.random.default_rng(seed)
    k_means = []
    k_ses = []
    for g in g_vals:
        k_mean, k_se = estimate_k(alpha, beta_param, m, N, runs, rng, c_fixed, g)
        k_means.append(k_mean)
        k_ses.append(k_se)
    plt.figure(figsize=(8, 5))
    plt.errorbar(g_vals, k_means, yerr=k_ses, fmt='o-', capsize=4)
    plt.axhline(1, color='gray', linestyle='--')
    plt.xlabel('Additional trials funded')
    plt.ylabel('Ratio of social values from extra trials vs extra candidates')
    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(f"Additional trials vs {c_fixed}x more candidates")
    plt.tight_layout()
    param_text = (f"candidates drawn from Beta(α={alpha}, β={beta_param}). baseline trials={N}, "
                  f"baseline candidates={m}")
    plt.figtext(0.5, 0.01, param_text, wrap=True, horizontalalignment='center', fontsize=7)
    if outfile:
        plt.savefig(outfile.replace('.png', '_k_vs_g.png'))
        print(f"Saved plot to {outfile.replace('.png', '_k_vs_g.png')}")
    else:
        plt.show()


def estimate_expected_social_value(alpha, beta_param, m, N, runs, rng):
    """
    Estimate the expected sum of the top N out of m Beta(alpha, beta_param) draws.
    """
    vals = np.empty(runs)
    for i in range(runs):
        draws = rng.beta(alpha, beta_param, size=m)
        vals[i] = np.sum(np.partition(draws, -N)[-N:])
    return np.mean(vals), np.std(vals, ddof=1) / np.sqrt(runs)

def plot_social_value_vs_m_or_N(alpha, beta_param, N_base, m_base, runs, seed, plot_vs, c_min, c_max, c_step, g_min, g_max, g_step, outfile=None):
    rng = np.random.default_rng(seed)
    if plot_vs == 'm':
        c_vals = np.arange(c_min, c_max, c_step)
        m_vals = [int(c * m_base) for c in c_vals]
        means = []
        ses = []
        for m_val in m_vals:
            mean, se = estimate_expected_social_value(alpha, beta_param, m_val, N_base, runs, rng)
            means.append(mean)
            ses.append(se)
        plt.figure(figsize=(8, 5))
        plt.errorbar(c_vals, means, yerr=ses, fmt='o-', capsize=4)
        plt.xlabel('Expansion factor for number of candidates (c)')
        plt.ylabel(f'Expected sum of top {N_base} success probabilities')
        plt.title(f'Sum of top {N_base} success probabilities vs candidate expansion factor')
        plt.tight_layout()
        if outfile:
            outname = outfile.replace('.png', '_social_value_vs_m.png')
            plt.savefig(outname)
            print(f"Saved plot to {outname}")
        else:
            plt.show()
    elif plot_vs == 'N':
        g_vals = np.arange(g_min, g_max + 1, g_step)
        N_vals = [N_base + g for g in g_vals]
        means = []
        ses = []
        for N_val in N_vals:
            mean, se = estimate_expected_social_value(alpha, beta_param, m_base, N_val, runs, rng)
            means.append(mean)
            ses.append(se)
        plt.figure(figsize=(8, 5))
        plt.errorbar(g_vals, means, yerr=ses, fmt='o-', capsize=4)
        plt.xlabel('Additional trials funded (g)')
        plt.ylabel(f'Expected sum of top N success probabilities (m={m_base})')
        plt.title(f'Sum of top N success probabilities vs additional trials')
        plt.tight_layout()
        if outfile:
            outname = outfile.replace('.png', '_social_value_vs_N.png')
            plt.savefig(outname)
            print(f"Saved plot to {outname}")
        else:
            plt.show()
    else:
        raise ValueError("plot_vs must be 'm' or 'N'")

def main():
    if N < 1:
        print("Error: N must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if mean_top_N < 0.0 or mean_top_N > 1.0:
        print("Error: mean_top_N must be between 0.0 and 1.0.", file=sys.stderr)
        sys.exit(1)
    if runs < 1:
        print("Error: runs must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if c_min <= 1.0 or c_max <= 1.0 or c_max <= c_min:
        print("Error: c_min and c_max must be > 1 and c_max > c_min.", file=sys.stderr)
        sys.exit(1)
    if g_fixed_for_c < 1:
        print("Error: g must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if g_min < 1 or g_max < g_min:
        print("Error: g_min must be >= 1 and g_max >= g_min.", file=sys.stderr)
        sys.exit(1)

    # Find alpha and beta to match mean_top_N
    print(f"Finding Beta(alpha, beta) such that mean of top {N} out of {m} is {mean_top_N:.2f}...")
    alpha, beta_param = find_beta_params_for_top_N_mean(mean_top_N, N, m, runs=50, seed=seed)
    print(f"Using alpha={alpha:.4f}, beta={beta_param:.4f}")

    # Plot k vs c (original exercise)
    plot_k_vs_c(alpha, beta_param, m, N, runs, seed, c_min, c_max, c_step, g_fixed_for_c, outfile)
    # Plot k vs g (new exercise)
    plot_k_vs_g(alpha, beta_param, m, N, runs, seed, c_fixed_for_g, g_min, g_max, g_step, outfile)

    # Plot social value vs m (using c sweep)
    plot_social_value_vs_m_or_N(alpha, beta_param, N, m, runs, seed, plot_vs='m', c_min=c_min, c_max=c_max, c_step=c_step, g_min=g_min, g_max=g_max, g_step=g_step, outfile=outfile)
    # Plot social value vs N (using g sweep)
    plot_social_value_vs_m_or_N(alpha, beta_param, N, m, runs, seed, plot_vs='N', c_min=c_min, c_max=c_max, c_step=c_step, g_min=g_min, g_max=g_max, g_step=g_step, outfile=outfile)

if __name__ == "__main__":
    main() 
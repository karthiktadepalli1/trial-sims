"""
Drug Trial Disease Comparison Simulation

Compares the marginal value of an additional candidate drug for two diseases:
- Disease H: higher social value per success, lower success probability
- Disease L: lower social value per success, higher success probability

Uses a similar simulation framework as drug_trial_equivalence.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import sys

# ---- PARAMETERS: Set these at the top ----
# Disease H (High value, low probability)
alpha_H = 1.0      # Beta alpha parameter for H
beta_H = 9.0       # Beta beta parameter for H
value_H = 10.0     # Social value per success for H

# Disease L (Low value, high probability)
alpha_L = 5.0      # Beta alpha parameter for L
beta_L = 2.0       # Beta beta parameter for L
value_L = 1.0      # Social value per success for L

N_H = 10           # Number of clinical-trial slots for H
N_L = 10           # Number of clinical-trial slots for L
m_H = 100          # Number of candidate drugs for H
m_L = 100          # Number of candidate drugs for L
runs = 1000        # Monte-Carlo replications per point
seed = 42          # RNG seed
outfile = "drug_trial_disease_comparison.png"  # Output plot file, or None to display
# ------------------------------------------

def draw_candidates(alpha, beta_param, m, rng):
    return rng.beta(alpha, beta_param, size=m)

def marginal_value_of_candidate(alpha, beta_param, m, N, value_per_success, runs, rng):
    """
    Estimate the marginal social value of adding one more candidate (from m to m+1).
    Returns mean and standard error over runs.
    """
    vals = np.empty(runs)
    for i in range(runs):
        draws = draw_candidates(alpha, beta_param, m+1, rng)
        # Social value is sum of top N, weighted by value per success
        base = np.sum(np.partition(draws[:m], -N)[-N:]) * value_per_success
        added = np.sum(np.partition(draws, -N)[-N:]) * value_per_success
        vals[i] = added - base
    return np.mean(vals), np.std(vals, ddof=1) / np.sqrt(runs)

def main():
    rng = np.random.default_rng(seed)
    # Marginal value for H
    mv_H_mean, mv_H_se = marginal_value_of_candidate(
        alpha_H, beta_H, m_H, N_H, value_H, runs, rng)
    # Marginal value for L
    mv_L_mean, mv_L_se = marginal_value_of_candidate(
        alpha_L, beta_L, m_L, N_L, value_L, runs, rng)
    # Bar plot
    plt.figure(figsize=(6, 5))
    diseases = ['Disease H', 'Disease L']
    means = [mv_H_mean, mv_L_mean]
    ses = [mv_H_se, mv_L_se]
    plt.bar(diseases, means, yerr=ses, capsize=8, color=['tab:blue', 'tab:orange'])
    plt.ylabel('Marginal social value of an additional candidate')
    plt.title('Marginal value of an additional candidate drug\nfor two diseases')
    param_text = (f"H: Beta(α={alpha_H}, β={beta_H}), value={value_H}, N={N_H}, m={m_H}\n"
                  f"L: Beta(α={alpha_L}, β={beta_L}), value={value_L}, N={N_L}, m={m_L}")
    plt.figtext(0.5, 0.01, param_text, wrap=True, horizontalalignment='center', fontsize=8)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if outfile:
        plt.savefig(outfile)
        print(f"Saved plot to {outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main() 
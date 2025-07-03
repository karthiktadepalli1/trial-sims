import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar

st.title("Drug Trial Equivalence Simulator")
st.write("""
This app simulates the tradeoff between funding additional clinical trial slots and screening more candidate drugs. Adjust the parameters below to see how the results change.
""")

# --- User Inputs ---
mean_top_N = st.sidebar.slider("Mean success rate among top N candidates", 0.01, 0.99, 0.4, 0.01)
N = st.sidebar.number_input("Number of clinical-trial slots (N)", min_value=1, max_value=1000, value=100)
m = st.sidebar.number_input("Number of candidates (m)", min_value=N+1, max_value=100000, value=10000)
runs = st.sidebar.number_input("Monte-Carlo replications per point", min_value=10, max_value=1000, value=100)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=100000, value=42)
c_min = st.sidebar.number_input("Min candidate expansion factor (c_min)", min_value=1.01, max_value=10.0, value=1.5, step=0.01)
c_max = st.sidebar.number_input("Max candidate expansion factor (c_max)", min_value=c_min+0.01, max_value=20.0, value=5.0, step=0.01)
c_step = st.sidebar.number_input("Step for c", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
g_fixed_for_c = st.sidebar.number_input("Additional trials for k vs c (g)", min_value=1, max_value=100, value=5)
g_min = st.sidebar.number_input("Min additional trials (g_min)", min_value=1, max_value=100, value=1)
g_max = st.sidebar.number_input("Max additional trials (g_max)", min_value=g_min, max_value=100, value=10)
g_step = st.sidebar.number_input("Step for g", min_value=1, max_value=20, value=1)
c_fixed_for_g = st.sidebar.number_input("Candidate expansion for k vs g (c)", min_value=1.01, max_value=10.0, value=2.0, step=0.01)

# --- Helper functions ---
def estimate_top_N_mean(alpha, beta_param, N, m, runs, rng):
    means = np.empty(runs)
    for i in range(runs):
        draws = rng.beta(alpha, beta_param, size=m)
        means[i] = np.mean(np.partition(draws, -N)[-N:])
    return np.mean(means)

def find_beta_params_for_top_N_mean(target_mean, N, m, runs=100, seed=42):
    rng = np.random.default_rng(seed)
    alpha = 1.0
    def objective(beta_param):
        if beta_param <= 0:
            return 1e6
        est_mean = estimate_top_N_mean(alpha, beta_param, N, m, runs, rng)
        return (est_mean - target_mean) ** 2
    mean_guess = target_mean
    beta0 = (1 - mean_guess) / mean_guess
    res = minimize_scalar(objective, bounds=(1e-3, 1e3), method='bounded')
    if not res.success:
        st.error(f"Could not find suitable beta: {res.message}")
        st.stop()
    return alpha, res.x

def draw_candidates(alpha, beta_param, m, rng):
    return rng.beta(alpha, beta_param, size=m)

def social_value(draws, N):
    return np.sum(np.partition(draws, -N)[-N:])

def estimate_k(alpha, beta_param, m, N, runs, rng, c, g):
    m_c = int(np.ceil(c * m))
    if m_c <= m:
        m_c = m + 1
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

def estimate_expected_social_value(alpha, beta_param, m, N, runs, rng):
    vals = np.empty(runs)
    for i in range(runs):
        draws = rng.beta(alpha, beta_param, size=m)
        vals[i] = np.sum(np.partition(draws, -N)[-N:])
    return np.mean(vals), np.std(vals, ddof=1) / np.sqrt(runs)

# --- Main logic ---
st.header("Inferred Beta distribution parameters")
alpha, beta_param = find_beta_params_for_top_N_mean(mean_top_N, N, m, runs=50, seed=seed)
st.write(f"**alpha = {alpha:.4f}, beta = {beta_param:.4f}**")

rng = np.random.default_rng(seed)

# --- k vs c plot ---
c_vals = np.arange(c_min, c_max, c_step)
k_means = []
k_ses = []
for c in c_vals:
    k_mean, k_se = estimate_k(alpha, beta_param, m, N, runs, rng, c, g_fixed_for_c)
    k_means.append(k_mean)
    k_ses.append(k_se)
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.errorbar(c_vals, k_means, yerr=k_ses, fmt='o-', capsize=4)
ax1.axhline(1, color='gray', linestyle='--')
ax1.set_xlabel('Expansion factor for number of candidates (c)')
ax1.set_ylabel('Ratio of social values from extra trials vs extra candidates')
ax1.set_title(f"Additional candidates vs +{g_fixed_for_c} trials")
st.pyplot(fig1)

# --- k vs g plot ---
g_vals = np.arange(g_min, g_max + 1, g_step)
k_means = []
k_ses = []
for g in g_vals:
    k_mean, k_se = estimate_k(alpha, beta_param, m, N, runs, rng, c_fixed_for_g, g)
    k_means.append(k_mean)
    k_ses.append(k_se)
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.errorbar(g_vals, k_means, yerr=k_ses, fmt='o-', capsize=4)
ax2.axhline(1, color='gray', linestyle='--')
ax2.set_xlabel('Additional trials funded (g)')
ax2.set_ylabel('Ratio of social values from extra trials vs extra candidates')
ax2.set_title(f"Additional trials vs {c_fixed_for_g}x more candidates")
st.pyplot(fig2)

# --- Social value vs m (using c sweep) ---
social_means = []
social_ses = []
m_vals = [int(c * m) for c in c_vals]
for m_val in m_vals:
    mean, se = estimate_expected_social_value(alpha, beta_param, m_val, N, runs, rng)
    social_means.append(mean)
    social_ses.append(se)
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.errorbar(c_vals, social_means, yerr=social_ses, fmt='o-', capsize=4)
ax3.set_xlabel('Expansion factor for number of candidates (c)')
ax3.set_ylabel(f'Expected sum of top {N} success probabilities')
ax3.set_title(f'Sum of top {N} success probabilities vs candidate expansion factor')
st.pyplot(fig3)

# --- Social value vs N (using g sweep) ---
social_means = []
social_ses = []
N_vals = [N + g for g in g_vals]
for N_val in N_vals:
    mean, se = estimate_expected_social_value(alpha, beta_param, m, N_val, runs, rng)
    social_means.append(mean)
    social_ses.append(se)
fig4, ax4 = plt.subplots(figsize=(7, 4))
ax4.errorbar(g_vals, social_means, yerr=social_ses, fmt='o-', capsize=4)
ax4.set_xlabel('Additional trials funded (g)')
ax4.set_ylabel(f'Expected sum of top N success probabilities (m={m})')
ax4.set_title(f'Sum of top N success probabilities vs additional trials')
st.pyplot(fig4) 
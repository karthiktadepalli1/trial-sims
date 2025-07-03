import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar

st.title("Drug Trial Resource Allocation Simulator")
st.markdown("""
This tool helps you explore the tradeoff between funding additional clinical trial slots and screening more candidate drugs. Adjust the parameters below to see how the expected number of successful drugs changes under different resource allocation strategies.
""")

# --- User Inputs ---
st.sidebar.header("Basic Parameters")
mean_top_N = st.sidebar.slider("Mean success rate among candidates that reach clinical trials", 0.01, 0.99, 0.4, 0.01)
N = st.sidebar.number_input("Number of clinical trial slots (N)", min_value=1, max_value=1000, value=100)
m = st.sidebar.number_input("Number of candidate drugs (m)", min_value=N+1, max_value=100000, value=10000)

st.sidebar.header("Main Comparison (Trials vs Candidates)")
g_fixed_for_c = st.sidebar.number_input("Additional clinical trial slots for main comparison", min_value=1, max_value=100, value=5)
percent_fixed_for_g = st.sidebar.number_input("% increase in candidate drugs for main comparison", min_value=1, max_value=1000, value=100)

st.sidebar.header("Sweep Ranges")
percent_min = st.sidebar.number_input("Minimum % increase in candidate drugs", min_value=1, max_value=1000, value=50)
percent_max = st.sidebar.number_input("Maximum % increase in candidate drugs", min_value=percent_min+1, max_value=2000, value=400)
percent_step = st.sidebar.number_input("Step for % increase", min_value=1, max_value=500, value=50)
g_min = st.sidebar.number_input("Minimum additional clinical trial slots", min_value=1, max_value=100, value=1)
g_max = st.sidebar.number_input("Maximum additional clinical trial slots", min_value=g_min, max_value=100, value=10)
g_step = st.sidebar.number_input("Step for additional clinical trial slots", min_value=1, max_value=20, value=1)

# Set fixed values for runs and seed
runs = 100
seed = 42

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

def estimate_k(alpha, beta_param, m, N, runs, rng, percent_increase, g):
    c = 1 + percent_increase / 100.0
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

def estimate_expected_successes(alpha, beta_param, m, N, runs, rng):
    vals = np.empty(runs)
    for i in range(runs):
        draws = rng.beta(alpha, beta_param, size=m)
        vals[i] = np.sum(np.partition(draws, -N)[-N:])
    return np.mean(vals), np.std(vals, ddof=1) / np.sqrt(runs)

# --- Beta PDF and Intuitive Facts ---
st.header("Distribution of Drug Success Probabilities")
alpha, beta_param = find_beta_params_for_top_N_mean(mean_top_N, N, m, runs=50, seed=seed)
mean_prob = alpha / (alpha + beta_param)
frac_80 = 1 - beta.cdf(0.8, alpha, beta_param)

x = np.linspace(0, 1, 500)
pdf = beta.pdf(x, alpha, beta_param)
fig_pdf, ax_pdf = plt.subplots(figsize=(7, 3))
ax_pdf.plot(x, pdf, label=f'Beta({alpha:.2f}, {beta_param:.2f})')
ax_pdf.set_xlabel('Probability of success for a candidate drug')
ax_pdf.set_ylabel('Density')
ax_pdf.set_title('Distribution of success probabilities across all candidate drugs')
ax_pdf.legend()
st.pyplot(fig_pdf)

st.markdown(f"""
- **Average candidate drug has a {100*mean_prob:.1f}% chance of success.**
- **Only {100*frac_80:.2f}% of candidate drugs have an 80% or higher chance of success.**
- **By design, the average success rate among the top {N} out of {m} candidates is {100*mean_top_N:.1f}%.**
""")

# --- k vs % increase in candidate drugs ---
st.header("Comparing Additional Clinical Trials vs. Additional Drug Candidates")
st.markdown(f"""
This plot compares two ways to increase the expected number of successful drugs:
- **Funding more clinical trial slots** (adding {g_fixed_for_c} slots)
- **Screening more candidate drugs** (increasing the number of candidates by a given percentage)

The y-axis shows the ratio of the expected increase in successful drugs from funding more trials to the expected increase from screening more candidates. A value above 1 means funding more trials is more effective; below 1 means screening more candidates is more effective.
""")
percent_vals = np.arange(percent_min, percent_max+1, percent_step)
k_means = []
k_ses = []
for percent in percent_vals:
    k_mean, k_se = estimate_k(alpha, beta_param, m, N, runs, np.random.default_rng(seed), percent, g_fixed_for_c)
    k_means.append(k_mean)
    k_ses.append(k_se)
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.errorbar(percent_vals, k_means, yerr=k_ses, fmt='o-', capsize=4)
ax1.axhline(1, color='gray', linestyle='--')
ax1.set_xlabel('% increase in number of candidate drugs')
ax1.set_ylabel('Relative value: more trials vs. more candidates')
ax1.set_title(f"Effectiveness of adding {g_fixed_for_c} clinical trial slots vs. more candidates")
st.pyplot(fig1)

# --- k vs additional clinical trial slots ---
st.header("Comparing Additional Drug Candidates vs. Additional Clinical Trials")
st.markdown(f"""
This plot compares the same tradeoff, but now for a fixed % increase in candidate drugs (here, {percent_fixed_for_g}%), as you vary the number of additional clinical trial slots.

The y-axis is interpreted as above.
""")
g_vals = np.arange(g_min, g_max + 1, g_step)
k_means = []
k_ses = []
for g in g_vals:
    k_mean, k_se = estimate_k(alpha, beta_param, m, N, runs, np.random.default_rng(seed), percent_fixed_for_g, g)
    k_means.append(k_mean)
    k_ses.append(k_se)
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.errorbar(g_vals, k_means, yerr=k_ses, fmt='o-', capsize=4)
ax2.axhline(1, color='gray', linestyle='--')
ax2.set_xlabel('Number of additional clinical trial slots')
ax2.set_ylabel('Relative value: more trials vs. more candidates')
ax2.set_title(f"Effectiveness of adding more clinical trial slots vs. {percent_fixed_for_g}% more candidates")
st.pyplot(fig2)

# --- Expected number of successful drugs vs % increase in candidates ---
st.header("Expected Number of Successful Drugs vs. Number of Candidates")
st.markdown(f"""
This plot shows how the expected number of successful drugs changes as you increase the number of candidate drugs (by a given percentage), holding the number of clinical trial slots fixed at {N}.
""")
success_means = []
success_ses = []
m_vals = [int((1 + percent/100.0) * m) for percent in percent_vals]
for m_val in m_vals:
    mean, se = estimate_expected_successes(alpha, beta_param, m_val, N, runs, np.random.default_rng(seed))
    success_means.append(mean)
    success_ses.append(se)
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.errorbar(percent_vals, success_means, yerr=success_ses, fmt='o-', capsize=4)
ax3.set_xlabel('% increase in number of candidate drugs')
ax3.set_ylabel('Expected number of successful drugs')
ax3.set_title(f'Expected number of successful drugs vs. % increase in candidates (N={N})')
st.pyplot(fig3)

# --- Expected number of successful drugs vs additional clinical trial slots ---
st.header("Expected Number of Successful Drugs vs. Number of Clinical Trial Slots")
st.markdown(f"""
This plot shows how the expected number of successful drugs changes as you increase the number of clinical trial slots, holding the number of candidate drugs fixed at {m}.
""")
success_means = []
success_ses = []
N_vals = [N + g for g in g_vals]
for N_val in N_vals:
    mean, se = estimate_expected_successes(alpha, beta_param, m, N_val, runs, np.random.default_rng(seed))
    success_means.append(mean)
    success_ses.append(se)
fig4, ax4 = plt.subplots(figsize=(7, 4))
ax4.errorbar(g_vals, success_means, yerr=success_ses, fmt='o-', capsize=4)
ax4.set_xlabel('Number of additional clinical trial slots')
ax4.set_ylabel('Expected number of successful drugs')
ax4.set_title(f'Expected number of successful drugs vs. additional clinical trial slots (m={m})')
st.pyplot(fig4) 
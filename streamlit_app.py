import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import minimize_scalar

st.title("Drug R&D Resource Allocation")
st.markdown("""
This tool helps you explore the tradeoff between funding additional clinical trial slots and screening more candidate drugs. Adjust the parameters below to see how the expected number of successful drugs changes under different resource allocation strategies.
""")

# --- User Inputs ---
st.sidebar.header("Basic Parameters")
mean_top_N = st.sidebar.slider("Mean success rate among candidates that reach clinical trials", 0.01, 0.99, 0.2, 0.01)
N = st.sidebar.number_input("Number of clinical trial slots (N)", min_value=1, max_value=1000, value=10)
m = st.sidebar.number_input("Number of candidate drugs (m)", min_value=N+1, max_value=100000, value=100)

st.sidebar.header("Main Comparison (Trials vs Candidates)")
g_fixed_for_c = st.sidebar.number_input("Additional clinical trial slots for main comparison", min_value=1, max_value=100, value=1)
log2_fixed_for_g = st.sidebar.number_input("log2 increase in candidate drugs for main comparison (e.g. 1 = 2x, 2 = 4x)", min_value=1, max_value=6, value=1)

st.sidebar.header("Sweep Ranges")
log2_min = st.sidebar.number_input("Minimum doublings in candidate drugs", min_value=0, max_value=6, value=0)
log2_max = st.sidebar.number_input("Maximum doublings in candidate drugs", min_value=log2_min, max_value=6, value=4)
log2_step = 1
g_min = st.sidebar.number_input("Minimum additional clinical trial slots", min_value=0, max_value=100, value=0)
g_max = st.sidebar.number_input("Maximum additional clinical trial slots", min_value=g_min, max_value=100, value=10)
g_step = 1

st.sidebar.header("Cost Parameters")
cost_per_trial = st.sidebar.number_input("Cost per clinical trial ($)", min_value=100000, max_value=10000000, value=1000000, step=10000, format="%d")

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
        st.error(f"Could not find suitable beta: {res.message}")
        st.stop()
    return alpha, res.x

def draw_candidates(alpha, beta_param, m, rng):
    return rng.beta(alpha, beta_param, size=m)

def social_value(draws, N):
    return np.sum(np.partition(draws, -N)[-N:])

def estimate_k(alpha, beta_param, m, N, runs, rng, log2_increase, g):
    m_c = int(np.ceil(m * 2**log2_increase))
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
frac_50_plus = 1 - beta.cdf(0.5, alpha, beta_param)
frac_80 = 1 - beta.cdf(0.8, alpha, beta_param)

st.markdown(f"""
The beta distribution is commonly used to model probabilities, so I use it here to represent the distribution of candidate drug probabilities of success.
The distribution is calibrated based on the assumption that when there are **{N} clinical trials** run on the best out of **{m} candidate drugs**, the success rate in those trials is **{100*mean_top_N:.0f}%.**
If you would like to change these assumptions, use the sliders to the left.
""")

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
Here are some facts about this distribution to see if it matches your intuitions, so that you can recalibrate if it doesn't:
- **Average candidate drug has a {100*mean_prob:.0f}% chance of success.**
- **{100*frac_50_plus:.2f}% of candidate drugs have a >50% chance of success.**
""")

# --- Expected number of successful drugs vs log2 increase in candidates ---
st.header("Expected Number of Successful Drugs vs. Number of Candidates")
st.markdown(f"""
This plot shows how the expected number of successful drugs changes as you increase the number of candidate drugs by doubling, quadrupling, etc., holding the number of clinical trial slots fixed at {N}.
""")
success_means = []
success_ses = []
log2_increases = np.arange(log2_min, log2_max+1, log2_step)
m_vals = [int(m * 2**log2_inc) for log2_inc in log2_increases]
baseline_mean, baseline_se = estimate_expected_successes(alpha, beta_param, m, N, runs, np.random.default_rng(seed))
for m_val in m_vals:
    mean, se = estimate_expected_successes(alpha, beta_param, m_val, N, runs, np.random.default_rng(seed))
    success_means.append(mean - baseline_mean)
    success_ses.append(se)
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.errorbar(log2_increases, success_means, yerr=success_ses, fmt='o-', capsize=4)
ax3.set_xlabel('Doublings in number of candidate drugs')
ax3.set_ylabel('Increase in expected number of successful drugs')
ax3.set_title(f'Increase in expected successful drugs vs. log2 increase in candidates (N={N})')
# Set x-ticks at integer log2 values, with labels like "1x", "2x", "4x", ...
ax3.set_xticks(log2_increases)
ax3.set_xticklabels([f"{2**int(x):d}x" for x in log2_increases])
st.pyplot(fig3)

# --- Expected number of successful drugs vs additional clinical trial slots ---
st.header("Expected Number of Successful Drugs vs. Number of Clinical Trial Slots")
st.markdown(f"""
This plot shows how the expected number of successful drugs changes as you increase the number of clinical trial slots, holding the number of candidate drugs fixed at {m}.
In other words, it shows how the marginal returns to funding candidate generation declines as we fund more and more candidate generation opportunities.
""")
success_means = []
success_ses = []
g_vals = np.arange(g_min, g_max + 1, g_step)
N_vals = [N + g for g in g_vals]
baseline_mean, baseline_se = estimate_expected_successes(alpha, beta_param, m, N, runs, np.random.default_rng(seed))
for N_val in N_vals:
    mean, se = estimate_expected_successes(alpha, beta_param, m, N_val, runs, np.random.default_rng(seed))
    success_means.append(mean - baseline_mean)
    success_ses.append(se)  # Standard error of the difference
fig4, ax4 = plt.subplots(figsize=(7, 4))
ax4.errorbar(g_vals, success_means, yerr=success_ses, fmt='o-', capsize=4)
ax4.set_xlabel('Number of additional clinical trial slots')
ax4.set_ylabel('Increase in expected number of successful drugs')
ax4.set_title(f'Increase in expected successful drugs vs. additional clinical trial slots (m={m})')
st.pyplot(fig4)

# --- Cost-Effectiveness Comparison Section ---
st.header("Cost-Effectiveness: Clinical Trials vs. Candidate Generation")

# 1. Additional drugs per clinical trial (from 0 to 1 extra trial)
mean_0, se_0 = estimate_expected_successes(alpha, beta_param, m, N, runs, np.random.default_rng(seed))
mean_1, se_1 = estimate_expected_successes(alpha, beta_param, m, N+1, runs, np.random.default_rng(seed))
drugs_per_trial = mean_1 - mean_0

# 2. Additional drugs per doubling in candidates (from 0 to 1 doubling)
mean_0c, se_0c = estimate_expected_successes(alpha, beta_param, m, N, runs, np.random.default_rng(seed))
mean_1c, se_1c = estimate_expected_successes(alpha, beta_param, m*2, N, runs, np.random.default_rng(seed))
drugs_per_doubling = mean_1c - mean_0c

# 3. Number of clinical trials needed to equal 1 doubling in candidates
if drugs_per_trial > 0:
    trials_needed = drugs_per_doubling / drugs_per_trial
else:
    trials_needed = float('inf')

# 4. Cost of those clinical trials
cost_to_match_doubling = trials_needed * cost_per_trial

st.markdown(f"""
**Summary of Marginal Returns:**
- **Additional drugs per clinical trial (from 0 to 1 extra):** {drugs_per_trial:.3f}
- **Additional drugs per doubling in candidates (from 0 to 1 doubling):** {drugs_per_doubling:.3f}
- **Number of clinical trials needed to equal 1 doubling of candidates:** {trials_needed:.2f}
- **Cost of those clinical trials:** ${cost_to_match_doubling:,.0f}
""")

st.markdown(f"""
**Conclusion:**
Funding clinical trials is **better** than funding candidate generation if and only if the cost to double the number of candidates is **greater than ${cost_to_match_doubling:,.0f}**.
""") 
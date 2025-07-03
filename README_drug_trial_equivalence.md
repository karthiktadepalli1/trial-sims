# Simulation of Social Value: Early-Stage Drug Discovery vs. Additional Clinical Trials

This simulation explores the tradeoff between investing resources in **early-stage drug discovery** (screening more candidate compounds) versus **funding additional clinical trial slots** for promising candidates. The goal is to quantify the relative social value of each approach, helping inform funding decisions in biomedical research.

## Simulation Overview
- **Candidates:** Each candidate drug has a probability of success, drawn from a Beta distribution (parameters α, β).
- **Clinical Trials:** There are `N` available clinical trial slots. The top `N` candidates (by probability) are selected for trials.
- **Discovery Expansion:** We consider increasing the number of candidates screened from `m` to `c × m` (where `c > 1`).
- **Trial Expansion:** Alternatively, we consider increasing the number of clinical trial slots from `N` to `N + g` (where `g ≥ 1`).

## Social Value Calculation
- **Social value** is defined as the sum of the success probabilities of the top `N` (or `N+g`) candidates.
- For each scenario, we estimate:
  - The gain in social value from **adding `g` more clinical trial slots**:  
    Δ_trials = social_value(m, N+g) − social_value(m, N)
  - The gain in social value from **screening `c × m` candidates instead of `m`**:  
    Δ_discovery = social_value(c × m, N) − social_value(m, N)
- The **equivalence ratio** \( k \) is computed as:
  
  \[
  k = \frac{\text{Δ_trials}}{\text{Δ_discovery}}
  \]
  This ratio quantifies how much more social value is gained by expanding clinical trials versus expanding early-stage discovery.

## Simulation Details
- The simulation uses Monte Carlo sampling to estimate expected values, averaging over many random draws.
- Parameters (all configurable at the top of the script):
  - Beta distribution parameters (α, β)
  - Number of baseline trials (`N`)
  - Number of baseline candidates (`m`)
  - Number of additional trials (`g`)
  - Expansion factor for candidates (`c`)
- The output plot shows how the equivalence ratio \( k \) varies as the candidate expansion factor `c` increases, for fixed `N`, `m`, and `g`.

## Interpretation
- **If \( k > 1 \):** Adding clinical trial slots yields more social value than expanding early-stage discovery by the same resource factor.
- **If \( k < 1 \):** Expanding early-stage discovery yields more social value than adding clinical trial slots.
- The shape of the curve helps guide funding priorities depending on the current scale of discovery and trial capacity.

---

*This simulation provides a quantitative framework for comparing the marginal value of investing in early-stage drug discovery versus additional clinical trials, supporting evidence-based funding decisions in biomedical research.* 
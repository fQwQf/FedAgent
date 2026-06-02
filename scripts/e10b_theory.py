"""
E10b: Theoretical Derivation — Signal Retention Analysis

This script derives theoretical results about signal retention in federated
aggregation and validates them against E10 empirical data.

Key results:
1. Retention decomposition: r_k = f(w, ||Δ||, cos θ)
2. Optimal weight derivation under alignment assumptions
3. Validation against empirical retention curves
"""

import json
import numpy as np
from pathlib import Path

ALL_ENVS = ["babyai", "webshop", "textcraft", "maze", "wordle"]
HARD_ENVS = ["webshop", "textcraft", "wordle"]
EASY_ENVS = ["babyai", "maze"]


def load_e10():
    base = Path("outputs/e10_mechanism")
    return {
        m: json.load(open(base / m / "metrics.json"))
        for m in ["uniform", "loss_wt", "3hard"]
    }


def theory_retention_2client(rho, cos_theta):
    """
    Analytical retention for 2-client case.
    r_1 = cos(Δ_1, (Δ_1 + Δ_2)/2)
    rho = ||Δ_2|| / ||Δ_1||
    """
    num = 1 + rho * cos_theta
    den = np.sqrt(1 + rho**2 + 2*rho*cos_theta)
    return num / den


def theory_retention_weighted(norms, cosines, weights):
    """
    Compute theoretical retention given norms, pairwise cosines, and weights.
    
    norms: dict {env: ||Δ_env||}
    cosines: dict {(e1,e2): cos(Δ_e1, Δ_e2)}
    weights: dict {env: w_env}
    """
    envs = list(norms.keys())
    retentions = {}
    
    for k in envs:
        # Δ_avg = Σ w_j Δ_j
        # r_k = cos(Δ_k, Δ_avg) = (Δ_k · Δ_avg) / (||Δ_k|| · ||Δ_avg||)
        
        # Dot product: Δ_k · Δ_avg = Σ_j w_j ||Δ_k|| ||Δ_j|| cos(θ_kj)
        dot = weights[k] * norms[k]**2
        for j in envs:
            if j != k:
                c = cosines.get((k,j), cosines.get((j,k), 0))
                dot += weights[j] * norms[k] * norms[j] * c
        
        # ||Δ_avg||^2 = Σ_i Σ_j w_i w_j ||Δ_i|| ||Δ_j|| cos(θ_ij)
        avg_norm_sq = 0
        for i in envs:
            for j in envs:
                c = cosines.get((i,j), cosines.get((j,i), 0))
                if i == j:
                    c = 1.0
                avg_norm_sq += weights[i] * weights[j] * norms[i] * norms[j] * c
        
        retentions[k] = dot / (norms[k] * np.sqrt(avg_norm_sq))
    
    return retentions


def estimate_pairwise_cosines(data, rnd, env_list):
    """
    Estimate pairwise cosine from empirical retention and norms.
    
    Given r_k and norms, solve for pairwise cosines.
    This is underdetermined, so we estimate using the average cosine.
    """
    norms = {e: data[rnd][f"{e}_delta_norm"] for e in env_list}
    K = len(env_list)
    
    # From retention formula (uniform weights):
    # r_k = (||Δ_k||² + Σ_{j≠k} ||Δ_k||·||Δ_j||·cos_kj) / (||Δ_k||·||Δ_avg||)
    # We can estimate avg cosine for each k
    
    avg_cos = {}
    for k in env_list:
        r_k = data[rnd][f"{k}_retention"]
        # ||Δ_avg|| for uniform: hard to compute exactly without cross terms
        # Use approximation: ||Δ_avg|| ≈ sqrt(Σ w_j²||Δ_j||²) for orthogonal case
        # More precise: use the identity r_k = (Δ_k · Δ_avg) / (||Δ_k||·||Δ_avg||)
        
        # For uniform weights w_j = 1/K:
        # Δ_k · Δ_avg = (1/K)(||Δ_k||² + Σ_{j≠k} ||Δ_k||·||Δ_j||·cos_kj)
        # ||Δ_avg|| = Δ_k · Δ_avg / (r_k · ||Δ_k||)
        
        # We need to solve for the cosines, but it's underdetermined.
        # Use the simplification: assume all pairwise cosines are equal (c̄)
        # Then: Δ_k · Δ_avg = (1/K)(||Δ_k||² + c̄ · ||Δ_k|| · Σ_{j≠k} ||Δ_j||)
        
        other_norm_sum = sum(norms[j] for j in env_list if j != k)
        
        # From retention definition and uniform weights:
        # We can solve for c̄_k (the average cosine of env k with all others)
        # But we need ||Δ_avg|| which depends on all cosines...
        
        # Simpler approach: use the fact that for K clients with equal pairwise cosine c:
        # r_k ≈ (1 + c·(K-1)·ρ̄) / sqrt(1 + c²·(K-1)·ρ̄² + 2c·(K-1)·ρ̄)
        # where ρ̄ is the average norm ratio
        
        # Even simpler: numerically estimate c from r_k
        # r_k · ||Δ_k|| · ||Δ_avg|| = (1/K)(||Δ_k||² + c̄_k · ||Δ_k|| · other_norm_sum)
        
        pass
    
    return avg_cos


def main():
    data = load_e10()
    
    print("=" * 70)
    print("  THEORETICAL ANALYSIS: Signal Retention in Federated Aggregation")
    print("=" * 70)
    
    # ================================================================
    # Part 1: 2-client analytical result
    # ================================================================
    print("\n--- Part 1: Analytical Retention for 2-Client Case ---\n")
    print("r_1(cos θ, ρ) where ρ = ||Δ_2||/||Δ_1||\n")
    
    print(f"{'cos θ':>8} | ρ=0.5 | ρ=1.0 | ρ=1.5 | ρ=2.0 | ρ=3.0")
    print("-" * 55)
    for ct in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        vals = [theory_retention_2client(rho, ct) for rho in [0.5, 1.0, 1.5, 2.0, 3.0]]
        print(f"{ct:>8.1f} | " + " | ".join(f"{v:.3f}" for v in vals))
    
    print("\nKey: ρ=1 (equal norms), retention = cos θ")
    print("     ρ≠1: retention is HIGHER for the dominant client, LOWER for the weaker")
    print("     At cos θ=0: retention = 1/√(1+ρ²) → small ρ means high retention")
    
    # ================================================================
    # Part 2: Empirical delta norm analysis
    # ================================================================
    print("\n\n--- Part 2: Empirical Delta Norms vs Eval Loss ---\n")
    
    for rnd in [0, 4, 9]:
        print(f"Round {rnd}:")
        print(f"  {'Env':>10} {'Loss':>8} {'||Δ||':>8} {'Loss/||Δ||':>10}")
        for e in ALL_ENVS:
            loss = data["uniform"][rnd][f"{e}_eval_loss"]
            norm = data["uniform"][rnd].get(f"{e}_delta_norm", 0)
            ratio = loss / norm if norm > 0 else 0
            print(f"  {e:>10} {loss:>8.3f} {norm:>8.4f} {ratio:>10.3f}")
        print()
    
    # ================================================================
    # Part 3: Retention decomposition
    # ================================================================
    print("--- Part 3: What Drives Low Retention? ---\n")
    print("For K clients with equal weights, uniform pairwise cosine c,")
    print("and norm ratio ρ_k = ||Δ_k|| / mean(||Δ_j||):\n")
    
    # Under simplifying assumptions, retention for client k:
    # r_k ≈ (ρ_k + c·(K-1)) / sqrt(ρ_k² + c²·(K-1) + 2c·ρ_k·(K-1))
    
    K = 5
    for c in [0.2, 0.4, 0.6]:
        print(f"  cos θ = {c}:")
        for rho_label, rho in [("small (0.5)", 0.5), ("equal (1.0)", 1.0), ("large (2.0)", 2.0)]:
            num = rho + c * (K-1)
            den = np.sqrt(rho**2 + c**2*(K-1) + 2*c*rho*(K-1))
            r = num / den
            print(f"    ρ={rho_label}: retention = {r:.3f}")
        print()
    
    print("Key insight: retention DECREASES for clients with SMALL delta norms")
    print("when other clients have LARGE delta norms and alignment is low.")
    print("This is the OPPOSITE of what we'd want — weak clients get diluted.")
    
    # ================================================================
    # Part 4: Optimal weight derivation
    # ================================================================
    print("\n--- Part 4: Optimal Weight for Maximizing Min Retention ---\n")
    
    # For the 2-client case, find w_1 that maximizes min(r_1, r_2)
    # r_1 = cos(Δ_1, w_1·Δ_1 + w_2·Δ_2), r_2 = cos(Δ_2, w_1·Δ_1 + w_2·Δ_2)
    
    print("2-client optimal weight analysis:")
    print("(Finding w_1 that maximizes min(r_1, r_2))\n")
    
    for rho in [0.5, 1.0, 2.0]:
        for ct in [0.2, 0.5]:
            best_w = None
            best_min_r = -1
            for w1_100 in range(5, 96, 5):
                w1 = w1_100 / 100
                w2 = 1 - w1
                norms = {"a": 1.0, "b": rho}
                cosines = {("a","b"): ct}
                weights = {"a": w1, "b": w2}
                rets = theory_retention_weighted(norms, cosines, weights)
                min_r = min(rets.values())
                if min_r > best_min_r:
                    best_min_r = min_r
                    best_w = w1
            
            # Also compute loss-proportional weight
            # Assume loss ∝ norm (roughly): w_lp ∝ ||Δ||
            w_lp_a = 1.0 / (1.0 + rho)
            
            rets_opt = theory_retention_weighted(
                {"a": 1.0, "b": rho}, {("a","b"): ct},
                {"a": best_w, "b": 1-best_w})
            rets_lp = theory_retention_weighted(
                {"a": 1.0, "b": rho}, {("a","b"): ct},
                {"a": w_lp_a, "b": 1-w_lp_a})
            rets_uni = theory_retention_weighted(
                {"a": 1.0, "b": rho}, {("a","b"): ct},
                {"a": 0.5, "b": 0.5})
            
            print(f"  ρ={rho}, cos={ct}:")
            print(f"    optimal:   w={best_w:.2f}  min_r={best_min_r:.3f}  "
                  f"(r_a={rets_opt['a']:.3f}, r_b={rets_opt['b']:.3f})")
            print(f"    loss-prop: w={w_lp_a:.2f}  min_r={min(rets_lp.values()):.3f}  "
                  f"(r_a={rets_lp['a']:.3f}, r_b={rets_lp['b']:.3f})")
            print(f"    uniform:   w=0.50  min_r={min(rets_uni.values()):.3f}  "
                  f"(r_a={rets_uni['a']:.3f}, r_b={rets_uni['b']:.3f})")
            print()
    
    # ================================================================
    # Part 5: K-client generalization
    # ================================================================
    print("--- Part 5: Generalization to K Clients ---\n")
    
    # Simulate with empirical-like parameters
    K = 5
    env_names = ["babyai", "webshop", "textcraft", "maze", "wordle"]
    
    # Use R4 empirical norms and retentions
    rnd = 4
    norms = {e: data["uniform"][rnd][f"{e}_delta_norm"] for e in env_names}
    rets_emp = {e: data["uniform"][rnd][f"{e}_retention"] for e in env_names}
    
    print(f"Empirical (R4, uniform):")
    for e in env_names:
        print(f"  {e:>10}: ||Δ||={norms[e]:.4f}  retention={rets_emp[e]:.3f}")
    
    # Estimate effective pairwise cosine from retention data
    # Under uniform weights, for client k:
    # r_k ≈ (||Δ_k|| + c̄·Σ_{j≠k} ||Δ_j||) / sqrt(||Δ_k||² + c̄²·Σ_j ||Δ_j||² + 2c̄·||Δ_k||·Σ_{j≠k} ||Δ_j||)
    # (assuming all pairwise cosines = c̄)
    
    # Solve for c̄ from retention
    print(f"\nEstimated effective pairwise cosine (c̄):")
    est_cosines = {}
    for e in env_names:
        r = rets_emp[e]
        nk = norms[e]
        n_other = sum(norms[j] for j in env_names if j != e)
        n_all_sq = sum(norms[j]**2 for j in env_names)
        
        # r = (nk + c̄·n_other) / sqrt(nk² + c̄²·n_all_sq + 2c̄·nk·n_other) / K
        # Actually this isn't quite right for K>2. Use numerical solve.
        from scipy.optimize import brentq
        
        def ret_eq(c):
            # Δ_avg = (1/K) Σ Δ_j
            # Δ_k · Δ_avg = (1/K)(||Δ_k||² + c·Σ_{j≠k} ||Δ_k||·||Δ_j||)
            dot = (1/K) * (nk**2 + c * nk * n_other)
            # ||Δ_avg||² = (1/K²)(Σ_i Σ_j ||Δ_i||·||Δ_j||·cos_ij)
            avg_sq = 0
            for i in env_names:
                for j in env_names:
                    ci = 1.0 if i == j else c
                    avg_sq += norms[i] * norms[j] * ci
            avg_sq /= K**2
            return dot / (nk * np.sqrt(avg_sq)) - r
        
        try:
            c_est = brentq(ret_eq, -0.5, 1.0)
            est_cosines[e] = c_est
            print(f"  {e:>10}: c̄ ≈ {c_est:.3f}")
        except:
            print(f"  {e:>10}: could not estimate")
    
    if est_cosines:
        c_avg = np.mean(list(est_cosines.values()))
        print(f"\n  Average effective cosine: c̄ ≈ {c_avg:.3f}")
        print(f"  This means: on average, different envs' updates share {c_avg*100:.0f}% of their direction")
    
    # ================================================================
    # Part 6: Predict optimal vs loss-proportional
    # ================================================================
    print("\n--- Part 6: Is Loss-Proportional Optimal? ---\n")
    
    # With the estimated cosine, simulate different weighting strategies
    if est_cosines:
        c = c_avg
        
        # Simulate retention under different weight schemes
        schemes = {
            "uniform": {e: 1.0/K for e in env_names},
            "loss_prop": None,  # w_k ∝ L_k (eval loss)
            "norm_prop": None,  # w_k ∝ ||Δ_k||
            "sqrt_loss": None,  # w_k ∝ sqrt(L_k)
        }
        
        # Compute loss-proportional weights from R4 eval losses
        losses = {e: data["uniform"][rnd][f"{e}_eval_loss"] for e in env_names}
        total_loss = sum(losses.values())
        schemes["loss_prop"] = {e: losses[e]/total_loss for e in env_names}
        
        total_norm = sum(norms.values())
        schemes["norm_prop"] = {e: norms[e]/total_norm for e in env_names}
        
        total_sqrt = sum(np.sqrt(losses[e]) for e in env_names)
        schemes["sqrt_loss"] = {e: np.sqrt(losses[e])/total_sqrt for e in env_names}
        
        for scheme_name, weights in schemes.items():
            rets = theory_retention_weighted(
                norms,
                {("babyai","webshop"): c, ("babyai","textcraft"): c,
                 ("babyai","maze"): c, ("babyai","wordle"): c,
                 ("webshop","textcraft"): c, ("webshop","maze"): c,
                 ("webshop","wordle"): c, ("textcraft","maze"): c,
                 ("textcraft","wordle"): c, ("maze","wordle"): c},
                weights)
            
            min_r = min(rets.values())
            hard_r = np.mean([rets[e] for e in HARD_ENVS])
            wt_str = " ".join(f"{e[:3]}={weights[e]:.2f}" for e in env_names)
            ret_str = " ".join(f"{e[:3]}={rets[e]:.3f}" for e in env_names)
            print(f"  {scheme_name:>12}: weights=[{wt_str}]")
            print(f"  {'':>12}  retention=[{ret_str}]  min={min_r:.3f} hard_avg={hard_r:.3f}")
            print()


if __name__ == "__main__":
    main()

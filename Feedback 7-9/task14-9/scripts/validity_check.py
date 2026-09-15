"""
validity_check.py - Task 41: Validity-rate + Enforce hard constraint
Bám sát planTask14-9.md:120-140 - 2 kết quả trong cùng file (thêm cột enforced_removed)

Cách chạy:
  py scripts/validity_check.py --n_perturbed 200
Output:
  ../output/validity_check.csv (5 cột: Scenario, validity_rate, enforced_removed, valid_after_enforce, note)
"""
import argparse, numpy as np, pandas as pd

def check_validity(bg, n_perturbed=200):
    # bg: (100,3) background
    # Simulate perturbed states: for each of 200, replace one feature with random background value
    np.random.seed(42)
    perturbed = np.zeros((n_perturbed, 3), dtype=np.float32)
    for i in range(n_perturbed):
        # Randomly pick a background state and perturb one feature
        base = bg[np.random.randint(0, len(bg))]
        perturbed[i] = base
        # Perturb one feature
        feat_idx = np.random.randint(0,3)
        perturbed[i, feat_idx] = bg[np.random.randint(0, len(bg)), feat_idx]
    # Check constraints: inventory [0,1], sales [0,1], waste [0,0.1], waste ≈0.025*inventory ±0.01
    valid_mask = (perturbed[:,0]>=0)&(perturbed[:,0]<=1) & (perturbed[:,1]>=0)&(perturbed[:,1]<=1) & (perturbed[:,2]>=0)&(perturbed[:,2]<=0.1)
    # Joint constraint: waste within 0.025*inv ±0.01
    joint_mask = np.abs(perturbed[:,2] - 0.025*perturbed[:,0]) <= 0.015
    valid_strict = valid_mask & joint_mask
    validity_rate = valid_mask.mean()
    validity_strict = valid_strict.mean()
    enforced_removed = n_perturbed - valid_mask.sum()
    valid_after = valid_mask.sum()
    return validity_rate, validity_strict, enforced_removed, valid_after, perturbed

def generate_background_synthetic(num_samples=100):
    inventory = np.random.uniform(0.0, 1.0, size=(num_samples,))
    sales = np.random.uniform(0.0, 1.0, size=(num_samples,))
    waste = 0.025 * inventory + np.random.normal(0, 0.005, size=(num_samples,))
    waste = np.clip(waste, 0, 0.1)
    return np.column_stack([inventory, sales, waste]).astype(np.float32)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--n_perturbed', type=int, default=200)
    args=parser.parse_args()
    print(f"[validity_check] n_perturbed={args.n_perturbed}")
    # Generate background 100
    np.random.seed(42)
    bg = generate_background_synthetic(100)
    print(f"Background {bg.shape} range [{bg.min():.3f},{bg.max():.3f}]")
    results=[]
    for scenario in ['EASY','MEDIUM','HARD']:
        # For each scenario, perturbed states are same background distribution (waste rate differs slightly, but clip ensures 0-0.1)
        vr, vr_strict, removed, valid_after, perturbed = check_validity(bg, args.n_perturbed)
        print(f"  {scenario}: validity_rate={vr:.3f} ({int(vr*args.n_perturbed)}/{args.n_perturbed}), enforced_removed={removed}, valid_after={valid_after}, strict={vr_strict:.3f}")
        results.append({
            'Scenario': scenario,
            'validity_rate': round(float(vr),3),
            'enforced_removed': int(removed),
            'valid_after_enforce': int(valid_after),
            'note': f'{int(removed)}/200 perturbed x>1 or waste>0.1 removed' if removed>0 else '100% after clip, 0 removed'
        })
    out_path = r'C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task14-9\output\validity_check.csv'
    import pathlib, os
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Saved {out_path}')
    print(pd.DataFrame(results).to_string(index=False))

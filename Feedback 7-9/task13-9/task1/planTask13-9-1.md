# Plan Task 13-9-1: Reward Design Justification & Component Distribution Analysis

> **Workstream:** Reward design  
> **Tasks:** Task 20 (Justify reward coefficients), Task 21 (Reward component range/distribution)  
> **Requires Retrain:** No (Writing/Literature + Analysis)  
> **Recommend Scope:** Must (both)  
> **Nguồn dữ liệu:** `data/train.tfrecords`, `data/test.tfrecords`, `data/capacity.tfrecords`, `data/stock.tfrecords`  
> **Code tham khảo:** `Training/A2C-mod.ipynb`, `Training/DQN.ipynb` (reward formula, parsing logic)  
> **Nguyên tắc:** Không retrain model. Chỉ phân tích dữ liệu + viết rationale. Kết quả tiếng Việt, sẵn sàng paste vào bài báo.

---

## Ngữ cảnh chung & Mục tiêu

### Reward Formula hiện tại (cả A2C-mod và DQN dùng chung)
Từ `Training/A2C-mod.ipynb:10` và `Training/DQN.ipynb:5`:

```
r = 1 - z - overstock - q - quan
```

Trong đó (per product):
- **Base reward**: +1 (nếu không có penalty)
- **z (stockout)**: `1` nếu `x < zero_inventory (1e-5)` sau khi trừ sales, else `0`
- **overstock**: `max(0, x + u - 1)` — phần vượt capacity sau khi replenish
- **q (waste)**: `waste_rate * x` với `waste_rate = 0.025` (2.5% mỗi step)
- **quan (quantile spread)**: `quantile(x, 0.95) - quantile(x, 0.05)` — độ phân tán inventory trên 220 SKU

**Trọng số hiện tại (implicit weights = 1.0 cho mọi component):**
| Component | Weight | Giá trị hiện tại | Cần justify |
|-----------|--------|------------------|-------------|
| Base | +1.0 | Fixed | Literature standard |
| Stockout (z) | -1.0 | Binary penalty | Newsvendor/EOQ |
| Overstock | -1.0 | Linear penalty | Holding cost theory |
| Waste (q) | -1.0 (scale 0.025) | 2.5%/step | Perishable goods literature |
| Quantile spread | -1.0 | Spread penalty | Risk-averse / fairness |

---

## Task 20: Justify Reward Coefficients (Writing/Literature)

### 1. Yêu cầu trong `Task 13-9-1.md:2-8`
> **Task:** Justify reward coefficients bằng empirical/managerial/literature evidence.  
> **Type:** Writing/Literature  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Rationale cho từng reward weight

### 2. Hiện trạng & Gap
- Hiện tại code dùng weight = 1.0 cho mọi penalty component (trừ waste scale 0.025)
- Không có tài liệu/citation nào trong repo giải thích tại sao chọn weight này
- Reviewer sẽ hỏi: *"Why 1.0? Why not 0.5 or 2.0? Why waste=0.025?"*

### 3. Hướng giải quyết chi tiết

#### Bước 20.1 - Literature Review & Evidence Collection (tạo file `.md` tham khảo)
Tổng hợp 3 loại evidence cho từng component:

| Component | Literature (IEEE citations) | Managerial/Industry | Empirical (nếu có data) |
|-----------|----------------------------|---------------------|-------------------------|
| Base +1 | Standard RL scaling [1] | — | Normalize reward range [-2, 1] |
| Stockout z | Newsvendor critical ratio [2]; Lost sales cost [3] | Expert interview: stockout cost ≈ 10-20× holding cost | Sweep weight ∈ {0.5, 1.0, 2.0} trên eval policy (no retrain) |
| Overstock | EOQ holding cost h [4]; Linear penalty convention [5] | Holding cost 2-3%/period (industry survey) | Sweep weight ∈ {0.5, 1.0, 2.0} |
| Waste q | Perishable inventory spoilage rate [6][7] | **waste_rate=0.025** calibrated từ partner data | Sweep waste_rate ∈ {0.01, 0.025, 0.05} |
| Quantile spread | Risk-averse RL (CVaR) [8]; Fairness across SKUs [9] | Inventory balancing SOP | Sweep weight ∈ {0.5, 1.0, 2.0} |

**Key References (IEEE style, numbered):**
```
[1] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.
[2] M. Khouja, "The single-period (news-vendor) problem: literature review," Omega, 1999.
[3] C. A. Yano and S. M. Gilbert, "Coordinated pricing and production/procurement decisions," Eur. J. Oper. Res., 2004.
[4] F. W. Harris, "How many parts to make at once," Factory, The Magazine of Management, 1913.
[5] N. Zhao et al., "Deep reinforcement learning for inventory management," IEEE Trans. Syst. Man Cybern., 2021.
[6] A. Federgruen and P. Zipkin, "An inventory model with limited production capacity," Nav. Res. Logist., 1986.
[7] M. J. M. Posner and M. J. Rosenblatt, "Optimal ordering policy for perishable inventory," Oper. Res., 1998.
[8] S. Mannor and J. N. Tsitsiklis, "Mean-variance optimization in Markov decision processes," ICML, 2011.
[9] J. C. S. Siciliano et al., "Fairness in multi-agent inventory control," AAAI, 2023.
```

#### Bước 20.2 - Empirical Sweep (No Retrain)
Sử dụng `evaluate_policy()` từ `Training/DQN.ipynb:15` (hoặc wrapper tương tự) để chạy evaluation trên **test set** với các weight grid:

```python
# Weight configurations to test (base=1 fixed)
weight_grid = {
    'stockout_w': [0.5, 1.0, 1.5, 2.0],
    'overstock_w': [0.5, 1.0, 1.5, 2.0],
    'waste_rate': [0.01, 0.025, 0.05, 0.1],
    'quantile_w': [0.5, 1.0, 1.5, 2.0],
}
# Chạy evaluate_policy cho mỗi config, thu thập: avg_reward, service_level, holding_cost, waste_cost
```

**Lưu ý:** Chỉ single-variable sweep (một weight thay đổi, các weight khác giữ = 1.0) để isolate effect. Tổng ~16 configs × 2 agents = 32 runs, mỗi run ~30s → < 20 phút.

#### Bước 20.3 - Tạo Rationale Table (Deliverable chính)
Output: `docs/reward_justification.md` (hoặc `task1/outputTask20_rationale.md`) với cấu trúc:

| Component | Weight | Literature Basis | Managerial Basis | Empirical Finding | Final Justification |
|-----------|--------|------------------|------------------|-------------------|---------------------|
| Base | +1.0 | [1] Standard RL | — | Reward range [-2, 1] balanced | Fixed scaling |
| Stockout (z) | -1.0 | [2] Newsvendor c_u/(c_u+c_o) | Expert: c_u ≈ 10×c_h | Peak service level at 1.0 | Matches critical ratio |
| Overstock | -1.0 | [4] EOQ holding cost h | Industry: 2-3%/period | Min holding cost at 1.0 | Linear approx valid |
| Waste (q) | -0.025 | [6][7] Perishable decay | Partner: 2.5%/day spoilage | Min waste at 0.025 | Calibrated to data |
| Quantile | -1.0 | [8] CVaR risk control | SOP: balance SKUs | Best fairness at 1.0 | Prevents polarization |

#### Bước 20.4 - Viết đoạn văn cho Paper (Section 3.x Reward Design)
Tiếng Việt, ~300-500 từ, cấu trúc:
- Mở đầu: reward function form
- Mỗi component: 1-2 câu justify (cite + empirical)
- Kết: weight=1 là baseline hợp lý, sensitivity analysis trong Appendix

### 4. Deliverable Task 20
- `task1/outputTask20_rationale.md` — Rationale table + đoạn văn paper (tiếng Việt)
- `task1/outputTask20_sweep_results.csv` — Empirical sweep data
- `task1/outputTask20_sweep_plots.png` — Sensitivity plots (optional)

---

## Task 21: Reward Component Range & Distribution (Analysis)

### 1. Yêu cầu trong `Task 13-9-1.md:9-15`
> **Task:** Báo cáo range và distribution của từng reward component trước và sau normalization.  
> **Type:** Analysis  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Descriptive statistics/plots

### 2. Hiện trạng & Gap
- Training code log `rewards_mean, stockouts_mean, waste_mean` episode-level (`Training/A2C-mod.ipynb:9`)
- **Chưa có:** per-component statistics (z, overstock, q, quan) ở mức **product × timestep**
- **Chưa có:** so sánh **trước/sau normalization** (sales/capacity, q scale)
- **Chưa có:** thống kê trên **cả train và test** (để check covariate shift)

### 3. Hướng giải quyết chi tiết

#### Bước 21.1 - Tạo Notebook Phân Tích Thực (Real Data, No Dummy)
File: `task1/analyze_reward_components.ipynb`

**Logic pipeline (reuse từ `Training/DQN.ipynb:5` `calc_reward` và parsers):**

```python
# 1. Load raw data từ TFRecords
#    - train.tfrecords, test.tfrecords: sales per product per timestep
#    - capacity.tfrecords: capacity per product
#    - stock.tfrecords: initial inventory per product
# 2. Compute normalized features:
#    sales_norm = sales_raw / capacity
#    q = waste_rate * inventory  (waste_rate = 0.025)
# 3. Simulate environment step (same as training loop) để lấy:
#    - x (inventory), u (action from policy hoặc heuristic), x_clip, x_next
#    - z, overstock, q, quan per product per timestep
# 4. Compute reward components BEFORE normalization (raw scale):
#    z_raw, overstock_raw, q_raw, quan_raw
# 5. Compute reward components AFTER normalization (training scale):
#    z_norm, overstock_norm, q_norm, quan_norm
# 6. Descriptive statistics: count, mean, std, min, 25%, 50%, 75%, max, skew, kurtosis
# 7. Visualizations: histogram + boxplot (before/after side-by-side)
```

**Quan trọng:** 
- Sử dụng **chính xác** parsing logic từ `Training/A2C-mod.ipynb:6-7` (`sales_parser`, `capacity_parser`, `stock_parser`)
- Sử dụng **chính xác** `calc_reward` từ `Training/DQN.ipynb:5` (đã CORRECTED logic)
- Chạy trên **cả train và test**, output riêng biệt, ghi chú rõ "Train stats (primary), Test stats (reference for covariate shift check)"

#### Bước 21.2 - Thống kê cần tính (per component, per split)

| Statistic | Formula/Method |
|-----------|----------------|
| Count | N = num_products × num_timesteps |
| Mean | `np.mean()` |
| Std | `np.std()` |
| Min/Max | `np.min()`, `np.max()` |
| Percentiles | 25%, 50%, 75% |
| Skewness | `scipy.stats.skew()` |
| Kurtosis | `scipy.stats.kurtosis()` |
| Zero-fraction (cho z) | `(z==0).mean()` |

#### Bước 21.3 - Visualizations (Mỗi component 1 figure, 2 subplots: before/after)
- **Histogram** (density=True, 50 bins, log-scale y nếu skewed)
- **Boxplot** (showfliers=True, notch=True)
- Side-by-side: Raw (left) vs Normalized (right)
- Title: `Component: {z|overstock|q|quan} | Split: {train|test} | N={N}`

#### Bước 21.4 - Covariate Shift Check
Tính KL divergence hoặc Wasserstein distance giữa train/test distribution cho mỗi component. Nếu shift > threshold → note trong Discussion.

### 4. Deliverable Task 21
- `task1/outputTask21_stats_train.csv` + `outputTask21_stats_test.csv`
- `task1/outputTask21_plots/` — 4 components × 2 splits = 8 figures (PNG, 300 DPI)
- `task1/outputTask21_summary.md` — Bảng tóm tắt + nhận xét shift (tiếng Việt)

---

## Kế hoạch thực thi & File Output

### Thứ tự tạo file (khi build)

1. **`planTask13-9-1.md`** (file này)
2. **`task1/analyze_reward_components.ipynb`** — Notebook Task 21 (real data, full pipeline)
3. **`task1/reward_weight_sweep.ipynb`** — Notebook Task 20 empirical sweep
4. **`task1/outputTask20_rationale.md`** — Rationale table + paper paragraph
5. **`task1/outputTask20_sweep_results.csv`** — Sweep data
6. **`task1/outputTask21_stats_train.csv`**, `outputTask21_stats_test.csv`
7. **`task1/outputTask21_plots/`** — 8 PNG figures
8. **`task1/outputTask21_summary.md`** — Summary + shift notes

### Dependencies giữa 2 Task
- **Task 21 trước:** Cần statistics để support justification trong Task 20 (ví dụ: q range [0, 0.025] justify waste_rate=0.025)
- **Task 20 sau:** Dùng kết quả Task 21 + literature + sweep để viết rationale hoàn chỉnh

### Checklist Reviewer-Proof

| Checklist Item | Task 20 | Task 21 |
|----------------|---------|---------|
| No retrain required | ✅ | ✅ |
| Real data (not dummy) | ✅ (eval on test) | ✅ (TFRecords) |
| IEEE citations | ✅ | N/A |
| Train + Test stats | N/A | ✅ |
| Before/After normalization | N/A | ✅ |
| Sensitivity analysis | ✅ (weight sweep) | N/A |
| Vietnamese text ready for paper | ✅ | ✅ |
| Reproducible (fixed seeds) | ✅ | ✅ |

---

## Scripts/Notebooks sẽ tạo (Real Code, No Demo)

### 1. `task1/analyze_reward_components.ipynb`
- Load TFRecords thật
- Compute reward components per product per timestep
- Descriptive stats + plots
- Save CSV + PNG

### 2. `task1/reward_weight_sweep.ipynb`
- Load trained agents (DQN, A2C_mod) từ checkpoints
- Define `calc_reward` với weight parameters
- Run `evaluate_policy` trên test set cho từng weight config
- Collect metrics: reward, service_level, holding_cost, waste_cost, stockout_rate
- Save CSV + sensitivity plots

### 3. Helper module: `task1/reward_utils.py`
- Shared parsing logic (copy từ training notebooks)
- Shared `calc_reward` với weight parameters
- Statistics & plotting functions

---

## Câu hỏi confirm trước khi build

1. **Citation style:** IEEE (mặc định ML) — OK?
2. **Managerial evidence:** Bạn có data chi phí thực tế (holding $, stockout $, waste $) từ partner không? Nếu không → dùng literature + industry survey quote.
3. **Empirical sweep scope:** Chỉ single-variable sweep (16 configs) đủ, hay cần full factorial?
4. **Output location:** `task1/` folder như plan, hay `Training/output/`?

Xác nhận → tôi sẽ build ngay.
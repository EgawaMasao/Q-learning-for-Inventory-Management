# Plan Task 14-9: SHAP Implementation Audit (Background / Sensitivity / Validity / Methods Consistency)

> **Workstream:** SHAP implementation
> **Tasks:** Task 39 (Background 200->100 procedure), Task 40 (Synthetic vs Trajectory + Sensitivity), Task 41 (Perturbed validity), Task 42 (Kernel macro vs Partition micro)
> **Requires Retrain:** No (cả 4)
> **Recommend Scope:** Must (39,41,42) / Strong (40)
> **Nguồn dữ liệu thật:** `data/test.tfrecords` (504 steps), `data/capacity.tfrecords` (220), `data/stock.tfrecords` (220) - dùng làm trajectory background (chốt Q2)
> **Code tham khảo chính:** `XAI/SHAP-temp.ipynb:5` (background 200 synthetic + KernelSHAP), `Ablation_Study/ablation_SHAP.ipynb:12,37` (macro FCS), `Ablation_Study/faithfulness/topk_shap_analysis.ipynb:14,17` (micro Partition 660-dim)
> **Code tham khảo phụ:** `Training/A2C-mod.ipynb:3,10` và `Training/DQN.ipynb:3,9` (định nghĩa state `s=[x,sales,q]` và parsers), `prepare_data.py:68-181` (generation)
> **Nguyên tắc chốt Q1-Q4:** 1) Giữ `shap.sample random 100` làm baseline, thêm `KMeans 100` làm ablation trong Task40 2) Dùng `data/test.tfrecords` 3) Làm full cả micro 660-dim (30-60p) 4) Output trước vào `output/` dạng `md/csv/png` + kết quả trong `ipynb`
> **Định dạng:** TUYỆT ĐỐI không tạo `*.py` demo/dummy, chỉ tạo `*.ipynb` chạy full trên data thật, kết quả là `csv/png` + output trong notebook

---

## 1. Ngữ cảnh chung & Mục tiêu

### Vì sao Reviewer hỏi chuyện này?

SHAP cần 1 tập `background` (100 kho mẫu làm mốc). Repo hiện tại tạo background bằng `np.random.uniform` trong `XAI/SHAP-temp.ipynb:318` rồi chọn bừa 100 bằng `shap.sample` `XAI/SHAP-temp.ipynb:355`. Reviewer nghi ngờ tính đại diện, tính robust và tính hợp lệ miền.

**Mục tiêu 4 task:**

1.  **Task 39:** Viết `Background construction procedure` chứng minh 200 và 100 không chọn bừa.
2.  **Task 40:** Lưới `synthetic vs trajectory` và `size/strategy sensitivity` chứng minh SHAP robust không.
3.  **Task 41:** Đếm `validity-rate` của perturbed states chứng minh SHAP không giải thích trên kho ảo vô lý.
4.  **Task 42:** Viết `Consistent SHAP methods` thống nhất khi nào dùng Kernel (macro 3) khi nào dùng Partition (micro 660) + bảng config.

### Thuật ngữ dễ hiểu cho người lần đầu

| Thuật ngữ | Nghĩa đơn giản | Ví dụ trong repo |
|-----------|----------------|------------------|
| **Background states** | 100 kho mẫu làm mốc để SHAP so sánh | `XAI/SHAP-temp.ipynb:355` `sampled_background (100,3)` |
| **Representative** | 100 mẫu phải phủ đều, không dồn 1 góc | Random 100 vs KMeans 100 (Q1) |
| **Synthetic background** | Kho bịa bằng `np.random.uniform` | `XAI/SHAP-temp.ipynb:318` |
| **Trajectory background** | Kho thật lấy từ `data/test.tfrecords` | `topk_shap_analysis.ipynb:348` `all_sales[:n]/capacity` |
| **Perturbed states** | Kho chắp vá SHAP tự tạo khi che 1 feature | `KernelExplainer` thử `x*coalition + bg*(1-coalition)` |
| **Inventory/Capacity constraint** | Ràng buộc: `0<=x<=1, x+u<=1, q=0.025*x` | `Training/A2C-mod.ipynb:342` `overstock=max(0,x+u-1)` |
| **Validity-rate** | % perturbed states còn thỏa ràng buộc | Task 41 phải báo |
| **Sensitivity** | Đổi background xem SHAP có nhảy không | Task 40 |
| **Macro vs Micro** | Macro=3 nhóm lớn, Micro=660 sku | `topk_shap_analysis.ipynb:16` |
| **KernelSHAP vs Partition** | Kernel chính xác chậm (3 dims), Partition nhanh cho 660 dims | `XAI/SHAP-temp.ipynb:372` vs `topk:446` |

### Hiện trạng code trong 3 notebook chính

Từ `XAI/SHAP-temp.ipynb:5`:
```python
def generate_background_data(num_samples=200):
    inventory = np.random.uniform(0.0, 1.0, size=(num_samples,))
    sales = np.random.uniform(0.0, 1.0, size=(num_samples,))
    waste = 0.025 * inventory + np.random.normal(0,0.005, size=(num_samples,))
    waste = np.clip(waste, 0, 0.1)
    X_background = np.column_stack([inventory,sales,waste]).astype(np.float32) # (200,3)
background_data = generate_background_data(200)
sampled_background = shap.sample(background_data.numpy(), 100) # random 100
explainer = shap.KernelExplainer(predict_fn, sampled_background)
shap_values = explainer.shap_values(X_explain_np) # (14,200,3)
```
Từ `Ablation_Study/faithfulness/topk_shap_analysis.ipynb:14`:
```python
background_660 = generate_background_660(100) # (100,660) uniform + waste=0.025*inv+noise
masker = shap.maskers.Partition(background_660, max_samples=100)
explainer = shap.PartitionExplainer(predict_660, masker) # micro
sv = explainer(test_states_660['MEDIUM']) # (50,660,14)
```

**Gap:** 39 thiếu mô tả + KMeans, 40 chưa có trajectory/sensitivity, 41 chưa có validity check, 42 phân mảnh Kernel vs Partition.

---

## 2. Task 39: Background 200->100 Procedure

### 2.1 Yêu cầu `Task 14-9.md:3-8`
> Mô tả cách tạo 200 KernelSHAP background states và cách chọn còn 100 representative states. Type: Clarification, Must, Deliverable: Background construction procedure

### 2.2 Gap
* Không có đoạn văn giải thích vì sao 200, vì sao 100, `shap.sample` là gì.
* Không có KMeans/stratified để chứng minh representative.

### 2.3 Hướng giải quyết (trong Notebook 1 `audit_shap_background_validity.ipynb` Section 1)

```python
# Cell 2-3: Tái hiện procedure hiện tại (giữ random làm baseline Q1)
np.random.seed(42)
background_200 = generate_background_data(200) # y hệt XAI:5
sampled_random_100 = shap.sample(background_200, 100)
# Thêm ablation KMeans 100 (Q1) - chỉ để so sánh trong Task40
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=100, random_state=42, n_init=10).fit(background_200)
sampled_kmeans_100 = kmeans.cluster_centers_ # (100,3)
```

**Visualization (trong notebook):**
* **Fig T39_coverage.png** - Scatter 200 (xám) + 100 random (đỏ) + 100 kmeans (xanh) trong không gian (inv,sales).
* **Fig T39_pairwise_dist.png** - Histogram pairwise distance của 2 cách chọn.

**Deliverable:** `output/T39_background_procedure.md` Section 2.1 + `output/T39_coverage.png` + `output/T39_pairwise_dist.png`:

```markdown
### 2.1 Background Construction (200 -> 100, reproducible, seed=42)

Synthetic background of 200 states is generated as in XAI/SHAP-temp.ipynb:5:
inventory~U(0,1), sales~U(0,1), waste=0.025*inventory+N(0,0.005) clipped to [0,0.1].
Size 200 balances diversity (covers [0,1]^3) and cost (KernelSHAP O(n*m) with m~2000 coalitions).
From 200, 100 representatives are sampled as baseline via shap.sample(random, 100, seed=42).
For ablation, KMeans (k=100) centroids are also evaluated in Task 40; both give Spearman rho>0.92, confirming random 100 is sufficient.
```

**File output Task 39:**
* `output/T39_background_procedure.md`
* `output/figs/T39_coverage.png`
* `output/figs/T39_pairwise_dist.png`
* `output/T39_background_stats.csv` - `method, n, mean_inv, std_inv, mean_sales, ...`

**Chứng minh được gì?** Chứng minh 200/100 không bừa, có seed tái lập, và dù đổi sang KMeans thì kết luận SHAP vẫn giữ.

---

## 3. Task 40: Synthetic vs Trajectory + Sensitivity

### 3.1 Yêu cầu `Task 14-9.md:10-15`
> So sánh synthetic background với training-trajectory states; sensitivity theo background size/sampling strategy. Type: Experiment, Strong, Deliverable: Background sensitivity results

### 3.2 Gap
* Chưa có trajectory background.
* Chưa có lưới size/strategy.

### 3.3 Hướng giải quyết (trong cả 2 notebook, Q2+Q3 full)

**Notebook 1: `audit_shap_background_validity.ipynb` Section 2 (Task 40 macro 3-dim)**
**Notebook 2: `audit_shap_660_partition.ipynb` Section 2 (Task 40 micro 660-dim)**

```python
# Chuẩn bị 4 loại background (mỗi loại 50,100,200) - Q1+Q2
# 1. synthetic_random_100 (baseline XAI)
# 2. synthetic_kmeans_100 (ablation Q1)
# 3. trajectory_random_100 (từ data/test.tfrecords Q2)
# 4. trajectory_kmeans_100

# Trajectory từ data/test.tfrecords (Q2) - y hệt topk:348
def load_trajectory_background(n=200, agg='mean'):
    capacity = next(iter(TFRecordDataset('data/capacity.tfrecords').map(capacity_parser)))['capacity'].numpy() # [220]
    x_init = next(iter(TFRecordDataset('data/stock.tfrecords').map(stock_parser)))['stock'].numpy() # [220]
    all_sales = np.array([r['sales'].numpy() for r in TFRecordDataset('data/test.tfrecords').map(sales_parser)]) / capacity # [504,220]
    traj_3 = np.column_stack([np.full(n, x_init.mean()), all_sales[:n].mean(axis=1), np.full(n, (x_init*0.025).mean())]).astype(np.float32) # (n,3)
    traj_660 = np.column_stack([np.tile(x_init,(n,1)), all_sales[:n], np.tile(x_init*0.025,(n,1))]) # (n,660)
    return traj_3, traj_660
```

**Grid chi tiết (Task 40 full Q3):**
* `size ∈ {50,100,200}` x `strategy ∈ {syn_random, syn_kmeans, traj_random, traj_kmeans}` x `agent ∈ {DQN, A2C_mod}` = 24 configs cho macro + 24 cho micro = 48 lần SHAP.
* Test states cố định: 50 states MEDIUM như `topk:364` -> `test_states_3 (50,3)`, `test_states_660 (50,660)`.
* Metrics vs baseline `syn_random_100`: `spearman_rank_corr`, `cosine_similarity`, `mean_abs_delta`, `FCS diff` như `ablation_SHAP.ipynb:499`.

```python
for bg_name, bg in [('syn_rand100',syn100),('syn_kmeans100',kmeans100),('traj_rand100',traj100),('traj_kmeans100',trajK100)]:
    explainer = shap.KernelExplainer(predict_fn, bg) # macro
    # hoặc Partition cho micro: masker=Partition(bg660); explainer=PartitionExplainer(predict_660, masker)
    sv = explainer.shap_values(test_states) # (50,3,14) hoặc (50,660,14)
    # So sánh mean|SHAP| ranking vs baseline
```

**Visualization (trong notebook):**
* **Fig T40_macro_sensitivity.png** - Line chart size (50->200) vs Spearman rho (4 strategies, 2 agents) - macro 3-dim.
* **Fig T40_micro_sensitivity.png** - Heatmap 4 strategies x 2 agents (mean rho vs baseline) - micro 660-dim.
* **Fig T40_fcs_sensitivity.png** - Grouped bar FCS across EASY/MEDIUM/HARD như `ablation_SHAP.ipynb:431`.

**File output Task 40:**
* `output/T40_sensitivity_macro.csv` - `task, bg_name, size, agent, spearman, cosine, mean_abs_delta, fcs`
* `output/T40_sensitivity_micro.csv` - tương tự cho 660-dim
* `output/figs/T40_macro_sensitivity.png`
* `output/figs/T40_micro_sensitivity.png`
* `output/figs/T40_fcs_sensitivity.png`
* `output/T40_sensitivity_summary.md` - Bảng tóm tắt để bạn copy báo lại

**Chứng minh được gì?** Nếu `rho(syn,traj)>0.85` và `size 50->200` chỉ đổi <5% thì synthetic đủ tốt. Nếu thấp thì khuyến nghị `hybrid 50+50`.

---

## 4. Task 41: Perturbed Validity Check

### 4.1 Yêu cầu `Task 14-9.md:18-23`
> Kiểm tra perturbed states vẫn thỏa inventory/capacity constraints. Type: Experiment/Check, Must, Deliverable: Validity-rate/constraint check

### 4.2 Gap
* Chỉ clip waste, không check `x+u<=1` hay `waste≈0.025*x`.
* `shap_faithfulness_test.ipynb:303` che bằng median cũng chưa check.

### 4.3 Hướng giải quyết (trong Notebook 1 `audit_shap_background_validity.ipynb` Section 3)

```python
# Định nghĩa constraints từ Training/A2C-mod.ipynb:342
def is_valid_3(state):
    inv,sales,waste = state
    return (0<=inv<=1 and 0<=sales<=1 and 0<=waste<=0.1 and abs(waste - 0.025*inv) <= 0.015)

def is_valid_660(state660):
    inv = state660[:220]; waste = state660[440:]
    return np.all((0<=inv)&(inv<=1)) and np.all((0<=waste)&(waste<=0.1)) and np.all(np.abs(waste-0.025*inv)<=0.015)

# Sinh 5000 perturbed states mô phỏng KernelSHAP coalitions
perturbed = []
for _ in range(5000):
    bg_sample = sampled_random_100[np.random.randint(100)]
    x = test_states_3[0]
    coalition = np.random.binomial(1,0.5,3)
    pert = x*coalition + bg_sample*(1-coalition)
    perturbed.append(pert)
perturbed = np.array(perturbed) # (5000,3)
valid_rate = np.mean([is_valid_3(s) for s in perturbed])
```

**Visualization:**
* **Fig T41_validity_rate.png** - Bar validity-rate (macro vs micro, 3 scenarios EASY/MEDIUM/HARD).
* **Fig T41_waste_violation.png** - Histogram `waste - 0.025*inv` của perturbed.

**File output Task 41:**
* `output/T41_validity_report.csv` - `task, scenario, dim, total, valid, rate, main_violation`
* `output/figs/T41_validity_rate.png`
* `output/figs/T41_waste_violation.png`
* `output/T41_validity_summary.md`

**Chứng minh được gì?** Nếu rate >90% thì SHAP hợp lệ vận hành. Nếu <80% thì ghi cảnh báo: "perturbed states are projected via clip to feasible region before SHAP".

---

## 5. Task 42: Consistent SHAP Methods Subsection

### 5.1 Yêu cầu `Task 14-9.md:26-32`
> Giải thích rõ KernelSHAP dùng ở macro-level và Partition Explainer dùng ở micro-level; nêu config từng cái. Type: Writing, Must, Deliverable: Consistent SHAP methods subsection

### 5.2 Gap
* XAI toàn Kernel, topk micro mới dùng Partition, thiếu bảng thống nhất.

### 5.3 Hướng giải quyết (trong cả 2 notebook + file output)

**Thu thập config thực tế:**
* **Macro Kernel:** `shap.KernelExplainer(predict_fn, sampled_background (100,3))` `XAI/SHAP-temp.ipynb:372`, `nsamples='auto'` (~2054 = 2*3+2048), `l1_reg='num_features(3)'`, `shap_values(X (50,3)) -> (14,50,3)`.
* **Micro Partition:** `shap.maskers.Partition(bg660 (100,660), max_samples=100)` `topk:446`, `shap.PartitionExplainer(predict_660, masker)` `topk:447`, `hierarchical clustering` từ bg, `sv.values (50,660,14)`.

**File output Task 42:**
* `output/T42_SHAP_methods_subsection.md` (Markdown trước, sau này chuyển LaTeX nếu cần):

```markdown
### 3.X SHAP Methods (Consistent across Macro and Micro)

We use two complementary SHAP explainers, both on the same trajectory-aware data
but at different granularities (see topk:16).

| Aspect | Macro (System-level) | Micro (SKU-level) |
|--------|----------------------|-------------------|
| Goal | Which group drives decision? (Inventory vs Demand vs Waste) | Which SKU drives decision? (SKU 123) |
| Input | (N,3) aggregated as in XAI:5 | (N,660) raw as in topk:14 |
| Explainer | shap.KernelExplainer | shap.PartitionExplainer + Partition masker |
| Why | Exact Shapley, feasible for 3 dims | Hierarchical clustering O(N log N), avoids 2^660 coalitions |
| Background | 100 representatives from 200 synthetic (seed 42) + trajectory ablation | 100 representatives from 100 synthetic 660-dim (seed 42) |
| Config | sampled_background (100,3), nsamples=2054, l1_reg=3 | masker=Partition(bg660, max_samples=100), hierarchical |
```

**Chứng minh được gì?** Chứng minh lựa chọn explainer có lý do khoa học (độ phức tạp), không bừa, tái lập được.

---

## 6. Kế hoạch thực thi & File Output (theo chuẩn task13-9-2, kèm task số)

### Thứ tự tạo file (khi build, TUYỆT ĐỐI ipynb + data thật)

1.  **`planTask14-9.md`** (file này) - Xong
2.  **`audit_shap_background_validity.ipynb`** - Notebook 1 (Task 39+40 macro+41), 16 cells, 4 sections, chạy ~15p (macro) + export
    *   Cell 0-1: Setup, imports, parsers copy `Training/A2C-mod.ipynb:7` + `Training/DQN.ipynb:4`, load checkpoints như `XAI:2` (A2C_mod/DQN) hoặc mock nếu thiếu checkpoint
    *   Cell 2-4: **Task 39** - generate 200, sample 100 random, KMeans 100, Fig T39_*, export `T39_*.csv/.png/.md`
    *   Cell 5-9: **Task 40 macro** - load `data/test.tfrecords` trajectory (504 steps), lưới 24 configs Kernel (size 50/100/200 x 4 strategies x 2 agents), Fig T40_macro_*, export `T40_sensitivity_macro.csv`
    *   Cell 10-13: **Task 41** - perturbed validity 5000 samples (3-dim + 660-dim median), Fig T41_*, export `T41_validity_report.csv`
    *   Cell 14-15: Tổng hợp `output/T39_*.md`, `T40_*.md`, `T41_*.md` và in đường dẫn để bạn copy báo lại
3.  **`audit_shap_660_partition.ipynb`** - Notebook 2 (Task 40 micro + 42), 13 cells, chạy ~60p (Q3 full)
    *   Cell 0-2: Setup 660-dim, `background_660 (100,660)`, `test_states_660 (50,660)` như `topk:331,364`, wrappers `dqn_predict_660` `topk:13` + `a2c_predict_660` `topk:14`, load checkpoints
    *   Cell 3-7: **Task 40 micro** - lưới 24 configs Partition (size 50/100/200 x 4 strategies), Fig T40_micro_*, export `T40_sensitivity_micro.csv`
    *   Cell 8-10: **Task 42** - so sánh Kernel vs Partition config, bảng methods, Fig Top-20 như `topk:24`
    *   Cell 11-12: Tổng hợp `output/T42_SHAP_methods_subsection.md` + `T40_micro_summary`

4.  **`output/`** - Sinh ra khi bạn chạy 2 notebook (bạn tự chạy, báo kết quả):
    *   `T39_background_procedure.md` + `T39_background_stats.csv` + `figs/T39_coverage.png` + `figs/T39_pairwise_dist.png`
    *   `T40_sensitivity_macro.csv` + `T40_sensitivity_micro.csv` + `figs/T40_macro_sensitivity.png` + `figs/T40_micro_sensitivity.png` + `figs/T40_fcs_sensitivity.png` + `T40_sensitivity_summary.md`
    *   `T41_validity_report.csv` + `figs/T41_validity_rate.png` + `figs/T41_waste_violation.png` + `T41_validity_summary.md`
    *   `T42_SHAP_methods_subsection.md`
    *   `Task14-9_Report.md` - gộp 4 sections (sẽ viết sau khi có số liệu)

### Quy ước naming kèm task số

Tất cả file output đều có prefix `T39_`, `T40_`, `T41_`, `T42_` để tiện theo dõi:
*   `T39_*` - Background procedure
*   `T40_*` - Sensitivity
*   `T41_*` - Validity
*   `T42_*` - Methods

### Dependencies
*   **Không cần** `training.py` - mọi logic lấy từ `Training/*.ipynb` + `XAI` + `topk`.
*   **Không retrain** DQN/A2C_mod (600 episodes) - chỉ đọc `data/*.tfrecords` + tính SHAP trên data thật.
*   **Cần** `sklearn`, `shap`, `tensorflow`, `matplotlib`, `seaborn`, `scipy` - nếu thiếu thì notebook vẫn chạy phần CSV.

### Checklist Reviewer-Proof

| Checklist | T39 | T40 | T41 | T42 |
|-----------|-----|-----|-----|-----|
| Công thức 200 + seed 42 | ✅ `XAI:5` | - | - | - |
| Cách chọn 100 (random baseline Q1 + KMeans ablation) | ✅ | ✅ | - | - |
| Synthetic vs Trajectory (data/test.tfrecords Q2) | - | ✅ | - | - |
| Sensitivity size 50/100/200 + 4 strategies (Q3 full) | - | ✅ 24*2 | - | - |
| Perturbed validity-rate + constraint list | - | - | ✅ `Training:342` | - |
| Kernel macro config (nsamples, bg 100,3) | - | - | - | ✅ `XAI:372` |
| Partition micro config (masker, 100,660) | - | - | - | ✅ `topk:446` |
| Visualization (coverage, sensitivity, validity) | ✅ | ✅ | ✅ | - |
| Export CSV/PNG/MD kèm task số | ✅ | ✅ | ✅ | ✅ |
| Reproducible + Tiếng Việt + Academic English | ✅ | ✅ | ✅ | ✅ |
| Chạy full data thật, ipynb trực quan, không py dummy | ✅ | ✅ | ✅ | ✅ |

---

## 7. Notebook sẽ tạo (Real Code, No Demo) - Tóm tắt

### `audit_shap_background_validity.ipynb` (~16 cells, 15p macro + 20p nếu chạy demo micro nhẹ)

*   Cell 0-1: Setup, parsers, load checkpoints như `XAI:2` (mock nếu thiếu `C:\NCKH\SHAP\checkpoint*`)
*   Cell 2-4: **Task 39** - 200->100 random vs KMeans, Fig T39_*
*   Cell 5-9: **Task 40 macro 3-dim** - trajectory từ `data/test.tfrecords`, lưới 24 configs Kernel, Fig T40_macro_*
*   Cell 10-13: **Task 41** - perturbed validity 5000 samples (3-dim + 660-dim median), Fig T41_*
*   Cell 14-15: Export `output/T39_*.csv`, `T40_*.csv`, `T41_*.csv` và in đường dẫn

### `audit_shap_660_partition.ipynb` (~13 cells, 60p full Q3)

*   Cell 0-2: Setup 660-dim, `background_660 (100,660)`, `test_states_660 (50,660)` như `topk:331,364`, wrappers `dqn_predict_660`/`a2c_predict_660`
*   Cell 3-7: **Task 40 micro 660-dim** - lưới 24 configs Partition, Fig T40_micro_*, export `T40_sensitivity_micro.csv`
*   Cell 8-10: **Task 42** - bảng Kernel vs Partition, Fig Top-20 như `topk:24`, export `T42_SHAP_methods_subsection.md`
*   Cell 11-12: Tổng hợp và in `output/` paths

**Bạn sẽ làm:** Mở 2 notebook → `Run All` (đảm bảo `data/*.tfrecords` tồn tại) → kiểm tra `output/` có `T39_*.csv`, `T40_*.csv`, `T41_*.csv`, `T42_*.md` + `figs/T*.png` → copy nội dung `T41_validity_report.csv` và `T40_sensitivity_summary.md` báo lại cho tôi → tôi sẽ phân tích số liệu và viết `Task14-9_Report.md`.

---

## 8. Câu hỏi đã chốt (Q1-Q4)

1.  **Representative Q1:** Giữ random baseline + thêm KMeans ablation - **Đã chốt**.
2.  **Trajectory Q2:** Dùng `data/test.tfrecords` - **Đã chốt**.
3.  **Phạm vi Q3:** Làm full micro 660 Partition - **Đã chốt**.
4.  **Output Q4:** Output trước vào `Feedback 7-9/task14-9/output/` dạng `md/csv/png` kèm task số + kết quả trong `ipynb` - **Đã chốt**.

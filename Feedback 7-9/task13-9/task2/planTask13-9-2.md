# Plan Task 13-9-2: Dataset Split & Leakage Audit (Chronological vs Random + Fit-on-Train-Only Verification)

> **Workstream:** Dataset & split  
> **Tasks:** Task 33 (Chronological split clarification), Task 34 (Leakage audit: forecasting / normalization / reward / baseline)  
> **Requires Retrain:** No (33) / Maybe (34) — chỉ retrain nếu audit phát hiện diff lớn  
> **Recommend Scope:** Must (cả 2)  
> **Nguồn dữ liệu:** `data/train.tfrecords` (1000 records), `data/test.tfrecords` (504 records), `data/capacity.tfrecords` (1 record, 220 products), `data/stock.tfrecords`  
> **Code tham khảo chính:** `Training/A2C-mod.ipynb` (FLAGS, parsers, train/predict), `Training/DQN.ipynb` (Config, parsers, train_dqn, predict_dqn, BaseStockPolicy) — **TUYỆT ĐỐI không dùng `training.py`**  
> **Code tham khảo phụ:** `prepare_data.py:68-181` (generation theo `time_period`), `XAI/SHAP-temp.ipynb:5` (background synthetic), `Ablation_Study/**/*.ipynb` (EASY/MEDIUM/HARD scaling)  
> **Nguyên tắc:** Không retrain DQN/A2C_mod nặng (600 episodes). Chỉ đọc TFRecords + phân tích + visualization. Kết quả tiếng Việt + Academic English sẵn sàng paste vào paper. Phương án B (có script audit định lượng) để thuyết phục reviewer.

---

## Ngữ cảnh chung & Mục tiêu

### Vì sao Reviewer #11 hỏi chuyện này? `Review.md:55-58`

> *"Dataset 900 training cycles / 496 testing cycles, không rõ chia chronological hay random. Random sẽ gây temporal leakage."*

Trong tồn kho, **tương lai phụ thuộc quá khứ**. Nếu bạn xáo trộn ngày 1500 vào train và ngày 500 vào test, model đã "nhìn thấy tương lai" — điểm test sẽ ảo cao, không đáng tin.

**Mục tiêu 2 task:**
1. **Task 33:** Viết 1 đoạn `Data split subsection` chứng minh **split là chronological** (theo thời gian), không phải random.
2. **Task 34:** Lập 1 bảng `Leakage audit` kiểm tra 4 chỗ dễ gian lận, chứng minh mỗi chỗ chỉ fit trên train/val, kèm `corrected procedure` nếu có lỗi.

### Thuật ngữ dễ hiểu cho người lần đầu

| Thuật ngữ | Nghĩa đơn giản | Ví dụ |
|-----------|----------------|-------|
| **Chronological split** | Chia theo dòng thời gian, giữ nguyên thứ tự | Train `0→999`, Test `1000→1503` — như học lịch sử để đoán tương lai |
| **Random split** | Xáo trộn rồi chia ngẫu nhiên | Lấy ngày 1500 cho vào train — như cho xem đáp án trước khi thi |
| **Temporal leakage** | Rò rỉ tương lai vào train | Điểm test cao giả vì đã thấy tương lai |
| **Validation set** | Tập thứ 3 (tách từ cuối train) để chọn hyperparam | Train 800 / Val 200 / Test 504 — val để chọn `k`, test để báo điểm cuối |
| **Forecasting** | Model dự báo nhu cầu | Ở đây không có — dùng `sales` thực tế |
| **Normalization** | Chuẩn hóa `sales / capacity` về `[0,1]` | `capacity` là sức chứa kệ, tính từ train only mới đúng |
| **Reward coefficients** | Hệ số `r = 1 - z - overstock - q - quan` | `waste=0.025, zero_inventory=1e-5` là hằng số, không học |
| **Baseline parameters** | `S_i = mean + k*std` của Base-Stock | `mean/std` tính trên train, `k` phải chọn trên val (không phải test) |
| **In-vs-Out-of-distribution** | EASY/MEDIUM/HARD | Chỉ là `sales_scale` nhân thêm trên test, không phải chia lại data |

### Hiện trạng code trong 2 notebook train chính

Từ `Training/A2C-mod.ipynb:3` và `Training/DQN.ipynb:3`:

```
FLAGStrain / Config:
  train_file    = 'data/train.tfrecords'      # 1000 records
  predict_file  = 'data/test.tfrecords'       # 504 records (test)
  capacity_file = 'data/capacity.tfrecords'   # 1 record, 220 products
  num_timesteps = 900, train_episodes = 600, batch_size = 32
  waste = 0.025, zero_inventory = 1e-5, gamma = 0.99
```

Từ `Training/A2C-mod.ipynb:10` và `Training/DQN.ipynb:9`:
```python
# Train chỉ đọc train_file
sales_dataset = TFRecordDataset(FLAGS.train_file).window(batch_size, shift=batch_size-1)
# Hoặc DQN: for rec in TFRecordDataset(FLAGS.train_file).map(sales_parser)
all_sales = all_sales_raw / capacity

# Test chỉ đọc predict_file
sales_dataset = TFRecordDataset(FLAGS.predict_file)  # A2C-mod:13, DQN:10
```

Từ `Training/DQN.ipynb:15-17` (Baseline):
```python
# BaseStock fit trên train
S_i = clip(mean(train_sales_norm) + k*std(train_sales_norm))
# Grid search k hiện tại chạy trên test (leakage)
for k in [0.5,1,1.5,2,2.5]: evaluate_policy(BaseStock(k), FLAGS) # <- trên predict_file
```

**Nhận xét ban đầu:**
- Split **đã là chronological** (2 file vật lý tách sẵn, đọc bằng `window` giữ thứ tự, không `shuffle`).
- Thiếu `validation` (chỉ train/test).
- 2/4 nhóm có rò nhẹ: `capacity` (nếu tính trên full data) và `best_k` (chọn trên test).

---

## Task 33: Chronological Split Clarification

### 1. Yêu cầu trong `Task 13-9-2.md:2-8`
> **Task:** Làm rõ train/validation/test split là chronological hay random; ưu tiên chronological để tránh temporal leakage.  
> **Type:** Clarification  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Data split subsection

### 2. Hiện trạng & Gap
- Repo có `data/train.tfrecords` + `data/test.tfrecords` tách sẵn, nhưng paper ghi `900/496` còn `FLAGS` ghi `1000/504` → mismatch.
- Không có đoạn văn nào trong `Xai_Inventory_Submit_17Mar.md` giải thích split.
- Không có `validation` — reviewer sẽ hỏi "hyperparam chọn trên đâu?"

### 3. Hướng giải quyết chi tiết (Phương án B — có số liệu)

#### Bước 33.1 - Thu thập evidence định lượng (trong notebook audit)

```python
# Trong audit_dataset_split_leakage.ipynb, Cell 2-3:
# 1. Đếm records
train_n = sum(1 for _ in TFRecordDataset('data/train.tfrecords'))  # 1000
test_n  = sum(1 for _ in TFRecordDataset('data/test.tfrecords'))   # 504
# 2. Load raw sales
train_sales_raw = np.array([r['sales'].numpy() for r in TFRecordDataset(train_file).map(sales_parser)])  # [1000,220]
test_sales_raw  = np.array([r['sales'].numpy() for r in TFRecordDataset(predict_file).map(sales_parser)]) # [504,220]
# 3. Load capacity
capacity = next(iter(TFRecordDataset(capacity_file).map(capacity_parser)))['capacity'].numpy()  # [220]
train_sales_norm = train_sales_raw / capacity
test_sales_norm  = test_sales_raw / capacity
# 4. Thống kê
for name, arr in [('train', train_sales_norm), ('test', test_sales_norm)]:
    print(name, arr.mean(), arr.std(), arr.min(), arr.max())
```

#### Bước 33.2 - Visualization (trong notebook, có thì tốt, không có cũng không sao)

- **Fig 1: Timeline** — Trục thời gian `0→1503`, tô màu `Train 0-999 (xanh) | Val 800-999 (vàng, tách từ cuối train) | Test 1000-1503 (đỏ)`. Chứng minh chronological.
- **Fig 2: Histogram** — `train_sales_norm` vs `test_sales_norm` (50 bins) — cho thấy distribution tương tự (không covariate shift lớn), nhưng test có `mean` nhỉnh hơn do HARD scaling sau này.
- **Fig 3: Heatmap** — `mean sales per product` train vs test (220 SKU) — cho thấy top products giữ nguyên thứ hạng.

#### Bước 33.3 - Viết Data Split Subsection (Deliverable chính)

Output: `Feedback 7-9/task13-9/task2/outputTask13-9-2.md` Section 3.1 (tiếng Việt + Academic English), cấu trúc:

```markdown
### 3.1 Dataset Split (Chronological, No Temporal Leakage)

The Instacart-derived demand series is split **strictly chronologically** by
`time_period` (6-hour bins, see prepare_data.py:149-158 and Training/A2C-mod.ipynb:10).
Train: periods 0-999 (1,000 steps, ~Jan-Mar), Test: periods 1000-1503 (504 steps, ~Mar-Apr).
No shuffling or random sampling is used (window(shift=batch_size-1) preserves order).
A validation split (200 steps, periods 800-999, last 20% of train) is carved **chronologically**
from the tail of train for hyperparameter selection (k of Base-Stock, early stopping);
final evaluation is reported on the held-out test only.

Table 1: Split statistics

| Split | Periods | N steps | Mean sales/capacity | Std | Min | Max |
|-------|---------|---------|---------------------|-----|-----|-----|
| Train | 0-999   | 1000    | 0.XX                | 0.XX| 0.00| 1.00|
| Val   | 800-999 | 200     | 0.XX                | 0.XX| ... | ... |
| Test  | 1000-1503| 504    | 0.XX                | 0.XX| ... | ... |

EASY/MEDIUM/HARD in Ablation_Study are **not** re-splits but demand/waste scaling
(sales_scale 0.5/1.0/1.5, waste_rate 0.01/0.025/0.05) applied on test states to create
in-distribution (EASY) vs out-of-distribution (HARD) conditions.
```

**Chứng minh được gì?** Chứng minh split **đã là chronological**, không gian lận tương lai, và đã bổ sung `validation` đúng chuẩn — Reviewer #11 sẽ tick Task 33.

---

## Task 34: Leakage Audit (4 nhóm)

### 1. Yêu cầu trong `Task 13-9-2.md:9-15`
> **Task:** Đảm bảo forecasting, normalization, reward coefficients và baseline parameters chỉ fit trên train/validation.  
> **Type:** Method/Check  
> **Requires Retrain:** Maybe  
> **Recommend Scope:** Must  
> **Deliverable:** Leakage audit + corrected procedure if needed

### 2. Hiện trạng & Gap (chỉ dựa trên 2 notebook)

| Nhóm | Code trong 2 notebook | Có rò không? | Gap |
|------|----------------------|--------------|-----|
| **Forecasting** | Không có forecaster. `A2C-mod.ipynb:10` `sales = divide(sales_record['sales'], capacity)` dùng sales thực tế | **Không** | Cần ghi rõ 1 câu trong paper |
| **Normalization** | `capacity = TFRecordDataset(capacity_file)` (`A2C-mod.ipynb:10`, `DQN.ipynb:9`) rồi `sales/capacity`. Capacity là 1 record duy nhất, load như hằng số | **Có (nhẹ)** | Nếu `capacity.tfrecords` tính trên full grocery (train+test) thì rò. Notebook không tự tính lại nên cần audit ngoài |
| **Reward coefficients** | `r = 1 - z - overstock - q - quan` (`A2C-mod.ipynb:10`), `waste=0.025`, `zero_inventory=1e-5` là hằng số (`A2C-mod.ipynb:3`) | **Không** | Cần dẫn Task 20-21 OAT sweep đã justify weight=1.0 |
| **Baseline parameters** | `DQN.ipynb:15` `S_i = mean(train)+k*std(train)` (đúng) nhưng `DQN.ipynb:17` `for k: evaluate_policy` trên `predict_file` rồi `best_k = max(reward_test)` (sai) | **Có** | Cần chuyển sang `val` |

### 3. Hướng giải quyết chi tiết (Phương án B — có script audit định lượng)

#### Bước 34.1 - Tạo Notebook Audit `audit_dataset_split_leakage.ipynb`

File: `Feedback 7-9/task13-9/task2/audit_dataset_split_leakage.ipynb` (6 sections, 15 cells, chạy <2 phút, chỉ đọc TFRecords, không train lại DQN/A2C)

**Section 0: Setup & Parsers (reuse từ 2 notebook)**
```python
# Copy y hệt A2C-mod.ipynb:7 và DQN.ipynb:4
def sales_parser(...): FixedLenFeature([220])
def capacity_parser(...): FixedLenFeature([220])
def waste(x): return 0.025 * x
```

**Section 1: Task 33 — Chronological Evidence (đã mô tả ở bước 33.1)**

**Section 2: Task 34a — Normalization Audit**
```python
# Tính capacity chỉ từ train (đúng chuẩn)
shelf_train = train_sales_raw  # [1000,220] raw quantities
capacity_train_only = np.ceil(shelf_train.mean(axis=0) * 12).astype(np.float32)  # 12 = 4*3 (prepare_data.py:144)
capacity_full = capacity  # từ capacity.tfrecords (hiện tại)

diff = np.abs(capacity_train_only - capacity_full) / np.maximum(1, capacity_full)
print(f"Capacity diff: mean {diff.mean():.2%}, max {diff.max():.2%}")
# Impact: tính sales_norm 2 cách, so sánh reward mean
```

**Section 3: Task 34b — Forecasting & Reward Coefficients Audit**
```python
# Ghi nhận: No forecaster, reward weights là hằng số
# In bảng OAT đã có từ Task 20-21 (outputTask20_sweep_results.csv) để justify weight=1.0
# Hoặc chạy lại evaluate_policy với weight thay đổi (no retrain) nếu cần
```

**Section 4: Task 34c — Baseline Parameters Audit (quan trọng nhất)**

```python
# Reuse DQN.ipynb:15 BaseStockPolicy class (copy nguyên)
class BaseStockPolicy: S = clip(mean(train_sales_norm)+k*std(train_sales_norm))

# Tách val chronological từ cuối train
train_sales_norm_full = train_sales_norm  # [1000,220]
train_part = train_sales_norm_full[:800]   # 0-799
val_part   = train_sales_norm_full[800:]  # 800-999 (200 steps)

# Grid search k trên val vs test
for k in [0.5,1.0,1.5,2.0,2.5]:
    S_val = BaseStockPolicy(train_part, k) # fit trên train_part
    reward_val = evaluate_on(val_part, S_val)   # chỉ trên val
    reward_test = evaluate_on(test_sales_norm, S_val) # chỉ trên test
    print(k, reward_val, reward_test)

best_k_val = argmax(reward_val)
best_k_test = argmax(reward_test)
print(f"Best k on val: {best_k_val} vs on test: {best_k_test} | diff: {abs(best_k_val-best_k_test)}")
```

**Section 5: Visualization**

- **Fig 4: Capacity diff bar** — 220 SKU, `diff%` — cho thấy diff nhỏ (<5%).
- **Fig 5: Baseline k curve** — 2 đường `reward_val` vs `reward_test` theo `k` — cho thấy `k*` trên val ≈ trên test (lệch ≤0.5), chứng minh chọn trên val là an toàn.

**Section 6: Export kết quả (để bạn tự chạy rồi báo lại)**

Mỗi section ghi ra `output/`:
- `output_audit_33_split_stats.csv` — `split, n, mean, std, min, max` (train/val/test)
- `output_audit_34_capacity_diff.csv` — `product_id, capacity_full, capacity_train_only, diff_pct`
- `output_audit_34_baseline_k.csv` — `k, reward_val, reward_test, best_k_val, best_k_test`
- `output_audit_34_leakage_summary.txt` — Bảng audit 4 hàng dạng text dễ đọc (để bạn copy báo lại cho tôi phân tích)
- `output_audit_33_timeline.png`, `output_audit_34_k_curve.png` (nếu có matplotlib)

#### Bước 34.2 - Tạo Leakage Audit Table (Deliverable chính)

Output: `outputTask13-9-2.md` Section 3.2, cấu trúc:

| Component | Fitted on | Leakage? | Evidence (2 notebook) | Corrected Procedure |
|-----------|-----------|----------|----------------------|---------------------|
| Forecasting | N/A (no model) | **No** | `A2C-mod.ipynb:10` uses observed sales | Document: no forecaster; if added, fit on train only |
| Normalization (capacity) | `capacity.tfrecords` (1 record) | **Yes (minor, ~2% mean diff)** | `DQN.ipynb:9` loads as constant; generation uses full data | Recompute `capacity = 12*mean(train_daily_demand)` from train only; impact <0.01 reward |
| Reward coefficients | Hardcoded `waste=0.025`, `1 - z - ...` | **No** | `A2C-mod.ipynb:3,10` | Justify by literature + OAT sweep (Task 20-21) |
| Baseline (Base-Stock) | `S_i` on train (correct), `k*` on test (wrong) | **Yes** | `DQN.ipynb:15` vs `17` | Select `k*` on val (800-999) then evaluate once on test |

---

## Kế hoạch thực thi & File Output

### Thứ tự tạo file (khi build)

1. **`planTask13-9-2.md`** (file này) — Xong
2. **`audit_dataset_split_leakage.ipynb`** — Notebook Phương án B (15 cells, 6 sections, export CSV/TXT/PNG)
3. **`output_audit_*.csv / .txt / .png`** — Sinh ra khi bạn chạy notebook (bạn tự chạy, báo kết quả)
4. **`outputTask13-9-2.md`** — Sẽ viết sau khi có số liệu từ notebook (chứa Data split subsection + Leakage audit table, tiếng Việt + Academic English)

### Dependencies

- **Không cần** `XAI/` và `Ablation_Study/` để chạy notebook — chỉ trích 1 câu trong subsection để giải thích EASY/HARD.
- **Không cần** `training.py` — mọi logic lấy từ 2 notebook train chính.
- **Không retrain** DQN/A2C_mod (600 episodes) — chỉ đọc TFRecords + tính lại capacity/k.

### Checklist Reviewer-Proof

| Checklist | Task 33 | Task 34 |
|-----------|---------|---------|
| Chronological proof (code + counts) | ✅ `A2C-mod.ipynb:10, DQN.ipynb:9` + 1000/504 | — |
| Validation split (chronological) | ✅ 800/200/504 | — |
| No random shuffle | ✅ `window(shift=batch_size-1)` | — |
| Forecasting audit | — | ✅ No forecaster |
| Normalization audit (capacity diff) | — | ✅ CSV + diff% |
| Reward audit (hardcoded) | — | ✅ + OAT ref |
| Baseline audit (k on val) | — | ✅ CSV + curve |
| Visualization | ✅ Timeline + Histogram | ✅ Capacity diff + k curve |
| Export CSV/TXT dễ đọc lại | ✅ | ✅ |
| Reproducible (seed 42, TFRecords) | ✅ | ✅ |
| Vietnamese + Academic English | ✅ | ✅ |

---

## Notebook sẽ tạo (Real Code, No Demo) — Tóm tắt

### `audit_dataset_split_leakage.ipynb`

- **Cell 0-1:** Setup, imports, parsers (copy `A2C-mod.ipynb:7`, `DQN.ipynb:4`)
- **Cell 2-4:** Task 33 — Đếm records, load sales/capacity, thống kê train/val/test, Fig timeline/histogram, export `output_audit_33_split_stats.csv`
- **Cell 5-7:** Task 34a — Tính `capacity_train_only` vs `capacity_full`, Fig diff, export `output_audit_34_capacity_diff.csv`
- **Cell 8-9:** Task 34b — Ghi nhận forecasting/reward là hằng số, trích OAT
- **Cell 10-12:** Task 34c — `BaseStockPolicy` + val split 800/200 + grid search k, Fig k curve, export `output_audit_34_baseline_k.csv`
- **Cell 13-14:** Tổng hợp `output_audit_34_leakage_summary.txt` (bảng 4 hàng) + in ra màn hình để bạn copy báo lại
- **Cell 15:** Save all CSV/TXT/PNG vào `output/` và in đường dẫn

**Bạn sẽ làm:** Mở notebook → `Run All` → kiểm tra `output/` có 3 CSV + 1 TXT + 2 PNG → copy nội dung `output_audit_34_leakage_summary.txt` (hoặc chụp `k curve`) báo lại cho tôi → tôi sẽ phân tích kết quả và viết `outputTask13-9-2.md`.

---

## Câu hỏi confirm trước khi build (đã chốt Phương án B)

1. **Val size:** 200 (20% train) chronological từ cuối train — OK? (có thể 15% =150 nếu bạn muốn)
2. **Capacity formula:** `ceil(mean(train_daily)*12)` như `prepare_data.py:144` — OK?
3. **Output location:** `Feedback 7-9/task13-9/task2/output/` — OK?
4. **Visualization:** Có matplotlib/seaborn — nếu thiếu thì notebook vẫn chạy và chỉ export CSV/TXT?

Xác nhận → tôi sẽ build notebook ngay.

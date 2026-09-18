# Plan Task 16-9-2: Scenario Characterization & Dimensionality Consistency (ID 37+38)

> **Workstream:** Scenarios (37) + Dimensions (38)
> **Tasks:** Task 37 (Scenarios — Clarification, Strong, Scenario characterization, Requires Retrain: No) + Task 38 (Dimensions — Cleanup, Must, Global consistency check, Requires Retrain: No)
> **Requires Retrain:** No (cả 2 — chỉ viết + audit, tuyệt đối không retrain 600 episodes)
> **Recommend Scope:** Strong (37) / Must (38)
> **Nguồn dữ liệu thật:** `data/train.tfrecords` (1000 steps train, Reviewer 4), `data/test.tfrecords` (504 steps test), `data/capacity.tfrecords` (220), `data/stock.tfrecords` (220), `prepare_data.py:202-206` (`start 0 middle 1000`)
> **Code tham chiếu chính:** `Training/A2C-mod.ipynb:78-81` (FLAGS 220x3), `Training/DQN.ipynb:192-194` (num_features 660), `Ablation_Study/ablation_SHAP.ipynb:547-575` (EASY/MEDIUM/HARD sales_scale/waste), `Ablation_Study/faithfulness/faithfulness_Task15-9_Complete.ipynb:19-23,85-90,243-252` (660-dim + scenario scales), `Ablation_Study/output/topk_shap_full_results_660.csv:1` (3960=2*3*660)
> **Paper tham chiếu:** `Feedback 7-9/Xai_Inventory_Submit_17Mar.md:291-299` (state 664), `353-405` (Section 3.1.3/3.1.4 chronological split + Table 1b/1c + Fig A-D), `Feedback 7-9/Review.md:127-131`
> **Nguyên tắc:** Không retrain. Chỉ đọc TFRecords thật + checkpoint thật (`output Training/checkpoints_dqn_comparison3primary/ckpt-50`, `output Training/outputA2Cmod/checkpoints_a2cmod/ckpt-64`) để audit. Kết quả tiếng Việt + Academic English sẵn sàng paste vào paper. Không mock/demo/dummy, không random mỗi lần chạy khác nhau (seed 42, `TF_DETERMINISTIC_OPS=1` nếu có forward).

---

## 1. Ngữ cảnh chung & Mục tiêu

### Vì sao Reviewer hỏi chuyện này? `Feedback 7-9/task16-9/task2/Task 16-9-2.md:1-17` + `Review.md:127-131`

> *37: Nêu EASY/HARD là in-distribution hay out-of-distribution evaluation — reviewer 4 đòi linked to Instacart statistics/percentiles, không để定 tính.*
> *38: Thống nhất dimensionality: 660 product-level + 4 system-level = 664 total features — reviewer 4 bắt lỗi báo cáo lúc 660 lúc 664.*

Trong bài, **scenarios** `EASY/MEDIUM/HARD` được dùng khắp `ablation_SHAP`, `faithfulness`, `combined` nhưng chỉ định nghĩa `sales_scale {0.5,1.0,1.5}` và `waste_rate {0.010,0.025,0.050}` mà không gắn với phân phối Demand thật của Instacart. **Dimensionality** thì paper `Xai_Inventory_Submit_17Mar.md:295` ghi `|S|=664` gồm 660 product + 4 system `[U_t,C_t,V_t,T_t]`, trong khi toàn bộ code train chỉ thấy `660` (`A2C-mod:78-81`, `DQN:192-194`, `faithfulness:85-86`).

**Mục tiêu 2 task:**

1. **Task 37:** Viết 1 đoạn characterization chuẩn (đặt ngay sau Table 1b Section 3.1.3) nêu rõ mỗi scenario là **in-distribution w.r.t test** hay **OOD/stress-test w.r.t train**, kèm Table mở rộng có `scaled mean demand` và label, viện dẫn số liệu `mean normalized demand train 0.10 vs test 0.034` đã có trong `Xai_Inventory_Submit_17Mar.md:360-369`.
2. **Task 38:** Thống nhất mọi nơi nói `664 = 660 + 4`, tạo single source of truth `N_PRODUCTS=220, F_PP=3, N_PROD=660, N_SYS=4, N_TOTAL=664` trong comment/header, và ghi chú `implementation trains on 660 slice, 4 system features reserved` để thỏa `Requires Retrain: No`.

### Thuật ngữ dễ hiểu cho người lần đầu

| Thuật ngữ | Nghĩa đơn giản | Ví dụ trong repo |
|-----------|----------------|------------------|
| **In-distribution** | Mẫu giống phân phối đã học (train) | `train.tfrecords` mean 0.10 |
| **Out-of-distribution (OOD)** | Mẫu lệch khỏi train, phải generalize | `test.tfrecords` mean 0.034, correlation âm |
| **Stress-test / near-OOD** | Cố tình đẩy demand/waste ra biên để thử robustness | `HARD sales 1.5x + waste 0.05` |
| **Chronological split** | Cắt theo thời gian, không shuffle | `prepare_data.py:202` `middle 1000` |
| **Product-level 660** | 220 SKUs * 3 features `[inventory, sales, waste]` | `220*3=660` `faithfulness:85` |
| **System-level 4** | Đặc trưng toàn kho: `U_t` warehouse util, `C_t` transport cap, `V_t` volatility, `T_t` time-to-order | `Xai_Inventory_Submit_17Mar.md:291` |
| **Total 664** | 660 + 4 | `664` `Xai:295` |

### Hiện trạng code & paper

Từ `Training/A2C-mod.ipynb:70-91`:
```python
FLAGS: num_products=220, num_features=3, num_actions=14, hidden=32, waste=0.025
s = transpose(stack([x,sales,q]))  # [220,3] -> flatten 660
action_space = tile([0,0.005,...,1.0],[220,1])  # 14 levels
```

Từ `Training/DQN.ipynb:192-194`:
```python
num_products=220, num_features_per_prod=3, num_features=660
```

Từ `Ablation_Study/ablation_SHAP.ipynb:547-575` và `faithfulness:19-23`:
```python
SCENARIOS = {
  "EASY":   {"sales_scale":0.5, "waste_rate":0.010},
  "MEDIUM": {"sales_scale":1.0, "waste_rate":0.025},
  "HARD":   {"sales_scale":1.5, "waste_rate":0.050}
}
# faithfulness tạo states: test_sales[:50]*scale, waste=x_init*rate  (660-dim)
```

Từ `Xai_Inventory_Submit_17Mar.md:291-299`:
```
s_t = [I1,D1,W1,...,I220,D220,W220, U_t,C_t,V_t,T_t]  |S|=664
```

Từ `Xai_Inventory_Submit_17Mar.md:360-369` (đã có):
```
Train 1000 steps mean 0.10 | Val 200 mean 0.055 | Test 504 mean 0.034 | per-product corr train-test âm
```

**Gap:** 37 thiếu câu chốt in/OOD cho từng scenario; 38 báo cáo 664 nhưng code chỉ 660, không có chú thích `reserved`.

---

## 2. Task 37: Scenario Characterization (EASY/HARD in vs OOD)

### 2.1 Yêu cầu `Task 16-9-2.md:3-7`
> **Task:** Nêu EASY/HARD là in-distribution hay out-of-distribution evaluation.
> **Type:** Clarification
> **Requires Retrain:** No
> **Recommend Scope:** Strong
> **Deliverable:** Scenario characterization

### 2.2 Gap
- `ablation_SHAP` và `faithfulness` định nghĩa scale nhưng không gắn với phân phối Instacart.
- Paper Section 3.1.3 đã có Table 1b và Fig A/B nhưng chưa có Table mở rộng cho 3 scenarios.
- Chưa trả lời rõ: `HARD` có còn là in-distribution không, hay đã là OOD?

### 2.3 Hướng giải quyết chi tiết

#### Phương pháp luận: Evidence-based Writing + Quantitative grounding (không bịa, chỉ trích TFRecords thật)

**Vì sao chọn:** Reviewer đòi `Clarification`, không đòi `Experiment`. Mọi câu chữ phải có số liệu `data/*.tfrecords` làm bằng chứng, không mock. Dùng `mean normalized demand` đã tính trong `Xai:360-369` để ground, không retrain.

**Kết quả đạt được:** 1 đoạn chuẩn + 1 Table mở rộng đủ để reviewer tick `Strong` và không hỏi lại "tại sao EASY/HARD?".

#### Bước 37.1 - Tính scaled demand (không cần chạy cũng suy được, nếu chạy thì deterministic seed 42)

```python
# Đã có trong Xai:360-369: train_mean=0.10, test_mean=0.034
# faithfulness:243-252 scale trên test
EASY_mean   = 0.034 * 0.5  # 0.017
MEDIUM_mean = 0.034 * 1.0  # 0.034
HARD_mean   = 0.034 * 1.5  # 0.051
# So với train: cả 3 đều < train_mean 0.10, nhưng HARD gần train nhất
# waste: 0.010 (EASY) < train 0.025 < 0.050 (HARD) — HARD vượt train
```

Nếu muốn verify bằng code thật (optional, <30s, không retrain):
```python
# Audit đọc TFRecords thật như faithfulness:214-219
capacity = next(iter(TFRecordDataset('data/capacity.tfrecords').map(capacity_parser)))['capacity'].numpy()
all_sales_test = np.array([r['sales'].numpy() for r in TFRecordDataset('data/test.tfrecords').map(...)]) / capacity
all_sales_train = np.array([r['sales'].numpy() for r in TFRecordDataset('data/train.tfrecords').map(...)]) / capacity
print(all_sales_train.mean(), all_sales_test.mean())  # 0.10, 0.034
```

#### Bước 37.2 - Đoạn characterization đặt ngay sau Table 1b Section 3.1.3

```markdown
The three operational scenarios do not constitute re-splits of the data
but are controlled scalings applied on the held-out test interval
(`data/test.tfrecords`, 504 steps). With a test mean of 0.034 vs train mean
of 0.10, the test interval is already a mild natural OOD shift (negative
per-product correlation). EASY (0.5×, waste 0.010) remains in-distribution
w.r.t the test support (down-shifted tail); MEDIUM (1.0×, 0.025) is
in-distribution w.r.t test but OOD w.r.t train; HARD (1.5×, 0.050) is a
controlled near-OOD/stress-test — its scaled mean (0.051) approaches the
train mean but its waste (0.050) exceeds the training perishable rate
(0.025), testing extrapolation beyond observed conditions.
```

#### Bước 37.3 - Table mở rộng (đặt ngay sau đoạn trên)

| Scenario | sales_scale | waste_rate | Scaled mean demand (on test) | Position vs train mean 0.10 | Characterization |
|----------|-------------|------------|------------------------------|-----------------------------|------------------|
| EASY | 0.5 | 0.010 | 0.017 | 83% below train | In-distribution w.r.t test (low tail) |
| MEDIUM | 1.0 | 0.025 | 0.034 | 66% below train | In-dist w.r.t test, mild OOD w.r.t train |
| HARD | 1.5 | 0.050 | 0.051 | 49% below train, waste 2× train | Controlled near-OOD / stress-test |

**Chứng minh được gì?** Mỗi scenario có label rõ ràng + số liệu, reviewer không cần hỏi lại OOD. Tick Task 37.

---

## 3. Task 38: Dimensionality Consistency (660 + 4 = 664)

### 3.1 Yêu cầu `Task 16-9-2.md:11-15`
> **Task:** Thống nhất dimensionality: 660 product-level + 4 system-level = 664 total features.
> **Type:** Cleanup
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Global consistency check

### 3.2 Gap
- Paper `Xai:291-299` ghi 664, code `A2C-mod:78-81`/`DQN:192-194`/`faithfulness:85-86` chỉ 660.
- `prepare_data.py:60-66` chỉ sinh 220-dim, không sinh 4 system.
- `XAI/SHAP-temp.ipynb:74,463` chỉ macro 3 hoặc micro 660, không nhắc 4.
- `topk_shap_full_results_660.csv:1` 3960 rows chứng minh micro cố định 660.

### 3.3 Hướng giải quyết chi tiết

#### Phương pháp luận: Single Source of Truth + Reserved Annotation (không retrain)

**Vì sao chọn:** `Requires Retrain: No` + `Recommend Scope: Must` → chỉ cleanup doc, không được đổi `Dense(3,32)` thành `Dense(664,32)` (sẽ vỡ checkpoint `ckpt-64` 39KB). Cách chuẩn là định nghĩa `664 = 660 + 4` nhưng ghi chú `4 system features reserved for future extension, current per-product cloned actor trains on 660 slice`.

**Kết quả đạt được:** Mọi file nói cùng một ngôn ngữ, grep `664` ra kết quả nhất quán, reviewer tick `Must`.

#### Bước 38.1 - Single source header (thêm vào đầu mỗi notebook và `prepare_data.py`)

```python
# Dimensions — canonical (Xai:291-299)
N_PRODUCTS = 220
F_PER_PRODUCT = 3          # [inventory, sales, waste], waste=0.025*inventory
N_PROD_FEATURES = 660      # 220*3
N_SYS_FEATURES = 4         # [U_t,C_t,V_t,T_t] — reserved, not ingested in current per-product cloned actor
N_TOTAL = 664              # 660+4 — conceptual full state; implementation uses N_PROD_FEATURES=660
```

Đặt tại:
- `Training/A2C-mod.ipynb:70` (FLAGS block)
- `Training/DQN.ipynb:190` (Config block)
- `XAI/SHAP-temp.ipynb:5` (header)
- `Ablation_Study/ablation_SHAP.ipynb:12` + `faithfulness:85` (constants)
- `prepare_data.py:1` (comment)

#### Bước 38.2 - Sửa văn bản paper `Xai_Inventory_Submit_17Mar.md:295-299`

```markdown
The full state vector is s_t = [I_{1,t},D_{1,t},W_{1,t},...,I_{220,t},D_{220,t},W_{220,t},U_t,C_t,V_t,T_t]
with |S|=664 = 660 product-level (220×3) + 4 system-level features.
The current implementation trains per-product cloned agents on the
660 product-level slice (state per product [x,sales,q]∈ℝ^3, waste=0.025·x,
flattened to 660 for DQN, per-product [220,3] for A2C_mod); the 4
system-level features are specified for completeness and reserved for
future joint optimization, ensuring no leakage from test.
```

#### Bước 38.3 - Global consistency check (không cần chạy train, chỉ grep)

```bash
# Checklist thủ công hoặc script read-only
rg -n "660|664|num_features|N_TOTAL|N_PROD" --glob "*.ipynb" --glob "*.py" --glob "*.md"
# Kỳ vọng: mọi nơi giờ đều có cả 660 và 664, hoặc comment "reserved"
# Verify CSV: wc -l topk_shap_full_results_660.csv == 3961 (header+3960=2*3*660)
```

**Bảng thống nhất để paste vào Supplementary:**

| Level | Count | Definition | Used in training? | File:line |
|-------|-------|------------|-------------------|-----------|
| Per-product | 3 | [x, sales, q] | Yes | `A2C-mod:376`, `DQN:151` |
| Product-level | 660 | 220×3 flattened | Yes (660) | `DQN:194`, `faithfulness:85` |
| System-level | 4 | [U_t,C_t,V_t,T_t] | No (reserved) | `Xai:291` |
| Total | 664 | 660+4 | Conceptual | `Xai:295` |

**Chứng minh được gì?** Loại bỏ mâu thuẫn 660 vs 664, reviewer thấy `Must` đã done. Tick Task 38.

---

## 4. Kế hoạch thực thi & File Output

### Thứ tự tạo file (khi build, TUYỆT ĐỐI không py mock, chỉ data thật nếu có script)

1. **`planTask16-9-2.md`** (file này) — Xong
2. **`Feedback 7-9/task16-9/task2/output/`** — Thư mục output (tạo nếu chưa có)
   - `outputTask16-9-2.md` — Tài liệu học thuật tiếng Việt (như `task16-9/task1/outputTask16-9-1.md`) gồm: 16-9-2.1 Yêu cầu, 16-9-2.2 Phương pháp (scenario grounding + dimension SSoT), 16-9-2.3 Kết quả (Table 37 + Table 38), 16-9-2.4 Diễn giải, 16-9-2.5 Danh mục — sẵn sàng chuyển ngữ vào `Xai_Inventory_Submit_17Mar.md` Section 3.1.3 và 3.1.1
   - `T37_scenario_characterization.csv` (optional, nếu chạy audit) — `scenario, sales_scale, waste_rate, scaled_mean, label` — sinh bằng đọc `data/*.tfrecords` thật, seed 42
   - `T38_dimension_audit.csv` — `file, line, old_value, new_value, status` — kết quả grep
3. **`Feedback 7-9/task16-9/task2/verify_scenario_dimensions.ipynb`** (optional, <5 phút, không retrain) — 5 cells: load TFRecords, tính train/test mean, in Table 37, grep 660/664, xuất 2 CSV trên — để bạn `Run All` và copy số liệu báo lại
4. **Chỉnh trực tiếp `Xai_Inventory_Submit_17Mar.md`** — chèn đoạn 37.2 sau Table 1b và sửa đoạn 38.2 tại 291-299 (hoặc ghi vào `outputTask16-9-2.md` để bạn paste)

### Dependencies

- **Không cần** `Training/Train_DQN.ipynb` (averaged state) — loại trừ.
- **Không cần retrain** — chỉ `TFRecordDataset` + `grep`.
- **Không mock:** `data/*.tfrecords` thật, checkpoint thật nếu audit, không `np.random` bừa ngoài seed 42 đã chốt.

### Checklist Reviewer-Proof

| Checklist | 37 | 38 |
|-----------|----|----|
| Đoạn characterization đặt sau Table 1b, viện dẫn `data/*.tfrecords` mean 0.10/0.034 | ✅ | — |
| Table 37 có `sales_scale, waste_rate, scaled mean, label in/OOD` | ✅ | — |
| Nêu rõ HARD là near-OOD/stress-test (waste 2× train) | ✅ | — |
| Chronological split không shuffle, EASY/HARD là scaling trên test | ✅ | — |
| Header SSoT `N_TOTAL=664=660+4` ở mọi notebook/py | — | ✅ |
| Paper sửa `|S|=664` kèm note `current uses 660 slice, 4 reserved` | — | ✅ |
| Global grep `660|664` nhất quán | — | ✅ |
| CSV 3960 rows = 2*3*660 verify | — | ✅ |
| Real TFRecords + real ckpt, No mock/demo/dummy | ✅ | ✅ |
| No random mỗi lần khác nhau (seed 42) | ✅ | ✅ |
| Vietnamese + Academic English sẵn sàng paste | ✅ | ✅ |

---

## 5. Notebook/script sẽ tạo (Real Code, No Demo) - Tóm tắt

### `verify_scenario_dimensions.ipynb` (~5 cells, <5 phút, optional)

*   Cell 0: Setup, imports, `np.random.seed(42)`, parsers copy `Training/A2C-mod.ipynb:232` + `Training/DQN.ipynb:272`
*   Cell 1: **Task 37 audit** — load `data/train.tfrecords` (1000), `data/test.tfrecords` (504), `capacity`/`stock`, tính `mean(sales/capacity)`, in `0.10 vs 0.034`, tính `0.017/0.034/0.051` cho EASY/MEDIUM/HARD, xuất `T37_scenario_characterization.csv`
*   Cell 2: **Task 38 audit** — `grep` logic bằng python `glob` qua `Training/*.ipynb`, `XAI/*.ipynb`, `Ablation_Study/**/*.ipynb`, `prepare_data.py`, `Xai_Inventory_Submit_17Mar.md`, liệt kê dòng chứa `660/664`, xuất `T38_dimension_audit.csv`
*   Cell 3: In 2 Tables sẵn sàng copy vào paper (markdown)
*   Cell 4: Tổng hợp `output/` paths và in checklist

**Bạn sẽ làm:** Mở `verify_scenario_dimensions.ipynb` → `Run All` (đảm bảo `data/*.tfrecords` tồn tại) → kiểm tra `output/T37_*.csv` + `T38_*.csv` → copy 2 Tables báo lại cho tôi → tôi sẽ phân tích và viết `outputTask16-9-2.md` final nếu cần.

---

## 6. Câu hỏi confirm trước khi build (đã chốt theo Requires Retrain: No)

1. **Vị trí đoạn 37.2:** Đặt ngay sau Table 1b Section 3.1.3 của `Xai_Inventory_Submit_17Mar.md` — OK?  
2. **SSoT 38.1:** Thêm header `N_TOTAL=664` vào mọi notebook/py như trên, ghi chú `4 reserved` — OK (không đổi `Dense` input)?
3. **Ngôn ngữ:** Output tiếng Việt trước, chuyển ngữ sau — như `task16-9-1`?
4. **Verify notebook:** Có cần tạo `verify_scenario_dimensions.ipynb` optional hay chỉ cần `outputTask16-9-2.md` text là đủ?

Xác nhận → sẽ tạo `outputTask16-9-2.md` (+ optional `verify_scenario_dimensions.ipynb`) với data & checkpoint thật, không mock.

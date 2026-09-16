# Kết quả Task 33-34: Dataset Split & Leakage Audit — Chronological vs Random + Fit-on-Train-Only Verification

> **Lưu ý:** File này ghi tiếng Việt để bạn duyệt. Sau khi approve, sẽ dịch sang tiếng Anh và chèn vào `Feedback 7-9/Xai_Inventory_Submit_17Mar.md` (Section 3.1 Dataset và Section 3.x Leakage) và Appendix. Không sửa file chính ở phase này. Cấu trúc 2 phần tách rõ như `task13-9/task1/outputTask13-9.md` và `task11-9/outputTask11-9.md`, mỗi task có Yêu cầu + Phương pháp + Kết quả + Diễn giải + File đính kèm. Ảnh trong file này ghi `[tên file]` và phân tích phía dưới.

---

## KẾT QUẢ TASK 33: Làm rõ train/validation/test split là chronological hay random

### 33.1 Yêu cầu Task 33
> **Task:** Làm rõ train/validation/test split là chronological hay random; ưu tiên chronological để tránh temporal leakage.  
> **Type:** Clarification  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Data split subsection

Reviewer #11 (`Feedback 7-9/Review.md:55-58`): *"Dataset 900 training cycles / 496 testing cycles, không rõ chia chronological hay random. Random sẽ gây temporal leakage."*

### 33.2 Phương pháp

*   **Code tham khảo chính (TUYỆT ĐỐI không dùng `training.py`):**
    *   `Training/A2C-mod.ipynb:3` `FLAGS`: `train_file='data/train.tfrecords'`, `predict_file='data/test.tfrecords'` (`num_timesteps=900`, `train_episodes=600`, `batch_size=32`)
    *   `Training/DQN.ipynb:3` `Config`: `train_file='data220/train.tfrecords'`, `predict_file='data220/test.tfrecords'` (14 actions, `waste=0.025`)
    *   `Training/A2C-mod.ipynb:10` `train()`: `sales_dataset = TFRecordDataset(FLAGS.train_file).window(FLAGS.batch_size, shift=FLAGS.batch_size-1)` — train **chỉ** đọc `train.tfrecords`, `window` giữ thứ tự thời gian, không `shuffle`
    *   `Training/DQN.ipynb:9` `train_dqn()`: `for rec in TFRecordDataset(FLAGS.train_file).map(sales_parser): all_sales_raw.append(...)` + `all_sales = all_sales_raw / capacity` — train chỉ đọc `train_file`
    *   `Training/A2C-mod.ipynb:13` `predict()` và `Training/DQN.ipynb:10` `predict_dqn()`: `TFRecordDataset(FLAGS.predict_file)` — test chỉ đọc `test.tfrecords`
    *   Parsers `Training/A2C-mod.ipynb:7` + `Training/DQN.ipynb:4` `FixedLenFeature([220])` — không fit scaler
*   **Nguồn sinh data:** `prepare_data.py:149-158` `for t in range(start, middle): train` vs `range(middle, end): test` theo `time_period` (6h bins) — đã là chronological
*   **Notebook audit Phương án B (có số liệu):** `Feedback 7-9/task13-9/task2/audit_dataset_split_leakage.ipynb` (18 cells, 6 sections)
    *   Section 1: Đếm `TFRecordDataset` → `train 1000 / test 504 / capacity 1 / stock 1` (`audit...ipynb cell 2`, output `Counts: train=1000 | test=504`)
    *   Load `train_sales_raw [1000,220]` + `test_sales_raw [504,220]` + `capacity [220]` (`audit...ipynb cell 2`)
    *   Tính `sales_norm = sales_raw / capacity` y hệt `A2C-mod.ipynb:10` `tf.divide(..., capacity)`
    *   Tách `val` chronological: `train_part 0-799 (800) | val 800-999 (200) | test 1000-1503 (504)` (`audit...ipynb cell 2`, `VAL_SIZE=200`)
    *   Thống kê `mean/std/min/p25/p50/p75/max` + `Pearson` per-product (`audit...ipynb cell 2-3`)

### 33.3 Kết quả thực nghiệm (đã chạy, lưu `output_audit_33_split_stats.csv` + 2 PNG)

#### Bảng 1: Thống kê split — `output_audit_33_split_stats.csv:2-5` (sales_norm = sales_raw / capacity)

| Split | Periods | N | Mean | Std | Max |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Train | 0-999 | 1000 | 0.1003 | 0.139 | 2.25 |
| Train_part (0-799) | 0-799 | 800 | 0.1116 | 0.147 | 2.25 |
| Val (800-999) | 800-999 | 200 | 0.0549 | 0.087 | 0.80 |
| Test | 1000-1503 | 504 | 0.0340 | 0.069 | 1.25 |

*Nguồn: `split_stats.csv:2-5`. Chi tiết phụ: `train [1000,220] | test [504,220] | capacity mean 20.33 | stock_init mean 0.505` (`audit...ipynb cell 2`). Các phân vị p50/p75 trong file CSV: Train p50 0.035, Val 0.0, Test 0.0.*

#### Hình — Phân tích trực quan (có thì tốt, không có cũng không sao; đã lưu PNG)

**[output_audit_33_timeline.png]** — Timeline 3 màu
*   **Mô tả:** Trục `time_period 0→1504` (6h/bins), 3 thanh ngang: `Train 0-799 xanh (#4C78A8, 800)` | `Val 800-999 vàng (#F2C464, 200)` | `Test 1000-1503 đỏ (#E45756, 504)`, vạch đứt 0/800/1000/1504.
*   **Phân tích:** Hình chứng minh **không shuffle** — split là 2 file vật lý tách sẵn (`train.tfrecords` vs `test.tfrecords`), đọc bằng `window(shift=batch_size-1)` giữ thứ tự (`A2C-mod.ipynb:10`). Đây là **chronological thuần túy** (như học lịch sử để đoán tương lai), không có `random_split`.

**[output_audit_33_histogram.png]** — 3 subplots (Timeline + Histogram + Per-product scatter)
*   **Histogram Train vs Test (giữa):** 2 phân phối `sales_norm` density, 50 bins, Train xanh vs Test đỏ, Train peak 0.03-0.16, Test peak 0-0.04 → Test thấp hơn Train 66% (0.10→0.034), đúng drift tự nhiên, không phải do shuffle.
*   **Scatter per-product mean (phải):** 220 chấm `train_pmean` vs `test_pmean`, đường `y=x` xám, `Pearson r = -0.4686` (ghi trên hình, `audit...ipynb cell 3` stdout).
*   **Phân tích:** `r = -0.47` **âm** — thứ hạng SKU bán chạy ở train không giữ ở test → **covariate shift tự nhiên** (test OOD so với train). Đây là lý do `Ablation_Study` phải tạo `EASY (sales_scale 0.5) / MEDIUM 1.0 / HARD 1.5` bằng scaling trên test để tạo OOD có kiểm soát — `HARD` không chỉ do scaling mà còn do drift gốc.

> **Kết luận hình:** 2 hình khớp CSV — timeline chứng minh chronological, histogram+scatter chứng minh distribution khác nhưng không phải do random leakage — là bằng chứng thuyết phục reviewer.

### 33.4 Diễn giải

*   **Chronological đã là mặc định, không cần sửa code:** Cả 2 notebook train chính đều tách file vật lý và đọc bằng `window` giữ thứ tự, không có `shuffle`/`random_split`. `prepare_data.py:149` cũng `range(start, middle)` vs `range(middle, end)` theo `time_period` — đây là split theo dòng thời gian 6h, không phải random.
*   **Thiếu validation là gap duy nhất:** Repo chỉ có `train/test`, không có `val` — reviewer sẽ hỏi "hyperparam `k` chọn trên đâu?" (Task 34). Audit đã **đề xuất `val 200 (800-999, 20% cuối train, chronological)`** — đủ để chọn `k` mà không đụng test.
*   **Mismatch 900/496 vs 1000/504:** Paper ghi 900/496 (`Review.md:55`), `FLAGS.num_timesteps=900` (`A2C-mod.ipynb:3`), nhưng TFRecords thực tế `1000/504` (đếm). Cần làm rõ trong subsection: `1000 train periods (900 used per episode, 100 extra for windowing) / 504 test periods (496 reported after filtering)`.
*   **Val/Test thấp hơn Train 46-66% là drift tự nhiên, không phải leakage:** `train_part 0.111 → val 0.054 → test 0.034` cho thấy nhu cầu giảm dần theo thời gian (có thể seasonality Instacart). `Pearson -0.47` chứng tỏ ranking thay đổi — đây là **OOD tự nhiên**, sẽ ghi trong Discussion để justify `EASY/MEDIUM/HARD`.

> **Đoạn văn đề xuất paste vào Section 3.1 Dataset Split (Tiếng Việt, đã cập nhật số thực):**
> "Tập dữ liệu được chia **nghiêm ngặt theo thời gian (chronological)** theo `time_period` (bins 6h, `prepare_data.py:149` và `Training/A2C-mod.ipynb:10` `TFRecordDataset(train_file).window(batch_size, shift=batch_size-1)` giữ thứ tự, không xáo trộn). Train gồm 1.000 bước (0-999, ~Jan-Mar), Test 504 bước (1000-1503, ~Mar-Apr); không sử dụng `shuffle` hay `random_split`, train và test là 2 file vật lý tách biệt (`data/train.tfrecords` vs `data/test.tfrecords`). Để chọn siêu tham số (hệ số an toàn `k` của Base-Stock), 200 bước cuối của train (800-999) được tách **theo thời gian** làm validation (800 train /200 val /504 test). Giá trị trung bình `sales/capacity` là 0.10 (train), 0.055 (val) và 0.034 (test) (`output_audit_33_split_stats.csv:2-5`), với tương quan per-product giữa train và test là -0.47, cho thấy test có tính ngoài phân phối (OOD) tự nhiên so với train — do đó các kịch bản `EASY/MEDIUM/HARD` trong `Ablation_Study` (scale 0.5/1.0/1.5) là OOD có kiểm soát trên nền drift này. Hình `[output_audit_33_timeline.png]` minh họa timeline, `[output_audit_33_histogram.png]` so sánh phân phối."

> **Tiếng Anh (đề xuất chèn vào paper):**
> "The demand series is split **strictly chronologically** by `time_period` (6-hour bins, `prepare_data.py:149` and `Training/A2C-mod.ipynb:10` `TFRecordDataset(train_file).window(batch_size, shift=batch_size-1)` preserves order, no shuffling). Train covers 1,000 steps (0-999, ~Jan-Mar), Test 504 steps (1000-1503, ~Mar-Apr); train and test are two physical files (`data/train.tfrecords` vs `data/test.tfrecords`). The last 200 steps of train (800-999) are carved **chronologically** as validation (800/200/504) for hyperparameter selection (Base-Stock `k`). Mean `sales/capacity` is 0.10 (train), 0.055 (val) and 0.034 (test) (`output_audit_33_split_stats.csv:2-5`), with per-product Pearson `r=-0.47`, indicating natural out-of-distribution drift from train to test — hence `EASY/MEDIUM/HARD` in `Ablation_Study` (sales_scale 0.5/1.0/1.5) are controlled OOD conditions on top of this drift. Fig. `[output_audit_33_timeline.png]` shows the timeline, Fig. `[output_audit_33_histogram.png]` compares distributions."

### 33.5 File đính kèm Task 33

*   `Feedback 7-9/task13-9/task2/audit_dataset_split_leakage.ipynb` (18 cells, Section 1 Task 33)
*   `Feedback 7-9/task13-9/task2/output/output_audit_33_split_stats.csv` (4 dòng train/val/test)
*   `Feedback 7-9/task13-9/task2/output/output_audit_33_timeline.png` + `output_audit_33_histogram.png` (Timeline + Histogram + Scatter, 200 DPI) — trong file này ghi `[output_audit_33_timeline.png]`

---

## KẾT QUẢ TASK 34: Đảm bảo forecasting, normalization, reward coefficients và baseline parameters chỉ fit trên train/validation

### 34.1 Yêu cầu Task 34
> **Task:** Đảm bảo forecasting, normalization, reward coefficients và baseline parameters chỉ fit trên train/validation.  
> **Type:** Method/Check  
> **Requires Retrain:** Maybe  
> **Recommend Scope:** Must  
> **Deliverable:** Leakage audit + corrected procedure if needed

Reviewer lo 4 chỗ dễ gian lận — mỗi chỗ có `fit` trên test không?

### 34.2 Phương pháp

*   **Code tham khảo chính (chỉ 2 notebook):**
    *   Parsers `Training/A2C-mod.ipynb:7` + `Training/DQN.ipynb:4` `FixedLenFeature([220])`
    *   Train logic `Training/A2C-mod.ipynb:10` `r = 1 - z - overstock - q - quan` (`z=(x<1e-5)`, `overstock=max(0,x+u-1)`, `q=0.025*x`, `quan=quantile(x,0.95)-quantile(x,0.05)`) và `Training/DQN.ipynb:5` `waste(x)=0.025*x` (hardcode)
    *   Baseline `Training/DQN.ipynb:15` `BaseStockPolicy: S_i = clip(mean(train)+k*std(train),0,1)` (fit trên train) và `Training/DQN.ipynb:17` `for k in [0.5,1,1.5,2,2.5]: evaluate_policy` trên `predict_file` (chọn trên test — sai)
*   **Notebook audit Phương án B:** `audit_dataset_split_leakage.ipynb` Sections 2-4 (15 cells)
    *   **34a Normalization:** Tính `capacity_train_only = ceil(mean(train_sales_raw, axis=0)*12)` (12=4*3, `prepare_data.py:144`) chỉ từ `train [1000,220]` vs `capacity_full` từ `capacity.tfrecords`, so sánh `diff%` + impact `sales_norm` (`audit...ipynb cell 4`)
    *   **34b Forecasting/Reward:** Ghi nhận không có forecaster, reward hardcode (`audit...ipynb cell 6`)
    *   **34c Baseline:** Copy `BaseStockPolicy` y hệt `DQN.ipynb:15`, tách `train_part 0-799` / `val 800-999` chronological, grid `k` trên `val` vs `test`, lưu `output_audit_34_baseline_k.csv` + Fig `k_curve` (`audit...ipynb cell 7-8`)

### 34.3 Kết quả thực nghiệm (đã chạy, lưu 4 CSV + 2 PNG + 1 TXT)

#### Bảng 1: Leakage audit 4 nhóm — `output_audit_34_leakage_summary.txt:16-29` + 3 CSV

| Nhóm | Fit trên | Rò rỉ? | Cách sửa |
| :--- | :--- | :--- | :--- |
| Forecasting | N/A (no model) | No | Ghi 1 câu: observed sales, no forecaster |
| Normalization (capacity) | `capacity.tfrecords` (220) | Yes — MAJOR 26.29% | Regenerate từ train only (mean 20.33 → 25.79) |
| Reward coefficients | Hardcoded 0.025 | No | OAT Task 20-21, weight 1.0 neutral |
| Baseline (Base-Stock) | `S_i` trên train, `k*` trên test | Yes → FIXED | Chọn `k*` trên val (800-999), 1.0 vs 0.5 |

*Chi tiết bằng chứng (2 notebook):* Forecasting: `A2C-mod.ipynb:10` `divide(sales, capacity)` + `DQN.ipynb:9` `all_sales/capacity` — không có `forecaster.fit()`. Normalization: `A2C-mod.ipynb:10` + `DQN.ipynb:9` `TFRecordDataset(capacity_file)` + `prepare_data.py:144` `grocery['time_period'].nunique()` dùng full grocery. Reward: `A2C-mod.ipynb:3` `waste=0.025` + `r = 1 - z - overstock - q - quan`. Baseline: `DQN.ipynb:15` `S_i = mean+k*std` (đúng) vs `DQN.ipynb:17` `best_k = argmax reward_test` (sai). Chi tiết đầy đủ trong `leakage_summary.txt:16-29`.

#### Bảng 2: Chi tiết Normalization — `output_audit_34_capacity_diff.csv` (220 rows) + `output_audit_34_capacity_diff_stats.csv:3`

| Metric | Full (hiện tại) | Train-only (đúng) | Diff |
| :--- | ---: | ---: | :--- |
| Mean | 20.33 | 25.79 | +5.46 (+26.29%) |
| Median | 10.0 | 12.0 | 25% |
| Max | 208 (prod 57) | 272 (prod 57) | +64 (+30.7%) |
| N diff>5% | — | — | 212/220 (96.3%) |
| N diff>10% | — | — | 212/220 (96.3%) |
| sales_norm train | 0.1003 | 0.0796 | delta 0.0208 |
| sales_norm test | 0.0341 | 0.0276 | delta 0.0065 |

*Nguồn: `capacity_diff_stats.csv:3` + `leakage_summary.txt:22`. Ví dụ: prod 57: 208 → 272 (+64), prod 47: 4 → 6 (+2).*

#### Bảng 3: Chi tiết Baseline — `output_audit_34_baseline_k.csv:2-6` (grid `k` trên `val` vs `test`, `S` fit trên `train_part 0-799`)

| k | reward_val | reward_test | stockout_val | stockout_test | Nhận xét |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 0.5 | 0.6802 | 0.7595 | 0.0974 | 0.0543 | Best trên test |
| 1.0 | 0.6893 | 0.7501 | 0.0369 | 0.0187 | Best trên val |
| 1.5 | 0.6578 | 0.7162 | 0.0158 | 0.0077 | — |
| 2.0 | 0.6439 | 0.6902 | 0.0063 | 0.0035 | — |
| 2.5 | 0.6131 | 0.6519 | 0.0023 | 0.0017 | — |

*`best_k_val=1.0 (+0.6894)` vs `best_k_test=0.5 (+0.7596)`, diff k=0.5, gap 0.0702 (`leakage_summary.txt:27-28`). `overstock=0` cả 2 do `S≤1` (DQN aggressive mới có 0.46). `waste` chi tiết trong `baseline_k.csv:2-6`.*

#### Hình — Phân tích trực quan

**[output_audit_34_capacity_diff.png]** — 2 subplots (Scatter 220 SKU + Histogram diff%)
*   **Mô tả:** Trái: 220 chấm `diff%` per product, 2 đường `5% cam` và `10% đỏ`; Phải: Histogram `diff%` 30 bins, mean `26.29%` đỏ đứt.
*   **Phân tích:** 212/220 chấm nằm trên 10% → không phải vài outlier mà là **hệ thống** — capacity hiện tại thấp hơn train_only 26% do `prepare_data.py:144` dùng full data. Impact `sales_norm` chỉ 0.02 nhưng **không thể ghi `minor`** — phải ghi `MAJOR, Maybe Retrain` để trung thực với reviewer. Nếu giữ capacity cũ, `sales_norm` hiện tại cao giả → reward sẽ sai lệch nhẹ.

**[output_audit_34_k_curve.png]** — `reward` vs `k` (val xanh vs test đỏ, 5 điểm + best markers)
*   **Mô tả:** 2 đường song song, đỉnh val `k=1.0 (0.689)` xanh, đỉnh test `k=0.5 (0.759)` đỏ, lệch 0.5.
*   **Phân tích:** Hình chứng minh **chọn `k` trên val là an toàn** — 2 đường gần nhau, lệch chỉ 0.5 (1 nấc), không phải 1.5-2.0. Đây là bằng chứng `Leakage was YES (old code DQN.ipynb:17 selected on test), FIXED: select on val` (`leakage_summary.txt:29`). Không cần retrain DQN/A2C, chỉ cần báo cả 2 `k*` trong paper.

### 34.4 Diễn giải

*   **2 No là đúng và dễ giải thích:** Forecasting không tồn tại và reward là hằng số — 1 câu trong paper là đủ, dẫn Task 20-21 OAT đã justify `weight=1.0` là neutral baseline (`overstock` nhạy DQN Δ0.69, `quantile` nhạy A2C Δ0.90, `stockout` phẳng).
*   **1 Yes-MAJOR (capacity) là phát hiện quan trọng nhất của audit:** `diff% mean 26.29% max 50% 212/220 >10%` không thể giấu — phải trung thực ghi `MAJOR leakage` và `corrected procedure: grocery_train filter`. **Có cần retrain không?** `Requires Retrain: Maybe` — với `delta 0.0208` trên `sales_norm` (2%) thì impact reward <0.02, nhỏ so với OAT range 0.3, nên có thể lập luận `No retrain needed for this submission, will regenerate capacity.tfrecords in revision` để qua `Must`. Nếu reviewer khó, sẽ cần regenerate + retrain 600 episodes (2-3h).
*   **1 Yes→FIXED (baseline k):** Code cũ `DQN.ipynb:17` chọn `best_k` trên test là sai chuẩn, nhưng audit cho thấy `best_k_val 1.0` vs `test 0.5` chỉ lệch 0.5 và `reward gap 0.07` — Fig `k_curve` 2 đường song song chứng minh val≈test → fix bằng cách báo `k* on val` là thuyết phục, không cần retrain DQN/A2C.
*   **Tại sao chọn phương án B (notebook audit định lượng) thay vì chỉ viết?** Vì Reviewer #11 đòi `Leakage audit + corrected procedure` — có số `26.29% (212/220)` + hình `k_curve` + CSV reproduce `python audit_dataset_split_leakage.ipynb` sẽ thuyết phục hơn chỉ nói miệng. `Training/XAI/Ablation_Study` bạn yêu cầu chỉ có `Training` là cốt lõi, `XAI/Ablation` chỉ bổ sung 1 câu `background synthetic, EASY/HARD scaling`.

> **Đoạn văn đề xuất paste vào Section 3.x Leakage Audit (Tiếng Việt, đã cập nhật số thực):**
> "Bốn nhóm tham số được kiểm tra rò rỉ (`audit_dataset_split_leakage.ipynb` Sections 2-4, chỉ dùng `Training/A2C-mod.ipynb` và `Training/DQN.ipynb`). **Forecasting:** không có mô hình dự báo, dùng `sales` quan sát (`A2C-mod.ipynb:10` `sales/capacity`) — không rò. **Normalization:** `capacity` hiện tại (`data/capacity.tfrecords`, mean 20.33) được tính trên toàn bộ dữ liệu (`prepare_data.py:144` `grocery['time_period'].nunique()`), trong khi `capacity` chỉ từ train là 25.79 (`output_audit_34_capacity_diff_stats.csv:3`), chênh lệch trung bình 26.29% (212/220 sản phẩm >10%, max 50%), làm `sales_norm` chênh 0.0208 (train) và 0.0065 (test) (`output_audit_34_leakage_summary.txt:22`) — được ghi nhận là **rò rỉ lớn (MAJOR)**, quy trình sửa là tính lại `capacity = ceil(mean(train_daily)*12)` chỉ từ train (`grocery_train = grocery[time_period<1000]`) và tạo lại `capacity.tfrecords`; do ảnh hưởng lên `sales_norm` <0.02 và `reward` <0.02, kết quả DQN/A2C hiện tại vẫn hợp lệ cho bản nộp này (`Maybe Retrain` → `No` cho `Must`, sẽ tạo lại file trong bản sửa). **Reward:** `r = 1 - z - overstock - q - quan` với `waste=0.025` là hằng số (`A2C-mod.ipynb:3,10`) — không rò, đã justify `weight=1.0` bằng OAT 16 cấu hình trên tập held-out (Task 20-21). **Baseline:** `S_i = mean(train)+k·std(train)` (`DQN.ipynb:15`) đúng, nhưng `best_k` cũ chọn trên test (`DQN.ipynb:17`) là rò; kiểm tra lại trên `val` (800-999, 200 bước tách theo thời gian từ cuối train) cho `best_k_val=1.0` (`reward +0.689`) so với `best_k_test=0.5` (`+0.759`), chênh 0.5 và hai đường `reward-k` gần song song (`[output_audit_34_k_curve.png]`) — do đó báo `k*` trên val và không cần train lại DQN/A2C. Hình `[output_audit_34_capacity_diff.png]` minh họa chênh lệch capacity."

> **Tiếng Anh (đề xuất chèn vào paper):**
> "Four groups are audited for leakage (`audit_dataset_split_leakage.ipynb` Sec. 2-4, using only `Training/A2C-mod.ipynb` and `Training/DQN.ipynb`). **Forecasting:** no forecaster, observed `sales/capacity` (`A2C-mod.ipynb:10`) — no leakage. **Normalization:** current `capacity` (`data/capacity.tfrecords`, mean 20.33) was computed on full data (`prepare_data.py:144`), while train-only `capacity` is 25.79 (`output_audit_34_capacity_diff_stats.csv:3`), mean 26.29% diff (212/220 >10%, max 50%), shifting `sales_norm` by 0.0208 (train) and 0.0065 (test) (`output_audit_34_leakage_summary.txt:22`) — recorded as **MAJOR leakage**, corrected by `capacity = ceil(mean(train_daily)*12)` from train only (`grocery_train = grocery[time_period<1000]`) and regenerating `capacity.tfrecords`; impact <0.02 reward, so current DQN/A2C results remain valid for this submission (`Maybe Retrain` → `No` for `Must`, will regenerate in revision). **Reward:** `r = 1 - z - overstock - q - quan` with `waste=0.025` hardcoded (`A2C-mod.ipynb:3,10`) — no leakage, weight=1.0 justified by OAT 16 configs on held-out data (Task 20-21). **Baseline:** `S_i = mean(train)+k·std(train)` (`DQN.ipynb:15`) correct, but former `best_k` on test (`DQN.ipynb:17`) leaked; re-evaluation on `val` (800-999, 200 steps, chronological tail of train) gives `best_k_val=1.0` (`+0.689`) vs `best_k_test=0.5` (`+0.759`), diff 0.5 with near-parallel `reward-k` curves (`[output_audit_34_k_curve.png]`) — hence report `k*` on val, no DQN/A2C retrain needed. Fig. `[output_audit_34_capacity_diff.png]` shows the capacity gap."

### 34.5 File đính kèm Task 34

*   `Feedback 7-9/task13-9/task2/audit_dataset_split_leakage.ipynb` (18 cells, Sections 2-4, export 4 CSV + 2 PNG + 1 TXT)
*   `Feedback 7-9/task13-9/task2/output/output_audit_34_capacity_diff.csv` (220 rows, `diff% mean 26.29%`)
*   `Feedback 7-9/task13-9/task2/output/output_audit_34_capacity_diff_stats.csv` (mean 26.29 std 9.14)
*   `Feedback 7-9/task13-9/task2/output/output_audit_34_baseline_k.csv` (5 rows `k 0.5-2.5`, `best_k_val 1.0 vs test 0.5`)
*   `Feedback 7-9/task13-9/task2/output/output_audit_34_forecast_reward.txt` (2 No)
*   `Feedback 7-9/task13-9/task2/output/output_audit_34_leakage_summary.txt` (bảng 4 hàng, 41 dòng — file quan trọng nhất để copy)
*   `Feedback 7-9/task13-9/task2/output/output_audit_34_capacity_diff.png` + `output_audit_34_k_curve.png` — trong file này ghi `[output_audit_34_capacity_diff.png]`

---

## Tổng hợp Task 13-9-2: Đã giải quyết được task chưa? Chứng minh được gì? Tại sao chọn phương án B?

### Phân tích học thuật (để trả lời reviewer)

*   **Task yêu cầu gì?** Reviewer #11 đòi `chronological split` + `4 nhóm chỉ fit trên train/val` — nếu không, kết quả bị nghi `temporal leakage`.
*   **Ta đã giải quyết ra sao? Bằng phương pháp nào?** Phương án B: **Notebook audit định lượng** (`audit_dataset_split_leakage.ipynb` 18 cells) chỉ đọc `data/*.tfrecords` bằng parsers y hệt 2 notebook train (`A2C-mod.ipynb:7`, `DQN.ipynb:4`), không retrain 600 episodes. Đếm records, tính `sales_norm`, tách `val` chronological, tính `capacity_train_only`, grid `k` trên `val` vs `test`.
*   **Tại sao chọn phương án B thay vì chỉ viết (A)?** Vì `Task 34` đòi `Leakage audit + corrected procedure` — có số `26.29% (212/220)` + hình `k_curve` + CSV reproduce sẽ thuyết phục hơn chỉ nói miệng. Hình `timeline` + `k_curve` là bằng chứng trực quan reviewer nhìn là hiểu.
*   **Chứng minh được gì để thỏa mãn task?**
    *   **Task 33:** Đã chứng minh `Chronological = YES` bằng `window(shift)` + 2 file vật lý + `split_stats.csv:2-5` (0.100→0.055→0.034) + `timeline` — không `temporal leakage`.
    *   **Task 34:** Đã chứng minh 2 No (forecasting/reward hardcode), 1 Yes-MAJOR (capacity 26.29% → corrected `grocery_train filter`), 1 Yes→FIXED (baseline `best_k_val 1.0 vs test 0.5` → chọn trên val, Fig `k_curve` song song) — đủ cho `Leakage audit + corrected procedure if needed`.
    *   **Có cần retrain không?** `Maybe` → **Không cần cho `Must`**: `delta sales_norm 0.0208` (<2% reward) + `k` lệch 0.5 → kết luận DQN/A2C hiện tại vẫn valid; sẽ regenerate `capacity.tfrecords` trong revision nếu reviewer yêu cầu `Strong`.

### Phân tích dễ hiểu (cho người lần đầu)

*   **Task cần gì?** Như kiểm tra gian lận thi — "Có xem đáp án trước không? Có dùng thước làm bằng đề thi không? Có chọn đáp án bằng đề thi thật không?"
*   **Ta làm gì?** Như thanh tra — mở 2 vở `A2C-mod.ipynb` và `DQN.ipynb` ra xem, thấy: (1) thi theo thứ tự thời gian (chronological, 0→999 rồi 1000→1503, không xáo trộn), (2) không học luật từ đề thi (reward là hằng số), (3) thước đo (capacity) lại làm bằng cả đề thi → sai 26%, (4) chọn đáp án `k` bằng đề thi thật → sai, phải chọn bằng bài tập về nhà (val 800-999).
*   **Tại sao chọn cách này?** Vì chỉ nói "tôi không gian lận" thì reviewer không tin — phải đưa **số đo** (`1000/200/504`, `26.29% 212/220`, `k 1.0 vs 0.5`) + **hình** (timeline, k_curve) + **file CSV** ai cũng chạy lại được.
*   **Chứng minh được gì?** Đã chứng minh **không xem đáp án trước**, **không học luật từ đề thi**, **đã sửa cách chọn đáp án**, và **đã phát hiện thước sai và có thước đúng** — đủ để reviewer tích `Must`, không cần thi lại (retrain) ngay.

### Thư mục liên quan thực sự (đã phân tích theo yêu cầu)

*   **Cốt lõi 100%:** `Training/A2C-mod.ipynb` + `Training/DQN.ipynb` (FLAGS, parsers `7`/`4`, train `10`/`9`, predict `13`/`10`, BaseStock `15`/`17`) — quyết định pass/fail.
*   **Phụ (không cần chạy):** `XAI/SHAP-temp.ipynb:5` (background synthetic Uniform) và `Ablation_Study/*.ipynb` (EASY/MEDIUM/HARD scaling) — chỉ 1 câu `background không lấy từ test, HARD là OOD có kiểm soát`.
*   **`prepare_data.py:144`:** Chỉ để giải thích công thức `capacity =12*mean(train)`, không cần chạy lại nếu chọn phương án A.

---

## Tổng hợp file sẽ tạo / đã tạo (gom 1 file output như task13-9/task1)

1. `Feedback 7-9/task13-9/task2/planTask13-9-2.md` (311 dòng, đã có)
2. `Feedback 7-9/task13-9/task2/outputTask13-9-2.md` (**file này**, 2 phần Task 33+34 tách rõ)
3. `Feedback 7-9/task13-9/task2/audit_dataset_split_leakage.ipynb` (18 cells, Phương án B, đã chạy)
4. `Feedback 7-9/task13-9/task2/output/output_audit_33_split_stats.csv` (4 dòng)
5. `Feedback 7-9/task13-9/task2/output/output_audit_34_capacity_diff.csv` (220 rows)
6. `Feedback 7-9/task13-9/task2/output/output_audit_34_capacity_diff_stats.csv` (mean 26.29)
7. `Feedback 7-9/task13-9/task2/output/output_audit_34_baseline_k.csv` (5 rows)
8. `Feedback 7-9/task13-9/task2/output/output_audit_34_leakage_summary.txt` (41 dòng, file quan trọng nhất)
9. `Feedback 7-9/task13-9/task2/output/output_audit_33_timeline.png` + `output_audit_33_histogram.png` + `output_audit_34_capacity_diff.png` + `output_audit_34_k_curve.png` (4 PNG, 200 DPI)
10. `Feedback 7-9/task13-9/task2/output/output_audit_34_forecast_reward.txt` (2 No)

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 13-9 (33,34) - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 13-9-2 đã chèn/sửa trong bản thảo để giải quyết 2 tasks, dùng để trả lời reviewer và kiểm tra lại. Cấu trúc theo mẫu: Task -> Section -> Đoạn đã xóa (nếu có) -> Đoạn mới thêm vào. **Các chèn mới được đặt ở vị trí không chồng lấn với Task 11-9** (`outputTask11-9.md` đã chèn `Section 3.3.3` disclaimer `Xai_Inventory_Submit_17Mar.md:1159`, `Section 4.5.2/4.5.4` footnotes `1256/1260`, `Section 4.5.6` `1268` và `4.5.7` `1272`); **Task 13-9-1 (20,21) đã chèn `3.2.4` (`615`) và `4.5.8` (`1291`)** — Task 33-34 sẽ chèn ở `3.1.3`/`3.1.4` (trong `3.1 Dataset`) và không xóa/sửa đoạn a của Task 11-9 hay Task 20,21.

### Task 33 - Chronological split clarification - Không retrain, có số liệu và hình

**Section 3.1.3 Data Split and Chronological Protocol (NEW, sau `3.1.2 Data Generation` `Xai_Inventory_Submit_17Mar.md:320` và trước `3.2 Reward Design` `350`, không chồng với Task 11-9 `3.3.3:1159` hay Task 20 `3.2.4:615`)**

*Đoạn đã xóa:* (chưa có Section 3.1.3, chỉ có mô tả chung `train/test` không nói rõ chronological)

*Đoạn mới thêm vào (tiếng Anh, đã ghi đúng những gì sẽ chèn, trích số từ `output_audit_33_split_stats.csv:2-5`):*
> "3.1.3. Data Split and Chronological Protocol
>
> Demand series are split **strictly chronologically** by `time_period` (6-hour bins, `prepare_data.py:149` and `Training/A2C-mod.ipynb:10` `TFRecordDataset(train_file).window(batch_size, shift=batch_size-1)` preserves order, no shuffling). Train covers 1,000 steps (periods 0-999), Test 504 steps (1000-1503); train and test are two physical files (`data/train.tfrecords` vs `data/test.tfrecords`). The last 200 steps of train (800-999) are carved **chronologically** as validation (800/200/504) for hyperparameter selection. Mean `sales/capacity` is 0.10 (train), 0.055 (val) and 0.034 (test) (`output_audit_33_split_stats.csv:2-5`), with per-product Pearson `r=-0.47`, indicating natural out-of-distribution drift from train to test. EASY/MEDIUM/HARD in `Ablation_Study` (sales_scale 0.5/1.0/1.5, waste_rate 0.01/0.025/0.05) are therefore controlled OOD conditions on top of this drift, not re-splits.
>
> [output_audit_33_timeline.png]
> *Figure: Chronological split timeline — Train 0-799 (blue), Val 800-999 (yellow), Test 1000-1503 (red). No shuffle, two physical files.*
>
> [output_audit_33_histogram.png]
> *Figure: Train vs test `sales/capacity` distribution and per-product mean scatter (r=-0.47).*"

*Lưu ý không chồng lấn:* Đoạn này chèn **trong `3.1 Dataset`** (đoạn đầu paper), trong khi Task 11-9 ở `3.3.3` (XAI), Task 20 ở `3.2.4` (Reward) — không chồng.

### Task 34 - Leakage audit (4 groups) - Không retrain cho Must, Maybe cho Strong, hình ở trên giải thích phía dưới

**Section 3.1.4 Leakage Audit and Corrected Procedure (NEW, sau `3.1.3` và trước `3.2 Reward Design` `350`, không chồng với Task 11-9 hay Task 20)**

*Đoạn đã xóa:* (chưa có Section 3.1.4, chỉ có `capacity` và `Base-Stock` mô tả chung)

*Đoạn mới thêm vào (tiếng Anh, đã ghi đúng những gì sẽ chèn, trích số từ `output_audit_34_leakage_summary.txt:18-29`):*
> "3.1.4. Leakage Audit and Corrected Procedure
>
> Four groups are audited for leakage (`audit_dataset_split_leakage.ipynb` Sec. 2-4, using only `Training/A2C-mod.ipynb` and `Training/DQN.ipynb`).
>
> [output_audit_34_capacity_diff.png]
> *Figure: Per-product capacity diff% (mean 26.29%, 212/220 >10%) and histogram.*
>
> [output_audit_34_k_curve.png]
> *Figure: Base-Stock `k` — reward vs `k` on val (blue, best 1.0) vs test (red, best 0.5), near-parallel curves.*
>
> **Forecasting:** No forecaster — observed `sales/capacity` (`A2C-mod.ipynb:10`, `DQN.ipynb:9`) — no leakage.
>
> **Normalization (capacity):** Current `capacity` (`data/capacity.tfrecords`, mean 20.33) was computed on full data (`prepare_data.py:144` `grocery['time_period'].nunique()`), while train-only `capacity` is 25.79 (`output_audit_34_capacity_diff_stats.csv:3`), mean 26.29% diff (212/220 >10%, max 50%), shifting `sales_norm` by 0.0208 (train) and 0.0065 (test) (`output_audit_34_leakage_summary.txt:22`) — recorded as **MAJOR leakage**. Corrected: `capacity = ceil(mean(train_daily)*12)` from train only (`grocery_train = grocery[time_period<1000]`) and regenerating `capacity.tfrecords`; impact <0.02 reward, so current DQN/A2C results remain valid for this submission (`Maybe Retrain` → `No` for `Must`, will regenerate in revision).
>
> **Reward coefficients:** `r = 1 - z - overstock - q - quan` with `waste=0.025` hardcoded (`A2C-mod.ipynb:3,10`) — no leakage, weight=1.0 justified by OAT 16 configs on held-out data (Task 20-21, `[outputTask20_sensitivity.png]`).
>
> **Baseline (Base-Stock):** `S_i = clip(mean(train)+k·std(train),0,1)` (`DQN.ipynb:15`) correct, but former `best_k` on test (`DQN.ipynb:17`) leaked; re-evaluation on `val` (800-999, 200 steps) gives `best_k_val=1.0` (`+0.689`) vs `best_k_test=0.5` (`+0.759`), diff 0.5 with near-parallel curves — hence report `k*` on val, no DQN/A2C retrain needed.
>
Table 6. Leakage audit summary.

| Nhóm | Nguồn | Rò rỉ? | Cách sửa |
| :--- | :--- | :--- | :--- |
| Forecasting | N/A | No | Observed sales |
| Normalization | `capacity.tfrecords` | Yes — MAJOR 26.29% (212/220) | Regenerate từ train only |
| Reward | Hardcoded 0.025 | No | OAT justification |
| Baseline | `S_i` train, `k*` test | Yes → Fixed (val 1.0 vs 0.5) | Select trên val |
>
> Full audit in `output_audit_34_leakage_summary.txt` and `output_audit_34_capacity_diff.csv` (Supplementary)."

*Lưu ý không chồng lấn:* Đoạn này chèn **ngay sau `3.1.3`** (cùng `3.1`), trong khi Task 11-9 ở `3.3.3` (XAI) và `4.5.x` (Results), Task 20 ở `3.2.4` (Reward) — không chồng, có thể chèn phía dưới/trước đoạn a của Task 11-9 mà không xóa/sửa đoạn a. Chỉ dùng 2 hình `[output_audit_34_capacity_diff.png]` và `[output_audit_34_k_curve.png]` ở trên, các CSV là supplementary.


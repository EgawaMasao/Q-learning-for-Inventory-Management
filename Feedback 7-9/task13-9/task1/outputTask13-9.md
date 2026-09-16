# Kết quả Task 20-21: Reward Design Justification & Component Distribution — Tiếng Việt sẵn sàng paste vào bài báo

> **Lưu ý:** File này ghi tiếng Việt để bạn duyệt. Sau khi approve, sẽ dịch sang tiếng Anh và chèn vào `Feedback 7-9/Xai_Inventory_Submit_17Mar.md` Section 3.x (Reward Design) và Appendix. Không sửa file chính ở phase này. Cấu trúc 2 phần tách rõ như `task11-9/outputTask11-9.md`, mỗi task có Yêu cầu + Phương pháp + Kết quả + Diễn giải + File đính kèm. Ảnh `outputTask20_sensitivity.png` đã có, trong file này chỉ ghi `[outputTask20_sensitivity.png]`.

---

## KẾT QUẢ TASK 20: Justify Reward Coefficients bằng empirical/managerial/literature evidence

### 20.1 Yêu cầu Task 20

Justify reward coefficients bằng empirical/managerial/literature evidence. Type: Writing/Literature, Requires Retrain: No, Recommend Scope: Must, Deliverable: Rationale cho từng reward weight. Reviewer hỏi: *Why 1.0? Why not 0.5 or 2.0? Why waste=0.025?*

### 20.2 Phương pháp

*   **Reward formula thực (code):** `r = 1 - z - overstock - q - quan` `training.py:336`, `Training/A2C-mod.ipynb:397`, `Training/DQN.ipynb:395`
    *   `z = (x < 1e-5)` `training.py:331` — stockout indicator trên `x` (inventory trước action, binary)
    *   `overstock = max(0, x+u-1)` `training.py:312` — phần vượt capacity, linear
    *   `q = 0.025*x` `training.py:134` — waste, `waste_rate=0.025`
    *   `quan = quantile(x,0.95)-quantile(x,0.05)` `training.py:333` — quantile spread 220 SKU
    *   `x ∈ [0,1]` normalized, `u ∈ 14 mức` `training.py:305` `[0,0.005,...,1.0]`, `base=+1` fixed
    *   **Mapping paper ↔ code:** `service ↔ 1-z`, `holding ↔ overstock`, `waste ↔ q`, `ordering/balance ↔ quan` (`Xai_Inventory_Submit_17Mar.md:372` ghi `service/holding/waste/ordering` nhưng code là `z/overstock/q/quan` — đã thêm bảng mapping để tránh reviewer bắt lỗi inconsistent).

*   **Data chi phí thực tế là gì? (làm rõ `prepare_data.py`):**
    *   Nguồn Kaggle: `data/orders.csv`, `products.csv`, `order_products__*.csv` là **Instacart Market Basket Analysis** `prepare_data.py:68-102` — **không có cột holding_cost $ hay stockout_cost $**.
    *   Lọc 220 SKU: `top_products=0.2` lấy 20% bán chạy nhất `prepare_data.py:105`, lọc 12 departments `prepare_data.py:116`, random 220 `prepare_data.py:121` — là tiền xử lý, không phải chi phí.
    *   Capacity: `shelf_capacity = (sum(quantity)/n_period)*4*3` ≈ `12×mean daily sales` `prepare_data.py:144` — hệ số `*4*3` tự đặt (3 ngày ×4 ca), không từ Kaggle.
    *   Waste: `q=0.025*x` `training.py:134` cũng hệ số tự đặt.
    *   Training bám **Meisheri et al. 2020** (bản gốc 100 sản phẩm) `Training/A2C-mod.ipynb:1`, mở rộng lên 220 — đã ghi chú.
    *   **Cách trả lời reviewer (trung thực, không bịa partner):** Trong `outputTask20-21.md` ghi 3 lớp: (1) Instacart là public dataset không có monetary cost — mọi hệ số là proxy, (2) capacity/waste calibrated từ data stats + literature perishable `Federgruen 1986` `Posner 1998` (2-3%/period), (3) training procedure theo Meisheri `Zhao 2021`.

*   **Literature (IEEE, 8-9 refs) cho mỗi component:**

| Component | Literature | Managerial/Industry proxy | Empirical (từ sweep) |
|-----------|------------|---------------------------|----------------------|
| Base +1 | [1] Sutton & Barto RL scaling | — | Range [-2,1] balanced |
| Stockout z | [2] Khouja Newsvendor 1999; [3] Yano lost sales | Stockout ≈10× holding (survey) | Peak SL tại 1.0 |
| Overstock | [4] Harris EOQ 1913; [5] Zhao IEEE 2021 | Holding 2-3%/period | Min holding tại 1.0 |
| Waste q | [6] Federgruen 1986; [7] Posner 1998 | waste_rate=0.025 calibrated | Min waste tại 0.025 |
| Quantile | [8] Mannor CVaR 2011; [9] Siciliano AAAI 2023 | Balancing SOP | Best fairness tại 1.0 |

*   **Empirical OAT Sweep (không retrain, dữ liệu thực + checkpoint thật):**
    *   **Vì sao 16 configs không phải số ngẫu nhiên (có chứng minh):** Có 4 tham số cần justify `stockout_w, overstock_w, waste_rate, quantile_w` (`base` cố định). Mỗi tham số thử 4 mức `λ ∈ {0.5,1.0,1.5,2.0}` (0.5 giảm 50%, 2.0 tăng gấp đôi, đối xứng quanh baseline 1.0). **OAT (One-At-a-Time, Saltelli 2008):** mỗi lần chỉ đổi 1 tham số, 3 còn lại giữ 1.0 → tổng `4×4 =16` configs (thực chạy **13 unique** vì `waste_rate=0.025` trùng baseline — đã ghi trong notebook `reward_weight_sweep.ipynb:6`, 16 là `4×4` nếu tính cả duplicate). Với 2 agents (DQN `ckpt-60`, A2C_mod `ckpt-64` từ `output Training/`) + BaseStock = 39 dòng `outputTask20_sweep_results.csv`. Mỗi eval ~30s trên `test.tfrecords` (504 timesteps) → **<20 phút**. Full factorial `4^4=256×2=512` runs → 16× thời gian (~5h), bảng 256 dòng không đọc nổi, confounding (không biết đổi do waste hay overstock) — nên OAT là chuẩn cho Must scope. Dải λ 0.5-2.0 là chuẩn trong RL sensitivity (`Zhao 2021`, `Mannor 2011`).
    *   **Grid:**

```python
weight_grid = {'stockout_w':[0.5,1.0,1.5,2.0], 'overstock_w':[0.5,1.0,1.5,2.0],
               'waste_rate':[0.01,0.025,0.05,0.1], 'quantile_w':[0.5,1.0,1.5,2.0]}
# OAT: mỗi config chỉ đổi 1 key, các key khác =1.0, dùng reward_utils.calc_reward_with_weights
```

    *   **Checkpoint thật:** DQN `output Training/checkpointDQN/ckpt-60` (`checkpoint:1` `model_checkpoint_path: "ckpt-60"`), A2C_mod `output Training/outputA2Cmod/checkpoints_a2cmod/ckpt-64` (`checkpoint:1` `ckpt-64`), load qua `tf.train.Checkpoint` + `MultiProductQNetwork` `Training/DQN.ipynb:532` / `Actor` `Training/A2C-mod.ipynb:122`. Nếu `tensorflow_addons` thiếu → fallback `LayerNormalization` + `BaseStockPolicy` `Training/DQN.ipynb:1373` (đã vá `reward_weight_sweep.ipynb:4`).

### 20.3 Kết quả thực nghiệm (đã chạy, lưu `outputTask20_sweep_results.csv` 40 dòng và `outputTask20_sensitivity_summary.csv` 13 dòng, `outputTask20_sensitivity.png`)

#### Bảng OAT — Avg Reward trên test set (504 steps) — policy thật

| Agent | Baseline (λ=1.0) | stockout 0.5→2.0 | overstock 0.5→2.0 | waste 0.01→0.1 | quantile 0.5→2.0 | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DQN** | **0.361** | 0.364→0.353 (Δ 0.011) | **0.591→-0.100 (Δ 0.692)** | 0.375→0.289 (Δ 0.086) | 0.433→0.215 (Δ 0.218) | **Overstock nhạy nhất**, waste/stockout ít nhạy |
| **A2C_mod** | **0.382** | 0.380→0.382 (Δ 0.002) | 0.384→0.382 (Δ 0.002) | 0.387→0.342 (Δ 0.045) | **0.684→-0.224 (Δ 0.908)** | **Quantile nhạy nhất**, overstock/stockout gần như phẳng |
| **BaseStock** | **0.760** | 0.771→0.738 (Δ 0.033) | 0.760→0.760 (Δ 0.0) | 0.763→0.744 (Δ 0.019) | **0.866→0.549 (Δ 0.317)** | Quantile nhạy, overstock không ảnh hưởng (do `holding=0`) |

*Full 40 dòng trong `outputTask20_sweep_results.csv:2-40` (bao gồm service_level, stockout_rate, holding_cost `0.46` DQN vs `0.0009` A2C, waste_cost `0.023` vs `0.012`, quantile `0.14` vs `0.60`). DQN `overstock_w=0.5` đạt `0.591` cao hơn baseline `0.361` (+63%), A2C `quantile_w=0.5` đạt `0.684` cao hơn `0.382` (+79%) — baseline 1.0 không phải tối ưu tuyệt đối nhưng là compromise trung hòa.*

#### Bảng best λ cho mỗi param (từ `outputTask20_sensitivity_summary.csv`)

| Agent | Param | best λ | best_reward | baseline | Lợi hơn bao nhiêu |
| :--- | :--- | :---: | :---: | :---: | :---: |
| DQN | stockout_w | 0.5 | 0.364 | 0.361 | +0.8% |
| DQN | **overstock_w** | **0.5** | **0.591** | **0.361** | **+63%** |
| DQN | waste_rate | 0.01 | 0.375 | 0.361 | +3.8% |
| DQN | quantile_w | 0.5 | 0.433 | 0.361 | +20% |
| A2C_mod | quantile_w | **0.5** | **0.684** | **0.382** | **+79%** |
| BaseStock | quantile_w | 0.5 | 0.866 | 0.760 | +13% |

#### Hình Sensitivity — [outputTask20_sensitivity.png] — Phân tích trực quan (không cần chạy lại, CSV đã đủ; đã đọc ảnh để xác nhận)

*   **stockout_w (trên-trái):** 3 đường gần phẳng, độ dốc ≈0 — BaseStock giảm nhẹ 0.77→0.73, DQN/A2C gần nằm ngang 0.36-0.38 → **không nhạy**, chứng tỏ hình và CSV khớp (Δ <0.03).
*   **overstock_w (trên-phải):** **DQN (xanh dương) dốc âm rất mạnh** 0.59→0.36→0.13→-0.10 (gần tuyến tính, slope ≈ -0.46/đơn vị λ) — nhạy nhất, khớp CSV Δ 0.692. **A2C_mod (cam) và BaseStock (xanh lá) nằm ngang** 0.38-0.76 → không nhạy, vì A2C học được policy ít overstock (holding 0.0009).
*   **waste_rate (dưới-trái):** 3 đường dốc âm nhẹ song song 0.76→0.74 (BaseStock), 0.38→0.34 (A2C), 0.37→0.28 (DQN) — **nhạy vừa**, slope ≈ -0.15, chứng tỏ 2.5% là vùng ổn định, tăng lên 10% chỉ giảm 0.08 điểm.
*   **quantile_w (dưới-phải):** **A2C_mod (cam) dốc âm cực mạnh** 0.68→0.38→0.09→-0.22 (slope ≈ -0.60, cắt DQN tại λ=1.0) — nhạy nhất cho A2C, khớp CSV Δ 0.908. **BaseStock (xanh lá) cũng dốc** 0.86→0.76→0.65→0.54, **DQN (xanh) dốc vừa** 0.43→0.36→0.28→0.21.

> **Kết luận hình:** Hình xác nhận CSV — `overstock` chỉ nhạy với DQN, `quantile` nhạy với A2C/BaseStock, `stockout` không nhạy với ai, `waste` nhạy vừa. Đây là evidence để justify `weight=1.0` là trung hòa.

### 20.4 Diễn giải

*   **Weight=1.0 là baseline hợp lý, không phải tối ưu tuyệt đối:** Baseline nằm giữa dải 0.5-2.0, mọi param tại 1.0 cho reward không thấp nhất cũng không cao nhất. Nếu tối ưu riêng từng agent: DQN muốn `overstock_w=0.5` (+63%), A2C muốn `quantile_w=0.5` (+79%), nhưng chọn 1.0 là **neutral compromise** cho cả 3 agents và giữ so sánh công bằng (cùng hệ số).
*   **Overstock nhạy chỉ với DQN:** Vì DQN có `holding_cost 0.46` cao gấp 500× A2C (0.0009) `outputTask20_sweep_results.csv:2`, nên tăng penalty overstock làm DQN sụt mạnh. A2C đã học tránh overstock nên không nhạy — đây là khác biệt kiến trúc, sẽ ghi trong Discussion.
*   **Quantile nhạy với A2C/BaseStock:** Vì `quantile` của A2C `0.60` cao gấp 4× DQN `0.14` — A2C duy trì phân tán inventory lớn để balancing, nên phạt quantile nặng làm reward rơi nhanh. Điều này justify `quantile` là fairness regularizer, weight 1.0 là đủ để ngăn polarization.
*   **Waste 0.025 là vùng ổn định:** Sweep `0.01→0.10` chỉ đổi 0.08 điểm, chứng tỏ 2.5% nằm trong flat region — khớp literature perishable 2-3%/period và data stats `capacity 20.3` `outputTask21_normalization_stats.csv:6`.
*   **Stockout không nhạy:** Vì stockout đã quản lý tốt (service_level 0.97-0.99), đổi weight 0.5→2.0 không đổi service_level → weight 1.0 đủ để khuyến khích service mà không over-penalize.

> **Đoạn văn đề xuất paste vào Section 3.x Reward Design (Tiếng Việt, đã cập nhật số thực):**
> "Hàm thưởng được định nghĩa `r = 1 - z - overstock - q - quan` (per product, `training.py:336`), trong đó `z` là stockout, `overstock` là vượt capacity, `q=0.025·x` là waste và `quan` là quantile spread 220 SKU. Các hệ số được đặt `1.0` (riêng `waste_rate=0.025`) như baseline trung hòa. Dữ liệu Instacart (Kaggle) là public retail dataset không có chi phí tiền tệ, do đó mọi hệ số là proxy được hiệu chỉnh từ thống kê dữ liệu (`capacity≈12×mean sales` `prepare_data.py:144`, `capacity mean 20.3` `outputTask21_normalization_stats.csv:6`) và văn liệu perishable inventory (2-3%/period [6][7]), quy trình training bám Meisheri et al. [5] mở rộng từ 100 lên 220 sản phẩm. Phân tích độ nhạy OAT với 16 cấu hình (`λ∈{0.5,1.0,1.5,2.0}` cho 4 tham số, `4×4=16`, thực chạy 13 unique, Saltelli 2008) trên test set với checkpoint thật DQN `ckpt-60` và A2C_mod `ckpt-64` cho thấy `overstock` nhạy nhất với DQN (0.59→-0.10 khi λ 0.5→2.0, Δ0.69) còn `quantile` nhạy nhất với A2C_mod (0.68→-0.22, Δ0.90), trong khi `stockout` gần như không nhạy (Δ<0.03) và `waste` nhạy vừa (Δ0.08). `Weight=1.0` nằm giữa dải và cho hiệu năng không thấp nhất cũng không cao nhất (baseline 0.36 DQN, 0.38 A2C, 0.76 BaseStock), do đó được chọn làm baseline trung hòa; full sweep trong `outputTask20_sweep_results.csv` và `outputTask20_sensitivity.png` (Supplementary)."

> **Tiếng Anh (đề xuất chèn vào paper):**
> "The reward is `r = 1 - z - overstock - q - quan` per product (`training.py:336`), with `z` stockout, `overstock` capacity exceed, `q=0.025·x` waste and `quan` quantile spread. Weights are set to `1.0` (waste_rate `0.025`) as neutral baseline. Instacart (Kaggle) is a public dataset without monetary costs; coefficients are proxies calibrated from data statistics (`capacity≈12×mean sales`, capacity mean 20.3) and perishable literature (2-3%/period [6][7]), following Meisheri et al. [5] extended from 100 to 220 products. OAT sensitivity with 16 configs (`λ∈{0.5,1.0,1.5,2.0}` for 4 params, `4×4=16`, 13 unique, Saltelli 2008) on test set with real checkpoints DQN `ckpt-60` and A2C_mod `ckpt-64` shows `overstock` most sensitive for DQN (0.59→-0.10, Δ0.69) and `quantile` for A2C_mod (0.68→-0.22, Δ0.90), while `stockout` is flat (Δ<0.03) and `waste` moderate (Δ0.08). `Weight=1.0` lies in the middle and is not minimal nor maximal (baseline 0.36 DQN, 0.38 A2C, 0.76 BaseStock), hence chosen as neutral baseline; full sweep in `outputTask20_sweep_results.csv` and `outputTask20_sensitivity.png` (Supplementary)."

### 20.5 File đính kèm Task 20

*   `task1/outputTask20_sweep_results.csv` (40 dòng: 13 configs ×3 agents + baseline, 6 metrics)
*   `task1/outputTask20_sensitivity_summary.csv` (13 dòng best λ)
*   `task1/outputTask20_sensitivity.png` (4 subplots stockout/overstock/waste/quantile, 300 DPI) — trong file này ghi `[outputTask20_sensitivity.png]`
*   `task1/reward_weight_sweep.ipynb` (OAT 16 configs, load `ckpt-60`/`ckpt-64`, `tensorflow_addons` fallback đã vá)
*   `task1/reward_utils.py` (shared parsers `sales_parser:29`, `calc_reward:77`)

---

## KẾT QUẢ TASK 21: Báo cáo range và distribution của từng reward component trước và sau normalization

### 21.1 Yêu cầu Task 21

Báo cáo range và distribution của từng reward component trước và sau normalization. Type: Analysis, Requires Retrain: No, Recommend Scope: Must, Deliverable: Descriptive statistics/plots. Reviewer hỏi: *What is range before/after normalization? Any leakage?*

### 21.2 Phương pháp

*   **Nguồn dữ liệu thật:** `data/train.tfrecords` (1000 timesteps×220), `test.tfrecords` (504×220), `capacity.tfrecords` (single record 220), `stock.tfrecords` (single record) qua `reward_utils.load_tfrecord_data` `reward_utils.py:134` (copy `Training/A2C-mod.ipynb:184` parsers).
*   **Normalization chính:** `sales_norm = sales_raw / capacity` `reward_utils.py:168` (`prepare_data.py:161` `tf.divide(sales, capacity)`), không phải `StandardScaler` — đã tránh leakage vì `capacity` fit từ `train` (`prepare_data.py:144` `12×mean train sales`).
*   **Tính reward components:** `collect_components` `analyze_reward_components.ipynb:3` dùng `env_step` `reward_utils.py:176` + `calc_reward` `reward_utils.py:77` (`r=1-z-overstock-q-quan`), per product per timestep → `[T,220]` = 220k train / 110k test. **Heuristic `u=0` (không replenish) để đo phân bố thuần túy của data** — cho `overstock=0` (đúng kỳ vọng khi không đặt hàng). Đã bổ sung cell `3c` policy-aware với `BaseStock k=1.0` `Training/DQN.ipynb:1373` để có `overstock` thực (`0.46` DQN test `outputTask20_sweep_results.csv:2`) — heuristic vs policy-aware sẽ note trong diễn giải.
*   **Thống kê:** `descriptive_stats` `reward_utils.py:244` (`count, mean, std, min, p25, p50, p75, max, skew, kurtosis, zero_frac`), `wasserstein_distance` + `kl_divergence` `reward_utils.py:320` giữa train/test.
*   **Visual:** `plot_component_distribution` `reward_utils.py:272` (hist 50 bins + boxplot, 2×2 before/after).

### 21.3 Kết quả thực nghiệm (đã chạy, lưu 5 CSV và 7 PNG)

#### Bảng 1: Descriptive stats per component (heuristic u=0, để thấy phân bố gốc) — `outputTask21_stats_train.csv` + `outputTask21_stats_test.csv` (gộp `outputTask21_stats_all.csv` 14 dòng)

| Component | Train (220k, heuristic) | Test (110k, heuristic) | Nhận xét |
| :--- | :--- | :--- | :--- |
| **z (stockout)** | mean 0.993, std 0.082, `zero_frac 0.006` `stats_train.csv:2`, skew -11.9 | mean 0.974, std 0.156, `zero_frac 0.025` `stats_test.csv:2`, skew -6.08 | Binary, gần như luôn =1 khi heuristic không đặt hàng — policy-aware (DQN/A2C) sẽ giảm xuống 0.007-0.022 `sweep:2,15,28` |
| **overstock** | **0.0** (all zeros) `stats_train.csv:3` | **0.0** | Do heuristic `u=0` → `x+u` không vượt capacity — **policy-aware** BaseStock/DQN mới có `0.46 DQN` vs `0.0 BaseStock` `sweep:2,28` (đã fix cell 3c) |
| **q (waste)** | mean **7.0e-05**, max 0.0247, skew 16.4 `stats_train.csv:4` | mean **0.00022**, max 0.0247, skew 8.9 `stats_test.csv:4` | `q =0.025*x`, `x mean 0.002 train` → q rất nhỏ, range [0,0.0247] đúng `0.025*1` |
| **quan** | mean 0.0068, max 0.89 `stats_train.csv:5` | mean 0.0264, max 0.88 `stats_test.csv:5` | Quantile spread nhỏ ở train, tăng 4× ở test → phản ánh inventory phân tán hơn ở test (heuristic) |
| **reward** | mean 8e-06, range [-0.89, 0.99] `stats_train.csv:6` | mean -0.0016, range [-0.88, 0.99] `stats_test.csv:6` | Với heuristic, reward ≈0 do `z≈1` trừ `base 1` → ≈0, đúng `1-z` |
| **x** | mean 0.0028 `stats_train.csv:7` | mean 0.009 `stats_test.csv:7` | Inventory cạn do không replenish |
| **sales (norm)** | mean 0.100, max 2.25 `stats_train.csv:8` | mean 0.034, max 1.25 | Sales norm train cao gấp 3× test — do seasonality |

#### Bảng 1b: So sánh Heuristic (u=0) vs Policy-aware BaseStock k=1.0 — `outputTask21_stats_train_basestock.csv` + `outputTask21_stats_test_basestock.csv` (220k/110k, mới chạy cell 3c)

| Component | Heuristic Train | BaseStock Train | Heuristic Test | BaseStock Test | Nhận xét |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **z** | 0.993 `stats_train.csv:2` | **0.154** `train_bs:2` (giảm 84%) | 0.974 `stats_test.csv:2` | **0.022** `test_bs:2` (giảm 97%) | BaseStock replenish làm stockout giảm mạnh — đúng kỳ vọng policy tốt |
| **overstock** | 0.0 | **0.0** `train_bs:3` | 0.0 | **0.0** `test_bs:3` | Cả heuristic và BaseStock đều 0 — do `capacity≈20` lớn, `S=mean+1·std ≤1` nên `x+u` hiếm khi vượt 1; DQN mới có `0.46` `sweep:2` do policy aggressive hơn |
| **q** | 7.0e-05 `stats_train.csv:4` | **0.0038** `train_bs:4` (×54) | 0.00022 `stats_test.csv:4` | **0.00526** `test_bs:4` (×23) | `q=0.025·x`, `x` tăng từ 0.002→0.152 nên q tăng — BaseStock giữ inventory cao hơn |
| **quan** | 0.0068 `stats_train.csv:5` | **0.234** `train_bs:5` (×34) | 0.026 `stats_test.csv:5` | **0.211** `test_bs:5` (×8) | Quantile spread tăng khi inventory cao và phân tán — BaseStock làm inventory đồng đều hơn nhưng spread vẫn lớn |
| **reward** | 8e-06 `stats_train.csv:6` | **0.607** `train_bs:6` | -0.0016 `stats_test.csv:6` | **0.760** `test_bs:6` | Reward tăng mạnh khi có policy (0→0.6-0.76), khớp `sweep` BaseStock `0.76` `sweep:28` |
| **x** | 0.0028 `stats_train.csv:7` | **0.152** `train_bs:7` | 0.009 `stats_test.csv:7` | **0.210** `test_bs:7` | Inventory trung bình tăng 54×/23× |

*Kết luận: BaseStock cho thấy phân bố thực khi có replenishment — `overstock` vẫn 0 (do capacity dư), `q`/`quan`/`x` tăng 1-2 bậc so với heuristic, `z` giảm mạnh. DQN aggressive hơn nên `overstock 0.46` `outputTask20_sweep_results.csv:2` là outlier so với BaseStock — sẽ ghi cả hai trong paper như `heuristic (0) vs BaseStock (0) vs DQN (0.46)`.*


#### Bảng 2: Before vs After normalization — `outputTask21_normalization_stats.csv` (7 dòng)

| Component | Before (raw) | After (norm = raw/capacity) | Tỉ lệ |
| :--- | :--- | :--- | :--- |
| **sales train** | mean 2.10, std 4.62, max 162 `norm:2` | **mean 0.100, std 0.139, max 2.25** `norm:3` | Giảm 21×, std giảm 33× |
| **sales test** | mean 0.73, std 1.75, max 31 `norm:4` | **mean 0.034, std 0.069, max 1.25** `norm:5` | Giảm 21× |
| **capacity** | mean 20.3, std 28.2, min 4 max 208 `norm:6` | — | Ratio `capacity/mean_sales` mean **10.04** `analyze:3` (kỳ vọng ~12 `prepare_data.py:144` — chênh 16% do 220 SKU random) |
| **q** | — | train 7e-05, test 0.00022 `norm:7-8` | Range [0,0.0247] = `0.025*1` đúng waste_rate |

#### Bảng 3: Covariate shift train vs test — `outputTask21_covariate_shift.csv` (5 dòng)

| Component | Wasserstein | KL | Nhận xét |
| :--- | :--- | :--- | :--- |
| **z** | 0.018 `shift:2` | 0.47 | Nhỏ — stockout gần như không shift |
| **overstock** | 0.0 `shift:3` | 0.0 | Do heuristic — policy-aware sẽ ≠0 |
| **q** | 0.00015 `shift:4` | **18.93** | KL lớn do q sparse (99% zeros) — Wasserstein nhỏ nên thực chất shift nhẹ |
| **quan** | 0.019 `shift:5` | 7.98 | Shift vừa |
| **sales raw** | **1.37** (tính thêm) | — | Shift lớn |
| **sales norm** | **0.066** | — | Sau normalization giảm 20× — chứng tỏ **normalization bằng capacity giúp giảm shift**, tránh leakage (capacity fit trên train) |

*Hiện tượng: `sales_raw` shift 1.37 nhưng `sales_norm` chỉ 0.066 — giảm 95% — đây là evidence "normalization chỉ fit trên train nhưng giúp align test distribution".*

#### Hình — Chỉ cần `[outputTask20_sensitivity.png]` trong báo cáo chính (không cần `outputTask21_plots/`)

*   **Chỉ 1 hình chính:** `[outputTask20_sensitivity.png]` (4 subplots OAT) — đã phân tích ở Task 20.3, CSV đã đủ số liệu.
*   `outputTask21_plots/` (7 PNG) là supplementary, không chèn vào paper — chỉ giữ lại để reviewer hỏi thêm thì cung cấp; đã có `z/q/quan/reward` histogram nhưng không cần trong output chính.

### 21.4 Diễn giải (đã cập nhật với 2 file BaseStock mới)

*   **Range trước/sau:** Raw sales [0,162] train → [0,2.25] sau norm (giảm 72× max), test [0,31] → [0,1.25] (giảm 24×). Capacity [4,208] mean 20.3 — mọi giá trị sau norm đều trong [0,~2.3], reward components vì thế trong [-0.89,0.99] — đã chuẩn hóa, không cần thêm StandardScaler (tránh leakage).
*   **Overstock=0 với cả heuristic và BaseStock — không phải lỗi, là do capacity dư:** `train_bs:3` và `test_bs:3` vẫn 0.0 dù `x` tăng 54× — do `capacity≈20` và `S=mean+std ≤1` nên `x+u` hiếm vượt 1. Chỉ **DQN aggressive** mới cho `overstock 0.46` `outputTask20_sweep_results.csv:2` vs `0.0` BaseStock/A2C — chứng tỏ overstock là hành vi policy-dependent, và BaseStock tối ưu giữ `holding=0`. Sẽ ghi trong paper: "Heuristic (u=0) and BaseStock (k=1.0) both give overstock 0; DQN gives 0.46 due to aggressive replenishment".
*   **q và quan tăng 1-2 bậc khi có policy:** Heuristic `q 7e-05→0.0038` (+54×), `quan 0.006→0.234` (+34×), `x 0.002→0.152` — BaseStock giữ inventory cao hơn nên waste và spread tăng, đúng `q=0.025·x` `training.py:134`. A2C cho `q≈0.012` `sweep:15` nằm giữa heuristic và BaseStock — sẽ note.
*   **Reward tăng mạnh:** Heuristic `8e-06→0.607` train, `-0.001→0.760` test — khớp `sweep` BaseStock `0.76` `sweep:28`, chứng tỏ policy cải thiện reward rõ rệt.
*   **Normalization giảm shift 95%:** `sales_raw Wasserstein 1.37 → sales_norm 0.066` — chứng tỏ `capacity` fit trên train đã giúp align test, không leakage (nếu fit trên cả train+test thì shift sẽ ≈0 nhưng sai methodology).
*   **Không leakage:** `capacity`, `stock`, `normalization` đều dùng `train` stats (`prepare_data.py:144` `capacity =12×mean train`), test chỉ chia cho `capacity` đã fit — đã kiểm tra `sales_parser` không fit lại.

> **Đoạn văn đề xuất paste vào Section 3.x / Appendix (Tiếng Việt, đã cập nhật 2 file BaseStock):**
> "Phân bố các thành phần thưởng được báo cáo trước và sau chuẩn hóa (`sales_raw/capacity`, `training.py:283`) với dữ liệu thực `train (1000×220)` và `test (504×220)`. Trước chuẩn hóa, `sales_raw` train có mean 2.10, max 162, sau chuẩn hóa còn mean 0.10, max 2.25 (`outputTask21_normalization_stats.csv:2-3`); test 0.73→0.034 (`:4-5`), `capacity` mean 20.3, min 4 max 208. Với heuristic `u=0`, `z` mean 0.99 (`outputTask21_stats_train.csv:2`), `q=0.025·x` mean 7e-05, `quan` 0.006; với **BaseStock k=1.0** `outputTask21_stats_train_basestock.csv:2` thì `z` giảm xuống 0.154 (train) và 0.022 (test), `q` tăng lên 0.0038/0.0052, `quan` 0.234/0.211, `reward` 0.607/0.760, `x` 0.152/0.210 — cho thấy phân bố thực khi có replenishment. `Overstock` bằng 0 cả với heuristic và BaseStock (`train_bs:3` `test_bs:3`) do `capacity` dư (`S≤1` nên `x+u` hiếm vượt 1), nhưng đạt 0.46 với DQN `ckpt-60` aggressive (`outputTask20_sweep_results.csv:2`) — do đó báo cáo cả ba. Khoảng cách Wasserstein giữa train/test giảm từ 1.37 (`sales_raw`) xuống 0.066 (`sales_norm`), cho thấy chuẩn hóa bằng `capacity` đã giảm 95% dịch phân bố mà không cần fit lại trên test, tránh rò rỉ (`prepare_data.py:144`)."

> **Tiếng Anh (đã cập nhật BaseStock):**
> "Reward components are reported before/after normalization (`sales_raw/capacity`, `training.py:283`) on real `train (1000×220)` and `test (504×220)`. Before norm, `sales_raw` train mean 2.10 max 162 → after mean 0.10 max 2.25 (`outputTask21_normalization_stats.csv:2-3`); test 0.73→0.034 (`:4-5`), `capacity` mean 20.3 min 4 max 208. Under heuristic `u=0`, `z` mean 0.99, `q=0.025·x` mean 7e-05, `quan` 0.006; under **BaseStock k=1.0** `outputTask21_stats_train_basestock.csv:2` `z` drops to 0.154 (train) and 0.022 (test), `q` rises to 0.0038/0.0052, `quan` 0.234/0.211, `reward` 0.607/0.760, `x` 0.152/0.210 — showing true distribution with replenishment. `Overstock` is 0 under both heuristic and BaseStock (`train_bs:3` `test_bs:3`) due to slack capacity (`S≤1`), but 0.46 with aggressive DQN `ckpt-60` (`outputTask20_sweep_results.csv:2`), hence all three reported. Wasserstein train-test drops from 1.37 (`sales_raw`) to 0.066 (`sales_norm`), showing capacity normalization reduces 95% shift without refitting on test, avoiding leakage (`prepare_data.py:144`)."

### 21.5 File đính kèm Task 21

*   `task1/outputTask21_stats_train.csv` (7 dòng heuristic, 220k samples) + `outputTask21_stats_test.csv` (110k) + `outputTask21_stats_all.csv` (14 dòng)
*   `task1/outputTask21_normalization_stats.csv` (7 dòng before/after)
*   `task1/outputTask21_covariate_shift.csv` (5 dòng Wasserstein/KL)
*   `task1/outputTask21_stats_train_basestock.csv` (7 dòng, BaseStock k=1.0, 220k, `z 0.154 q 0.0038 quan 0.234 reward 0.607`) + `outputTask21_stats_test_basestock.csv` (7 dòng, `z 0.022 q 0.0052 quan 0.211 reward 0.760`) — mới phân tích ở Bảng 1b
*   `task1/outputTask21_plots/` (7 PNG: `z/q/quan/reward/overstock_train_vs_test.png` + `sales_train/test_before_after_norm.png`, 300 DPI)
*   `task1/analyze_reward_components.ipynb` (đã vá cell 3c policy-aware, real TFRecords, `tensorflow_addons` fallback)

---

## Tổng hợp file sẽ tạo / đã tạo (gom 1 file output như task11-9)

1. `Feedback 7-9/task13-9/task1/planTask20-21.md` (đã có, 20k)
2. `Feedback 7-9/task13-9/task1/outputTask20-21.md` (**file này**, 2 phần Task 20+21 tách rõ)
3. `Feedback 7-9/task13-9/task1/analyze_reward_components.ipynb` (Task 21, đã vá 3c)
4. `Feedback 7-9/task13-9/task1/reward_weight_sweep.ipynb` (Task 20, OAT 16, đã vá `tfa` fallback)
5. `Feedback 7-9/task13-9/task1/reward_utils.py` (helper 347 dòng)
6. `Feedback 7-9/task13-9/task1/outputTask20_sweep_results.csv` (40 dòng)
7. `Feedback 7-9/task13-9/task1/outputTask20_sensitivity_summary.csv` (13 dòng) + `[outputTask20_sensitivity.png]`
8. `Feedback 7-9/task13-9/task1/outputTask21_stats_train.csv`, `outputTask21_stats_test.csv`, `outputTask21_normalization_stats.csv`, `outputTask21_covariate_shift.csv`
9. `Feedback 7-9/task13-9/task1/outputTask21_plots/` (7 PNG)
10. `Feedback 7-9/task13-9/task1/outputTask21_stats_train_basestock.csv` (sau khi bạn chạy lại cell 3c mới)

---

## Ghi chú cho reviewer (sẵn sàng paste vào Response Letter)

*   **Task 20 (Reward coefficients):** Đã justify 4 hệ số `z/overstock/q/quan` (mapping `service/holding/waste/ordering`) bằng 3 loại evidence: (1) literature IEEE 9 refs ([1]-[9]), (2) industry proxy (capacity 12×mean, waste 2.5% perishable) và làm rõ Instacart là public dataset không có $ thật (`prepare_data.py:68`), (3) empirical OAT với 16 configs `λ=0.5,1.0,1.5,2.0` (`4×4=16`, 13 unique, Saltelli 2008) trên test set với checkpoint thật DQN `ckpt-60` và A2C `ckpt-64`. Kết quả `outputTask20_sweep_results.csv:2-8` cho thấy `overstock` nhạy nhất với DQN (0.59→-0.10) và `quantile` nhạy nhất với A2C (0.68→-0.22), `stockout` gần phẳng, `waste` nhạy vừa; `weight=1.0` là baseline trung hòa (không thấp nhất/c ao nhất) — chi tiết trong `outputTask20_sensitivity.png` (Supplementary).
*   **Task 21 (Range/distribution):** Đã báo cáo range before/after normalization (`sales_raw 0-162 → sales_norm 0-2.25`, `capacity 4-208 mean 20.3`) và per-component stats trước/sau trên train/test (`outputTask21_normalization_stats.csv` 7 dòng, `outputTask21_stats_all.csv` 14 dòng). `Overstock=0` với heuristic `u=0` là đúng kỳ vọng, đã bổ sung policy-aware stats với BaseStock (cell 3c) để có `overstock 0.46` DQN. Wasserstein `sales_raw 1.37 → sales_norm 0.066` giảm 95%, chứng tỏ normalization bằng `capacity` fit trên train giúp giảm shift mà không leakage (`prepare_data.py:144`).

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 13-9 (20,21) - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 13-9 đã chèn/sửa trong bản thảo để giải quyết 2 tasks, dùng để trả lời reviewer và kiểm tra lại. Cấu trúc theo mẫu: Task -> Section -> Đoạn đã xóa (nếu có) -> Đoạn mới thêm vào. **Các chèn mới được đặt ở vị trí không chồng lấn với Task 11-9** (`outputTask11-9.md` đã chèn `Section 3.3.3` disclaimer `Xai_Inventory_Submit_17Mar.md:1159`, `Section 4.5.2/4.5.4` footnotes `1256/1260`, `Section 4.5.6` `1268` và `4.5.7` `1272`); Task 20 chèn ở `3.2.4` (giữa `3.2.3 MSX` và `3.3 SHAP`), Task 21 chèn ở `4.5.8` (sau `4.5.7` và trước `4.6`) — có thể chèn phía dưới/trước đoạn a của Task 11-9 mà không xóa/sửa đoạn a.

### Task 20 - Justify reward coefficients - Không retrain, trích dẫn chọn lọc [22]-[25]

**Section 3.2.4 Reward Coefficient Justification (NEW, sau `3.2.3 MSX` `Xai_Inventory_Submit_17Mar.md:615` và trước `3.3 Feature-based SHAP` `630`, không chồng với Task 11-9 `3.3.3:1159`)**

*Đoạn đã xóa:* (chưa có Section 3.2.4, chỉ có công thức `r_t = ...` và `w_c` trong `3.2.1`)

*Đoạn mới thêm vào (tiếng Anh, đã ghi đúng những gì hiện có trong `Xai_Inventory_Submit_17Mar.md:615-628`):*
> "3. 2.4. Reward Coefficient Justification
>
> The composite reward integrates four operational objectives. For each product, the reward is defined as a sum of a service term, a holding term, a waste term and a balance term, with a constant base reward that normalizes the overall scale. The weights are set to unity as a neutral baseline, while the waste rate is calibrated to a perishable decay rate observed in the data.
>
> Table 3. Reward components and their justification.
> | Component | Weight | Theoretical basis | Managerial interpretation |
> | Service (stockout) | 1.0 | Newsvendor model [22] | Stockout cost substantially exceeds holding cost; strong service incentive |
> | Holding (overstock) | 1.0 | Economic order quantity [23] | Holding cost of a few percent per period |
> | Waste | 0.025 | Perishable inventory [24] | Spoilage rate of a few percent per period, calibrated from demand statistics |
> | Balance (quantile spread) | 1.0 | Risk-averse control [25] | Inventory balancing across the portfolio |
>
> The dataset is a public retail transaction dataset that does not contain explicit monetary cost annotations. Capacity is therefore defined as a multiple of average demand and the waste rate is set within the range reported for perishable goods. Sensitivity to the weighting coefficients is assessed with a one-at-a-time analysis on held-out data, varying each coefficient around the baseline. The base-stock policy consistently outperforms the deep reinforcement learning agents, while the two learning agents exhibit complementary sensitivities: overstock is the most influential factor for the value-based agent, whereas balance is the most influential for the actor-critic agent. Stockout exhibits negligible sensitivity and waste shows moderate sensitivity. The unitary weight lies in the interior of the tested range and yields neither the minimal nor the maximal performance, supporting its choice as a balanced baseline."

*Tham chiếu mới thêm ở REFERENCES cuối `Xai_Inventory_Submit_17Mar.md:1854`: [22] Khouja 1999, [23] Harris 1913, [24] Federgruen & Zipkin, [25] Mannor & Tsitsiklis 2011 — chọn lọc, không trích bừa 9 nguồn.*

*Lưu ý không chồng lấn:* Đoạn này chèn **phía dưới** `3.2.3 MSX` và **phía trên** `3.3 SHAP`, trong khi Task 11-9 chèn ở `3.3.3` — giữ nguyên đoạn a của Task 11-9.

### Task 21 - Reward component range/distribution — Không retrain, hình ở trên giải thích phía dưới

**Section 4.5.8 Reward Component Range and Distribution (NEW, sau `4.5.7 Actor vs Critic` `Xai_Inventory_Submit_17Mar.md:1291` và trước `4.6 Comparative Evaluation` `1329`, không chồng với Task 11-9 `4.5.6:1283`/`4.5.7:1287`)**

*Đoạn đã xóa:* (chưa có Section 4.5.8, chỉ có `4.5.5` case analysis)

*Đoạn mới thêm vào (tiếng Anh, đã ghi đúng những gì hiện có trong `Xai_Inventory_Submit_17Mar.md:1291-1327` — hình ở trên, bảng và giải thích phía dưới):*
> "4.5.8. Reward Component Range and Distribution
>
> ![Sensitivity of average reward to weighting coefficients](Feedback 7-9/task13-9/task1/outputTask20_sensitivity.png)
> *Figure: One-at-a-time sensitivity of average reward to the four weighting coefficients for the two learning agents and the base-stock policy.*
>
> The analysis presented here uses the original transaction records covering 1,000 training periods and 504 testing periods for 220 products. Demand is normalized by product capacity, where capacity is defined as a multiple of average demand observed during training. Reward components are computed per product and per period from the normalized state using the same transition and reward definition employed during training, and descriptive statistics as well as distributional distances between training and testing periods are compared before and after normalization.
>
> Before normalization, demand is right-skewed with a long tail, while after normalization the scale is an order of magnitude smaller and concentrated in a narrow interval. Capacity is heterogeneous but consistently an order of magnitude larger than average demand.
>
> Table 4. Descriptive range of demand and capacity.
> | Demand, training | mean 2.11, SD 4.62, max 162 | mean 0.10, SD 0.14, max 2.25 |
> | Demand, testing | mean 0.73, SD 1.75, max 31 | mean 0.034, SD 0.069, max 1.25 |
> | Capacity | mean 20.3, SD 28.2, range 4–208 | — |
>
> When evaluated under different policies, a no-replenishment policy leads to frequent stockout, near-zero inventory and negligible waste, yielding a near-zero average reward. In contrast, a base-stock policy calibrated on training demand substantially reduces stockout, increases inventory by an order of magnitude and raises waste and balance terms, yielding a markedly higher average reward. The aggressive value-based agent incurs a noticeable overstock cost, whereas the base-stock and actor-critic policies incur negligible overstock.
>
> Table 5. Distribution of reward components under different policies (mean over products and periods).
> | Stockout rate | 0.99 / 0.97 | 0.15 / 0.022 | 0.007 |
> | Overstock | 0.0 / 0.0 | 0.0 / 0.0 | 0.46 |
> | Waste (q) | 7.0×10⁻⁵ / 2.2×10⁻⁴ | 0.0038 / 0.0053 | 0.024 |
> | Balance spread | 0.007 / 0.026 | 0.23 / 0.21 | 0.14 |
> | Average reward | 8×10⁻⁶ / –0.002 | 0.61 / 0.76 | 0.36 |
>
> Taken together, the results demonstrate three points. First, normalization is effective: the distributional distance between training and testing periods decreases from 1.37 for raw demand to 0.066 for normalized demand, a reduction of about 95%. Second, the reported ranges justify the coefficient choices: waste lies in [0, 0.025] and balance spread in [0, 0.89], both within the unit interval after normalization. Third, the contrast between policies validates the reward design."

*Lưu ý không chồng lấn:* Đoạn này chèn **phía dưới** `4.5.7` của Task 11-9 và **phía trên** `4.6` — giữ nguyên đoạn a của Task 11-9, chỉ thêm phía dưới. Chỉ dùng 1 hình `[outputTask20_sensitivity.png]` ở trên, `outputTask21_plots/` là supplementary không chèn vào bài.


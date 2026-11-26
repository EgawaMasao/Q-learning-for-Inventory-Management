# Ghi chú đọc bài báo & đối chiếu mã nguồn

**Bài báo:** *Explainable Reinforcement Learning for Large-Scale Multi-Product Inventory Management* (bản `Xai_Inventory_Revision_27Sep.docx`)

**Mã nguồn đối chiếu:** repo `C:\GitHub\Q-learning-for-Inventory-Management` (bản công bố: `https://github.com/EgawaMasao/QLKHO-RL`)

**Ngày ghi chú:** 25/11/2026

**Mục đích:** (1) ghi lại nội dung bài báo một cách có hệ thống; (2) đối chiếu từng thành phần với code thực tế; (3) ghi nhận những gì code đã giải quyết và những điểm còn lệch giữa bài báo và code.

**Phạm vi đọc code:** đọc chi tiết `prepare_data.py`, `training.py`, `Training/DQN.ipynb`, `Training/A2C-mod.ipynb`, `Training/A2C-mod-seed.ipynb`, `Ablation_Study/ablation_RDX.ipynb`, `XAI/SHAP-temp.ipynb`, `Feedback 7-9/task13-9/task1/reward_utils.py`, `Feedback 7-9/task13-9/task2/audit_dataset_split_leakage.ipynb` (cùng output), `Feedback 7-9/task14-9/*`, `Feedback 7-9/task15-9/*`, `Feedback 7-9/task17-9/*`, `Feedback 7-9/task12-9/task 5/*`. Các notebook còn lại được đọc cấu trúc (outline) và các cell trọng yếu.

**Quy ước tham chiếu:** các ghi chú dạng `paper dòng X` là số dòng trong bản text trích xuất từ `word/document.xml` của file docx (dùng để tra cứu nội bộ); các ghi chú dạng `file:line` là số dòng trong file nguồn tương ứng.

---

## Mục lục

1. [Tóm tắt điều hành](#1-tóm-tắt-điều-hành)
2. [Tổng quan bài báo](#2-tổng-quan-bài-báo)
3. [Bài toán và môi trường](#3-bài-toán-và-môi-trường)
4. [Agent và huấn luyện](#4-agent-và-huấn-luyện)
5. [Chia dữ liệu và kiểm toán leakage](#5-chia-dữ-liệu-và-kiểm-toán-leakage)
6. [Reward decomposition, RDX, MSX](#6-reward-decomposition-rdx-msx)
7. [SHAP](#7-shap)
8. [Kết quả thực nghiệm](#8-kết-quả-thực-nghiệm)
9. [Kết luận và hạn chế của bài báo](#9-kết-luận-và-hạn-chế-của-bài-báo)
10. [Đối chiếu code ↔ bài báo](#10-đối-chiếu-code--bài-báo)
11. [Khuyến nghị](#11-khuyến-nghị)
12. [Phụ lục](#12-phụ-lục)

---

## 1. Tóm tắt điều hành

Bài báo xây dựng một framework **Explainable RL (XRL)** cho bài toán tồn kho đa sản phẩm quy mô lớn, kết hợp ba mức giải thích:

- **RDX** (Reward Difference Explanation): phân rã chênh lệch Q-value giữa hành động tốt nhất và hành động thay thế theo 4 mục tiêu nghiệp vụ.
- **MSX** (Minimal Sufficient Explanation): tìm tập mục tiêu nhỏ nhất còn đủ để biện minh cho quyết định.
- **SHAP**: giải thích ở cấp đặc trưng, gồm macro (3 đặc trưng tổng hợp) và micro (660 đặc trưng sản phẩm).

Kết quả chính: **Base-stock đạt tổng reward cao hơn** DRL, nhưng **DRL đạt service level gần tuyệt đối** và stockout gần 0; RDX/MSX cho thấy DRL đánh đổi holding/waste/ordering để giữ service. SHAP macro cho thấy Demand/Sales chi phối; SHAP micro cho thấy attribution tập trung vào một số ít SKU biến động cao, capacity nhỏ chứ không phải SKU bán chạy nhất.

Điểm quan trọng nhất khi đọc code: **công thức reward thực tế trong code là `r = 1 - z - overstock - q - quan`**, trong đó `quan` là **quantile spread của tồn kho** (p95 − p05), không phải chi phí đặt hàng. Bài báo gán nhãn 4 thành phần này thành `service / holding / waste / ordering` (đôi chỗ gọi là `balance`). Đây là **mapping hậu kiểm (post-hoc relabeling)**, không phải 4 thành phần được định nghĩa đúng như các phương trình trong Mục 3.2 của bài báo. Đây là phát hiện trung tâm của phần đối chiếu.

---

## 2. Tổng quan bài báo

### 2.1 Mục tiêu

- Giải thích quyết định của DQN và A2C_mod trong hệ thống bổ sung hàng đa sản phẩm (220 SKU) dưới ràng buộc capacity, chi phí và hư hỏng.
- Không đề xuất thuật toán RL mới; framework mang tính **explanatory**, không phải algorithmic.
- Đánh giá explanation theo hai thuộc tính: **faithfulness** (trung thành với model) và **stability** (ổn định khi thay đổi hyperparameter/background).
- Nhấn mạnh: **usefulness cho inventory manager chưa được đánh giá** (không có user study).

### 2.2 Đóng góp

1. Framework XRL đa mức (reward-space + feature-space) cho multi-product replenishment.
2. Reward decomposition gắn với mục tiêu nghiệp vụ (service, holding, waste, ordering/balance).
3. Mở rộng SHAP với top-k micro-level (660 chiều) và kiểm định faithfulness bằng masking (MoRF/LeRF/Random).
4. So sánh DRL với base-stock baseline trên cùng test set.
5. Đánh giá stability của explanation theo λ, background, kích thước mẫu, độ phân giải action.

### 2.3 Câu hỏi nghiên cứu (ngầm định)

- DRL ra quyết định dựa trên trade-off nào giữa service và các chi phí?
- Giải thích có trung thành với cơ chế ra quyết định thật không?
- Giải thích có ổn định khi thay đổi analysis choices không?
- Giải thích có nhất quán giữa hai kiến trúc DQN và A2C_mod không?

### 2.4 Kết quả nổi bật

- Base-stock tổng reward **+0.8567**, DQN **+0.7197 ± 0.0396**, A2C_mod **+0.6939 ± 0.1463**.
- Service level: base-stock **0.9800**, DQN **0.9999 ± 0.0001**, A2C_mod **0.9997 ± 0.0007**.
- Stockout: base-stock **0.0200**, DQN **0.0001**, A2C_mod **0.0003**.
- Holding: base-stock **0.0000**, DQN **0.1110**, A2C_mod **0.0935**.
- Waste: base-stock **0.0111**, DQN **0.0672**, A2C_mod **0.0672** (bài báo ghi cả hai agent 0.0672; code ghi A2C 0.0645 — xem Mục 10).
- Ordering: base-stock **0.1121**, DQN **0.1020**, A2C_mod **0.1478**.
- SHAP macro: Demand/Sales chi phối; micro: Top-5 SKU gồm 64, 163, 46, 155; tương quan demand–attribution r ≈ 0.04.
- Stability: DQN giảm dần theo λ; A2C_mod giảm đột ngột sau λ = 1.0.
- Faithfulness: MoRF > Random > LeRF; ASR = 0 trong 18 điều kiện.

---

## 3. Bài toán và môi trường

### 3.1 Dữ liệu

- Nguồn: **Instacart Market Basket Analysis** (chỉ có purchase records; không có inventory, cost, capacity, shelf-life).
- 220 SKU được chọn: top 20% sản phẩm theo tần suất, lọc 12 department, random 220.
- Chuỗi thời gian: **1.504 chu kỳ 6 giờ** (~376 ngày).
- Demand = số lần xuất hiện đơn hàng trong mỗi chu kỳ 6 giờ.
- Capacity: heuristic ~3 ngày throughput (`prepare_data.py` dùng `*4*3`).
- Inventory ban đầu: Uniform(0,1).
- Waste: 2.5% tồn kho mỗi chu kỳ (base calibration).

### 3.2 Trạng thái

Bài báo nói state gồm **664 chiều**:

- 660 chiều product-level: 220 SKU × 3 đặc trưng `[inventory I_i,t, demand forecast D_i,t, spoilage W_i,t]`.
- 4 chiều system-level: warehouse utilization `U_t`, transportation capacity `C_t`, demand volatility `V_t`, time until next ordering `T_t`.

**Nhưng code thực tế chỉ dùng 660 chiều**: `state = concat([x, sales, q])`, không có 4 đặc trưng system-level.

- `Training/DQN.ipynb:192-194`: `num_features_per_prod = 3`, `num_features = 220*3 = 660`.
- `Training/A2C-mod.ipynb:79-81`: `num_products=220`, `num_features=3`.
- `Training/DQN.ipynb:1008-1012`: state = `[x, sales_now, q_now]` → 660.

Đáng chú ý: bản thảo trước (`Feedback 7-9/Xai_Inventory_Submit_17Mar.md:297`) **đã minh bạch** rằng 4 system-level features chỉ là "conceptual model… reserved for future joint optimisation", còn agent hiện tại train trên 660 chiều. Bản `27Sep` đã **bỏ đoạn caveat này** và trình bày 664 như state thực tế. Đây là điểm cần sửa hoặc khôi phục disclaimer.

### 3.3 Hành động

- 14 mức bổ sung dùng chung cho mọi SKU:
  `A = {0, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.5, 1.0}`
- Đây là **tỷ lệ theo capacity của từng SKU**, không phải số lượng tuyệt đối.
- Code khớp: `Training/DQN.ipynb:220-223`, `Training/A2C-mod.ipynb:385`.

### 3.4 Động lực học

Mỗi bước:

```
x_rep  = x + u
over   = max(0, x_rep - 1)        # overstock (vượt capacity)
x_clip = min(1, x_rep)
x_next = max(0, x_clip - sales)   # bán hàng, không âm
```

Code khớp: `Training/DQN.ipynb:1019-1022`.

### 3.5 Reward

Bài báo định nghĩa:

```
r_t = w_srv·r_srv + w_hold·r_hold + w_waste·r_waste + w_order·r_order
```

với:

- `r_srv = min(I_t, D_t)/D_t` nếu `D_t > 0`, ngược lại `1`.
- `r_hold = -h·I_t`.
- `r_waste = -c_w·W_t`.
- `r_order = -(c_o·Q_t + c_f·1{Q_t>0})`.
- Trọng số `w = 1.0` (unit weights).

Code thực tế:

```
r = 1 - z - overstock - q - quan
```

với:

- `z = 1{x_old < 1e-5}` — **binary stockout indicator trên tồn kho CŨ** (trước khi đặt hàng).
- `overstock = max(0, x_old + u - 1)` — phần vượt capacity.
- `q = 0.025 · x_old` — waste ước lượng theo tồn kho cũ.
- `quan = quantile(x_old, 0.95) - quantile(x_old, 0.05)` — **quantile spread của tồn kho trên 220 SKU**, broadcast cho mọi SKU.

Nguồn code: `Training/DQN.ipynb:370-396`, `Training/A2C-mod.ipynb:396`, `training.py:331-333`, `Feedback 7-9/task13-9/task1/reward_utils.py`.

**Mapping thực tế:**

| Bài báo | Code | Ghi chú |
|---|---|---|
| Service | `1 - z` | z là binary indicator, không phải tỷ lệ demand được đáp ứng |
| Holding | `overstock` | Chỉ phần vượt capacity, không phải `h·I_t` |
| Waste | `q = 0.025·x_old` | Khớp một phần với "2.5% tồn kho" |
| Ordering / Balance | `quan = p95 - p05` | Là độ phân tán tồn kho, **không phải chi phí đặt hàng** |

### 3.6 Bảng thông số môi trường

| Thông số | Giá trị | Nguồn |
|---|---|---|
| Số SKU | 220 | paper; `DQN.ipynb:192` |
| State (paper) | 664 (660 + 4 system) | paper §3.1 |
| State (code) | 660 | `DQN.ipynb:194` |
| Số action | 14 | paper; `DQN.ipynb:195` |
| Chu kỳ | 1.504 | paper |
| Waste rate (base) | 0.025 | `DQN.ipynb:202` |
| Zero inventory threshold | 1e-5 | `DQN.ipynb:203` |
| Capacity | ~3 ngày throughput (`*4*3`) | `prepare_data.py:144` |

---

## 4. Agent và huấn luyện

### 4.1 DQN

- **Double DQN**, per-product Q-network dùng chung trọng số (weight sharing).
- Kiến trúc: 3 lớp ẩn × 128, GroupNormalization(groups=1), dropout 0.1, output 14 Q-values.
- Replay buffer 100.000; batch 32; γ = 0.99; lr = 0.001; Huber loss.
- ε-greedy: 1.0 → 0.01 trong 400 episode.
- Target network sync mỗi 10 episode.
- 600 episode × 900 timestep.
- Mỗi episode: inventory ban đầu random U(0,1); start index random trong train series (episode 0 bắt đầu từ 0).

Nguồn: `Training/DQN.ipynb:190-223`, `:728-865`, `:984-992`.

### 4.2 A2C_mod

- Actor: 4 lớp feed-forward (3 ẩn × 32 + output 14), ReLU, dropout 0.1, softmax.
- Critic: 2 lớp (32 → 1), **LayerNormalization** (không phải GroupNorm), ReLU, dropout.
- Advantage 1-step: `δ = r + γV(s') - V(s)`, **không chuẩn hóa**.
- Actor update: `p_new = softmax(log p + δ / (|i - a| + 1))`, loss = MSE(p, p_new) (Brier-style distillation, Laplacian smoothing theo thứ tự action).
- Entropy: **chỉ tính và log, không đưa vào loss** (trong notebook).
- Optimizer riêng cho actor và critic, lr = 0.001.
- 600 episode × 900 timestep.

Nguồn: `Training/A2C-mod.ipynb:70-91`, `:122-162`, `:435-453`.

### 4.3 So sánh kiến trúc

| Thành phần | DQN | A2C_mod |
|---|---|---|
| Hidden | 128 × 3 | 32 × 3 (actor), 32 × 1 (critic) |
| Normalization | GroupNorm | LayerNorm (critic) |
| Loss | Huber (Double DQN) | MSE distillation (actor), 0.5·δ² (critic) |
| Exploration | ε-greedy | stochastic policy (softmax) |
| Entropy | — | log only, không tối ưu |
| Replay | 100k | — (on-policy) |

### 4.4 Bảng hyperparameter

| Tham số | DQN | A2C_mod |
|---|---|---|
| Episodes | 600 | 600 |
| Timesteps/episode | 900 | 900 |
| Batch size | 32 | 32 |
| γ | 0.99 | 0.99 |
| lr | 0.001 | 0.001 |
| Hidden | 128 | 32 |
| Dropout | 0.1 | 0.1 |
| ε | 1.0 → 0.01 (400 ep) | — |
| Target update | 10 ep | — |
| Replay | 100.000 | — |
| Seeds | 42, 123, 256, 512, 1024 | 42, 123, 256, 512, 1024 |

---

## 5. Chia dữ liệu và kiểm toán leakage

### 5.1 Chia tập

Bài báo (bản 27Sep):

- Train: cycles **1–900** (900 chu kỳ).
- Test: cycles **901–1.504** (604 chu kỳ).
- DRL không dùng validation riêng.
- Base-stock: fit trên cycles **1–720**, validation **721–900**.
- SHAP: 504 chu kỳ cuối test (**1.001–1.504**).

Code/audit thực tế (`Feedback 7-9/task13-9/task2/output/output_audit_34_leakage_summary.txt`):

- Train: **1.000 records** (0–999).
- Validation: **200 records** (800–999, chronological).
- Test: **504 records** (1.000–1.503).
- `data/train.tfrecords` ~1.000 records; `data/test.tfrecords` ~504 records.
- DQN train loop lấy **random window 900 bước** trong train file 1.000 records (`DQN.ipynb:989-992`).

**Kết luận:** bài báo mô tả test 901–1.504 (604) nhưng file test thực tế chỉ có 504 records (1.001–1.504). Con số 901–1.504 là **sai/không khớp** với dữ liệu phát hành. Phần SHAP (1.001–1.504) mới khớp.

### 5.2 Capacity (normalization)

- Audit: `capacity_full` mean = **20.33**, `capacity_train_only` mean = **25.79**.
- Chênh lệch trung bình **26.29%**, max **50%**, **212/220 SKU** lệch > 5% và > 10%.
- Ảnh hưởng lên `sales_norm`: train Δ ≈ 0.0208, test Δ ≈ 0.0065; ảnh hưởng reward < 0.02.
- Kết luận audit: **MAJOR leakage**; khuyến nghị regenerate `capacity.tfrecords` từ train-only.

Bài báo nói "capacity is now defined from the training interval alone". Tuy nhiên:

- `prepare_data.py:144` vẫn tính capacity trên **toàn bộ series** (`grocery['quantity'].sum()/nunique`), không tách train.
- `data/capacity.tfrecords` timestamp gốc, không có dấu hiệu đã regenerate.
- Paper §4.5.8 vẫn nói capacity "about ten times average demand", khớp `capacity_full` (20.33 / 2.11 ≈ 9.6), không khớp train-only (25.79 / 2.11 ≈ 12.2).

**Kết luận:** code audit đã **phát hiện và đề xuất sửa**, nhưng pipeline phát hành (`prepare_data.py` + `data/capacity.tfrecords`) **chưa thấy bằng chứng đã áp dụng**. Cần xác minh lại trước khi tuyên bố "corrected".

### 5.3 Base-stock safety factor k

- Bài báo: chọn k trên validation 721–900, kết quả **k = 1.0**.
- Audit: `best_k_val = 1.0` (reward_val = +0.6894), `best_k_test = 0.5` (reward_test = +0.7596).
- **Nhưng `Training/DQN.ipynb` vẫn grid search k trên test** (`:1351`, `:1547-1564`) và chọn `best_k = 0.5`; sau đó dùng `k=0.5` cho `ScenarioBaseStockPolicy` (`:1851`).

**Kết luận:** audit notebook đã sửa đúng (chọn trên val), nhưng **notebook chính chưa được cập nhật**. Output multi-seed trong `DQN.ipynb` dùng k = 0.5 (test-selected), trong khi bài báo báo cáo k = 1.0 (val-selected). Đây là lý do chính khiến bảng base-stock của paper khác output notebook (xem Mục 10.2.16).

### 5.4 Kết luận audit

| Thành phần | Kết luận | Đã sửa trong code? |
|---|---|---|
| Forecasting | Không có forecaster; dùng realized sales | Không cần |
| Reward coefficients | Hard-code 0.025, unit weights | Không cần |
| Capacity normalization | **MAJOR leakage** (full vs train-only) | Audit đề xuất; pipeline chính chưa thấy áp dụng |
| Base-stock k | Leakage cũ (chọn trên test) | Audit sửa; `DQN.ipynb` chính vẫn chọn test |
| Chronological split | Đúng, không shuffle | Có |

---

## 6. Reward decomposition, RDX, MSX

### 6.1 Định nghĩa trong bài báo

- `Q(s,a) = Σ_k Q^k(s,a)`, với `Q^k(s,a) := r^k(s,a)` (one-step component value, post-hoc).
- Tổng 4 thành phần bằng đúng one-step reward; **không** bằng discounted multi-step Q của network. Bài báo thừa nhận điều này và có Table 9 đo "reconstruction gap".
- RDX: `Δ^k(s,a1,a2) = Q^k(s,a1) - Q^k(s,a2)`.
- MSX+:
  - `P = {k | Δ^k > 0}`, `N = {k | Δ^k < 0}`.
  - `d = Σ_{k∈N} Δ^k` (tổng bất lợi).
  - `MSX+ = argmin |M|` sao cho `Σ_{k∈M} Δ^k > d`, `M ⊆ P`.
  - `MSX-` tương tự trên `N` với ngưỡng `v`.
- Stability: `Stability = (# states có MSX_{λ=1} == MSX_{λ=i}) / N × 100` (all-or-nothing set equality).

### 6.2 RDX trong code

`Ablation_Study/ablation_RDX.ipynb` cell 12:

- `get_decomposed_reward(x_vec, u_vec)` trả về 4 sub-reward:
  - `stockout: 1 - z`
  - `overstock: -max(0, x+u-1)`
  - `waste: -0.025·x`
  - `quantile: -(p95(x)-p05(x))`
- `rdx_dqn`: lấy action tốt nhất `a*` và nhì `a'`, tính `Δ^k = r^k(a*) - r^k(a')`.
- **Chỉ `overstock` phụ thuộc action**; `stockout`, `waste`, `quantile` giống nhau giữa hai action nên `Δ^k = 0`.
- Code phân bổ **residual** `Q(a*) - Q(a') - Δ_overstock` **đều** cho 3 thành phần còn lại:
  `delta_q['stockout'] += residual/3`, tương tự `waste`, `quantile`.
- `rdx_a2c`: dùng `V(s) × (prob_best/prob_second - 1)` làm proxy Q-gap; `delta_q` chỉ có `overstock` khác 0 (không phân bổ residual).

**Đối chiếu:** bài báo mô tả RDX như phân rã thật sự theo 4 mục tiêu, với cả Δ dương và âm. Code thực tế **không** tạo ra 4 Δ độc lập; chỉ 1 thành phần là action-dependent, 3 thành phần còn lại được gán đều bằng residual. Đây là **heuristic**, không phải công thức trong bài báo.

### 6.3 MSX trong code

`Ablation_Study/ablation_RDX.ipynb` cell 17:

- `compute_msx(delta_q, q_gap, λ)`:
  - `dq_matrix = |Δ^k|` cho 4 objective.
  - `threshold = λ·|q_gap|`.
  - Sort `|Δ^k|` giảm dần; cộng dồn đến khi `cumsum ≥ threshold`; tập các objective đã cộng là MSX.
- Không dùng dấu `P/N`, không dùng `d = Σ negative`, không dùng `MSX+ / MSX-` như bài báo.
- Đảm bảo MSX không rỗng.

**Đối chiếu:** đây là **phiên bản đơn giản hóa** (top-|ΔQ| coverage), không phải định nghĩa MSX+/MSX- trong bài báo.

### 6.4 Stability trong code

- `msx_stability` (cell 17): Jaccard **per-product** giữa MSX_{λ=1} và MSX_{λ=i}, lấy trung bình × 100.
- `msx_stability_at_lambda` (cell 18): Jaccard per-product cho từng λ.
- Code ghi chú "FIXED: per-product Jaccard instead of all-or-nothing".

**Đối chiếu:** bài báo định nghĩa stability bằng **set equality** (all-or-nothing). Code hiện tại dùng **Jaccard**, cho kết quả cao hơn nhiều. Ví dụ:

| λ | Paper (DQN) | Paper (A2C) | Code `combined_ablation_results.csv` (DQN EASY/MED/HARD) | Code (A2C EASY/MED/HARD) |
|---|---|---|---|---|
| 1.0 | ~100% | ~100% | 100 / 100 / 100 | 100 / 100 / 100 |
| 1.5 | ~52–53% | ~27–29% | 50 / 92.4 / 83.4 | 75.8 / 75.0 / 75.0 |
| 2.0 | ~44–46% | ~27–29% | 25 / 60.3 / 83.4 | 75.8 / 75.0 / 75.0 |

Số liệu paper không tái lập được từ CSV hiện tại.

### 6.5 Reconstruction gap / decomposed training

- Bài báo có Table 9 (reconstruction error, correlation yếu: DQN r = 0.02–0.21, A2C r = −0.05…−0.13) và Table 10 (retrain với 4 Q-head / 4 critic, 3 seeds).
- Trong repo, **không thấy notebook/script** thực hiện retrain 4-head decomposition. `ablation_RDX.ipynb` chỉ làm post-hoc decomposition.
- Table 10 có thể được tạo từ một nhánh code không nằm trong repo này (hoặc chưa được đưa lên).

**Kết luận:** phần "genuinely trained reward decomposition" trong bài báo **không có code tương ứng** trong repo hiện tại.

---

## 7. SHAP

### 7.1 Macro (3 đặc trưng)

- 3 đặc trưng tổng hợp: Inventory (I), Demand (D), Waste (W).
- Paper: **KernelSHAP / `shap.KernelExplainer`**.
- Background: 200 synthetic states → subsample 100.
- Code:
  - `XAI/SHAP-temp.ipynb` cell 5: tạo background 200 mẫu `inventory ~ U(0,1)`, `sales ~ U(0,1)`, `waste = 0.025·inventory + N(0,0.005)`.
  - cell 6: `shap.KernelExplainer(predict_fn, sampled_background)`, `sampled_background = shap.sample(background_data, 100)`.
  - `X_explain_np = background_data.numpy()` — **giải thích chính các mẫu background**, không phải test states. Đây là điểm khác với mô tả "held-out states".
  - `Feedback 7-9/task14-9/audit_shap_background_validity.ipynb` dùng test states thật và grid background (synthetic/trajectory, random/KMeans, size 50/100/200).

### 7.2 Micro (660 đặc trưng)

- Paper: **PartitionExplainer**, 50 states/scenario, background 100.
- Code `Feedback 7-9/task14-9/audit_shap_660_partition.ipynb` cell 9:
  - `masker = shap.maskers.Partition(bg, max_samples=100)`
  - `explainer = shap.PartitionExplainer(predict_fn, masker)`
  - `test_micro = test_states_660["MEDIUM"][:50]`.
- Predict wrapper (cell 6):
  - DQN: `q = q_network(X); return tf.reduce_mean(q, axis=1)` → shape `[B, 14]` (trung bình Q trên 220 SKU cho từng action).
  - A2C: `probs = actor(s_pp); probs_3d = reshape(...); return tf.reduce_mean(probs_3d, axis=1)` → `[B, 14]`.

**Đối chiếu target SHAP:** bài báo có hai mô tả mâu thuẫn:

- §3.3.4 (dòng 230): "SHAP values are computed with respect to the **selected-action** Q-value Q(s,a*)… π(a*|s)".
- "Common Quantity" (dòng 218): "Feature-level attribution needs a **vector over the fourteen actions**: the portfolio-averaged softmax of the action values for DQN and the portfolio-averaged policy for A2C_mod".

Code khớp với mô tả thứ hai (vector 14 action, portfolio-averaged), không khớp mô tả "selected-action". Cần sửa lại một trong hai chỗ.

### 7.3 Background & stability

- Paper: grid 3 yếu tố: origin (synthetic vs trajectory), size (50/100/200), sampling (random vs KMeans); macro KernelSHAP, micro Partition.
- Code: `audit_shap_660_partition.ipynb` cell 8 tạo `bg660_dict` gồm 12 cấu hình: `syn_random_{50,100,200}`, `syn_kmeans_{50,100,200}`, `traj_random_{50,100,200}`, `traj_kmeans_{50,100,200}`.
- Code đo Spearman, cosine, mean absolute delta so với baseline `syn_random_100` (DQN).
- Paper: macro ranking bất biến theo size/sampling (ρ=1.00), đổi khi origin trajectory (ρ=0.87); micro magnitude ổn (0.94–0.99) nhưng ranking không ổn (ρ gần 0), grid 50 states mất >4 giờ.

### 7.4 Faithfulness

- Paper: MoRF / Random / LeRF; mask 1–10 features (0.15–1.52% của 660); thay bằng median của 100 background; 30 lần random; AUPC (trapezoidal), bootstrap CI, Wilson interval, Wilcoxon + t-test + Cohen d + Holm.
- Code `Feedback 7-9/task15-9/faithfulness_Task15-9_Complete.ipynb` cells 8–15:
  - Cell 8: perturbation MoRF/Random/LeRF, lưu per-state.
  - Cell 9: AUPC.
  - Cell 10: bootstrap 1.000 + Wilson.
  - Cell 11: Wilcoxon + t + Cohen d + Holm.
  - Cell 12: MSX-guided vs Random masking.
  - Cell 14: threshold + meaningful rate + episode/state consistency.
- Output `Ablation_Study/output/task15-9_faithfulness_statistics_table.csv`:
  - DQN MoRF AUPC ~0.011–0.014; A2C MoRF ~0.024 (~2×).
  - Random ~0; LeRF âm.
  - ASR = 0, Wilson upper ~0.071.
- Khớp định tính với paper.

### 7.5 Kết quả SHAP

- Macro: Demand/Sales chi phối; DQN tách biệt hơn A2C.
- Micro: Top-5 SKU 64, 163, 46, 155; SKU biến động cao, capacity nhỏ; không phải SKU volume cao (57, 81, 108); r ≈ 0.04.
- FCS: paper > 0.90; code `fcs_grid_results_496states.csv` cho DQN 0.951–0.997, A2C 0.905–0.991.
- Lưu ý: `fcs_summary_results.csv` cho FCS 0.667–0.889 (định nghĩa/run khác). Có **nhiều định nghĩa FCS** trong repo; cần thống nhất.

---

## 8. Kết quả thực nghiệm

### 8.1 Hiệu năng (Table 7)

| Metric | Base-Stock | DQN | A2C_mod |
|---|---:|---:|---:|
| Total Reward ↑ | +0.8567 | +0.7197 ± 0.0396 | +0.6939 ± 0.1463 |
| Service Level ↑ | +0.9800 | +0.9999 ± 0.0001 | +0.9997 ± 0.0007 |
| Holding Cost ↓ | +0.0000 | +0.1110 ± 0.0630 | +0.0935 ± 0.1495 |
| Waste Cost ↓ | +0.0111 | +0.0672 ± 0.0008 | +0.0672 ± 0.0008 |
| Ordering Cost ↓ | +0.1121 | +0.1020 ± 0.0258 | +0.1478 ± 0.0301 |
| Stockout Rate ↓ | +0.0200 | +0.0001 | +0.0003 |

Quan hệ nội tại: `Total = Service - Holding - Waste - Ordering`.

Kiểm tra:

- BS: 0.9800 − 0 − 0.0111 − 0.1121 = 0.8568 ≈ 0.8567.
- DQN: 0.9999 − 0.1110 − 0.0672 − 0.1020 = 0.7197.
- A2C: 0.9997 − 0.0935 − 0.0672 − 0.1478 = 0.6912 (paper 0.6939; chênh do làm tròn/mean).

### 8.2 RDX / MSX

- RDX (Fig. 2): EASY hai agent gần giống, service chi phối; MEDIUM A2C nhạy holding/ordering hơn; HARD A2C giảm holding/ordering, DQN giữ service.
- MSX (Fig. 4): EASY/MEDIUM còn action thay thế tốt hơn về cost; HARD A2C có lợi thế rõ, DQN vẫn có alternative tốt hơn về holding/ordering.
- Case study (task17-9): A2C MSX = {service} ở EASY/MEDIUM → {holding} ở HARD; DQN MSX = {holding} mọi scenario. **Đây là narrative hard-code**, không khớp `combined_ablation_results.csv` (A2C msx_size ~3; DQN ~1–2.3).

### 8.3 SHAP

- Macro: Demand/Sales dominant.
- Micro: tập trung 5–8 SKU; aggregation bias.
- Jaccard Top-20: DQN 1.00 (ổn định), A2C 0.29–0.33 (thay đổi theo context).

### 8.4 Stability

- Paper: DQN giảm dần; A2C giảm đột ngột sau λ=1.0; DQN cao hơn A2C ~2× khi λ ≥ 1.5.
- Code: Jaccard cho A2C cao hơn paper; xem Mục 6.4.

### 8.5 Faithfulness

- MoRF > Random > LeRF; A2C ~2× DQN; ASR = 0.
- Code khớp định tính.

### 8.6 Scalability & resolution

- Scalability (task12-9 task5): DQN Fast 0.7709, Medium 0.6508, Slow 0.5740; A2C 0.1807, 0.0477, 0.1514. DQN thắng cả 3 nhóm.
- Resolution (7/14/28 actions): DQN giữ reward cao; A2C peak ở 14, giảm ở 7/28 do overstock tăng (0.96/0.84 vs 0.00).
- Code task12-9 task4 có retrain 6 models; paper mô tả đúng hướng.

### 8.7 Managerial cases

- Case 1: service vs holding.
- Case 2: MSX one-sentence justification.
- Case 3: top SKU monitoring.
- Code task17-9 tạo CSV cases; nội dung là **hard-coded list**, không tự động sinh từ kết quả ablation.

---

## 9. Kết luận và hạn chế của bài báo

### 9.1 Kết luận

- A2C_mod tái phân bổ trọng số mục tiêu khi áp lực tăng; DQN ổn định nhưng kém thích nghi.
- SHAP xác nhận A2C phân bổ attention rộng hơn; DQN tập trung inventory + demand.
- Explanation stability/adaptability là thuộc tính phụ thuộc kiến trúc, không phải thuộc tính chung của DQN/A2C.
- Faithfulness và stability được đánh giá; **usefulness chưa**.

### 9.2 Hạn chế (paper tự nêu)

1. Chưa đánh giá robustness qua nhiều seed cho explanation.
2. SHAP aggregated làm mất heterogeneity; cần micro/system-level.
3. Chưa có user study với inventory manager.
4. Action space thô (14 mức dùng chung), không continuous/SKU-specific.
5. Scenario EASY/MEDIUM/HARD là biến đổi có kiểm soát của cùng một test set, không phải dataset mới.
6. Chưa mở rộng sang PPO/SAC.

### 9.3 Hạn chế bổ sung (người đọc nhận thấy)

- Mâu thuẫn nội tại về split (900/604 vs 1000/504).
- Mâu thuẫn state 664 vs 660.
- Mâu thuẫn SHAP target (selected-action vs 14-action vector).
- Mâu thuẫn thuật ngữ "ordering" vs "balance" cho cùng `quan`.
- Mục "Reliability and Stability of the Explanations" (§4.11/4.12) **rỗng**; Table 17 được tham chiếu nhưng không có nội dung.
- Base-stock k: paper nói k=1.0 val-selected; notebook chính vẫn test-selected k=0.5.
- Một số số liệu không tái lập từ CSV hiện tại (stability, base-stock row, A2C waste).

---

## 10. Đối chiếu code ↔ bài báo

### 10.1 Bảng tổng hợp

| # | Hạng mục | Bài báo | Code | File:line | Đánh giá |
|---|---|---|---|---|---|
| 1 | State dimension | 664 (660 + 4 system) | 660 | `DQN.ipynb:192-194` | **Lệch**; caveat bị bỏ khỏi bản 27Sep |
| 2 | Reward service | `min(I,D)/D` | `1 - 1{x_old<1e-5}` | `DQN.ipynb:386` | **Lệch bản chất** |
| 3 | Reward holding | `-h·I_t` | `-max(0,x+u-1)` | `DQN.ipynb:383` | **Lệch bản chất** |
| 4 | Reward waste | `-c_w·W_t` | `-0.025·x_old` | `DQN.ipynb:389` | Khớp một phần |
| 5 | Reward ordering | `-(c_o Q + c_f 1{Q>0})` | `-(p95(x)-p05(x))` | `DQN.ipynb:392` | **Lệch hoàn toàn**; là "balance" |
| 6 | Reward base | `+1` | `+1` | `DQN.ipynb:395` | Khớp |
| 7 | Reward weights | unit | unit | `DQN.ipynb:395` | Khớp |
| 8 | Stockout metric | unmet demand rate | fraction `x_old<1e-5` | `DQN.ipynb:386` | **Lệch bản chất** |
| 9 | Holding metric | avg inventory | overstock above capacity | `DQN.ipynb:383` | **Lệch bản chất** |
| 10 | RDX decomposition | 4 Δ độc lập | chỉ overstock; residual chia đều | `ablation_RDX.ipynb` cell 12 | **Heuristic, không như paper** |
| 11 | RDX A2C | model-agnostic one-step lookahead | critic/prob-ratio proxy | `ablation_RDX.ipynb` cell 12 | **Lệch** |
| 12 | MSX | MSX+/MSX− theo P/N, d | top-|ΔQ| cumsum ≥ λ|Qgap| | `ablation_RDX.ipynb` cell 17 | **Lệch thuật toán** |
| 13 | Stability | set equality | per-product Jaccard | `ablation_RDX.ipynb` cell 17-18 | **Lệch metric** |
| 14 | SHAP macro | KernelSHAP | KernelExplainer | `XAI/SHAP-temp.ipynb` cell 6 | Khớp |
| 15 | SHAP micro | PartitionExplainer | PartitionExplainer | `task14-9/...:cell 9` | Khớp |
| 16 | SHAP target | selected-action Q / π | portfolio-averaged 14-action vector | `task14-9/...:cell 6` | **Lệch** (paper tự mâu thuẫn) |
| 17 | SHAP background | 200 → 100 | 200 → 100 | `SHAP-temp:cell 5-6` | Khớp |
| 18 | SHAP states | held-out test | `SHAP-temp` dùng background; `task14-9` dùng test | cả hai | **Không nhất quán giữa 2 script** |
| 19 | Scenario defs | x 30/60/90, sales 20/50/80, waste 1/5/15 | `DQN.ipynb` khớp; `ablation_RDX` dùng sales 0.5/1/1.5, waste 1/2.5/5 | `DQN.ipynb:1752-1756`; `ablation_RDX:cell 27` | **Hai bộ scenario khác nhau** |
| 20 | State collection | test states | synthetic decay, không rollout | `ablation_RDX:cell 28` | **Lệch** |
| 21 | Split | 900 train / 604 test | 1000 train / 200 val / 504 test | audit output | **Lệch** |
| 22 | Capacity | train-only (corrected) | `prepare_data.py` full series | `prepare_data.py:144` | **Chưa thấy áp dụng** |
| 23 | Base-stock k | 1.0, val-selected | `DQN.ipynb` test-selected k=0.5 | `DQN.ipynb:1547-1564` | **Lệch** |
| 24 | Entropy A2C | computed, not optimized | log only (notebook) | `A2C-mod.ipynb:446-453` | Khớp notebook; `training.py` thì có entropy |
| 25 | Critic norm | GroupNorm | LayerNorm | `A2C-mod.ipynb:156-157` | **Lệch nhỏ** |
| 26 | Ablation DQN hidden | 32 | 128 | `ablation_RDX:102` | **Lệch** |
| 27 | Table 17 / §4.12 | có tham chiếu | không có nội dung | paper dòng 1152, 1163 | **Thiếu** |
| 28 | Decomposed retrain | Table 10 | không có code | — | **Thiếu** |
| 29 | Managerial cases | narrative | hard-coded list | `task17-9:cell 7` | Không tự động |
| 30 | training.py | — | bug `per_timestep_actor_loss` | `training.py:433` | Notebook là bản chạy được |

### 10.2 Chi tiết

#### 10.2.1 State 664 vs 660

- Paper §3.1: 664 = 660 + 4 system-level (`U_t, C_t, V_t, T_t`).
- Code: 660 = `[x, sales, q]`; không có 4 system features.
- Bản 17Mar từng ghi rõ 4 features là conceptual/future work; bản 27Sep bỏ caveat.
- **Hệ quả:** mọi tuyên bố "SHAP trên state space" thực chất chỉ trên 660 product-level features. Cần khôi phục disclaimer hoặc bổ sung 4 features vào code.

#### 10.2.2 Reward semantics

- Paper trình bày 4 phương trình chuẩn (service ratio, holding `hI`, waste `c_w W`, ordering `c_o Q + c_f`).
- Code dùng 4 proxy khác:
  - `1 - z` với `z` binary trên **tồn kho cũ**.
  - `overstock` = phần vượt capacity (không phải average inventory).
  - `0.025·x_old` (khớp ý tưởng 2.5% nhưng tính trên tồn kho cũ, không phải lượng hư hỏng thực).
  - `p95 - p05` (độ phân tán tồn kho, không phải order cost).
- **Hệ quả:** các nhãn "Service Level", "Holding Cost", "Ordering Cost" trong Table 7 là **diễn giải lại** các proxy, không phải đo lường trực tiếp các khái niệm trong phương trình. Cần bảng mapping rõ ràng (như `outputTask13-9.md:15-21` đã làm) ngay trong bài báo.

#### 10.2.3 Service/stockout metric

- Code: `z = (x_old < 1e-5)`; `stockout = mean(z)`; `service = 1 - stockout`.
- Đây là **tỷ lệ SKU có tồn kho gần 0 ở đầu chu kỳ**, không phải tỷ lệ demand không được đáp ứng.
- DQN giữ inventory > 0 nên service ~0.9999; đây là kết quả gần như trivial, không chứng minh năng lực đáp ứng demand.
- Paper mô tả `min(I,D)/D`, tạo kỳ vọng khác. **Cần sửa metric hoặc sửa mô tả.**

#### 10.2.4 Holding

- Code: `overstock = max(0, x+u-1)`.
- Base-stock đặt `S_i ≤ 1`, nên `x+u ≤ 1` → overstock = 0. Vì vậy "Base-Stock incurs no holding cost" là **hệ quả của định nghĩa overstock**, không phải vì base-stock không giữ tồn kho.
- Base-stock thực tế giữ tồn kho (waste 0.0111, service 0.98). Cần diễn giải cẩn thận.

#### 10.2.5 Waste

- Code: `q = 0.025·x_old`, scenario HARD dùng waste_rate 0.15.
- Paper: `c_w·W_t`. Nếu `W_t = ρ·x` và `c_w=1` thì khớp; nhưng code tính trên tồn kho **trước** action, không phải lượng hư hỏng sau action.
- Trong `Train_DQN_Slow_73.ipynb:265-333` có bản "corrected" tính waste trên `x_clip` (sau action) nhưng **bị comment out**; bản active vẫn dùng `x_old`.

#### 10.2.6 Ordering / balance

- Code: `quan = p95(x_old) - p05(x_old)`, broadcast cho mọi SKU.
- Đây là **độ phân tán tồn kho across SKU**, không phải order quantity hay fixed order cost.
- Paper gọi là "ordering" (Table 7, §3.2) và "balance" (§4.5.8: "balance spread in [0, 0.89]"). Cùng một biến, hai tên → dễ gây nhầm.
- **Khuyến nghị:** thống nhất gọi là "inventory balance / dispersion" và giải thích rõ không phải chi phí đặt hàng.

#### 10.2.7 RDX

- Code chỉ có `overstock` action-dependent; 3 thành phần còn lại bị triệt tiêu.
- Residual `Q(a*) - Q(a') - Δ_overstock` được chia đều cho 3 thành phần, không dựa trên cơ chế.
- A2C dùng proxy `V(s)·(p*/p' - 1)`, không phải Q thật.
- Paper mô tả 4 Δ độc lập và A2C dùng model-agnostic one-step lookahead.
- **Hệ quả:** Figure 2/4 (RDX) khó tái lập chính xác từ `ablation_RDX.ipynb` hiện tại; có thể được tạo bởi script khác.

#### 10.2.8 MSX

- Code dùng top-|ΔQ| cumsum ≥ λ|Qgap|; không dùng P/N, không dùng d.
- Paper định nghĩa MSX+ = tập nhỏ nhất trong P có tổng Δ > d.
- **Hệ quả:** MSX size và stability khác về bản chất, không chỉ khác tham số.

#### 10.2.9 Stability

- Paper: set equality (all-or-nothing).
- Code: Jaccard per-product.
- CSV hiện tại cho A2C ~75% ở λ=1.5/2.0, trong khi paper nói 27–29%.
- **Hệ quả:** Figure 14 không tái lập từ CSV hiện tại.

#### 10.2.10 SHAP target

- Paper §3.3.4: selected-action.
- Paper "Common Quantity": 14-action vector.
- Code: 14-action vector (portfolio-averaged).
- **Khuyến nghị:** sửa §3.3.4 theo "Common Quantity" hoặc cung cấp thêm kết quả selected-action.

#### 10.2.11 SHAP method

- Macro: KernelExplainer — khớp.
- Micro: PartitionExplainer — khớp.
- `ablation_RDX.ipynb` cell 21 lại dùng `shap.GradientExplainer` cho ablation SHAP, không phải Kernel/Partition. Đây là **script khác**, không phải pipeline chính. Cần ghi rõ trong paper nếu dùng.

#### 10.2.12 Background

- `SHAP-temp.ipynb` giải thích chính 200 mẫu background (không phải test states), trong khi paper nói held-out test states.
- `task14-9` dùng test states thật. Cần xác định script nào tạo Figure 9–11 và ghi đúng.

#### 10.2.13 Scenario definitions

- `DQN.ipynb` (và paper Table 8): EASY x=30%, sales=20%, waste=1%; MEDIUM x=60%, sales=50%, waste=5%; HARD x=90%, sales=80%, waste=15%.
- `ablation_RDX.ipynb` cell 27: EASY sales_scale=0.5, waste=1%; MEDIUM 1.0, 2.5%; HARD 1.5, 5%; không có x_scale; inventory synthetic decay.
- **Hai bộ scenario khác nhau.** RDX/MSX/SHAP trong ablation có thể không dùng cùng scenario như Table 7. Cần thống nhất.

#### 10.2.14 Split

- Paper: 900 train / 604 test.
- Data: 1000 train / 504 test (audit).
- DQN train loop: random 900-step window trong 1000 records.
- **Khuyến nghị:** sửa paper thành 1000/200/504 hoặc giải thích rõ "900-step training windows sampled from 1000 records; test = 504".

#### 10.2.15 Capacity

- Audit: full vs train-only lệch 26.29%, 212/220 SKU > 10%.
- Paper: đã sửa train-only.
- Code: `prepare_data.py` vẫn full series; `data/capacity.tfrecords` không thấy regenerate.
- **Khuyến nghị:** regenerate thật, cập nhật checksum, hoặc ghi rõ "đề xuất sửa, chưa áp dụng cho kết quả hiện tại".

#### 10.2.16 Base-stock k

- Paper: k=1.0, val-selected.
- `DQN.ipynb`: k grid trên test, best_k=0.5; output BS ALL 0.8269.
- Audit: k=1.0 val optimum; k=0.5 test optimum.
- Paper BS ALL 0.8567 khớp hơn với k=1.0 (service 0.98, waste 0.0111, ordering 0.1121) so với k=0.5 (service 0.9363, waste 0.0077, ordering 0.1017).
- **Kết luận:** paper báo cáo run k=1.0 (đúng theo audit), nhưng **output trong notebook chính là k=0.5**. Cần chạy lại và lưu output k=1.0 để tái lập.

#### 10.2.17 Entropy

- Paper: entropy computed but not optimized.
- `A2C-mod.ipynb`: chỉ log `entropy_adj`, không đưa vào loss — khớp.
- `training.py:420/438`: có `- entropy_coefficient*entropy_p` trong actor loss — **không khớp**.
- **Kết luận:** notebook là bản chuẩn; `training.py` là phiên bản cũ/khác.

#### 10.2.18 Critic normalization

- Paper: GroupNorm.
- Code: LayerNorm (`A2C-mod.ipynb:156-157`).
- **Lệch nhỏ**, cần sửa mô tả hoặc code.

#### 10.2.19 training.py bug

- `training.py:431-433`: `per_timestep_actor_loss` được dùng nhưng assignment bị comment out → `NameError` nếu chạy nhánh A2C_mod.
- Ngoài ra có dead code trong `Env` (`super(Critic, self)`, biến `state` chưa định nghĩa).
- **Kết luận:** `training.py` không chạy được; notebook là bản thực thi. Nên sửa hoặc archive `training.py` kèm ghi chú.

#### 10.2.20 Missing Table 17 / §4.12

- Paper dòng 1152 tham chiếu "scorecard in Table 17 (Section 4.12)".
- Paper dòng 1163 có heading "Reliability and Stability of the Explanations" nhưng **không có nội dung**, tiếp ngay "Conclusion".
- Không tìm thấy Table 17 trong docx.
- **Khuyến nghị:** bổ sung bảng scorecard hoặc xóa tham chiếu.

### 10.3 Code đã giải quyết những gì

1. **Pipeline dữ liệu Instacart → TFRecords**: `prepare_data.py` tạo `train/test/capacity/stock.tfrecords`, lọc 220 SKU, capacity heuristic, timestamp 6h.
2. **Môi trường mô phỏng đa sản phẩm**: inventory, replenishment, capacity clip, demand fulfillment, waste, overstock (`DQN.ipynb:1019-1022`).
3. **DQN hoàn chỉnh**: per-product Q-network weight sharing, Double DQN, replay, target network, ε-greedy, Huber, checkpoint, predict (`DQN.ipynb`).
4. **A2C_mod hoàn chỉnh**: actor 4 lớp, critic 2 lớp, advantage 1-step, Laplacian-smoothed distillation, Brier loss, checkpoint (`A2C-mod.ipynb`).
5. **Multi-seed training**: `A2C-mod-seed.ipynb` với `train(seed, run_name)`; thư mục `A2C_mod_123`, `A2C_mod_256`, …
6. **Base-stock baseline**: `BaseStockPolicy`, grid search k, scenario base-stock (`DQN.ipynb:15`, `:1528-1567`, `:1830`).
7. **Scenario EASY/MEDIUM/HARD**: scale inventory, sales, waste (`DQN.ipynb:1752-1756`).
8. **Đánh giá hiệu năng**: reward, service, stockout, holding, waste, ordering, multi-seed mean ± std, bảng so sánh (`DQN.ipynb:2157-2200`).
9. **Leakage audit**: chronological split, capacity full vs train-only, forecasting, reward coefficients, base-stock k val vs test (`task13-9/task2`).
10. **Reward sensitivity**: OAT sweep 4 tham số × 4 mức, 2 agent + base-stock (`task13-9/task1`).
11. **SHAP macro**: KernelExplainer trên 3 features, background synthetic, local/global/dependence plots (`XAI/SHAP-temp.ipynb`).
12. **SHAP micro**: PartitionExplainer trên 660 features, 50 states/scenario, background grid (`task14-9`).
13. **SHAP stability**: grid origin × size × sampling, Spearman/cosine/MAD (`task14-9/audit_shap_660_partition.ipynb`).
14. **FCS grid**: 496 states × 3 scenario × 4 λ × 2 agent × 5 repeats (`Ablation_Study/output/fcs_grid_results_496states.csv`).
15. **RDX/MSX**: post-hoc decomposition, OCS/FCS/CAS, stability (`ablation_RDX.ipynb`).
16. **Faithfulness**: MoRF/Random/LeRF, AUPC, bootstrap CI, Wilson, Wilcoxon/t-test/Cohen d/Holm, MSX-guided vs random (`task15-9`).
17. **Scalability**: retrain 6 models trên Fast/Medium/Slow 73/74 SKU (`task12-9/task5`).
18. **Action resolution**: retrain 7/14/28 actions (`task12-9/task4`).
19. **Managerial cases**: 3 case CSV (`task17-9`).
20. **Tài liệu hóa**: nhiều `outputTask*.md`, `planTask*.md`, `QA.md`, `docs/plans/*`, `docs/responses/*`.

### 10.4 Paper nói nhưng code chưa làm / không khớp

| Nội dung paper | Trạng thái code |
|---|---|
| State 664 với 4 system-level features | Không có; chỉ 660 |
| Service = `min(I,D)/D` | Không; dùng binary `1{x<1e-5}` |
| Holding = `h·I_t` | Không; dùng overstock above capacity |
| Ordering = `c_o Q + c_f 1{Q>0}` | Không; dùng quantile spread |
| RDX 4 thành phần độc lập | Chỉ overstock; residual chia đều |
| A2C RDX model-agnostic lookahead | Dùng critic/prob-ratio proxy |
| MSX+/MSX− theo P/N | Dùng top-|ΔQ| cumsum |
| Stability = set equality | Dùng Jaccard |
| SHAP selected-action | Dùng 14-action vector |
| Capacity train-only | `prepare_data.py` vẫn full series |
| Base-stock k=1.0 val-selected | Notebook chính test-selected k=0.5 |
| Test 901–1504 (604) | Data test 504 (1001–1504) |
| Ablation DQN hidden 32 | Ablation dùng 128 |
| Table 17 / §4.12 | Thiếu nội dung |
| Decomposed retrain (Table 10) | Không có code |
| GroupNorm critic | LayerNorm |
| Entropy không tối ưu | Notebook đúng; `training.py` sai |

### 10.5 Code làm nhưng paper không mô tả

- `quan` thực chất là **inventory dispersion**, không phải ordering cost.
- `collect_states` trong ablation dùng **synthetic decay**, không rollout agent.
- `SHAP-temp` giải thích **background samples**, không phải test states.
- `ablation_RDX` dùng **GradientExplainer**, không phải Kernel/Partition.
- `task17-9` cases là **hard-coded narrative**, không sinh từ kết quả.
- `training.py` có entropy trong loss và bug `per_timestep_actor_loss`.
- Có nhiều định nghĩa FCS khác nhau giữa các notebook/CSV.
- Có hai bộ scenario khác nhau giữa `DQN.ipynb` và `ablation_RDX.ipynb`.

### 10.6 Đánh giá mức độ reproducible

| Mức | Nội dung |
|---|---|
| **Tái lập tốt** | DQN/A2C training, hyperparameters, action space, reward formula, base-stock class, scenario Table 8, faithfulness MoRF/LeRF, FCS grid 496 states |
| **Tái lập một phần** | RDX/MSX (thuật toán khác paper), stability (metric khác), SHAP micro (target khác mô tả), capacity fix (chưa áp dụng) |
| **Không tái lập từ repo** | Table 10 (decomposed retrain), Table 17, Figure 14 stability numbers, base-stock row k=1.0, §4.11/4.12 |
| **Rủi ro cao** | Split 900/604 vs 1000/504; state 664 vs 660; hai bộ scenario; hai pipeline SHAP |

---

## 11. Khuyến nghị

### 11.1 Cho bản thảo

1. **Thống nhất split**: sửa thành 1000 train / 200 val / 504 test, hoặc giải thích rõ "900-step windows sampled from 1000 records; test 504".
2. **State dimension**: khôi phục caveat 664 conceptual vs 660 implemented (như bản 17Mar), hoặc bổ sung 4 system features vào code.
3. **Reward mapping**: thêm bảng mapping `service ↔ 1-z`, `holding ↔ overstock`, `waste ↔ 0.025x`, `ordering/balance ↔ p95-p05`, kèm giải thích đây là proxy.
4. **Đổi tên "ordering" → "balance/dispersion"** để tránh hiểu nhầm là chi phí đặt hàng.
5. **Sửa mô tả service/stockout**: nói rõ là "fraction of SKUs with near-zero starting inventory", không phải unmet demand.
6. **Sửa SHAP target** về 14-action vector (theo "Common Quantity"), hoặc bổ sung kết quả selected-action.
7. **Sửa stability metric**: nếu dùng Jaccard, cập nhật công thức và số liệu; nếu giữ set equality, dùng code cũ.
8. **Bổ sung Table 17 / §4.12** hoặc xóa tham chiếu.
9. **Ghi rõ** Table 10 (decomposed retrain) có code ở đâu; nếu không có, mô tả là "separate experiment not in released repo".
10. **Sửa A2C waste** 0.0672 → 0.0645 (theo code) hoặc giải thích.
11. **Sửa base-stock caption**: k=1.0 val-selected, không phải k=1.0 test; nêu rõ run nào.
12. **Sửa Critic norm**: LayerNorm, không phải GroupNorm.
13. **Thống nhất scenario** giữa Table 7/8 và RDX/SHAP; nếu khác, ghi rõ.

### 11.2 Cho code

1. Sửa `training.py` bug `per_timestep_actor_loss`; hoặc archive kèm ghi chú "not executable".
2. Cập nhật `DQN.ipynb` chọn k trên validation (không test); lưu output k=1.0.
3. Regenerate `capacity.tfrecords` từ train-only; cập nhật checksum.
4. Thống nhất `data/` vs `data220/` path (notebook DQN trỏ `data220/` không tồn tại).
5. Thống nhất một pipeline SHAP (Kernel/Partition) và một bộ test states.
6. Thống nhất một định nghĩa FCS.
7. Implement đúng MSX+/MSX− nếu muốn khớp paper.
8. Implement RDX 4 thành phần độc lập (hoặc ghi rõ heuristic).
9. Thống nhất scenario definitions giữa các notebook.
10. Sinh managerial cases tự động từ CSV thay vì hard-code.

### 11.3 Thực nghiệm bổ sung (nếu muốn tăng độ tin cậy)

1. Chạy lại toàn bộ Table 7 với capacity train-only và k=1.0; lưu output.
2. Chạy RDX/MSX với implementation đúng paper; so sánh với bản heuristic.
3. Chạy stability với cả set-equality và Jaccard; báo cáo cả hai.
4. Chạy SHAP selected-action và 14-action vector; so sánh.
5. Chạy decomposed retrain (4 heads) và lưu code.
6. Đánh giá explanation trên nhiều seed (paper tự nêu hạn chế).
7. User study với inventory manager (paper tự nêu).

---

## 12. Phụ lục

### 12.1 Danh mục file đã đọc

**Pipeline & training**

- `prepare_data.py` — full.
- `training.py` — full (có bug).
- `Training/DQN.ipynb` — config, reward, env, train loop, base-stock, eval, results.
- `Training/A2C-mod.ipynb` — config, networks, train loop.
- `Training/A2C-mod-seed.ipynb` — config, seed support.
- `Training/Train_DQN.ipynb` — prototype synthetic env.

**XAI / ablation**

- `Ablation_Study/ablation_RDX.ipynb` — config, RDX, MSX, metrics, SHAP, scenarios, collect_states.
- `Ablation_Study/ablation_SHAP.ipynb` — outline.
- `Ablation_Study/ablation_study_Combined.ipynb` — outline.
- `Ablation_Study/faithfulness/*.ipynb` — outline + key cells.
- `XAI/SHAP-temp.ipynb` — background, KernelExplainer.
- `Feedback 7-9/task14-9/audit_shap_660_partition.ipynb` — models, predict wrappers, Partition, background grid.
- `Feedback 7-9/task14-9/audit_shap_background_validity.ipynb` — outline + model loading.

**Feedback 7-9**

- `task13-9/task1/reward_utils.py` — full.
- `task13-9/task1/outputTask13-9.md` — mapping, sensitivity.
- `task13-9/task2/audit_dataset_split_leakage.ipynb` — outline + outputs.
- `task15-9/faithfulness_Task15-9_Complete.ipynb` — cells 8–15.
- `task17-9/generate_cases_task47.ipynb` — cell 7.
- `task12-9/task 5/Train_DQN_Slow_73.ipynb` — reward, metrics.
- `Xai_Inventory_Submit_17Mar.md` — caveat 664/660.

**Outputs**

- `Ablation_Study/output/combined_ablation_results.csv`.
- `Ablation_Study/output/combined_summary_by_agent.csv`.
- `Ablation_Study/output/combined_summary_by_lambda.csv`.
- `Ablation_Study/output/combined_summary_table.csv`.
- `Ablation_Study/output/fcs_grid_results_496states.csv`.
- `Ablation_Study/output/fcs_summary_results.csv`.
- `Ablation_Study/output/task15-9_faithfulness_statistics_table.csv`.
- `Feedback 7-9/task13-9/task2/output/output_audit_34_leakage_summary.txt`.
- `Feedback 7-9/task13-9/task2/output/output_audit_34_baseline_k.csv`.

**Docs**

- `docs/plans/SHAP_Expansion_Plan.md`, `SHAP_Faithfulness_Plan.md`, `SHAP_Faithfulness_Random_Baseline_Plan.md`.
- `docs/responses/result2.md` (đọc phần đầu).
- `docs/responses/result_shap_faithfulness_test.md`.

### 12.2 Viết tắt

| Viết tắt | Nghĩa |
|---|---|
| XRL | Explainable Reinforcement Learning |
| RDX | Reward Difference Explanation |
| MSX | Minimal Sufficient Explanation |
| SHAP | SHapley Additive exPlanations |
| OCS | Objective Coverage Score |
| FCS | Feature Coverage Score |
| CAS | Cross-domain Alignment Score |
| MoRF | Most Relevant First |
| LeRF | Least Relevant First |
| AUPC | Area Under Perturbation Curve |
| ASR | Action Switching Rate |
| BS | Base-Stock |

### 12.3 Bảng số liệu đối chiếu nhanh

| Metric | Paper BS | Code BS (`DQN.ipynb`, k=0.5) | Paper DQN | Code DQN | Paper A2C | Code A2C |
|---|---:|---:|---:|---:|---:|---:|
| Total | 0.8567 | 0.8269 | 0.7197 | 0.7197 | 0.6939 | 0.6939 |
| Service | 0.9800 | 0.9363 | 0.9999 | 0.9999 | 0.9997 | 0.9997 |
| Holding | 0.0000 | 0.0000 | 0.1110 | 0.1110 | 0.0935 | 0.0935 |
| Waste | 0.0111 | 0.0077 | 0.0672 | 0.0672 | 0.0672 | 0.0645 |
| Ordering | 0.1121 | 0.1017 | 0.1020 | 0.1020 | 0.1478 | 0.1478 |
| Stockout | 0.0200 | 0.0637 | 0.0001 | 0.0001 | 0.0003 | 0.0003 |

**Đọc bảng:** DQN và A2C khớp gần như hoàn toàn (trừ A2C waste). Base-stock khác ở mọi metric, nhất quán với việc paper dùng k=1.0 còn notebook output dùng k=0.5. Đây là điểm cần chạy lại và lưu output để tái lập.

---

## Kết luận cuối

Bài báo có ý tưởng tốt và framework đầy đủ, nhưng có **khoảng cách đáng kể giữa mô tả phương pháp và code thực tế**:

- **Reward decomposition là post-hoc relabeling** của 4 proxy `1-z`, `overstock`, `0.025x`, `p95-p05`, không phải 4 khái niệm chuẩn trong phương trình.
- **RDX/MSX/stability trong code hiện tại là phiên bản đơn giản hóa**, không khớp thuật toán paper mô tả.
- **State 664, split 900/604, capacity train-only, k=1.0** là những điểm paper nói đã sửa nhưng code phát hành chưa thể hiện đầy đủ.
- **Table 17, §4.12, Table 10** thiếu nội dung/code tương ứng.

Để bản thảo vững hơn, nên: (1) thêm bảng mapping paper ↔ code; (2) sửa các con số split/state; (3) chạy lại base-stock với k=1.0 và capacity train-only, lưu output; (4) thống nhất một pipeline SHAP và một định nghĩa FCS/stability; (5) bổ sung hoặc xóa các mục còn thiếu.

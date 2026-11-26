# Plan Task 1 — Kiểm chứng độ ổn định SHAP qua 5 seed A2C_mod

## 1. Mục tiêu

Kiểm chứng **feature importance của SHAP có ổn định qua 5 lần huấn luyện độc lập** (seed 42, 123, 256, 512, 1024) của A2C_mod hay không.

Cụ thể:

- Chạy **macro SHAP** (3 nhóm: Inventory, Demand, Waste) và **micro top-k SHAP** (660 đặc trưng: 220 SKU × 3 biến).
- Chạy cho **cả 3 scenario**: EASY, MEDIUM, HARD.
- Chạy cho **cả 2 target**: policy probabilities `π` và pre-softmax `logits`.
- Báo cáo **Jaccard top-k** (k = 10/20/50) và **Spearman** giữa các cặp seed.
- Báo cáo **mean ± SD |SHAP|** theo từng feature.
- Báo cáo **riêng seed 256** vì đã xác nhận bị collapse.

## 2. Cấu hình chốt

### 2.1. Checkpoint

| Seed | Đường dẫn | Checkpoint | Step | Episodes |
|---|---|---|---|---|
| 42 | `Feedback 23-11/checkpoints_a2c_42` | `ckpt-53` | 17.490 | 530 |
| 123 | `Feedback 23-11/A2C_mod_123/checkpoints` | `ckpt-60` | 19.800 | 600 |
| 256 | `Feedback 23-11/A2C_mod_256/checkpoints` | `ckpt-60` | 19.800 | 600 |
| 512 | `Feedback 23-11/checkpoints_a2c_512` | `ckpt-60` | 19.800 | 600 |
| 1024 | `Feedback 23-11/checkpoints_a2c_1024` | `ckpt-60` | 19.800 | 600 |

- Quy ước: checkpoint lưu mỗi 10 episode → `ckpt-60` = 600 episode.
- Seed 256 còn `ckpt-76`, seed 1024 còn `ckpt-64`: chỉ dùng để minh họa collapse, không dùng cho phân tích chính.
- Seed 42 có `ckpt-50`: dùng cho kiểm tra công bằng 500 episode (tùy chọn, mặc định tắt).

### 2.2. Scenario

| Scenario | sales_scale | waste_rate |
|---|---|---|
| EASY | 0.5 | 0.010 |
| MEDIUM | 1.0 | 0.025 |
| HARD | 1.5 | 0.050 |

### 2.3. State & background

- **Micro:** 50 state 660-dim mỗi scenario:
  - `[x_init | sales_t × scale | x_init × waste_rate]`
- **Macro:** gộp mỗi state 660-dim thành 3 feature:
  - `[mean(inventory), mean(demand), mean(waste)]`
- **Background:**
  - Macro: 100 state synthetic 3-dim, seed 42.
  - Micro: 100 state synthetic 660-dim, seed 42.
  - Cố định background cho mọi seed để chỉ mô hình thay đổi.

### 2.4. Top-k

- Micro: `k = 10, 20, 50`.
- Macro: `k = 1, 2` (vì chỉ có 3 feature); báo cáo thêm Spearman trên cả 3.

## 3. Vì sao dùng cả `π` và `logits`

- **`π` (policy probabilities)** là đầu ra quyết định thực tế. SHAP trên `π` trả lời: *“đặc trưng này làm tăng/giảm xác suất chọn từng action bao nhiêu?”* Đây là target chính của paper cho A2C_mod, nên kết quả so sánh được trực tiếp với paper.
- **`logits` (pre-softmax)** là điểm ưu tiên nội bộ trước khi chuẩn hóa. SHAP trên `logits` trả lời: *“đặc trưng này làm thay đổi xu hướng ưu tiên của mạng bao nhiêu?”*
- **Vì sao cần cả hai:** 5 seed có policy rất bão hòa:
  - Seed 256: 88.7% SKU chọn action 13.
  - Seed 123: 82.9% action 13.
  - Seed 1024: 66.7% action 13.
  - Khi `π` gần one-hot, SHAP trên `π` có thể gần 0 và rất nhiễu; `logits` vẫn giữ cấu trúc ưu tiên và nhạy hơn với thay đổi đặc trưng.
- **Lợi ích:** tách được “ổn định ở mức quyết định” (`π`) và “ổn định ở mức biểu diễn” (`logits`). Nếu hai target cho cùng kết luận → kết luận vững; nếu khác nhau → đó là phát hiện quan trọng (paper cũng đã ghi nhận target sensitivity: π vs logits vs V(s), Jaccard chỉ 0.25).
- Notebook tính thêm **Jaccard/Spearman giữa `π` và `logits` trên cùng seed** để lượng hóa mức khác biệt này.

## 4. Thiết kế notebook (chỉ 1 file)

**File:** `Feedback 23-11/task1/task1_shap_seed_stability.ipynb`

| Cell | Nội dung |
|---|---|
| 0 | Markdown: mục tiêu, cấu hình, giải thích π vs logits |
| 1 | Config: paths, seeds, steps, scenarios, targets, k, cờ tùy chọn |
| 2 | Imports, TF config, reproducibility |
| 3 | Model classes: `Dense`, `Actor` có `logits()` và `__call__()` (dropout=0) |
| 4 | Load data, build state 660-dim + macro 3-dim, build background |
| 5 | Load checkpoint, metadata, action stats, rollout diagnostics |
| 6 | SHAP wrappers: macro π/logits, micro π/logits |
| 7 | Hàm chạy SHAP + lưu raw NPZ + resume (bỏ qua nếu đã có) |
| 8 | Chạy toàn bộ: target × scenario × seed (cell lâu nhất) |
| 9 | Load kết quả, tính mean±SD |SHAP|, group-level |
| 10 | Jaccard/Spearman/RBO, pairwise, summary, cross-target |
| 11 | Vẽ hình: heatmap, top-20, action distribution, Jaccard vs k |
| 12 | Tự sinh `output/Task1_Report.md` với bảng + hình + kết luận |
| 13 | (Tùy chọn, mặc định tắt) matched-500 và noise floor |

Notebook có **resume**: mỗi kết quả lưu thành `output/raw/{level}_{target}_{scenario}_seed{seed}.npz`; nếu đã có thì bỏ qua. Bị ngắt giữa chừng chỉ cần chạy lại cell 8.

## 5. Chi tiết kỹ thuật

### 5.1. SHAP macro

- `shap.KernelExplainer` trên 50 state 3-dim.
- Background 100 synthetic 3-dim.
- Target `π` và `logits`.
- `mean|SHAP|` qua 50 state và 14 action → 3 giá trị/seed.
- Lưu ý: state macro tổng hợp có inventory/waste gần như cố định trong một scenario, nên SHAP macro chủ yếu phản ánh demand. Điều này sẽ được ghi rõ trong report. Có cờ `MACRO_PER_PRODUCT` để chạy thêm biến thể per-product nếu cần.

### 5.2. SHAP micro

- `shap.PartitionExplainer` với `shap.maskers.Partition(bg660, max_samples=100)`.
- 50 state 660-dim.
- Wrapper:
  - `π`: reshape `[B,3,220] → [B×220,3]`, actor → `[B×220,14]`, reshape `[B,220,14]`, mean qua 220 SKU → `[B,14]`.
  - `logits`: tương tự nhưng dùng `actor.logits`.
- `mean|SHAP|` qua 50 state và 14 action → 660 giá trị/seed.

### 5.3. Metric

- **Jaccard top-k:** `|A∩B| / |A∪B|` cho từng cặp seed (10 cặp).
- **Spearman:** trên toàn bộ 660 feature; bổ sung Spearman trên union top-k.
- **Mean ± SD |SHAP|** theo feature qua 5 seed; thêm CV và SD của rank.
- **Group-level:** tổng |SHAP| theo Inventory/Demand/Waste.
- **Cross-target:** Jaccard/Spearman giữa `π` và `logits` trên cùng seed.
- **Summary 3 phiên bản:**
  - tất cả 5 seed,
  - loại seed 256,
  - chỉ 2 seed không bão hòa (42, 512).

### 5.4. Collapse diagnostics

- Mean argmax action, histogram 14 action, % action 13, entropy, max probability.
- Rollout 100 bước trên test sales theo đúng dynamics training:
  - reward, stockout, waste, overstock, `quan`.
- So sánh seed 256 `ckpt-60` vs `ckpt-76`.
- Ghi chú seed 123/1024 cũng bão hòa mạnh.

## 6. File sinh ra

| File | Nội dung |
|---|---|
| `output/checkpoints_used.csv` | Seed, path, checkpoint, step, episode |
| `output/collapse_diagnostics.csv` | Action distribution, entropy, % action 13 |
| `output/rollout_metrics.csv` | Reward/stockout/waste theo seed & scenario |
| `output/raw/macro_{target}_{scenario}_seed{seed}.npz` | Raw SHAP macro |
| `output/raw/micro_{target}_{scenario}_seed{seed}.npz` | Raw SHAP micro |
| `output/shap_macro_importance.csv` | Mean±SD |SHAP| macro |
| `output/shap_micro_importance.csv` | Mean±SD |SHAP| micro |
| `output/stability_pairwise.csv` | Jaccard/Spearman từng cặp seed |
| `output/stability_summary.csv` | Trung bình ± SD theo k/scenario/target |
| `output/cross_target_stability.csv` | So sánh π vs logits |
| `output/mean_sd_by_feature.csv` | Mean, SD, CV, rank SD từng feature |
| `output/figs/*.png` | Heatmap, top-20, action distribution, Jaccard vs k |
| `output/Task1_Report.md` | Báo cáo tự sinh: bảng, hình, kết luận, seed 256 |

## 7. Thời gian dự kiến

- Macro: rất nhanh, vài phút cho 5 seed × 3 scenario × 2 target.
- Micro: ~70 giây/lần chạy → 5 seed × 3 scenario × 2 target ≈ **35–40 phút**.
- Analysis + hình + report: ~5 phút.
- **Tổng chính:** khoảng **45–60 phút**.
- Tùy chọn matched-500/noise floor: cộng thêm tùy cấu hình.

## 8. Tiêu chí kiểm chứng

- Đủ 5 checkpoint được load đúng, có metadata step/episode.
- Đủ 60 file raw SHAP (2 level × 2 target × 3 scenario × 5 seed).
- Có bảng Jaccard/Spearman cho từng scenario, target, k.
- Có mean ± SD |SHAP| theo feature.
- Có phần seed 256 riêng + action distribution + rollout.
- Có so sánh π vs logits.
- Report kết luận rõ: ổn định / không ổn định / ổn định có điều kiện, kèm lý do.

## 9. Rủi ro & giả định

- Seed 42 chỉ 530 episode (ckpt-53) vs 600 của các seed khác → có thể ảnh hưởng; sẽ ghi rõ và có tùy chọn matched-500.
- 3/5 seed bão hòa → SHAP trên π có thể suy biến; logits là target bổ sung bắt buộc.
- Macro tổng hợp làm inventory/waste gần như cố định → sẽ ghi chú; có tùy chọn per-product.
- PartitionExplainer có yếu tố ngẫu nhiên → cố định seed, lưu raw, có thể lặp noise floor.
- Không dùng lại output SHAP cũ vì checkpoint `output Training` không trùng 5 seed này.

## 10. Việc làm ngay sau khi được duyệt

1. Tạo `Feedback 23-11/task1/planTask1.md` với nội dung này.
2. Tạo notebook `task1_shap_seed_stability.ipynb` theo đúng 13 cell ở trên.
3. Chạy cell 1–6, kiểm tra load 5 checkpoint + action stats.
4. Chạy cell 8 (SHAP toàn bộ), theo dõi tiến độ và resume nếu cần.
5. Chạy cell 9–12, xuất bảng/hình/report.
6. Đối chiếu kết quả với paper và ghi chú các phát hiện chính.

# Plan Task 3 — Báo cáo kết quả SHAP ở kịch bản HARD với Trajectory Background (100% hợp lệ) vs. Synthetic Background

## 1. Phân tích yêu cầu Task 3 (Hiểu ngắn gọn)

- **Vị trí trong bài báo** ([Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx)):
  - **P163–P164** (`2.3.4.2 Background Sensitivity`, tương ứng §3.3.4.2 theo đánh số bài gốc)
  - **P165–P166** (`2.3.4.3 Validity of Perturbed States`, tương ứng §3.3.4.3)
- **Vấn đề hiện tại trong bài báo:**
  - Ở `2.3.4.2`, độ nhạy của SHAP theo background mới chỉ được chạy trên kịch bản **MEDIUM** (từ thực nghiệm cũ `Feedback 7-9/task14-9`).
  - Ở `2.3.4.3`, kiểm định tính hợp lệ vật lý của trạng thái nhiễu loạn cho thấy: ở kịch bản **HARD**, **synthetic background chỉ đạt 84.5% hợp lệ** (4.224/5.000 mẫu), trong khi **trajectory background đạt 100% hợp lệ** (5.000/5.000 mẫu).
  - Do đó, reviewer/editor đặt câu hỏi: *nếu 15.5% trạng thái nhiễu loạn ở HARD không hợp lệ khi dùng synthetic background, liệu kết luận SHAP ở HARD có thay đổi khi chuyển sang trajectory background (100% hợp lệ) hay không?*
- **Việc cần làm:**
  1. Chạy lại thực nghiệm SHAP trên kịch bản **HARD** cho cả **DQN** và **A2C_mod** với **2 loại background** cạnh nhau:
     - `syn_random_100` (Synthetic background — baseline hiện tại của bài báo, 84.5% hợp lệ ở HARD)
     - `traj_random_100` (Trajectory background — 100% hợp lệ vật lý ở HARD)
  2. Chạy ở **cả 2 mức độ** giống đúng thiết kế bài báo:
     - **Macro (3 đặc trưng, `KernelExplainer`)**: so sánh thứ hạng (Inventory / Demand / Waste), `mean|SHAP|`, Spearman $\rho$, Cosine similarity, MAD, FCS.
     - **Micro (660 đặc trưng, `PartitionExplainer`)**: so sánh `mean|SHAP|`, Cosine similarity, Spearman $\rho$, Jaccard Top-k ($k = 10, 20, 50$), và Top-5/Top-10 SKU chi phối.
  3. Tổng hợp thành **bảng so sánh song song (Synthetic vs. Trajectory ở HARD)** có thể ghép trực tiếp vào bảng độ nhạy background của **R3-02 / Task 4.1** tại §2.3.4.2–2.3.4.3, kèm đoạn phân tích chuẩn bài báo khoa học.

---

## 2. Cấu hình thực nghiệm chốt

### 2.1. Mô hình & Checkpoint (tái sử dụng đúng checkpoint của `task14-9`)
| Agent | Đường dẫn checkpoint | Kiến trúc |
|---|---|---|
| `A2C_mod` | `output Training/outputA2Cmod/checkpoints_a2cmod` | Actor 4 lớp (3×32 + 14 softmax), dropout=0 khi suy luận |
| `DQN` | `output Training/checkpoints_dqn_comparison3primary` | `MultiProductQNetwork` (3×128 + GroupNorm + 14 Q-values), `training=False` |

> **Cải tiến kỹ thuật so với notebook cũ (`task14-9`):**
> - Dùng đúng lớp `MultiProductQNetwork` (có `GroupNormalization`) cho cả macro và micro DQN để đảm bảo 100% trọng số checkpoint được nạp đúng, không dùng lớp `DQNAgentRDX` thiếu GroupNorm như notebook macro cũ.
> - Kiểm tra xác nhận tải checkpoint thành công (`assert_existing_objects_matched` / kiểm tra norm trọng số khác khởi tạo ngẫu nhiên).

### 2.2. Kịch bản HARD & Trạng thái kiểm định
- Kế thừa đúng định nghĩa HARD của pipeline SHAP (`Feedback 7-9/task14-9/audit_shap_660_partition.ipynb` & `task1`):
  - `sales_scale = 1.5`, `waste_rate = 0.05` trên 50 bước kiểm định đầu tiên của `data/test.tfrecords` (`n = 50` states).
- **Xây dựng Trajectory Background (100% hợp lệ) ở HARD:**
  - *Vấn đề ở notebook cũ:* `build_trajectory_background` trong `task14-9` giữ cố định `x_init` ở mọi bước nên Inventory và Waste có phương sai = 0 → KernelSHAP gán attribution = 0 cho Inventory và Waste.
  - *Giải pháp trong Task 3:* Chạy **rollout thật** của môi trường tồn kho trên chuỗi cầu HARD (bước $t = 0 \dots 199$ của `test.tfrecords`, áp dụng đúng động lực học $x_{t+1} = \max(0, \min(1, x_t + u_t) - d_t)$ và $w_t = 0.05 \cdot x_t$) để thu **100 trạng thái quỹ đạo thật (`traj_rollout_100`)** nơi cả 3 biến Inventory, Demand, Waste đều biến thiên tự nhiên và thỏa mãn **100% ràng buộc vật lý**.
  - Đồng thời kiểm định lại tỷ lệ hợp lệ vật lý (5.000 mẫu nhiễu loạn) của cả `syn_random_100` và `traj_rollout_100` ở HARD để xác nhận con số 100% vs ~84.5%.
  - *(Tùy chọn đối chiếu nhanh cho macro — tốn < 2 giây)*: chạy thêm biến thể `traj_static_100` (kiểu cũ của `task14-9`) để giải thích rõ vì sao rollout trajectory hợp lý hơn.

### 2.3. Chỉ số đánh giá (Metrics)
| Cấp độ | Explainer | Target | Chỉ số báo cáo |
|---|---|---|---|
| **Macro (3-dim)** | `shap.KernelExplainer` | DQN: `softmax(Q)` (hoặc `Q` trung bình); A2C: `π` | `mean\|SHAP\|` của `[Inventory, Demand, Waste]`, thứ hạng đặc trưng, **Spearman $\rho$**, **Cosine similarity**, **MAD**, **FCS**, tỷ lệ hợp lệ trạng thái nhiễu loạn (%). |
| **Micro (660-dim)** | `shap.PartitionExplainer` (`max_samples=100`) | Portfolio-averaged 14-action vector (khớp `task14-9` và `task1`) | **Cosine similarity**, **Spearman $\rho$** (toàn 660 & union top-20), **Jaccard Top-k** ($k = 10, 20, 50$), **MAD**, phân bổ tỷ trọng nhóm (Inventory/Demand/Waste), và Top-5 SKU chi phối của Synthetic vs. Trajectory. |

---

## 3. Thiết kế script / notebook thực nghiệm

Tạo 1 notebook duy nhất có cơ chế **lưu raw & resume** tại `Feedback 23-11/task3/task3_hard_trajectory_shap.ipynb`:

| Cell | Nội dung |
|---|---|
| 0 | Markdown: mục tiêu Task 3, cấu hình HARD, giải thích `syn_random_100` vs `traj_rollout_100` |
| 1 | Imports, seed=42, đường dẫn dữ liệu & checkpoint |
| 2 | Load `test.tfrecords`, `capacity.tfrecords`, `stock.tfrecords` và load thật 2 checkpoint (`DQN`, `A2C_mod`) + kiểm tra xác nhận nạp trọng số |
| 3 | Tạo tập test HARD (50 states) và 2 tập background (100 states): `syn_random_100` và `traj_rollout_100` (từ rollout thật) + kiểm định 100% tính hợp lệ vật lý |
| 4 | Chạy **Macro SHAP** (3-dim, `KernelExplainer`) cho 2 agent × 2 background (~5–10 giây), lưu `output/raw/macro_*.npz` |
| 5 | Chạy **Micro SHAP** (660-dim, `PartitionExplainer`) cho 2 agent × 2 background (có resume, lưu `output/raw/micro_*.npz`) |
| 6 | Tổng hợp chỉ số: bảng Macro, bảng Micro, so sánh Synthetic vs. Trajectory (Spearman, Cosine, Jaccard Top-10/20/50, Top-5 SKU, nhóm đặc trưng) |
| 7 | Vẽ biểu đồ đối chiếu (Macro bar comparison + Micro group/top-k comparison) |
| 8 | Xuất các file CSV + sinh bảng sẵn sàng ghép vào **R3-02** và soạn `Feedback 23-11/task3/outputTask3.md` (song ngữ Anh–Việt theo cấu trúc `Vị trí / Nội dung sửa`) |

---

## 4. Thời gian chạy dự kiến

- Load dữ liệu + Rollout trajectory + Kiểm định hợp lệ (5.000 mẫu): **~15 giây**.
- Macro SHAP (2 agent × 2 background × 50 states): **~10 giây**.
- Micro SHAP (`PartitionExplainer`, 50 states):
  - `A2C_mod` × 2 background: **~4–6 phút**.
  - `DQN` × 2 background: **~25–35 phút** (có lưu checkpoint `.npz` sau từng lượt chạy nên ngắt giữa chừng vẫn resume được ngay).
- **Tổng thời gian dự kiến:** **~30–40 phút**.

---

## 5. Sản phẩm bàn giao

1. `Feedback 23-11/task3/planTask3.md` (file kế hoạch này).
2. `Feedback 23-11/task3/task3_hard_trajectory_shap.ipynb` (notebook thực nghiệm tự chạy & có resume).
3. `Feedback 23-11/task3/output/`:
   - `hard_validity_check.csv` (xác nhận tỷ lệ hợp lệ synthetic vs. trajectory ở HARD)
   - `hard_macro_shap_comparison.csv` (bảng chi tiết macro 3-dim)
   - `hard_micro_shap_comparison.csv` (bảng chi tiết micro 660-dim: Cosine, Spearman, Jaccard Top-10/20/50, Top-5 SKU)
   - `hard_r3_02_combined_table.csv` (bảng tổng hợp sẵn sàng ghép vào bảng R3-02 của Task 4.1)
   - `figs/hard_shap_syn_vs_traj.png` (biểu đồ minh họa đối chiếu)
4. `Feedback 23-11/task3/outputTask3.md`: trình bày kết quả và phần **Ghi vào bài báo** theo đúng cấu trúc `Vị trí / Nội dung sửa` (song ngữ Anh–Việt) cho **§2.3.4.2–2.3.4.3** (P163–P166 trong file `.docx`).

---

## 6. Tiêu chí kiểm chứng

- [ ] Nạp đúng trọng số thật của cả `DQN` và `A2C_mod` (không fallback mock).
- [ ] Xác nhận `traj_rollout_100` đạt **100% hợp lệ vật lý** ở HARD và cả 3 biến (Inventory, Demand, Waste) đều có phương sai > 0.
- [ ] Đủ kết quả cho cả **Macro (3-dim)** và **Micro (660-dim)** trên cả 2 agent (`DQN`, `A2C_mod`) và 2 background (`Synthetic` vs. `Trajectory`).
- [ ] Có bảng số liệu đặt cạnh nhau rõ ràng, tích hợp mượt với bảng độ nhạy background của R3-02 / Task 4.1.
- [ ] Kết luận trung thực dựa trên số liệu thực tế (xác nhận kết luận SHAP ở HARD có giữ nguyên hay dịch chuyển khi dùng trajectory background 100% hợp lệ).

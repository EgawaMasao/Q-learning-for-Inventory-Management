# Plan Task 6 — Bằng chứng định lượng cho sửa lỗi rò rỉ Normalization Capacity (§2.1.3, Table 3)

> **Phương án được chọn:** **Phương án B** *(đánh giá các checkpoint hiện có trên môi trường đã sửa `capacity_train_only`, báo cáo $\Delta$ của mọi metric + kiểm định thống kê; không retrain để tối ưu thời gian. Phương án A giữ làm dự phòng nếu phản biện yêu cầu huấn luyện lại).*

## 1. Phân tích hiện trạng bài báo & dữ liệu có sẵn
- **Vị trí trong bài báo** ([Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx)):
  - **§2.1.3 *Leakage Audit and Corrected Procedure*** (`d.p[067–071]`) và **Table 3** (`d.tables[2]`).
  - Liên quan đến **Table 6 & Table 7** (§3.2).
- **Vấn đề cần giải quyết:**
  - Bài báo khẳng định *"the resulting shift in normalized demand and in reward is under 0.02 ... Neither correction requires retraining the DRL agents, and the reported results remain valid"*, nhưng **chưa trình bày bảng số liệu $\Delta$ của tất cả các metric và kiểm định thống kê** chứng minh nhận định này.
  - **Phát hiện thêm 1 lỗi ngược chiều trong câu văn gốc (`d.p[068]`):** Bài viết gốc ghi `capacity_full` *lớn hơn* (`exceeded`) `capacity_train_only` ~25%, trong khi số liệu audit thực tế ([output_audit_34_leakage_summary.txt](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%207-9/task13-9/task2/output/output_audit_34_leakage_summary.txt)) là `capacity_full mean = 20.33` **nhỏ hơn** `capacity_train_only mean = 25.79` (lệch trung bình 26.29%, tối đa 50.0%, 212/220 SKU lệch > 10%) vì nhu cầu giai đoạn train cao hơn test → cần sửa chiều so sánh cho chính xác.
- **Tài nguyên đã có sẵn trên máy:**
  - Bảng 220 sản phẩm `capacity_full` vs. `capacity_train_only` đã tính sẵn tại [output_audit_34_capacity_diff.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%207-9/task13-9/task2/output/output_audit_34_capacity_diff.csv).
  - Checkpoint hiện có: **5 seed `A2C_mod`** (`42, 123, 256, 512, 1024`) + checkpoint chính `A2C_mod` (`ckpt-64`) + checkpoint chính `DQN` (`ckpt-50`).

---

## 2. Hai phương án thực hiện theo [Task6.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task6/Task6.md)

### Phương án B *(ĐÃ CHỌN)* — Đánh giá checkpoint cũ trên môi trường đã sửa + Kiểm định thống kê *(Nhanh: ~1–2 phút, chạy ngay được)*
1. **Tạo `capacity_train_only.tfrecords`** từ [output_audit_34_capacity_diff.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%207-9/task13-9/task2/output/output_audit_34_capacity_diff.csv) (lưu trong `Feedback 23-11/task6/data/`).
2. **Đánh giá ghép cặp (paired evaluation)** trên 504 bước test × 4 kịch bản (`ALL`, `EASY`, `MEDIUM`, `HARD`) giữa 2 môi trường **`capacity_full`** (cũ) vs. **`capacity_train_only`** (đã sửa) cho:
   - **Base-Stock (`k = 1.0`, val-selected)**: refit đúng target $S_i$ trên tập train cho từng bộ capacity.
   - **DQN** (`ckpt-50`): chạy rollout ghép cặp trên 504 bước.
   - **`A2C_mod`**: chạy trên **đủ 5 seed** (`42, 123, 256, 512, 1024`) và checkpoint chính (`ckpt-64`).
3. **Tính đầy đủ 6 metric + $\Delta$ + Kiểm định thống kê:**
   - 6 metric: *Total Reward, Service Level, Holding Cost, Waste Cost, Ordering/Balance Cost, Stockout Rate*.
   - $\Delta = \text{Metric}(\text{train-only}) - \text{Metric}(\text{full})$, kèm 95% bootstrap CI.
   - Kiểm định ghép cặp theo bước thời gian ($n = 504$) và theo seed ($n = 5$): **Paired t-test**, **Wilcoxon signed-rank test**, **Cohen's $d$ / Hedges' $g$**, và **TOST equivalence test** với biên $\pm 0.02$ để kiểm chứng trực tiếp khẳng định `"shift in reward is under 0.02"`.
4. **Cập nhật phần viết (§2.1.3 & Table 3):**
   - Sửa lỗi ngược chiều (`capacity_full` nhỏ hơn `capacity_train_only`).
   - Bổ sung bảng định lượng **Old vs. Corrected Capacity** ($\Delta$ mọi metric + p-value / TOST) ngay trong §2.1.3 (hoặc Supplementary) và nêu trung thực phạm vi (đánh giá checkpoint cũ trên môi trường đã sửa).

---

### Phương án A *(Dự phòng nếu phản biện yêu cầu)* — Huấn luyện lại DQN & `A2C_mod` với `capacity_train_only` (5 seed)
1. **Tạo `capacity_train_only.tfrecords`** từ [output_audit_34_capacity_diff.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%207-9/task6/data/).
2. **Huấn luyện lại 5 seed (`42, 123, 256, 512, 1024`)** trên `capacity_train_only.tfrecords`:
   - **`A2C_mod` (5 seed):** dùng kiến trúc GPU `@tf.function` tối ưu (tương tự `A2C_512_GPU.ipynb`, ~5–15 phút/seed trên Colab GPU).
   - **`DQN` (5 seed):** dùng `Training/DQN.ipynb` đã gắn `seed` và trỏ `capacity_file` sang `capacity_train_only.tfrecords`.
3. **Đánh giá & Đối chiếu Table 6 / Table 7 cũ vs. mới:**
   - Tổng hợp **Mean ± SD qua 5 seed** cho `Base-Stock`, `DQN`, `A2C_mod` dưới `capacity_train_only`.
   - Lập bảng đối chiếu **Original (`capacity_full`) vs. Retrained (`capacity_train_only`)** với $\Delta$ mọi metric + kiểm định Welch/paired + TOST.
4. **Cập nhật §2.1.3, Table 3, và (nếu cần) Table 6 / Table 7** trong bài báo.

---

## 3. So sánh nhanh để chọn phương án

| Tiêu chí | Phương án B (Eval checkpoint cũ) | Phương án A (Retrain 5 seed DQN + A2C_mod) |
|---|---|---|
| Thời gian chạy | **~1–2 phút** (chạy ngay trên máy local) | ~1–2 giờ trên Colab GPU (10 lần train) |
| Mức độ thuyết phục reviewer | Tốt (có $\Delta$ mọi metric + kiểm định thống kê đầy đủ) | **Cao nhất** (loại bỏ hoàn toàn leakage từ khâu train) |
| Tài nguyên cần | 1 notebook `task6_capacity_eval.ipynb` | `capacity_train_only.tfrecords` + 10 notebook train (5 DQN + 5 A2C_mod) + 1 notebook tổng hợp |

> **Đề xuất thực tế:** Chạy **Phương án B trước** (chỉ mất ~2 phút, có ngay số liệu $\Delta$ và kiểm định để biết chính xác mức ảnh hưởng thực tế của `capacity_train_only`); nếu giảng viên yêu cầu bắt buộc retrain thì dùng chính `capacity_train_only.tfrecords` đó để chạy tiếp **Phương án A** trên Colab GPU.

---

## 4. Sản phẩm dự kiến
- `Feedback 23-11/task6/data/capacity_train_only.tfrecords`
- Notebook thực nghiệm (`task6_capacity_eval_optionB.ipynb` cho Phương án B, và/hoặc bộ notebook retrain 5 seed cho Phương án A)
- `Feedback 23-11/task6/output/` (các file CSV kết quả $\Delta$, kiểm định thống kê)
- `Feedback 23-11/task6/outputTask6.md` (song ngữ Anh–Việt theo đúng format **`Ghi vào bài báo: Vị trí / Nội dung sửa`** cho §2.1.3 & Table 3).

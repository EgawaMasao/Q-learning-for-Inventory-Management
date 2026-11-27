# Plan Task 5 — Kiểm tra đối chiếu mã nguồn & Thống nhất toàn bài cho C-02 (Task 5.1) và C-03 (Task 5.2)

## 1. Mục tiêu & Phạm vi
- **Loại việc:** Kiểm tra đối chiếu mã nguồn + chuẩn hóa phần viết (100% không chạy huấn luyện/thực nghiệm mới).
- **Mục tiêu:** Đóng vai trò **báo cáo xác minh nguồn gốc (code-to-paper audit)** cho hai mâu thuẫn nội tại quan trọng nhất của bài báo ([Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx)):
  - **Task 5.1 (C-02):** Xác nhận kiến trúc thật của `A2C_mod` từ mã nguồn và đối chiếu thống nhất giữa **Table 2** và **§3.2 (`A2C_mod`)**.
  - **Task 5.2 (C-03):** Xác nhận đầu ra DQN được giải thích bởi SHAP từ toàn bộ các notebook SHAP trong repo và thống nhất ở **mọi vị trí** trong bài báo (`§2.2.6 Common Quantity`, `§2.3.1 SHAP for DQN`, `§2.3.3 SHAP Implementation & Table 4`, `§3.6 SHAP Results`).

---

## 2. Kết quả kiểm tra trực tiếp từ mã nguồn

### 2.1. Task 5.1 (C-02) — Kiến trúc thật của `A2C_mod`
Đã kiểm tra 3 file huấn luyện: `Training/A2C-mod.ipynb` (cell 5), `Training/A2C-mod-seed.ipynb` (cell 5), và `training.py` (dòng 31–83):
- **Actor (`class Actor`):** **4 lớp `Dense` tổng cộng** = **3 lớp ẩn** (`3 → 32`, `32 → 32`, `32 → 32`, mỗi lớp có ReLU + Dropout 0.1) + **1 lớp đầu ra** (`32 → 14` + softmax). Không có lớp chuẩn hóa.
- **Critic (`class Critic`):** **2 lớp `Dense` tổng cộng** = **1 lớp ẩn** (`3 → 32` + `LayerNormalization` trong 2 notebook thực thi + ReLU + Dropout 0.1) + **1 lớp đầu ra** (`32 → 1` tuyến tính).
- **Nguyên nhân mâu thuẫn trong bài báo:**
  - `Table 2` (`P064 / TBL02`) chỉ ghi `3 hidden layers, 32 units each` (bỏ sót Critic và không nói rõ "+ lớp đầu ra = 4 lớp").
  - `§2.2.5 / §3.2` (`P140`) gọi là `four-layer feed-forward network` (tính cả lớp đầu ra) và `two-layer critic`.
- **Phương án chốt:** Giữ **nhất quán 100% với `outputTask2.md`** — ghi tường minh ở cả Table 2 và §3.2: *Actor: 3 hidden layers (32 units each) + 1 output layer (4 layers total); Critic: 1 hidden layer (32 units) + 1 output layer (2 layers total), với `LayerNormalization` ở lớp ẩn Critic*.

### 2.2. Task 5.2 (C-03) — Đầu ra thật của DQN trong các script SHAP
Đã kiểm tra toàn bộ 5 notebook SHAP trong repo:

| File mã nguồn | Cấp độ / Mục đích | Đầu ra DQN thực tế (`dqn_predict`) | Đầu ra A2C_mod thực tế |
|---|---|---|---|
| `XAI/SHAP-temp.ipynb` (cell 2, 6, 9–17) | Global & Local SHAP chính (Fig. 10–11) | **`tf.nn.softmax(q_values)`** (14-dim; local chọn chỉ số `argmax`) | **`π(·\|s)`** (14-dim softmax) |
| `Feedback 7-9/task14-9/audit_shap_background_validity.ipynb` (cell 10) | Macro background sensitivity (3-dim) | **`tf.nn.softmax(q)`** (14-dim) | **`π(·\|s)`** (14-dim) |
| `Feedback 23-11/task3/task3_hard_trajectory_shap.ipynb` (cell 2) | HARD Trajectory vs. Synthetic (Macro & Micro) | **`tf.nn.softmax(q)`** (trung bình 220 SKU ở micro) | **`π(·\|s)`** (trung bình 220 SKU ở micro) |
| `Feedback 7-9/task14-9/audit_shap_660_partition.ipynb` (cell 6) & `Ablation_Study/ablation_SHAP.ipynb` (cell 14) | Micro Partition ablation (660-dim) / Common-target logits | **`tf.reduce_mean(q, axis=1)`** (raw pre-softmax Q, 14-dim) | `π(·\|s)` hoặc pre-softmax `logits` |
| `Ablation_Study/ablation_RDX.ipynb` (cell 21) | Ablation GradientExplainer | `sum(max_a Q)` | `max_a π` |

- **Nguyên nhân mâu thuẫn trong bài báo:**
  - `P143` (*Common Quantity*) và `P284` / `P289` (*SHAP Results*) mô tả đúng pipeline chính: **`softmax(Q)`** cho DQN và **`π(·|s)`** cho `A2C_mod`.
  - Nhưng **`P148–P150` (*SHAP for DQN*)**, **`P156` (*SHAP Implementation Details*)**, và **`Table 4` (`P157 / TBL04`)** vẫn viết là **`action value function Q(s, a)` / `selected-action Q-value Q(s, a*)`**.
- **Phương án chốt:** Thống nhất toàn bộ 5 vị trí trong bài báo (`P143`, `P148–P150`, `P156`, `Table 4`, `P284/P289`) theo đúng thực tế mã nguồn:
  1. **Mục tiêu SHAP chính (Primary target):** $\operatorname{softmax}(Q(s,\cdot))$ cho DQN (14 hành động, trung bình trên 220 SKU ở cấp vi mô; khi giải thích cục bộ một quyết định cụ thể thì trích xuất thành phần ứng với hành động được chọn $a^*$) và phân phối chính sách $\pi(\cdot \mid s)$ cho `A2C_mod`.
  2. **Ghi chú đối chứng minh bạch (Ablation note):** Nêu rõ 1 câu ở `P156` / `Table 4` rằng giá trị Q thô trước softmax (pre-softmax logits) chỉ được dùng trong phần kiểm tra độ nhạy mục tiêu chung (`§3.6.6`, `P310`) và đối chứng Partition 660 chiều.
  3. **Bổ sung phần còn thiếu từ Task 4:** Viết chỉnh sửa cho **`P148–P150` (*SHAP for DQN*)** để khớp 100% với `P156`, `Table 4` và `P284` đã sửa ở `outputTask4.md`.

---

## 3. Sản phẩm bàn giao sau khi duyệt
Tạo `Feedback 23-11/task5/outputTask5.md` gồm:
1. **Bảng bằng chứng đối chiếu mã nguồn** (file, cell/dòng code) cho cả **Task 5.1 (C-02)** và **Task 5.2 (C-03)**.
2. **Bản đồ đồng bộ toàn bài** liên kết `outputTask2.md`, `outputTask4.md` và `outputTask5.md` (đảm bảo không có bất kỳ vị trí nào còn sót mâu thuẫn).
3. Mục **`Ghi vào bài báo`** (`- Vị trí:` / `- Nội dung sửa:`, song ngữ Anh–Việt) tập trung vào phần **cập nhật bổ sung tại `P148–P150` (*SHAP for DQN*)** và bản tóm tắt chốt cuối cho **Table 2**, **§3.2**, **Table 4**, **`P156`**, **`P284`**.

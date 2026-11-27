# Plan Task 4 — Hoàn thiện 3 nhóm chỉnh sửa phần viết (Task 4.1, 4.2, 4.3)

## 1. Phạm vi & Nguyên tắc
- **Loại việc:** 100% xử lý **phần viết** trên bản thảo [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx) (không chạy code, không làm thêm thực nghiệm; tận dụng trực tiếp số liệu MEDIUM từ `Feedback 7-9/task14-9/output/` và số liệu HARD vừa chạy từ **Task 3**).
- **Định dạng đầu ra:** Xuất toàn bộ sang `Feedback 23-11/task4/outputTask4.md` theo đúng cấu trúc **`Ghi vào bài báo: Vị trí / Nội dung sửa`** kèm **song ngữ Anh–Việt** (văn phong súc tích, chuẩn bài báo khoa học).

---

## 2. Kế hoạch chi tiết cho từng tiểu mục

### Task 4.1 — Làm rõ chọn Background SHAP & Đưa kết quả Background Sensitivity thành 1 bảng trong bài chính (R3-02)
- **Vị trí trong `.docx`:**
  - `Materials and Methods` → `2.3.4.1 SHAP Background Construction` (`P161–P162`)
  - `2.3.4.2 Background Sensitivity` (`P163–P164`) & `2.3.4.3 Validity of Perturbed States` (`P165–P166`)
- **Nội dung chỉnh sửa:**
  1. **Tại `P162` (`2.3.4.1`):** Bổ sung 2–3 câu giải thích rõ lý do chọn **`synthetic + random-100`** làm baseline chính:
     - Bao phủ đồng đều toàn bộ miền trạng thái chuẩn hóa $[0,1]$ cho phép so sánh công bằng trên các chế độ nhu cầu/tồn kho khác nhau;
     - Giảm chi phí tính toán (KernelSHAP tăng tuyến tính theo kích thước tập nền) trong khi vẫn giữ thống kê mô tả và phân bố khoảng cách cặp gần như đồng nhất với `full-200` và `KMeans-100`;
     - Tái lập tuyệt đối với hạt giống cố định (`seed = 42`), đồng thời được đối chứng bằng tập nền quỹ đạo (`trajectory`) ở cả MEDIUM và HARD.
  2. **Tại `P164–P166` (`2.3.4.2–2.3.4.3`):** Chuyển các con số đang nằm rải rác trong văn xuôi thành **1 bảng tổng hợp duy nhất (Table R3-02)** trong bài chính, **gộp liền mạch cả kết quả MEDIUM (`task14-9`) và HARD (`Task 3`)**:
     - **MEDIUM — Macro (3-dim, KernelSHAP):** Synthetic Random/KMeans (50/100/200) → $\rho = 1.00$; Trajectory Random/KMeans (50/100/200) → $\rho = 0.87$ (1 lần hoán đổi thứ hạng).
     - **MEDIUM — Micro (660-dim, Partition):** Cosine similarity `0.94–0.99`, Spearman $\rho \approx 0$ (`-0.11` đến `+0.17`).
     - **HARD — Macro & Micro (Synthetic vs. Trajectory 100% hợp lệ, từ Task 3):** DQN Macro $\rho = 1.00$ / Cosine `0.987`; A2C_mod Macro = `0.000` (bão hòa trên miền quỹ đạo); Micro Cosine `0.971` (DQN) & `0.895` (A2C_mod), Spearman $\approx 0$, Top-20 Jaccard $\le 0.053$.

---

### Task 4.2 — Khắc phục so sánh độ lớn SHAP không tương đương & Sửa mâu thuẫn C-03
- **Vị trí trong `.docx`:**
  - `SHAP Implementation Details` (`P156`) & **Table 4** (`P157`, `TBL04`) — sửa **C-03**
  - `SHAP Results` (`P284`, `P288–P289`, `P297`) — sửa **C-03** và các câu so sánh độ lớn
  - `SHAP-only FCS` (`P352`) — làm rõ giới hạn ngưỡng $\varepsilon$ giữa 2 thang đo
  - `SHAP Faithfulness / AUPC` (`P401`) — bỏ so sánh định lượng *"about twice as large"*
- **Nội dung chỉnh sửa:**
  1. **Sửa triệt để mâu thuẫn C-03 (Thống nhất SHAP Target):**
     - Sửa `P156`, **Table 4** và `P284` cho khớp 100% với `P143` (*Common Quantity*) và mã nguồn:
       - **DQN:** giải thích đầu ra `softmax(Q)` (xác suất chuẩn hóa từ 14 giá trị Q; lấy trung bình trên 220 SKU ở cấp micro).
       - **A2C_mod:** giải thích phân phối chính sách `π(·|s)` (lấy trung bình trên 220 SKU ở cấp micro).
     - Lưu ý rõ: mặc dù cả hai đều nằm trên simplex $[0,1]^{14}$, `softmax(Q)` phụ thuộc vào biên độ/nhiệt độ của hàm Q chưa chuẩn hóa nên **độ lớn tuyệt đối của SHAP vẫn không tương đương giữa DQN và A2C_mod**; mọi so sánh giữa hai tác nhân chỉ dựa trên **thứ hạng đặc trưng và cấu trúc định tính**.
  2. **Rà và sửa 4 vị trí so sánh độ lớn giữa DQN và A2C_mod:**
     - **`P288–P289` (Global SHAP):** Bỏ các cụm so sánh độ lớn tuyệt đối (`larger dispersion`, `smaller SHAP magnitudes of A2C_mod`) → chỉ diễn đạt theo mức tập trung tương đối vào đặc trưng chủ đạo trong từng mô hình.
     - **`P297` (Local SHAP):** Thay các cụm `very small SHAP values close to zero` vs. `large magnitudes` bằng mô tả cấu trúc thứ hạng/tính tập trung nội bộ của từng tác nhân.
     - **`P352` (FCS):** Làm rõ chênh lệch độ giảm FCS dưới ngưỡng cố định $\varepsilon = 0.01\lambda$ giữa DQN và A2C_mod chịu ảnh hưởng bởi khác biệt thang đo đầu ra, chỉ mang tính mô tả định tính.
     - **`P401` (Faithfulness — AUPC):** Bỏ hẳn cụm **`showing changes about twice as large as the value-based agent`** → chỉ báo cáo thứ tự hợp lệ nội bộ trong từng agent (`MoRF > Random > LeRF`, khoảng tin cậy không giao nhau) và nhấn mạnh không so sánh trực tiếp độ lớn AUPC giữa hai thang đo khác nhau (tỷ lệ giảm tương đối của Q vs. mức giảm tuyệt đối của xác suất $\pi$).

---

### Task 4.3 — Thừa nhận rõ Trade-off Base-Stock vs. DRL & Bỏ hàm ý vượt trội tổng thể
- **Vị trí trong `.docx`:**
  - `Abstract` (`P009`)
  - `§3.2 Operational Performance Comparison` (`P198`, tiêu đề phụ)
  - `§3.3.2 RDX Results` (`P214` và kiểm tra thêm các đoạn lân cận nếu còn cụm hàm ý vượt trội)
  - `Conclusion` (`P441`)
- **Nội dung chỉnh sửa:**
  1. **`Abstract` (`P009`):** Sửa câu *"While the base-stock policy provides a strong reference, DRL agents prioritize service levels and stockout reduction"* → nêu thẳng trade-off ngay trong Abstract: **Base-Stock đạt tổng reward cao hơn với chi phí tồn kho/hao hụt thấp hơn**, trong khi **DQN và A2C_mod gần như triệt tiêu stockout nhưng phải đánh đổi bằng chi phí tồn kho và hao hụt cao hơn**; không có chính sách nào vượt trội tuyệt đối trên mọi tiêu chí.
  2. **`P198` (Tiêu đề phụ trong §3.2):** Đổi tiêu đề mang sắc thái thiên vị *"Why Prefer a DRL Policy When Base-Stock Attains a Higher Total Reward?"* → trung lập hơn: *"Interpreting the Operational Trade-Off When Base-Stock Attains a Higher Total Reward"*.
  3. **`P214` (RDX Results):** Tinh chỉnh câu *"When a DRL agent achieves better service-level performance..."* để gắn rõ với đánh đổi chi phí tồn kho/hao hụt; đảm bảo không còn cụm hàm ý `A2C_mod` hay `DQN` có hiệu năng vận hành vượt trội tổng thể.
  4. **`Conclusion` (`P441`):** Giữ câu thừa nhận trade-off hiện có và bổ sung thêm mệnh đề khẳng định **không có mô hình nào vượt trội toàn diện trên mọi chỉ tiêu**, và việc lựa chọn phụ thuộc vào trọng số chi phí thực tế của nhà bán lẻ.

---

## 3. Sản phẩm bàn giao sau khi duyệt
- Tạo `Feedback 23-11/task4/outputTask4.md` trình bày rõ ràng theo đúng cấu trúc **`Ghi vào bài báo`** (`- Vị trí:` / `- Nội dung sửa:`, đầy đủ song ngữ **Tiếng Anh chuẩn bài báo + Tiếng Việt để theo dõi**) cho cả **Task 4.1**, **Task 4.2**, và **Task 4.3**, kèm checklist đối chiếu hoàn thành.

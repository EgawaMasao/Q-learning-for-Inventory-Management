# Ghi chú các chỉnh sửa đã thực hiện (Tasks 1–5)

- **File gốc giữ nguyên:** `Feedback 23-11/Xai_Inventory_Revision_27Sep (1).docx`
- **File bản thảo đã cập nhật:** `Feedback 23-11/Xai_Inventory_Revision_27Sep_Tasks1to5.docx`
- **Nguyên tắc:** Chỉ thêm/sửa đúng các nội dung thuộc kết quả của Task 1–5; không xóa hoặc thay đổi bất kỳ phần nào khác của bài báo. Toàn bộ số mục bên dưới khớp **100%** với số mục hiển thị trên màn hình Microsoft Word (`numId = 1`).

---

## Task 1 — Độ ổn định SHAP qua 5 seed độc lập (`A2C_mod`)
- Thêm 1 câu nối vào cuối đoạn dẫn **§3.13 Reliability and Stability of the Explanations** (giữ nguyên hình ảnh Table 18 gốc).
- Thêm tiểu mục mới **§3.13.1 Multi-Seed Stability of SHAP Feature Importance and Seed-256 Collapse** ngay trước **4. Conclusion**, gồm 3 đoạn văn (thiết kế 5 seed, kết quả Macro/Micro + cảnh báo tie 648–651/660 feature, và phân tích riêng **seed 256 bị collapse** với 87.7–88.7% action 13).
- Thêm **Table 19** (Jaccard / Spearman top-$k$ cho All 5 seeds, Excl. seed 256, và Seed 256 vs. others) và **Table 20** (`Mean ± SD |SHAP|` cho 3 đặc trưng Macro và Top-5 Micro ở MEDIUM).
- Cập nhật 1 câu ở **5. Study Limitations and Future Work** (ghi nhận đã đánh giá multi-seed SHAP cho `A2C_mod` tại §3.13.1; mở rộng sang DQN và nhiều lần chạy hơn để lại cho hướng nghiên cứu tiếp theo).

## Task 2 — Mô tả đầy đủ `A2C_mod` và so sánh với A2C tiêu chuẩn *(chỉ phần viết, không huấn luyện A2C chuẩn 5 seed, giữ nguyên Table 7)*
- Sửa **Table 2** (§2.1.1, cột `A2C_mod`): cập nhật `Hidden layers / width` (Actor 3 lớp ẩn + đầu ra = 4 lớp; Critic 1 lớp ẩn + đầu ra = 2 lớp), `Normalization` (`LayerNormalization` ở Critic), `Entropy coefficient` (`0.001`, chỉ tính/log không cộng vào loss), và chuẩn hóa ký hiệu `Loss function`.
- Viết lại **§2.2.6 *A2C_mod Algorithm and Its Relation to Conventional A2C*** *(không chạm vào §2.2.5 Reward Coefficient Justification)*: bổ sung 4 phương trình đánh số `(1)–(4)` cho $\delta_{t,i}$, $\mathcal{L}_{\text{critic}}$, $p^{\text{new}}$, $\mathcal{L}_{\text{actor}}$; thêm khung **Algorithm 1** (pseudo-code khớp `Training/A2C-mod.ipynb`); thêm đoạn **Theoretical Motivation** (làm mượt hành động thứ tự kiểu Laplacian, gradient bị chặn kiểu Brier, và thiên lệch khai thác do không tối ưu entropy); thêm **Table X** so sánh trực tiếp Conventional A2C vs. `A2C_mod` (8 tiêu chí) và *Scope note*.

## Task 3 — Chạy lại & báo cáo SHAP kịch bản HARD với Trajectory Background (100% hợp lệ) vs. Synthetic
- Thực nghiệm chạy tại `Feedback 23-11/task3/task3_hard_trajectory_shap.ipynb` trên 2 checkpoint thật (`DQN ckpt-50`, `A2C_mod ckpt-64`), xác nhận `traj_rollout_100` đạt 99.2–100.0% hợp lệ vật lý ở HARD (so với 51.9–76.4% của synthetic).
- Bổ sung kết quả HARD (DQN giữ nguyên thứ hạng Macro `Inventory > Demand > Waste` với $\rho = 1.00$, cosine $= 0.987$; `A2C_mod` về `0.000` trên quỹ đạo HARD do bão hòa chính sách; Micro cosine $= 0.971$ / $0.895$, thứ hạng SKU lẻ không ổn định với $\rho \approx 0$, Top-20 Jaccard $\le 0.053$) vào **§2.3.4 (`2.3.4.2 Background Sensitivity`)** và gộp trực tiếp vào **Table R3-02** (chung với Task 4.1).

## Task 4 — Hoàn thiện 3 nhóm chỉnh sửa phần viết
- **Task 4.1 (Background SHAP & R3-02):**
  - Sửa **§2.3.4 (`2.3.4.1 SHAP Background Construction`)**: bổ sung 3 lý do chọn `synthetic + random-100` làm baseline chính (độ phủ miền $[0,1]$, giảm một nửa thời gian chạy KernelSHAP mà thống kê tương đương `full-200`/`KMeans-100`, tái lập bằng seed 42).
  - Sửa **§2.3.4 (`2.3.4.2 Background Sensitivity`)** và thêm **Table R3-02** (5 cột gọn) trong bài chính, gom trọn kết quả độ nhạy background của cả **MEDIUM** (`task14-9`) và **HARD** (Task 3) thành 1 bảng duy nhất.
- **Task 4.2 (Thang đo SHAP & mâu thuẫn C-03):**
  - Sửa **§2.3.3 SHAP Implementation Details**, **Table 4**, và đoạn mở đầu **§3.6 SHapley Additive exPlanations** để thống nhất đầu ra giải thích (`softmax(Q)` cho DQN, `π(·|s)` cho `A2C_mod`, nhất quán với **§2.2.7 Common Quantity**) và nêu rõ không tương đương về độ lớn tuyệt đối.
  - Sửa **§3.6.2 Experimental Results and Analysis** (Global SHAP & Local SHAP — bỏ `smaller SHAP magnitudes`, `larger dispersion`, `large magnitudes`), **§3.7.4 SHAP-only** (FCS), và **§3.8.4 SHAP Faithfulness: Area Under Perturbation Curve and Confidence Intervals** (bỏ hẳn cụm *"about twice as large"* ở AUPC) để chuyển sang so sánh thứ hạng và cấu trúc định tính nội bộ từng agent.
- **Task 4.3 (Trade-off Base-Stock vs. DRL):**
  - Sửa **Abstract** và **4. Conclusion**: thêm câu thừa nhận thẳng Base-Stock đạt tổng reward cao hơn với chi phí tồn kho/hao hụt thấp hơn, còn DRL đánh đổi chi phí cao hơn để gần như triệt tiêu stockout; không có chính sách nào vượt trội trên mọi tiêu chí.
  - Sửa tiêu đề phụ **§3.2.1** (*"Why Prefer a DRL Policy..."* → *"Interpreting the Operational Trade-Off When Base-Stock Attains a Higher Total Reward"*) và tinh chỉnh 2 câu ở **§3.3.2 Experimental Results and Analysis** (RDX) để bỏ hàm ý vượt trội tổng thể.

## Task 5 — Xác minh mã nguồn & Đồng bộ toàn bài cho C-02 và C-03
- **Task 5.1 (C-02 — Kiến trúc `A2C_mod`):** Xác minh từ `A2C-mod.ipynb`, `A2C-mod-seed.ipynb`, `training.py` và đồng bộ 100% giữa **Table 2** (§2.1.1) và **§2.2.6** (Actor 3 lớp ẩn + 1 lớp đầu ra = 4 lớp; Critic 1 lớp ẩn có `LayerNormalization` + 1 lớp đầu ra = 2 lớp).
- **Task 5.2 (C-03 — Đầu ra DQN cho SHAP):** Xác minh từ 5 notebook SHAP trong repo; giữ nguyên công thức gốc ở **§2.3.1 SHAP for DQN** và nối thêm 2 câu làm rõ đầu ra triển khai là $\operatorname{softmax}(Q(s,\cdot))$ (nhất quán với **§2.2.7**), đồng bộ với **§2.3.3**, **Table 4** (kèm ghi chú pre-softmax logits dùng trong kiểm tra độ nhạy **§3.6.6**) và **§3.6**.

## Guideline §1C, §7 — Data availability & Supplementary materials + CRediT author statement
- **Data availability and supplementary materials** (đặt trước CRediT): đổi tiêu đề `Data and code availability` → `Data availability and supplementary materials`, giữ nguyên link GitHub/Zenodo gốc và bổ sung danh mục đầy đủ các tài liệu nộp kèm trong file Supplementary Material (lệch capacity & lưới safety factor §2.1.3; bảng độ nhạy SHAP logits & actor/critic §3.6.4, §3.6.6–3.6.7; kiểm định thống kê masking §3.8.5; bảng/heatmap multi-seed SHAP & seed-256 §3.13.1).
- **CRediT author statement:** đổi tiêu đề `Credit author statement` → `CRediT author statement`, bỏ cụm sai `"Conceptualization of the review article"`, và viết lại theo đúng bộ thuật ngữ chuẩn CRediT (`Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization, Supervision`) cho từng tác giả.

---

## Lưu ý nhỏ trước khi nộp tạp chí
- **MathType:** Chuyển 4 phương trình mới ở Task 2 (`(1)–(4)`) sang đối tượng **MathType** và kiểm tra ký hiệu $\sum$ *(công thức ở §2.3.1 đã giữ nguyên đối tượng gốc)*.
- **Đánh số bảng:** Đổi tên tạm `Table X` (Task 2) và `Table R3-02` (Task 3 + 4.1) theo thứ tự bảng chính thức của toàn bài khi dàn trang cuối cùng.
- **Table 18:** Do Table 18 trong file gốc là ảnh chèn nên được giữ nguyên; nếu cần thêm hàng *Reproducibility (5-seed SHAP)* vào Table 18 thì cập nhật từ file nguồn tạo ảnh.

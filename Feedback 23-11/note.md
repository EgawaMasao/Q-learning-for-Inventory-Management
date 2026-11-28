# Ghi chú các chỉnh sửa đã thực hiện (Tasks 1–8)

- **File bản thảo nền (người dùng biên tập):** `Feedback 23-11/Xai_Inventory_Revision_27Sep (4).docx`
- **File bản thảo hoàn chỉnh (Tasks 1–8, chữ màu cam cho cập nhật thực nghiệm Task 2):** `Feedback 23-11/task1-8.docx`
- **Nguyên tắc:** Chỉ thêm/sửa đúng các nội dung thuộc kết quả của Task 1–8; không xóa hoặc thay đổi bất kỳ phần nào khác của bài báo. Toàn bộ số mục bên dưới khớp **100%** với số mục hiển thị trên màn hình Microsoft Word (`numId = 1`).

---

## Task 1 — Độ ổn định SHAP qua 5 seed độc lập (`A2C_mod`)
- Thêm 1 câu nối vào cuối đoạn dẫn **§3.13 Reliability and Stability of the Explanations** (giữ nguyên hình ảnh Table 18 gốc).
- Thêm tiểu mục mới **§3.13.1 Multi-Seed Stability of SHAP Feature Importance and Seed-256 Collapse** ngay trước **4. Conclusion**, gồm 3 đoạn văn (thiết kế 5 seed, kết quả Macro/Micro + cảnh báo tie 648–651/660 feature, và phân tích riêng **seed 256 bị collapse** với 87.7–88.7% action 13).
- Thêm **Table 19** (Jaccard / Spearman top-$k$ cho All 5 seeds, Excl. seed 256, và Seed 256 vs. others) và **Table 20** (`Mean ± SD |SHAP|` cho 3 đặc trưng Macro và Top-5 Micro ở MEDIUM).
- Cập nhật 1 câu ở **5. Study Limitations and Future Work** (ghi nhận đã đánh giá multi-seed SHAP cho `A2C_mod` tại §3.13.1; mở rộng sang DQN và nhiều lần chạy hơn để lại cho hướng nghiên cứu tiếp theo).

## Task 2 — Thực nghiệm 5 seed Standard A2C & Mô tả đầy đủ `A2C_mod` *(Thực nghiệm + Viết)*
- **Đã có từ trước (Phần viết):**
  - Sửa **Table 2** (§2.1.1, cột `A2C_mod`): cập nhật `Hidden layers / width` (fix C-02: Actor 3 lớp ẩn + đầu ra = 4 lớp; Critic 1 lớp ẩn + đầu ra = 2 lớp), `Normalization` (`LayerNormalization` ở Critic), `Entropy coefficient` (`0.001`, chỉ tính/log ở `A2C_mod`, có cộng vào loss ở bản đối chứng Standard A2C), và chuẩn hóa ký hiệu `Loss function`.
  - Viết lại **§2.2.6 *A2C_mod Algorithm and Its Relation to Conventional A2C***: bổ sung 4 phương trình đánh số `(1)–(4)`, khung **Algorithm 1**, đoạn **Theoretical Motivation**, và **Table X** (8 tiêu chí Conventional A2C vs. `A2C_mod`).
- **Mới thêm sau thực nghiệm 5 seed Standard A2C (`42, 123, 256, 512, 1024`, 600 episodes, `ckpt-60`):**
  - Thực nghiệm đánh giá thống nhất 4 chính sách tại `Feedback 23-11/task2/task2_eval_standard_a2c.ipynb` (dữ liệu tại `Feedback 23-11/task2/output/`).
  - **§2.2.6 (cuối mục, ngay dưới Table X):** thay *Scope note* cũ (*"left to future work"*) bằng đoạn **Empirical Benchmark Setup** dẫn sang §3.2 (Tables 6 & 7).
  - **§3.2 (`P192`, Table 6, Table 7, và đoạn phân tích ngay sau `P196`):** cập nhật câu mở đầu, **thêm cột `Standard A2C (5-seed)` vào Table 6 và Table 7** trên cùng khung đánh giá thống nhất 504 bước (Standard A2C đạt `+0.8291 ± 0.0088`, $\Delta = -0.0057$ so với Base-Stock, vượt `A2C_mod` ở 5/5 seed), và thêm đoạn **Empirical Comparison: Conventional A2C versus `A2C_mod`** (chứng minh entropy bonus $\beta = 0.001$ tăng entropy ~22×, triệt tiêu hoàn toàn collapse về `action 13` ở 0/5 seed so với 2/5 seed của `A2C_mod`, và giảm SD reward 78×).

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

## Task 6 — Bằng chứng định lượng & kiểm định cho sửa lỗi rò rỉ Normalization Capacity *(Phương án B — đánh giá ghép cặp 5 seed không retrain)*
- Thực nghiệm chạy tại `Feedback 23-11/task6/task6_capacity_eval_optionB.ipynb` (504 bước kiểm định × 4 kịch bản `ALL/EASY/MEDIUM/HARD`, Base-Stock + 5 seed DQN + 5 seed `A2C_mod`).
- **§2.1.3 *Leakage Audit and Corrected Procedure*** (thay đoạn chính và chèn thêm 1 đoạn phân tích định lượng):
  - Sửa lỗi ngược chiều (`capacity_full` mean `20.33` **nhỏ hơn / fell below** `capacity_train_only` mean `25.79`, lệch trung bình `26.29%`, cực đại `50.0%`, `212/220` SKU lệch > `10%`).
  - Đính chính trung thực nhận định cũ `"shift in reward is under 0.02"`: tổng reward tăng `+0.0258` (DQN 5-seed), `+0.0345` (`A2C_mod` 5-seed), `+0.0512` (Base-Stock) hoàn toàn do **Ordering Cost giảm cơ học** vì mẫu số capacity lớn hơn, trong khi **4 chỉ số vận hành bất biến** ($|\Delta| \le 0.0013$, đạt TOST $\pm 0.02$) và **thứ hạng chính sách giữ nguyên 100%** ở mọi kịch bản.
- Cập nhật hàng `Normalization (capacity)` trong **Table 3** và chèn bảng gọn **Table 3b** (5 cột, ngay sau Table 3) báo cáo $\Delta$ của mọi metric + paired $p$-value / TOST ($\pm 0.02$).
- Thêm 1 câu ở **5. Study Limitations and Future Work** ghi nhận giới hạn đánh giá checkpoint hiện có (chưa huấn luyện lại từ đầu dưới `capacity_train_only`).

## Task 7 — Tái cấu trúc Related Work (XAI / XRL / Post-Hoc Attribution), Bảng so sánh RDX/MSX vs. SHAP & Tài liệu Explanation Stability
- **§1.2 *Explainable Reinforcement Learning*** (giữ nguyên toàn bộ **§1.1** và các trích dẫn gốc `[1]–[32]`):
  - Tái cấu trúc thành 3 tầng tách bạch: **(a) General Explainable AI (XAI)**, **(b) Explainable Reinforcement Learning (XRL): Intrinsic vs. Post-Hoc**, và **(c) Post-Hoc Feature Attribution and Explanation Stability**.
  - Thêm đoạn **Comparing RDX/MSX and SHAP, and Rationale for Combining Them** và bảng so sánh gọn **Table 1a** (3 cột × 5 hàng đủ 5 tiêu chí: *Object explained / Explanandum, Explanation space, Output form, Core assumptions, Key limitations*).
- **§1.3 *Explainable Reinforcement Learning in Inventory Management***: cập nhật 1 câu cuối để nối mạch trực tiếp với bảng so sánh và kiểm toán faithfulness/multi-seed stability.
- **References**: bổ sung 5 tài liệu chuẩn IEEE đã xác minh (`[33]` Milani et al. 2024; `[34]` Alvarez-Melis & Jaakkola 2018; `[35]` Adebayo et al. 2018; `[36]` Ghorbani et al. 2019; `[37]` Krishna et al. 2024).

## Task 8 — Làm rõ phạm vi phân tích độ nhạy với chính sách cố định *(Phương án 1 — chỉ viết, giữ nguyên công thức, Table 12 & Fig. 7–9)*
- **§3.5 *Sensitivity Analysis*** (cuối đoạn mở đầu): thêm 2 câu khẳng định rõ đây là **phân tích độ nhạy hậu kiểm với chính sách cố định** (*post-hoc explanation sensitivity under fixed policies*), không huấn luyện lại mạng và không đo mức thích nghi chính sách khi đổi trọng số lúc huấn luyện.
- **§3.5.1 *Total reward and action invariance under single-weight reweighting*** (đầu mục, ngay sau tiêu đề): thêm 1 câu làm rõ *action invariance* là hệ quả thiết kế hậu kiểm trên quỹ đạo cố định, nên Fig. 9 đo độ nhạy của thước đo đánh giá tổng phần thưởng chứ không phải tái tối ưu chính sách.
- **5. Study Limitations and Future Work**: thêm 1 câu ghi nhận hướng huấn luyện lại với các bộ trọng số thay thế ($w_{\text{srv}} = 2$, $w_{\text{hold}} = 2$) thuộc phạm vi học tăng cường đa mục tiêu cho nghiên cứu tiếp theo.

## Guideline §1C, §7 — Data availability & Supplementary materials + CRediT author statement
- **Data availability and supplementary materials** (đặt trước CRediT): đổi tiêu đề `Data and code availability` → `Data availability and supplementary materials`, giữ nguyên link GitHub/Zenodo gốc và bổ sung danh mục đầy đủ các tài liệu nộp kèm trong file Supplementary Material (bao gồm cả bảng kiểm định chi tiết Task 6 §2.1.3 / Table 3b).
- **CRediT author statement:** đổi tiêu đề `Credit author statement` → `CRediT author statement`, bỏ cụm sai `"Conceptualization of the review article"`, và viết lại theo đúng bộ thuật ngữ chuẩn CRediT cho từng tác giả.

---

## Lưu ý nhỏ trước khi nộp tạp chí
- **MathType:** Chuyển 4 phương trình mới ở Task 2 (`(1)–(4)`) sang đối tượng **MathType** và kiểm tra ký hiệu $\sum$ *(công thức ở §2.3.1 đã giữ nguyên đối tượng gốc)*.
- **Đánh số bảng:** Đổi tên tạm `Table 1a` (Task 7), `Table 3b` (Task 6), `Table X` (Task 2), và `Table R3-02` (Task 3 + 4.1) theo thứ tự bảng chính thức của toàn bài khi dàn trang cuối cùng.
- **Table 18:** Do Table 18 trong file gốc là ảnh chèn nên được giữ nguyên; nếu cần thêm hàng *Reproducibility (5-seed SHAP)* vào Table 18 thì cập nhật từ file nguồn tạo ảnh.

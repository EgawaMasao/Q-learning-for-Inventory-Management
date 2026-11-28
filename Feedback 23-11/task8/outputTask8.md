# Output Task 8 — Làm rõ phạm vi phân tích độ nhạy với chính sách cố định (§3.5, §3.5.1)

## Tóm tắt công việc
- **Phương án thực hiện:** **Phương án 1 (chỉ viết, không retrain)** theo đúng yêu cầu bắt buộc của [Task8.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task8/Task8.md) (ưu tiên Trung bình). Giữ nguyên toàn bộ công thức, bảng và hình (Table 12, Fig. 7–9) của §3.5.
- **3 điểm chỉnh sửa gọn:**
  1. **Đầu §3.5 (`P253`):** Khẳng định rõ ngay từ đầu đây là **phân tích độ nhạy hậu kiểm của lời giải thích với chính sách cố định** (*post-hoc, fixed-policy sensitivity*), không phải phân tích thích nghi chính sách khi thay đổi trọng số trong lúc huấn luyện.
  2. **Đầu §3.5.1 (`P274–P275`):** Chèn 1 câu nhắc rõ *action invariance* là hệ quả trực tiếp của việc tái trọng số hậu kiểm trên quỹ đạo cố định của các checkpoint huấn luyện với trọng số đơn vị.
  3. **§5 *Study Limitations* (`P443`):** Thêm 1 câu ghi nhận hướng mở rộng huấn luyện lại với các bộ trọng số khác nhau (ví dụ $w_{\text{srv}} = 2$, $w_{\text{hold}} = 2$).

---

## Ghi vào bài báo

### Chỉnh sửa 1 — Tuyên bố rõ phạm vi *fixed-policy sensitivity* ở đầu §3.5 (`P253`)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **3.5 *Sensitivity Analysis***, chèn **ngay cuối đoạn mở đầu `P253`** (trước `P254 The total reward at time t...`).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào cuối `P253`):**
  > Crucially, this analysis evaluates **post-hoc explanation sensitivity under fixed policies**: the weighting coefficients are perturbed *after* the unit-weighted agents have been trained and their actions and trajectories have been generated, without retraining the DRL networks or altering the selected actions. Accordingly, the results characterize how RDX magnitudes, MSX reason sets, and reweighted evaluation rewards respond to changing operational priorities for a fixed learned policy, rather than how an agent's policy would adapt if those weights were modified during training.

- **Bản Tiếng Việt (Để theo dõi):**
  > Quan trọng là phân tích này đánh giá **độ nhạy hậu kiểm của lời giải thích với chính sách cố định**: các hệ số trọng số chỉ được biến thiên *sau khi* các tác tử huấn luyện với trọng số đơn vị đã hoàn tất huấn luyện và tạo hành động cùng quỹ đạo, không huấn luyện lại mạng DRL và không thay đổi hành động đã chọn. Do đó, kết quả phản ánh cách cường độ RDX, tập lý do MSX và tổng phần thưởng đánh giá biến đổi theo mức ưu tiên vận hành trên một chính sách đã học cố định, chứ không thể hiện cách chính sách của tác tử sẽ thích nghi nếu các trọng số này thay đổi trong quá trình huấn luyện.

---

### Chỉnh sửa 2 — Nhấn mạnh lại ngữ cảnh *action invariance* ở đầu §3.5.1 (`P274–P275`)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **3.5.1 *Total reward and action invariance under single-weight reweighting***, chèn **ngay sau tiêu đề `P274`** (đầu đoạn `P275`, trước phần phân tích Fig. 9).

- **Nội dung sửa (Bản Tiếng Anh — dán vào đầu `P275`):**
  > Because single-weight reweighting is applied post hoc to fixed trajectories produced by the unit-weighted checkpoints, the selected actions remain invariant by construction; Figure 9 therefore measures the sensitivity of the reweighted total-reward evaluation metric—not policy re-optimization—to each weight coefficient.

- **Bản Tiếng Việt (Để theo dõi):**
  > Do việc tái trọng số từng thành phần được áp dụng hậu kiểm trên các quỹ đạo cố định sinh bởi các checkpoint huấn luyện với trọng số đơn vị, các hành động được chọn bất biến theo thiết kế; do đó Hình 9 đo lường độ nhạy của thước đo đánh giá tổng phần thưởng sau tái trọng số—không phải sự tái tối ưu hóa chính sách—đối với từng hệ số trọng số.

---

### Chỉnh sửa 3 — Ghi nhận hướng huấn luyện lại đa trọng số tại §5 (*Study Limitations*, `P443`)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **5 *Study Limitations and Future Work*** (`P443`, chèn 1 câu sau nhận định về giải thích phần thưởng).

- **Nội dung sửa (Bản Tiếng Anh — dán vào `P443`):**
  > Likewise, the reward-weight sensitivity analysis in Section 3.5 evaluates fixed policies trained under unity weights; retraining agents under alternative preference regimes (e.g., $w_{\mathrm{srv}} = 2$ or $w_{\mathrm{hold}} = 2$) to quantify policy adaptation under shifting operational priorities remains a valuable direction for multi-objective RL extensions.

- **Bản Tiếng Việt (Để theo dõi):**
  > Tương tự, phân tích độ nhạy trọng số phần thưởng tại Mục 3.5 đánh giá các chính sách cố định được huấn luyện dưới trọng số đơn vị; việc huấn luyện lại tác tử dưới các cấu hình ưu tiên thay thế (chẳng hạn $w_{\mathrm{srv}} = 2$ hoặc $w_{\mathrm{hold}} = 2$) để định lượng mức độ thích nghi của chính sách khi ưu tiên vận hành thay đổi là một hướng mở rộng có giá trị cho học tăng cường đa mục tiêu.

---

## Checklist kiểm chứng hoàn thành Task 8
- [x] Ghi rõ ngay đầu §3.5 (`P253`) rằng đây là phân tích độ nhạy hậu kiểm của lời giải thích với **chính sách cố định** (*post-hoc, fixed-policy sensitivity*).
- [x] Làm rõ lại ở đầu §3.5.1 (`P274–P275`) rằng *action invariance* là hệ quả thiết kế hậu kiểm, không phải tái tối ưu chính sách.
- [x] Ghi nhận minh bạch giới hạn chưa retrain theo trọng số khác (ví dụ $w_{\mathrm{srv}} = 2, w_{\mathrm{hold}} = 2$) tại §5 (*Study Limitations*, `P443`).
- [x] Giữ nguyên toàn bộ công thức, Table 12 và Fig. 7–9 hiện có; đúng format song ngữ gọn của `outputTask1–7.md`.

# Plan Task 7 — Tái cấu trúc Related Work (XAI / XRL / Post-Hoc Attribution), Bảng so sánh RDX/MSX vs. SHAP & Tài liệu Explanation Stability

## 1. Hiện trạng bài báo & Khoảng trống cần xử lý
- **Vị trí trong file Word gốc** ([Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx)):
  - Khối Related Work hiện nằm ở cuối Section 1 (`P28–P38`, ngay trước `2. Materials and Methods`), gồm 3 tiểu mục:
    - `1.1 Reinforcement Learning for Inventory Management` (`P28–P31`) — *đã ổn định, giữ nguyên*.
    - `1.2 Explainable Reinforcement Learning` (`P33–P35`) — *hiện chỉ có 2 đoạn ngắn, gộp chung XAI, XRL và post-hoc attribution*.
    - `1.3 Explainable Reinforcement Learning in Inventory Management` (`P36–P38`) — *cần tinh chỉnh ngắn để nối mạch với bảng so sánh và lý do kết hợp*.
  - Danh mục tài liệu tham khảo hiện có `[1]–[32]` (`P453–P484`).
- **3 khoảng trống theo [Task7.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task7/Task7.md):**
  1. **Chưa tách bạch 3 tầng khái niệm:** (a) XAI chung, (b) XRL (intrinsic vs. post-hoc), và (c) post-hoc feature attribution.
  2. **Chưa làm rõ RDX/MSX và SHAP khác nhau ở *đối tượng được giải thích* (*explanandum*)**: RDX/MSX giải thích *sự ưu tiên hành động tương phản* trong **không gian mục tiêu/phần thưởng**, còn SHAP giải thích *độ nhạy đầu ra mô hình ($Q$ hoặc $\pi$)* theo **không gian đặc trưng trạng thái**; chưa có bảng đối chiếu và lập luận chặt chẽ tại sao cần kết hợp cả hai.
  3. **Thiếu nền tảng tài liệu về *explanation stability* & *faithfulness*** (dù phần thực nghiệm §3.8 và §3.13 đánh giá rất sâu 2 tiêu chí này).

---

## 2. Kế hoạch triển khai (Loại việc: Viết — không cần chạy code)

### Bước 1 — Tái cấu trúc mục `1.2 Explainable Reinforcement Learning` (`P33–P35`) theo 3 lớp rõ ràng
Viết lại `1.2` súc tích, mạch lạc (3 đoạn/tiểu phần có tiêu đề in đậm để không phá vỡ đánh số mục lục Word):
- **(a) General Explainable AI (XAI):** phân biệt rõ *intrinsic/ante-hoc* vs. *post-hoc* và *local* vs. *global* trong học có giám sát [16, 7], nêu rõ vì sao XAI tĩnh không đủ cho bài toán quyết định tuần tự.
- **(b) Explainable Reinforcement Learning (XRL): Intrinsic vs. Post-Hoc:** làm rõ đặc thù RL (trạng thái tuần tự, phần thưởng trễ, đánh đổi đa mục tiêu) [5, 18, 33]; tách bạch:
  - *Intrinsic XRL:* tích hợp khả năng giải thích ngay trong kiến trúc/huấn luyện (ví dụ: học hàm $Q$ phân rã theo thành phần phần thưởng như RDX/MSX gốc [6], chính sách cấu trúc/ký hiệu).
  - *Post-hoc XRL:* giải thích mô hình đã huấn luyện xong (saliency, surrogate, phân tích đầu vào–đầu ra) [17–20], đồng thời lưu ý rõ điểm tinh tế rằng RDX/MSX một bước áp dụng trên checkpoint sẵn có kết hợp ngữ nghĩa phần thưởng với đánh giá hậu kiểm.
- **(c) Post-Hoc Feature Attribution and Explanation Stability:** phân tích riêng nhóm phân bổ đặc trưng hậu kiểm (SHAP [7], Shapley values cho RL [8], hạn chế của attention [32] — phản ánh độ nhạy thống kê thay vì quan hệ nhân quả), và **bổ sung nền tảng tài liệu về độ ổn định & tính trung thực của lời giải thích (*explanation stability & faithfulness*)**:
  - Tính nhạy/mong manh dưới nhiễu nhỏ và thay đổi thiết lập [34, 36],
  - Kiểm chứng tính trung thực qua perturbation/sanity checks [35],
  - Hiện tượng bất đồng (*disagreement*) và biến thiên giữa các lần huấn luyện độc lập (multi-seed) [37] — tạo tiền đề trực tiếp cho §3.8 và §3.13.

### Bước 2 — Thiết kế Bảng so sánh gọn `RDX / MSX vs. SHAP` + Lý do kết hợp
Chèn **1 bảng gọn 3 cột** (`Dimension | RDX / MSX | SHAP`, vừa khít trang A4 dọc) ngay cuối mục `1.2`, đối chiếu đủ 5 tiêu chí yêu cầu:
1. **Object explained (*Explanandum*):** Ưu tiên hành động tương phản ($a$ vs. $a'$) vs. Đầu ra mô hình ($Q(s,a)$, $\pi(a|s)$, hoặc `logits`).
2. **Explanation space:** Không gian mục tiêu/phần thưởng (Service, Holding, Waste, Ordering) vs. Không gian đặc trưng trạng thái (Inventory, Demand, Waste — macro 3 chiều / micro 660 chiều).
3. **Output form:** Vector chênh lệch phần thưởng $\Delta r_c$ (RDX) và tập lý do tối thiểu đủ (MSX) vs. Giá trị Shapley cộng tính $\phi_j$ cho từng đặc trưng đầu vào.
4. **Core assumptions:** Hàm thưởng phân rã cộng tính được theo các thành phần có nghĩa nghiệp vụ (và giả định 1 bước khi không huấn luyện lại $Q_c$) vs. Phân phối nền (*background distribution*) đại diện và xấp xỉ độc lập/nhóm đặc trưng.
5. **Key limitations:** Phụ thuộc thiết kế phần thưởng; RDX/MSX 1 bước đánh giá thấp hiệu ứng dài hạn vs. Chỉ đo độ nhạy thống kê (không nhân quả), nhạy với tập nền, bão hòa softmax và seed.

Kèm **1 đoạn ngắn giải thích lý do kết hợp (Complementarity Rationale):** hai họ phương pháp trả lời hai câu hỏi trực giao (*"Vì mục tiêu nghiệp vụ nào mà tác tử chọn đặt nhiều hơn thay vì ít hơn?"* vs. *"Tín hiệu trạng thái/SKU nào đang chi phối đầu ra của mạng?"*), nhờ đó khắc phục điểm mù của từng phương pháp đơn lẻ.

### Bước 3 — Tinh chỉnh nối mạch mục `1.3` (`P36–P38`) & Xác minh tài liệu tham khảo mới (`[33]–[37]`)
- Cập nhật nhẹ 2–3 câu cuối ở `P38` để liên kết trực tiếp với bảng so sánh và bài toán kiểm chứng độ ổn định (multi-seed, perturbation faithfulness).
- Tra cứu và xác minh chính xác 100% thông tin thư mục (tác giả, năm, tạp chí/hội nghị, trang, DOI) của **5 tài liệu tham khảo mới (`[33]–[37]`)**:
  - `[33]` Milani et al. (2024) — *Explainable Reinforcement Learning: A Survey and Comparative Review* (ACM Computing Surveys).
  - `[34]` Alvarez-Melis & Jaakkola (2018) — *On the Robustness of Interpretability Methods* (ICML Workshop / arXiv).
  - `[35]` Adebayo et al. (2018) — *Sanity Checks for Saliency Maps* (NeurIPS).
  - `[36]` Ghorbani, Abid & Zou (2019) — *Interpretation of Neural Networks Is Fragile* (AAAI).
  - `[37]` Krishna et al. (2024) — *The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective* (TMLR).

---

## 3. Sản phẩm dự kiến
- `Feedback 23-11/task7/planTask7.md` (file này).
- `Feedback 23-11/task7/outputTask7.md` (song ngữ Anh–Việt theo đúng chuẩn **`Ghi vào bài báo: Vị trí / Nội dung sửa`**, súc tích, sẵn sàng dán vào Word + 5 tài liệu tham khảo `[33]–[37]` đã xác minh DOI).

# Plan Task 8 — Làm rõ phạm vi phân tích độ nhạy với chính sách cố định (§3.5, §3.5.1)

> **Phương án được chọn:** **Phương án 1 — Chỉ viết (làm rõ phạm vi *fixed-policy sensitivity*)**. Phần thực nghiệm huấn luyện lại với 1–2 bộ trọng số (`w_srv=2`, `w_hold=2`) giữ làm tuỳ chọn dự phòng nếu phản biện yêu cầu ở vòng sau.

## 1. Hiện trạng & Vấn đề cần xử lý
- **Vị trí trong bài báo** ([Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx)):
  - **§3.5 *Sensitivity Analysis*** (`P252–P279`, tương ứng `§4.4` trong [Task8.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task8/Task8.md)).
  - **§3.5.1 *Total reward and action invariance under single-weight reweighting*** (`P274–P275`, tương ứng `§4.4.1`).
  - **§5 *Study Limitations and Future Work*** (`P443`).
- **Vấn đề:** Đoạn mở đầu §3.5 (`P253`) hiện chưa tuyên bố rõ ngay từ đầu rằng việc thay đổi hệ số trọng số $\lambda$ chỉ được thực hiện **sau khi chọn hành động trên chính sách cố định**, dễ gây hiểu nhầm rằng phân tích này cho thấy chính sách DRL tự thích nghi khi đổi ưu tiên lúc huấn luyện.

---

## 2. Kế hoạch 3 điểm chỉnh sửa gọn (không thay đổi công thức, bảng hay hình cũ)

1. **Chỉnh sửa 1 — Đầu mục §3.5 (*Sensitivity Analysis*, cuối đoạn `P253`):**
   - Thêm 2 câu xác định rõ phạm vi: đây là **phân tích độ nhạy hậu kiểm của lời giải thích với chính sách cố định (*post-hoc, fixed-policy sensitivity analysis*)** — trọng số $\lambda$ được biến thiên *sau khi* các tác tử huấn luyện với trọng số đơn vị đã tạo hành động và quỹ đạo, không huấn luyện lại mạng và không làm thay đổi chính sách.
   - Nêu rõ mục tiêu của phân tích: kiểm tra xem cường độ RDX, tập lý do MSX và tổng phần thưởng đánh giá biến thiên ra sao theo góc nhìn ưu tiên của người ra quyết định trên cùng một chính sách đã học, *chứ không* đo mức độ thích nghi của chính sách khi đổi hàm thưởng lúc huấn luyện.

2. **Chỉnh sửa 2 — Nhắc lại ngắn ở đầu §3.5.1 (`P274–P275`):**
   - Chèn 1 câu làm rõ ngay đầu §3.5.1 (trước câu bàn về Fig. 9): nhấn mạnh tính bất biến của hành động (*action invariance*) là hệ quả trực tiếp của thiết kế đánh giá hậu kiểm với chính sách cố định, nên Fig. 9 phản ánh độ nhạy của *hàm chấm điểm phần thưởng* khi tái trọng số các chính sách cố định, không phải đường cong của chính sách tái huấn luyện.

3. **Chỉnh sửa 3 — Nối mạch 1 câu tại §5 (*Study Limitations*, `P443`):**
   - Bổ sung 1 câu ghi nhận trung thực: phân tích độ nhạy trọng số ở §3.5 giữ nguyên chính sách đã huấn luyện dưới trọng số đơn vị; việc huấn luyện lại tác tử với các cấu hình trọng số khác nhau (ví dụ $w_{\text{srv}} = 2$ hoặc $w_{\text{hold}} = 2$) để định lượng mức thích nghi của chính sách thuộc phạm vi học tăng cường đa mục tiêu và dành cho nghiên cứu tiếp theo.

---

## 3. Sản phẩm dự kiến
- `Feedback 23-11/task8/planTask8.md` (file này).
- `Feedback 23-11/task8/outputTask8.md` (song ngữ Anh–Việt súc tích theo đúng format **`Ghi vào bài báo: Vị trí / Nội dung sửa`**, sẵn sàng dán vào Word).

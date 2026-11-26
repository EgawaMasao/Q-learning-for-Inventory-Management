Task 4.1
Vị trí trong bài báo: §3.3 SHAP Implementation, §3.3.4.1–3.3.4.2
Vấn đề: Cần làm rõ cách chọn background dataset và độ nhạy của SHAP với background.
Việc cần làm: Đưa kết quả background sensitivity (rank corr 1.00 / 0.87, similarity 0.94–0.99, rank corr ≈0 ở micro) thành 1 bảng trong bài chính thay vì chỉ văn xuôi; giải thích lý do chọn synthetic + random-100 làm baseline.
Loại việc: Viết
Ưu tiên Trung bình

Task 4.2
Vị trí trong bài báo: §3.3, §4.5, §4.7.4, Table 4
Vấn đề: So sánh SHAP giữa softmax(Q) và policy probability không tương đương về độ lớn.
Việc cần làm: Rà mọi câu so sánh độ lớn (vd. 'about twice as large', 'smaller SHAP magnitudes') → chỉ so sánh thứ hạng/định tính. Đồng thời sửa mâu thuẫn: §3.3/Table 4 ghi DQN giải thích Q(s,a*), còn §4.5 ghi softmax(Q) (xem C-03).
Loại việc: Viết
Ưu tiên Trung bình

Task 4.3
Vị trí trong bài báo: Abstract, §4.2, §4.3.2, §5
Vấn đề: Base-stock có total reward cao hơn DRL; không được tuyên bố vượt trội tổng thể.
Việc cần làm: Thêm câu thừa nhận trade-off vào Abstract và Kết luận; rà và bỏ các câu kiểu 'better operational performance', 'A2C_mod demonstrates greater flexibility' nếu hàm ý vượt trội.
Loại việc: Viết
Ưu tiên Trung bình


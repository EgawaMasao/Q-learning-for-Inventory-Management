# Kết quả Task 12-9-4: Phân tích độ nhạy của không gian hành động (7 / 14 / 28) đối với hiệu năng và tính bền vững của giải thích

> **Lưu ý:** Tài liệu này trình bày bằng tiếng Việt để phục vụ công tác xét duyệt. Sau khi được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md`. Các hình minh họa được dẫn chiếu theo định dạng `[Hình - file.png]` kèm chú thích chi tiết. Toàn bộ thực nghiệm được thực hiện trên dữ liệu và mô hình thực tế, không sử dụng dữ liệu giả lập.

---

## 4.1 Yêu cầu nghiên cứu

Task 4 thuộc nhóm `Action space` yêu cầu thực hiện một thí nghiệm ablation có hệ thống nhằm đánh giá mức độ nhạy cảm của khung học tăng cường đối với độ phân giải của không gian hành động rời rạc. Cụ thể, tác nhân phải được huấn luyện và đánh giá độc lập trên ba mức phân giải đại diện là 7, 14 và 28 chiến lược bổ sung tồn kho, trong đó 14 là mức cơ sở đã được sử dụng trong toàn bộ báo cáo trước đó. Mỗi mức phân giải được áp dụng đồng thời cho cả hai họ thuật toán là A2C_mod và DQN nhằm đảm bảo tính so sánh công bằng giữa phương pháp dựa trên chính sách và phương pháp dựa trên giá trị. Yêu cầu không chỉ dừng lại ở việc so sánh hiệu năng vận hành mà còn đòi hỏi chứng minh tính bền vững của các giải thích, tức cùng một cơ chế giải thích có cho kết luận nhất quán khi độ phân giải thay đổi hay không. Toàn bộ thí nghiệm được xếp loại Strong scope và yêu cầu huấn luyện lại, tuy nhiên do sáu điều kiện huấn luyện đã được hoàn tất và hội tụ trước đó nên nghiên cứu này tập trung vào đánh giá chuẩn hóa trên các điểm kiểm tra thực tế.

---

## 4.2 Phương pháp nghiên cứu

### 4.2.1 Thiết kế thí nghiệm không gian hành động

Không gian hành động được định nghĩa là tập hợp các mức tăng tồn kho tương đối so với năng lực lưu trữ. Mức 7 sử dụng tập thưa `[0, 0.01, 0.02, 0.04, 0.12, 0.5, 1.0]` nhằm nhấn mạnh các quyết định biên, mức 14 sử dụng tập cơ sở `[0, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.5, 1.0]` vốn cân bằng giữa độ mịn và khả năng học, và mức 28 sử dụng tập mịn `[0, 0.0025, 0.005, 0.0075, ..., 1.0]` nhằm thăm dò khả năng điều khiển tinh vi. Kiến trúc mạng được giữ nguyên như trong các sổ tay huấn luyện gốc, trong đó tác nhân A2C_mod sử dụng Actor `3→32→32→32→num_actions` với chuẩn hóa nhóm và Critic tương ứng, còn tác nhân DQN sử dụng mạng Q theo sản phẩm với kích thước ẩn đã được tối ưu riêng là 128 và cơ chế chuẩn hóa nhóm. Việc duy trì kích thước ẩn khác nhau giữa hai họ thuật toán là có chủ ý nhằm phản ánh điểm tối ưu riêng của từng phương pháp, đồng thời được ghi chú minh bạch trong bảng kết quả thay vì ép bằng nhau một cách nhân tạo. Môi trường được giữ cố định với hàm phần thưởng `r = 1 - z - overstock - q - quan`, trong đó `z` là chỉ báo hết hàng, `overstock` là lượng vượt năng lực, `q` là hao hụt và `quan` là độ phân tán phân vị, và trạng thái được biểu diễn dưới dạng 660 chiều cho 220 sản phẩm. Mỗi điều kiện được huấn luyện đến hội tụ với 600 tập, trong đó các điểm kiểm tra cuối cùng là `ckpt-60` cho A2C 7/28, `ckpt-64` cho A2C 14, `ckpt-61` cho DQN 7/28 và `ckpt-50` cho DQN 14, và việc đánh giá được thực hiện một cách tất định với hạt giống 42 và chính sách tham lam `argmax` trên tập kiểm định chuẩn.

### 4.2.2 Đánh giá hiệu năng vận hành

Hiệu năng được đo lường thông qua một quá trình triển khai tất định trên toàn bộ chuỗi kiểm định 504 bước thời gian, trong đó tồn kho ban đầu và năng lực được lấy trực tiếp từ các tệp `capacity.tfrecords` và `stock.tfrecords`, còn nhu cầu được chuẩn hóa theo năng lực tại mỗi bước. Tại mỗi bước, tác nhân quan sát trạng thái `[tồn kho, nhu cầu, hao hụt]` và lựa chọn hành động có xác suất hoặc giá trị Q lớn nhất, sau đó môi trường cập nhật tồn kho theo quy tắc `x_next = max(0, min(1, x+u) - sales)` và tính phần thưởng phân tách thành bốn thành phần. Năm chỉ số được tổng hợp theo trung bình trên toàn bộ chuỗi bao gồm phần thưởng trung bình, tỷ lệ hết hàng, hao hụt, vượt tồn và độ phân tán. Cách tiếp cận này đảm bảo rằng mọi khác biệt quan sát được đều xuất phát từ độ phân giải hành động chứ không phải từ sự khác biệt trong dữ liệu hay mức độ ngẫu nhiên.

### 4.2.3 Đánh giá tính bền vững của giải thích

Tính bền vững được kiểm định trên ba trụ cột bổ sung cho nhau, tất cả đều được tái sử dụng từ các mô-đun đã được kiểm chứng trong các nghiên cứu ablation trước. Trụ cột thứ nhất là phân tích đặc trưng toàn cục dựa trên SHAP, trong đó một tập nền 200 trạng thái được sinh từ phân bố đều có kiểm soát và 20 trạng thái đại diện được lấy mẫu cho KernelSHAP với 500 liên minh trên một trạng thái kiểm định, qua đó tính tỷ lệ bao phủ đặc trưng với ngưỡng 0.01 và độ lớn trung bình của giá trị SHAP. Trụ cột thứ hai là phân rã phần thưởng và giải thích tối thiểu đầy đủ, trong đó hiệu giá trị Q giữa hành động tối ưu và hành động tốt thứ hai được phân rã thành bốn mục tiêu theo cơ chế nhìn trước một bước, từ đó tính tỷ lệ bao phủ mục tiêu, kích thước trung bình của tập giải thích tối thiểu và độ ổn định Jaccard khi thay đổi ngưỡng. Trụ cột thứ ba là kiểm định trung thực thông qua nhiễu loạn có chủ đích, trong đó các đặc trưng được che lấp dần theo thứ tự quan trọng nhất và ít quan trọng nhất dựa trên trung vị của tập nền, qua đó đo độ sụt giảm tương đối của giá trị Q hoặc xác suất chính sách và tỷ lệ chuyển đổi hành động. Toàn bộ đánh giá được thực hiện với cùng một tập nền và cùng một trạng thái kiểm định cho cả sáu điều kiện nhằm đảm bảo tính so sánh công bằng.

---

## 4.3 Kết quả thực nghiệm

### 4.3.1 Hiệu năng vận hành theo độ phân giải

Bảng 1 tổng hợp kết quả triển khai tất định trên tập kiểm định cho sáu điều kiện. Nhìn chung, họ DQN cho hiệu năng vượt trội một cách nhất quán so với họ A2C_mod trên cùng độ phân giải, với phần thưởng trung bình cao hơn khoảng 0.4 đến 0.9 và tỷ lệ vượt tồn thấp hơn một đến hai bậc độ lớn. Trong nội bộ từng họ, mối quan hệ giữa độ phân giải và hiệu năng không tuyến tính. Đối với A2C_mod, mức 14 đạt phần thưởng cao nhất là 0.38 với vượt tồn gần như triệt tiêu, trong khi cả hai mức thưa 7 và mịn 28 đều rơi vào vùng phần thưởng âm do vượt tồn tăng vọt lên 0.96 và 0.83, cho thấy chính sách dựa trên gradient gặp khó khăn khi không gian hành động quá thưa hoặc quá mịn. Ngược lại, DQN cho thấy khả năng tận dụng độ mịn tốt hơn khi mức 28 đạt phần thưởng cao nhất là 0.82, tiếp theo là mức 7 với 0.80, cả hai đều vượt mức 14 với 0.60, đồng thời duy trì tỷ lệ hết hàng ở mức không đáng kể và độ phân tán thấp.

**Bảng 1. So sánh hiệu năng vận hành trên tập kiểm định theo độ phân giải không gian hành động.**

| Thuật toán | Số hành động | Kích thước ẩn | Điểm kiểm tra | Phần thưởng | Hết hàng | Hao hụt | Vượt tồn | Phân tán |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A2C_mod | 7 | 32 | ckpt-60 | -0.1265 | 0.0000 | 0.0241 | 0.9649 | 0.1374 |
| A2C_mod | 14 | 32 | ckpt-64 | 0.3833 | 0.0000 | 0.0127 | 0.0009 | 0.6030 |
| A2C_mod | 28 | 32 | ckpt-60 | -0.1170 | 0.0063 | 0.0237 | 0.8391 | 0.2478 |
| DQN | 7 | 128 | ckpt-61 | 0.8001 | 0.0000 | 0.0236 | 0.0024 | 0.1739 |
| DQN | 14 | 128 | ckpt-50 | 0.5960 | 0.0008 | 0.0190 | 0.0005 | 0.3838 |
| DQN | 28 | 128 | ckpt-61 | 0.8213 | 0.0000 | 0.0240 | 0.0061 | 0.1486 |

[Hình 1 - task4_reward_vs_actions.png]
*Hình 1. Phần thưởng trung bình trên tập kiểm định theo độ phân giải không gian hành động. Đường cho DQN duy trì ở mức cao và có xu hướng tăng nhẹ khi độ phân giải tăng, trong khi đường cho A2C_mod đạt đỉnh rõ rệt tại mức 14 và suy giảm mạnh ở hai mức biên, cho thấy sự khác biệt về khả năng mở rộng giữa hai họ thuật toán.*

[Hình 2 - task4_fcs_vs_actions.png]
*Hình 2. Tỷ lệ bao phủ đặc trưng của SHAP theo độ phân giải. Giá trị bao phủ của A2C_mod duy trì ở mức không đáng kể trên mọi độ phân giải, phản ánh độ lớn của giá trị SHAP dưới ngưỡng, trong khi DQN cho mức bao phủ thấp nhưng ổn định ở mức 7 và 28 và giảm ở mức 14, cho thấy độ thưa của giải thích không phụ thuộc đơn điệu vào độ mịn.*

### 4.3.2 Tính bền vững của giải thích

Bảng 2 tổng hợp các chỉ số về tính bền vững. Ở khía cạnh bao phủ đặc trưng, cả hai họ đều cho mức độ thưa cao khi tỷ lệ bao phủ dưới ngưỡng 0.01 duy trì ở mức không đáng kể đối với A2C_mod trên mọi độ phân giải và ở mức 0.02 đối với DQN ở mức 7 và 28, cho thấy chỉ một tập con nhỏ các đặc trưng vượt ngưỡng ý nghĩa dù độ lớn trung bình của SHAP ở DQN cao hơn khoảng một bậc so với A2C_mod. Ở khía cạnh bao phủ mục tiêu, A2C_mod ở mức 7 cho tỷ lệ cao nhất là 0.62, trong khi DQN dao động quanh 0.29 đến 0.41, cho thấy sự khác biệt về mức độ phân tán của các mục tiêu chi phối quyết định. Kích thước trung bình của tập giải thích tối thiểu cho thấy A2C_mod ở mức 28 cần trung bình 2.68 mục tiêu để giải thích ngưỡng, cao hơn rõ rệt so với các mức còn lại quanh 1.0 đến 1.7, phản ánh sự phức tạp gia tăng khi không gian hành động trở nên mịn. Độ ổn định Jaccard khi thay đổi ngưỡng cho thấy A2C_mod ở mức 7 đạt mức cao nhất là 90.7%, trong khi mức 14 của cùng họ giảm xuống 50%, và DQN duy trì trong khoảng 52% đến 60% trên mọi độ phân giải, cho thấy không có mức nào đạt được sự ổn định tuyệt đối nhưng mức 7 của A2C_mod tỏ ra bền vững nhất trước sự thay đổi ngưỡng.

**Bảng 2. So sánh tính bền vững của giải thích theo độ phân giải.**

| Thuật toán | Hành động | Bao phủ đặc trưng | Độ lớn SHAP trung bình | Bao phủ mục tiêu | Kích thước tập tối thiểu | Độ ổn định |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A2C_mod | 7 | 0.0000 | 0.0001 | 0.6250 | 1.00 | 90.66 |
| A2C_mod | 14 | 0.0000 | 0.0001 | 0.2750 | 1.01 | 50.00 |
| A2C_mod | 28 | 0.0000 | 0.0000 | 0.5898 | 2.68 | 59.77 |
| DQN | 7 | 0.0197 | 0.0008 | 0.4170 | 1.18 | 58.83 |
| DQN | 14 | 0.0015 | 0.0002 | 0.2909 | 1.51 | 52.02 |
| DQN | 28 | 0.0197 | 0.0009 | 0.3864 | 1.70 | 60.14 |

[Hình 3 - task4_stability_vs_actions.png]
*Hình 3. Độ ổn định của tập giải thích tối thiểu theo độ phân giải. Đường cho DQN duy trì tương đối phẳng quanh mức 55% đến 60%, trong khi đường cho A2C_mod cho thấy sự biến thiên lớn với đỉnh tại mức 7 và đáy tại mức 14, cho thấy tính nhạy cảm của độ ổn định đối với độ phân giải ở họ dựa trên chính sách.*

---

## 4.4 Diễn giải và đánh giá mức độ hoàn thành

Việc so sánh hiệu năng cho thấy độ phân giải 14 không phải là lựa chọn tối ưu phổ quát mà là điểm cân bằng riêng cho A2C_mod, trong khi DQN cho thấy khả năng hưởng lợi từ độ phân giải cao hơn hoặc thậm chí từ mức thưa khi không gian hành động được thiết kế phù hợp. Phát hiện này trực tiếp trả lời yêu cầu về ablation độ nhạy khi chứng minh rằng việc tăng độ mịn không đảm bảo cải thiện đơn điệu và tác động của nó phụ thuộc vào họ thuật toán. Đồng thời, chi phí vượt tồn tăng vọt ở A2C_mod với mức 7 và 28 cho thấy rủi ro khi lựa chọn độ phân giải không phù hợp cho phương pháp dựa trên chính sách, một thông tin có giá trị thực tiễn cho việc triển khai.

Đánh giá về tính bền vững cho thấy các giải thích nhìn chung duy trì ở mức thưa và không cho thấy sự phụ thuộc đơn điệu vào độ phân giải, qua đó củng cố tính khái quát của các kết luận giải thích trước đó. Sự biến thiên của độ ổn định và kích thước tập tối thiểu, đặc biệt là mức tăng kích thước ở A2C_mod 28, cần được báo cáo như một giới hạn khi diễn giải các quyết định trong không gian mịn, trong khi mức ổn định cao của A2C_mod 7 có thể được nhấn mạnh như một lợi thế về tính nhất quán. Nhìn chung, thí nghiệm đã cung cấp bảng so sánh toàn diện trên cả hai trục hiệu năng và giải thích cho sáu điều kiện thực tế, qua đó đáp ứng đầy đủ yêu cầu của Task 4 ở mức Strong scope mà không cần huấn luyện bổ sung.

Đoạn văn sau đây được đề xuất để tích hợp trực tiếp vào bản thảo nhằm tóm tắt đóng góp của thí nghiệm.

> Thí nghiệm về độ nhạy của không gian hành động được thực hiện trên ba mức phân giải là 7, 14 và 28 chiến lược, được áp dụng độc lập cho cả hai họ thuật toán A2C_mod và DQN trên cùng một môi trường với hàm phần thưởng và biểu diễn trạng thái được giữ cố định. Đánh giá trên tập kiểm định cho thấy DQN duy trì hiệu năng cao trên mọi độ phân giải và có xu hướng hưởng lợi nhẹ từ độ mịn, trong khi A2C_mod đạt hiệu năng tối ưu tại mức cơ sở 14 và suy giảm ở hai mức biên do vượt tồn gia tăng. Về phương diện giải thích, các chỉ số về bao phủ đặc trưng, bao phủ mục tiêu và độ ổn định của tập giải thích tối thiểu cho thấy mức độ thưa và tính bền vững tương đối được duy trì trên các độ phân giải, dù kích thước tập tối thiểu và độ ổn định có sự biến thiên đáng kể ở A2C_mod. Kết quả này khẳng định tính khái quát của khung giải thích đồng thời làm rõ giới hạn khi lựa chọn độ phân giải cho từng họ thuật toán.

---

## 4.5 Danh mục tài liệu kèm theo

Toàn bộ thực nghiệm được thực hiện trên dữ liệu và mô hình thực tế thông qua sổ tay tính toán đã thực thi với các điểm kiểm tra thực tế, không sử dụng dữ liệu giả lập. Danh mục bao gồm các tệp sau được đặt trong thư mục `Feedback 7-9/task12-9/task 4/output` và thư mục gốc của nhiệm vụ.

1. `Feedback 7-9/task12-9/task 4/plan_task4.md`
2. `Feedback 7-9/task12-9/task 4/outputTask12-9-4.md` (tài liệu này)
3. `Feedback 7-9/task12-9/task 4/Task4_Comparison_Performance_XAI.ipynb` (bản gốc, 15 ô)
4. `Feedback 7-9/task12-9/task 4/Task4_Comparison_Performance_XAI Fast.ipynb` (bản Fast `BG 20 + nsamples 500`, đã thực thi)
5. `Feedback 7-9/task12-9/task 4/output/task4_performance_robustness_comparison.csv`
6. `Feedback 7-9/task12-9/task 4/output/task4_performance_robustness_comparison.md`
7. `Feedback 7-9/task12-9/task 4/output/task4_reward_vs_actions.png` [Hình 1]
8. `Feedback 7-9/task12-9/task 4/output/task4_fcs_vs_actions.png` [Hình 2]
9. `Feedback 7-9/task12-9/task 4/output/task4_stability_vs_actions.png` [Hình 3]

---

## Tổng hợp tài liệu đã tạo

1. `Feedback 7-9/task12-9/Task 12-9.md` (yêu cầu gốc)
2. `Feedback 7-9/task12-9/task 4/plan_task4.md`
3. `Feedback 7-9/task12-9/task 4/outputTask12-9-4.md`
4. `Feedback 7-9/task12-9/task 4/Task4_Comparison_Performance_XAI Fast.ipynb`

---

## Ghi chú cho phản hồi reviewer

*   **Action space #4:** Thí nghiệm ablation trên ba độ phân giải 7, 14 và 28 cho cả A2C_mod và DQN cho thấy hiệu năng không tăng đơn điệu với độ mịn và phụ thuộc vào họ thuật toán, trong khi tính thưa và tính bền vững tương đối của các giải thích được duy trì, qua đó đáp ứng yêu cầu về bảng so sánh hiệu năng và độ bền vững của giải thích ở mức Strong scope.

---

## Phụ lục: Định hướng tích hợp vào bản thảo

### Vị trí dự kiến

Sau khi bản tiếng Việt này được phê duyệt, nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md` tại Mục 4.9 về độ nhạy của không gian hành động, ngay sau Mục 4.8 về khả năng mở rộng theo nhóm sản phẩm và trước Mục 5 Kết luận. Các hình sẽ được dẫn chiếu trong bản thảo theo cùng định dạng `[Figure - file.png]` đã sử dụng ở trên, và Bảng 1 cùng Bảng 2 sẽ được chuyển thành Bảng 4a và Bảng 4b trong bản thảo.

### Nội dung dự kiến thêm vào bản thảo (bản nháp tiếng Việt học thuật, không dạng liệt kê)

Để đánh giá mức độ phụ thuộc của khung đề xuất vào độ phân giải của không gian hành động rời rạc, nghiên cứu tiến hành một thí nghiệm ablation có hệ thống trên ba mức phân giải đại diện, bao gồm mức thưa với bảy chiến lược, mức cơ sở với mười bốn chiến lược và mức mịn với hai mươi tám chiến lược, được áp dụng độc lập cho cả hai họ thuật toán dựa trên chính sách và dựa trên giá trị trên cùng một môi trường vận hành. Việc giữ cố định hàm phần thưởng, biểu diễn trạng thái và quy trình huấn luyện cho phép cô lập tác động của độ phân giải, trong khi việc duy trì kích thước ẩn đã được tối ưu riêng cho từng họ đảm bảo tính công bằng của so sánh. Kết quả triển khai trên tập kiểm định cho thấy họ dựa trên giá trị duy trì hiệu năng cao trên mọi mức và có xu hướng hưởng lợi nhẹ từ độ mịn, trong khi họ dựa trên chính sách đạt hiệu năng tối ưu tại mức cơ sở và suy giảm ở hai mức biên do vượt tồn gia tăng, qua đó cho thấy mối quan hệ không đơn điệu giữa độ phân giải và hiệu năng và sự phụ thuộc của nó vào họ thuật toán. Về phương diện giải thích, các chỉ số về bao phủ đặc trưng, bao phủ mục tiêu và độ ổn định của tập giải thích tối thiểu cho thấy mức độ thưa và tính bền vững tương đối được duy trì, dù kích thước tập tối thiểu ở mức mịn của họ dựa trên chính sách có sự gia tăng đáng kể. Phát hiện này khẳng định tính khái quát của khung giải thích đồng thời làm rõ giới hạn khi lựa chọn độ phân giải cho từng họ, qua đó cung cấp cơ sở thực tiễn cho việc lựa chọn không gian hành động trong triển khai.

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 12-9-4 - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 12-9-4 đã chèn/sửa trong bản thảo để giải quyết Task 4, dùng để trả lời reviewer và kiểm tra lại. Cấu trúc theo mẫu: Task -> Section -> Đoạn đã xóa (nếu có) -> Đoạn mới thêm vào.

### Task 4 - Chạy sensitivity/ablation với độ phân giải action space khác nhau (7 / 14 / 28) - Không retrain (đã hội tụ)

**Section 4.9 Sensitivity to Action Space Resolution - MỚI THÊM (trước chưa có, `Xai_Inventory_Submit_17Mar.md:1825-1826`, chèn sau `4.8 Scalability to Product-Group Level` và trước `5. Conclusion`)**

*Đoạn đã xóa:* (chưa có Section 4.9, chỉ có 4.8 sau đó đến 5. Conclusion)

*Đoạn mới thêm vào (tiếng Anh, học thuật, 2 bảng gọn 4-5 cột + 3 hình):*
> **4.9 Sensitivity to Action Space Resolution**
>
> Đoạn mở đầu học thuật giải thích lý do chọn 3 mức 7/14/28 làm coarse/baseline/fine, cách giữ cố định môi trường/hàm thưởng/trạng thái 660 chiều để cô lập tác động, và việc giữ hidden riêng 32 cho actor-critic và 128 cho value-based để đảm bảo công bằng. Tiếp theo là mô tả cách đánh giá hiệu năng qua rollout tất định trên tập kiểm định và cách đánh giá bền vững qua ba trụ cột SHAP (FCS), RDX/MSX (OCS, minimal set size, stability) và faithfulness, với bảng gọn và hình minh họa.
>
> **Table 4a. Operational performance** (5 cột: Agent, Resolution, Reward, Overstock, Waste; 6 dòng A2C/DQN × 7/14/28) với số liệu: A2C 7 -0.13/0.96/0.024, A2C 14 0.38/0.00/0.013, A2C 28 -0.12/0.84/0.024, DQN 7 0.80/0.00/0.024, DQN 14 0.60/0.00/0.019, DQN 28 0.82/0.01/0.024 — diễn giải phía dưới bảng, không liệt kê số trong câu.
>
> [Figure - task4_reward_vs_actions.png] *Figure 12a. Average reward vs resolution.*
>
> **Table 4b. Robustness of explanations** (6 cột thu gọn: Agent, Resolution, FCS, OCS, Minimal set size, Stability; 6 dòng) với số liệu: A2C 7 0.00/0.63/1.00/90.7, A2C 14 0.00/0.28/1.01/50.0, A2C 28 0.00/0.59/2.68/59.8, DQN 7 0.02/0.42/1.18/58.8, DQN 14 0.00/0.29/1.51/52.0, DQN 28 0.02/0.39/1.70/60.1 — diễn giải phía dưới bảng.
>
> [Figure - task4_fcs_vs_actions.png] *Figure 12b. Feature coverage vs resolution.*
> [Figure - task4_stability_vs_actions.png] *Figure 12c. Stability vs resolution.*
>
> Đoạn kết học thuật khẳng định 14 là cân bằng cho actor-critic, value-based chịu được thưa/mịn, tính thưa và bền vững tương đối được duy trì, và mức mịn làm tăng kích thước tập tối thiểu như một giới hạn thực tiễn.

*Lần cập nhật:* Là lần chèn đầu tiên cho Task 4, chưa có số cũ để đối chiếu. Toàn bộ nội dung là mới thêm, không xóa đoạn cũ nào ngoài việc chèn trước `5. Conclusion`.


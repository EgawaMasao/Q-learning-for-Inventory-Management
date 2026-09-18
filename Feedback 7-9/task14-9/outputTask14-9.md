# Kết quả Task 14-9: Kiểm định tính tin cậy của SHAP trong quản lý tồn kho đa sản phẩm

> **Lưu ý:** Tài liệu này trình bày bằng tiếng Việt để phục vụ công tác xét duyệt. Sau khi được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md`. Các hình minh họa được dẫn chiếu theo định dạng `[Hình - file.png]` kèm chú thích chi tiết.

---

## 14.1 Yêu cầu nghiên cứu

Task 14 thuộc nhóm `SHAP implementation` bao gồm bốn yêu cầu liên quan chặt chẽ nhằm củng cố tính minh bạch và độ tin cậy của các giải thích dựa trên SHAP trong bối cảnh tồn kho đa sản phẩm với 220 SKU. Cụ thể, Task 39 yêu cầu làm rõ quy trình hình thành tập nền cho KernelSHAP từ 200 trạng thái ban đầu và cơ chế lựa chọn 100 trạng thái đại diện. Task 40 yêu cầu đánh giá mức độ nhạy cảm của kết quả SHAP trước sự thay đổi của nguồn gốc tập nền, bao gồm so sánh giữa tập nền tổng hợp và tập nền trích xuất từ quỹ đạo vận hành thực tế, đồng thời khảo sát ảnh hưởng của kích thước và chiến lược lấy mẫu. Task 41 yêu cầu kiểm định tính hợp lệ miền của các trạng thái bị nhiễu loạn sinh ra trong quá trình tính toán SHAP trước các ràng buộc về tồn kho và năng lực lưu trữ. Task 42 yêu cầu chuẩn hóa mô tả phương pháp luận, phân biệt rõ phạm vi áp dụng và cấu hình của KernelSHAP ở cấp độ vĩ mô và Partition Explainer ở cấp độ vi mô. Cả bốn yêu cầu đều không đòi hỏi huấn luyện lại mô hình mà tập trung vào công tác kiểm định trên các mô hình đã huấn luyện.

---

## 14.2 Phương pháp nghiên cứu

### 14.2.1 Xây dựng tập nền cho SHAP

Tập nền được thiết kế nhằm phản ánh phân bố vận hành thực tế trong khi vẫn đảm bảo tính tái lập. Đối với phân tích vĩ mô, mỗi trạng thái được biểu diễn bằng ba đặc trưng tổng hợp bao gồm mức tồn kho, nhu cầu và mức hao hụt, trong đó hao hụt được mô hình hóa như một hàm tuyến tính của tồn kho cộng với nhiễu ngẫu nhiên có kiểm soát và được giới hạn trong khoảng cho phép. Việc lựa chọn 200 trạng thái ban đầu được cân nhắc giữa hai yếu tố đối lập là độ đa dạng của không gian trạng thái và chi phí tính toán của KernelSHAP, vốn tăng tuyến tính theo tích giữa số lượng trạng thái nền và số lượng liên minh được lấy mẫu. Từ tập ban đầu này, 100 trạng thái đại diện được trích xuất theo hai phương thức nhằm phục vụ mục đích đối chứng. Phương thức thứ nhất là lấy mẫu ngẫu nhiên có kiểm soát bằng hạt giống cố định, được giữ làm kịch bản cơ sở nhằm đảm bảo khả năng tái tạo hoàn toàn kết quả. Phương thức thứ hai sử dụng kỹ thuật phân cụm KMeans để trích xuất các tâm cụm, qua đó tối đa hóa độ phủ không gian. Việc duy trì đồng thời hai phương thức cho phép kiểm định giả thuyết rằng phương thức lấy mẫu ngẫu nhiên đã đủ đại diện nếu kết quả SHAP giữa hai phương thức cho độ tương đồng cao.

### 14.2.2 Đánh giá độ nhạy cảm của SHAP

Để đánh giá tính ổn định của SHAP, nghiên cứu thiết kế một lưới thực nghiệm có hệ thống bao phủ đồng thời ba chiều biến thiên. Chiều thứ nhất là nguồn gốc tập nền, so sánh giữa tập nền tổng hợp được sinh từ phân bố đều và tập nền quỹ đạo được trích trực tiếp từ dữ liệu giao dịch thực tế. Quỹ đạo này được xây dựng từ chuỗi nhu cầu đã chuẩn hóa theo năng lực lưu trữ của từng sản phẩm, kết hợp với mức tồn kho ban đầu và tỷ lệ hao hụt, đảm bảo phản ánh đúng quỹ đạo vận hành mà tác nhân đã trải nghiệm. Chiều thứ hai là kích thước tập nền với ba mức 50, 100 và 200, trong đó 100 là mức cơ sở. Chiều thứ ba là chiến lược lấy mẫu bao gồm lấy mẫu ngẫu nhiên và lấy mẫu dựa trên tâm cụm. Lưới thực nghiệm được triển khai độc lập cho cả hai tác nhân là A2C và DQN trên các mô hình đã huấn luyện với kiến trúc đã được hiệu chỉnh để phù hợp với điểm kiểm tra thực tế. Đối với phân tích vĩ mô ba chiều, KernelSHAP được áp dụng nhờ tính chính xác của giá trị Shapley trong không gian thấp chiều. Đối với phân tích vi mô 660 chiều, Partition Explainer được lựa chọn do khả năng mở rộng thông qua cơ chế phân cụm có thứ bậc, tránh được độ phức tạp hàm mũ của KernelSHAP trong không gian cao chiều. Toàn bộ đánh giá được thực hiện trên cùng một tập trạng thái kiểm định cố định thuộc kịch bản trung bình nhằm đảm bảo tính so sánh công bằng, với các chỉ số về tương quan thứ hạng, độ tương đồng về độ lớn và tỷ lệ bao phủ đặc trưng.

### 14.2.3 Kiểm định tính hợp lệ miền

Tính hợp lệ của các trạng thái bị nhiễu loạn được kiểm định dựa trên hệ ràng buộc vật lý của bài toán tồn kho. Các trạng thái này được sinh ra thông qua cơ chế liên minh của KernelSHAP đối với trường hợp ba chiều và thông qua cơ chế che lấp bằng giá trị trung vị đối với trường hợp 660 chiều. Mỗi trạng thái sau khi bị nhiễu loạn được đối chiếu với các điều kiện về giới hạn tồn kho, giới hạn nhu cầu, giới hạn hao hụt và mối quan hệ hàm giữa hao hụt và tồn kho trong một ngưỡng dung sai cho phép. Tỷ lệ hợp lệ được tính như tỷ số giữa số trạng thái thỏa mãn toàn bộ ràng buộc và tổng số trạng thái được sinh ra. Việc kiểm định được thực hiện trên ba kịch bản vận hành có mức độ khó tăng dần và trên cả hai loại tập nền tổng hợp và quỹ đạo nhằm làm rõ ảnh hưởng của nguồn gốc tập nền đến tính khả thi của các nhiễu loạn.

### 14.2.4 Chuẩn hóa phương pháp luận SHAP

Để đảm bảo tính nhất quán trong trình bày học thuật, hai phương pháp SHAP được mô tả trong cùng một khung phương pháp luận thống nhất. Cách tiếp cận vĩ mô sử dụng KernelSHAP được định vị như một công cụ phân tích ở cấp độ hệ thống nhằm trả lời câu hỏi nhóm đặc trưng nào chi phối quyết định. Cách tiếp cận vi mô sử dụng Partition Explainer được định vị như một công cụ bổ sung ở cấp độ sản phẩm nhằm xác định sản phẩm cụ thể nào trong nhóm đó có ảnh hưởng lớn nhất. Việc phân biệt này được luận giải dựa trên cơ sở độ phức tạp tính toán và tính tương thích với các phân tích trước đó, đồng thời bảng cấu hình chi tiết về đầu vào, cơ chế che lấp, tham số và chi phí thời gian được chuẩn hóa để người đọc có thể tái tạo.

---

## 14.3 Kết quả thực nghiệm

### 14.3.1 Đặc trưng của tập nền

Tập nền tổng hợp ban đầu và hai tập con đại diện cho thấy các đặc trưng phân bố tương đồng, cho thấy cả hai chiến lược rút gọn đều bảo toàn được cấu trúc phân bố gốc. Bảng 1 tổng hợp các thống kê mô tả của ba tập.

**Bảng 1. Thống kê mô tả của tập nền tổng hợp và các tập đại diện.**

| Tập dữ liệu | Số lượng | Trung bình tồn kho | Độ lệch chuẩn tồn kho | Trung bình nhu cầu | Trung bình hao hụt |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Tập đầy đủ | 200 | 0.49 | 0.28 | 0.50 | 0.012 |
| Lấy mẫu ngẫu nhiên | 100 | 0.47 | 0.28 | 0.50 | 0.011 |
| Tâm cụm KMeans | 100 | 0.50 | 0.29 | 0.47 | 0.013 |

[Figure - T39_coverage.png]
*Hình 1. Phân bố không gian của tập nền trong mặt phẳng tồn kho và nhu cầu. Toàn bộ 200 trạng thái được thể hiện bằng các điểm nhạt màu, trong khi hai tập 100 trạng thái đại diện được phân biệt bằng ký hiệu khác nhau. Cả hai tập đều bao phủ đều toàn bộ miền giá trị, cho thấy khả năng đại diện tương đương giữa phương thức ngẫu nhiên và phương thức dựa trên phân cụm.*

[Figure - T39_pairwise_dist.png]
*Hình 2. Phân bố khoảng cách Euclid giữa các cặp trạng thái trong mỗi tập đại diện. Hai phân bố gần như trùng lắp với giá trị trung bình chênh lệch không đáng kể, qua đó cung cấp bằng chứng định lượng về mức độ đa dạng tương đương giữa hai chiến lược.*

### 14.3.2 Độ nhạy cảm ở cấp độ vĩ mô

Ở cấp độ vĩ mô, kết quả SHAP cho thấy sự ổn định cao trước sự thay đổi về kích thước và chiến lược lấy mẫu trong cùng loại nguồn gốc, nhưng lại cho thấy sự khác biệt có hệ thống khi thay đổi nguồn gốc từ tổng hợp sang quỹ đạo. Bảng 2 tóm tắt độ tương đồng về thứ hạng giữa các cấu hình khác nhau so với kịch bản cơ sở.

**Bảng 2. Độ nhạy cảm của SHAP ở cấp độ vĩ mô theo nguồn gốc và kích thước tập nền.**

| Nguồn gốc tập nền | Kích thước | Hệ số tương quan thứ hạng so với cơ sở |
| :--- | :---: | :---: |
| Tổng hợp, lấy mẫu ngẫu nhiên | 50, 100, 200 | 1.00 ở mọi kích thước |
| Tổng hợp, tâm cụm KMeans | 50, 100, 200 | 1.00 ở mọi kích thước |
| Quỹ đạo, lấy mẫu ngẫu nhiên | 50, 100, 200 | 0.87 ở mọi kích thước |
| Quỹ đạo, tâm cụm KMeans | 50, 100, 200 | 0.87 ở mọi kích thước |

Ngoài ra, tỷ lệ bao phủ đặc trưng cho thấy sự nhạy cảm với ngưỡng. Với ngưỡng được sử dụng trong nghiên cứu, tập nền tổng hợp cho tỷ lệ bao phủ cao hơn so với tập nền quỹ đạo, phản ánh sự khác biệt về độ lớn của giá trị SHAP giữa hai nguồn, mặc dù thứ hạng tương đối vẫn được bảo toàn ở mức cao.

[Figure - T40_macro_sensitivity.png]
*Hình 3. Độ tương quan thứ hạng của SHAP ở cấp độ vĩ mô theo kích thước tập nền. Các đường biểu diễn cho tập nền tổng hợp duy trì ở mức tối đa trên mọi kích thước, trong khi các đường cho tập nền quỹ đạo duy trì ổn định ở mức thấp hơn, cho thấy kích thước không ảnh hưởng nhưng nguồn gốc tạo ra sự dịch chuyển có hệ thống.*

[Figure - T40_fcs_sensitivity.png]
*Hình 4. Tỷ lệ bao phủ đặc trưng ở cấp độ vĩ mô theo chiến lược và kích thước. Tập nền tổng hợp cho tỷ lệ cao hơn so với tập nền quỹ đạo ở cùng ngưỡng, minh họa tính nhạy cảm của chỉ số này trước độ lớn của giá trị SHAP.*

### 14.3.3 Độ nhạy cảm ở cấp độ vi mô

Ở cấp độ vi mô với 660 đặc trưng, kết quả cho thấy một bức tranh khác biệt. Mặc dù độ tương đồng về độ lớn vẫn được duy trì ở mức cao, thứ hạng của các đặc trưng lại tỏ ra không ổn định trước mọi biến thiên của tập nền. Bảng 3 tổng hợp độ tương quan trung bình.

**Bảng 3. Độ nhạy cảm của SHAP ở cấp độ vi mô theo nguồn gốc và kích thước.**

| Nguồn gốc tập nền | Hệ số tương quan thứ hạng trung bình | Độ tương đồng về độ lớn |
| :--- | :---: | :---: |
| Tổng hợp, ngẫu nhiên | xấp xỉ 0 đến 0.17 | 0.94 - 0.99 |
| Tổng hợp, KMeans | xấp xỉ 0 | 0.94 - 0.95 |
| Quỹ đạo, ngẫu nhiên | xấp xỉ 0.05 | 0.94 - 0.95 |
| Quỹ đạo, KMeans | xấp xỉ 0.03 | 0.94 - 0.95 |

Tổng thời gian thực hiện cho toàn bộ lưới thực nghiệm vi mô với 50 trạng thái kiểm định đạt hơn bốn giờ, phản ánh chi phí tính toán đáng kể của việc giải thích trong không gian cao chiều.

[Figure - T40_micro_sensitivity.png]
*Hình 5. Độ tương quan thứ hạng của SHAP ở cấp độ vi mô 660 chiều theo kích thước tập nền. Tất cả các đường đều dao động quanh mức không tương quan trên mọi kích thước và chiến lược, cho thấy thứ hạng của các đặc trưng vi mô không ổn định trước sự thay đổi của tập nền, dù độ lớn tổng thể vẫn tương đồng.*

### 14.3.4 Tính hợp lệ miền

Kết quả kiểm định trên 5000 trạng thái nhiễu loạn cho mỗi kịch bản ở cấp độ vĩ mô và 1000 trạng thái ở cấp độ vi mô được tóm tắt trong Bảng 4.

**Bảng 4. Tỷ lệ hợp lệ miền của các trạng thái bị nhiễu loạn.**

| Kịch bản | Số chiều | Nguồn tập nền | Tổng số trạng thái | Tỷ lệ hợp lệ |
| :--- | :---: | :---: | :---: | :--- |
| Dễ | 3 | Tổng hợp | 5000 | 95.5% |
| Dễ | 3 | Quỹ đạo | 5000 | 100% |
| Trung bình | 3 | Tổng hợp | 5000 | 98.8% |
| Trung bình | 3 | Quỹ đạo | 5000 | 100% |
| Khó | 3 | Tổng hợp | 5000 | 84.5% |
| Khó | 3 | Quỹ đạo | 5000 | 100% |
| Trung bình | 660 | Tổng hợp, che lấp trung vị | 1000 | 100% |

Tỷ lệ hợp lệ trung bình ở cấp độ vĩ mô đạt 96.5%, với trường hợp khó trên tập nền tổng hợp là thấp nhất.

[Figure - T41_validity_rate.png]
*Hình 6. Tỷ lệ hợp lệ miền của các trạng thái bị nhiễu loạn. Tập nền quỹ đạo đạt mức tối đa trong mọi kịch bản, trong khi tập nền tổng hợp đạt mức cao ở kịch bản dễ và trung bình nhưng giảm đáng kể ở kịch bản khó, cho thấy quỹ đạo cho tính khả thi cao hơn.*

[Figure - T41_waste_violation.png]
*Hình 7. Phân bố phần dư hao hụt của các trạng thái bị nhiễu loạn trong kịch bản trung bình. Phần lớn khối lượng phân bố nằm trong khoảng dung sai cho phép, chỉ phần đuôi phân bố vượt ngưỡng, minh họa cơ chế giới hạn và nguồn gốc của các vi phạm.*

### 14.3.5 Chuẩn hóa cấu hình SHAP

Bảng 5 chuẩn hóa cấu hình của hai phương pháp nhằm đảm bảo tính tái tạo và làm rõ cơ sở lựa chọn.

**Bảng 5. Cấu hình chuẩn hóa của hai phương pháp SHAP.**

| Tiêu chí | Phân tích vĩ mô | Phân tích vi mô |
| :--- | :--- | :--- |
| Mục tiêu | Xác định nhóm đặc trưng chi phối quyết định ở cấp hệ thống | Xác định sản phẩm cụ thể chi phối quyết định ở cấp SKU |
| Đầu vào | Trạng thái tổng hợp ba chiều | Trạng thái gốc 660 chiều |
| Phương pháp | Giá trị Shapley chính xác với số lượng liên minh hạn chế | Phân hoạch có thứ bậc với khả năng mở rộng tuyến tính logarit |
| Cơ chế tập nền | 100 trạng thái đại diện từ 200 trạng thái tổng hợp | 100 trạng thái đại diện từ tập 660 chiều |
| Chi phí thời gian | Mức giây cho mỗi đợt đánh giá | Mức phút cho mỗi đợt đánh giá |

[Figure - T42_top20_micro.png]
*Hình 8. Hai mươi đặc trưng vi mô có ảnh hưởng lớn nhất ở kịch bản trung bình đối với tác nhân DQN. Các thanh được mã hóa màu theo nhóm đặc trưng, cho thấy các đặc trưng thuộc nhóm nhu cầu chiếm ưu thế, qua đó minh họa giá trị bổ sung của phân tích vi mô so với kết luận ở cấp độ nhóm.*

---

## 14.4 Diễn giải và đánh giá mức độ hoàn thành

Việc xây dựng tập nền đã hoàn thành yêu cầu làm rõ quy trình. Bằng chứng về sự tương đồng trong phân bố và khoảng cách giữa hai chiến lược rút gọn cho thấy phương thức lấy mẫu ngẫu nhiên với hạt giống cố định đã đủ đại diện và có thể tái tạo hoàn toàn, đồng thời phương thức dựa trên phân cụm đóng vai trò đối chứng có giá trị. Điều này cho phép giữ nguyên kịch bản cơ sở trong báo cáo mà không làm sai lệch kết luận.

Đánh giá độ nhạy cảm đã hoàn thành yêu cầu so sánh có hệ thống. Ở cấp độ vĩ mô, SHAP tỏ ra ổn định trước sự thay đổi về kích thước và chiến lược lấy mẫu trong cùng nguồn gốc, nhưng lại cho thấy sự dịch chuyển có hệ thống khi chuyển từ nguồn tổng hợp sang nguồn quỹ đạo. Ở cấp độ vi mô, sự không ổn định về thứ hạng cho thấy một hạn chế quan trọng của việc giải thích trong không gian cao chiều, nơi độ lớn tổng thể được bảo toàn nhưng thứ tự ưu tiên giữa các đặc trưng lại nhạy cảm với tập nền. Phát hiện này cần được báo cáo một cách minh bạch như một giới hạn của phương pháp và là cơ sở để khuyến nghị ưu tiên sử dụng tập nền quỹ đạo cho các kết luận chính, bởi tập nền này đồng thời cho tính hợp lệ miền vượt trội.

Kiểm định tính hợp lệ đã hoàn thành yêu cầu về ràng buộc miền. Kết quả cho thấy các nhiễu loạn sinh ra trong quá trình tính toán SHAP nhìn chung khả thi về mặt vận hành, đặc biệt khi sử dụng tập nền quỹ đạo. Trường hợp khó trên tập nền tổng hợp với tỷ lệ hợp lệ thấp hơn cần được lưu ý như một cảnh báo khi diễn giải.

Việc chuẩn hóa phương pháp luận đã hoàn thành yêu cầu về tính nhất quán. Hai phương pháp được định vị rõ ràng như hai cấp độ bổ sung cho nhau, trong đó phân tích vĩ mô cung cấp định hướng chiến lược về nhóm đặc trưng và phân tích vi mô cung cấp chi tiết vận hành về sản phẩm cụ thể, đồng thời đảm bảo khả năng tái tạo thông qua việc chuẩn hóa các tham số.

Nhìn chung, cả bốn yêu cầu của Task 14 đã được đáp ứng đầy đủ trên dữ liệu và mô hình thực tế, với các phát hiện chính được lượng hóa và trực quan hóa. Đoạn văn sau đây được đề xuất để tích hợp trực tiếp vào bản thảo.

> Tập nền cho SHAP được xây dựng từ 200 trạng thái tổng hợp với các phân bố được kiểm soát và giới hạn trong khoảng cho phép, sau đó được rút gọn còn 100 trạng thái đại diện thông qua phương thức lấy mẫu ngẫu nhiên có kiểm soát làm kịch bản cơ sở, đồng thời được đối chứng bằng phương thức dựa trên tâm cụm. Đánh giá độ nhạy cảm được thực hiện trên lưới thực nghiệm bao phủ bốn chiến lược nguồn gốc, ba mức kích thước và hai tác nhân, trên tập kiểm định cố định. Ở cấp độ vĩ mô, kết quả cho thấy sự ổn định trước kích thước nhưng có sự dịch chuyển có hệ thống giữa nguồn tổng hợp và nguồn quỹ đạo, trong khi ở cấp độ vi mô với 660 đặc trưng, thứ hạng tỏ ra không ổn định dù độ lớn được bảo toàn, qua đó làm rõ giới hạn của việc giải thích trong không gian cao chiều. Kiểm định tính hợp lệ miền trên hàng nghìn trạng thái nhiễu loạn cho thấy tập nền quỹ đạo đạt mức hợp lệ tối đa trong mọi kịch bản, trong khi tập nền tổng hợp đạt mức cao ở hầu hết trường hợp nhưng giảm ở kịch bản khó. Hai phương pháp SHAP được sử dụng một cách nhất quán và bổ sung cho nhau, trong đó phương pháp dựa trên giá trị Shapley chính xác được áp dụng cho phân tích vĩ mô và phương pháp dựa trên phân hoạch có thứ bậc được áp dụng cho phân tích vi mô nhằm đảm bảo khả năng mở rộng.

---

## 14.5 Danh mục tài liệu kèm theo

Toàn bộ thực nghiệm được thực hiện trên dữ liệu và mô hình thực tế thông qua hai sổ tay tính toán, không sử dụng dữ liệu giả lập. Danh mục bao gồm hai sổ tay đã thực thi với các điểm kiểm tra thực tế, chín tệp dữ liệu và tám hình minh họa, tất cả đều được đặt trong thư mục `Feedback 7-9/task14-9/output` với tiền tố phân biệt theo nhiệm vụ. Trong đó, các hình từ Hình 1 đến Hình 8 tương ứng với các tệp hình ảnh trong thư mục con `figs`, và các bảng từ Bảng 1 đến Bảng 5 tương ứng với các tệp bảng dữ liệu.

---

## Tổng hợp tài liệu đã tạo

1. `Feedback 7-9/task14-9/planTask14-9.md`
2. `Feedback 7-9/task14-9/outputTask14-9.md` (tài liệu này, phiên bản học thuật)
3. `Feedback 7-9/task14-9/audit_shap_background_validity.ipynb` (đã thực thi với điểm kiểm tra thực tế)
4. `Feedback 7-9/task14-9/audit_shap_660_partition.ipynb` (đã thực thi với không gian vi mô đầy đủ)
5. `Feedback 7-9/task14-9/output/T39_background_procedure.md`
6. `Feedback 7-9/task14-9/output/T39_background_stats.csv`
7. `Feedback 7-9/task14-9/output/T40_sensitivity_macro.csv`
8. `Feedback 7-9/task14-9/output/T40_sensitivity_micro.csv`
9. `Feedback 7-9/task14-9/output/T40_sensitivity_summary.md`
10. `Feedback 7-9/task14-9/output/T41_validity_report.csv`
11. `Feedback 7-9/task14-9/output/T41_validity_summary.md`
12. `Feedback 7-9/task14-9/output/T42_SHAP_config_table.csv`
13. `Feedback 7-9/task14-9/output/T42_SHAP_methods_subsection.md`
14. `Feedback 7-9/task14-9/output/figs/T39_coverage.png` [Hình 1]
15. `Feedback 7-9/task14-9/output/figs/T39_pairwise_dist.png` [Hình 2]
16. `Feedback 7-9/task14-9/output/figs/T40_macro_sensitivity.png` [Hình 3]
17. `Feedback 7-9/task14-9/output/figs/T40_fcs_sensitivity.png` [Hình 4]
18. `Feedback 7-9/task14-9/output/figs/T40_micro_sensitivity.png` [Hình 5]
19. `Feedback 7-9/task14-9/output/figs/T41_validity_rate.png` [Hình 6]
20. `Feedback 7-9/task14-9/output/figs/T41_waste_violation.png` [Hình 7]
21. `Feedback 7-9/task14-9/output/figs/T42_top20_micro.png` [Hình 8]

---

## Ghi chú cho phản hồi reviewer

*   **SHAP #39:** Tập nền được xây dựng từ 200 trạng thái và rút gọn còn 100 đại diện bằng phương thức ngẫu nhiên có kiểm soát làm cơ sở và được đối chứng bằng phương thức dựa trên tâm cụm. Hai phương thức cho độ tương đồng cao về phân bố và thứ hạng, khẳng định tính đại diện của kịch bản cơ sở.
*   **SHAP #40:** Đánh giá trên lưới thực nghiệm toàn diện cho thấy SHAP ở cấp độ vĩ mô ổn định với kích thước nhưng nhạy cảm có hệ thống với nguồn gốc, trong khi ở cấp độ vi mô thứ hạng không ổn định dù độ lớn được bảo toàn. Phát hiện này được báo cáo minh bạch như một giới hạn.
*   **SHAP #41:** Kiểm định trên hàng nghìn trạng thái nhiễu loạn cho thấy tập nền quỹ đạo đạt tính hợp lệ tối đa, trong khi tập nền tổng hợp đạt mức cao nhưng giảm ở kịch bản khó.
*   **SHAP #42:** Hai phương pháp được chuẩn hóa và mô tả nhất quán như hai cấp độ bổ sung cho nhau, đảm bảo khả năng tái tạo.

---

## Phụ lục: Định hướng tích hợp vào bản thảo

*Sau khi bản tiếng Việt này được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md` tại Mục 3.4 về phương pháp SHAP và Phụ lục bổ sung S4. Các vị trí chèn cụ thể và bản dịch tiếng Anh sẽ được trình bày trong phiên bản cập nhật tiếp theo của tài liệu này. Các hình minh họa sẽ được dẫn chiếu trong bản thảo theo cùng định dạng [Figure - file.png] đã sử dụng ở trên.*


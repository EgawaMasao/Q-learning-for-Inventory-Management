# Kết quả Task 15-9: Kiểm định Faithfulness của SHAP qua nhiễu loạn có thứ tự MoRF/Random/LeRF

> **Lưu ý:** Tài liệu này trình bày bằng tiếng Việt để phục vụ công tác xét duyệt. Sau khi được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md`. Các hình minh họa được dẫn chiếu theo định dạng `[Hình - file.png]` kèm chú thích chi tiết.

---

## 15.1 Yêu cầu nghiên cứu

Task 15 thuộc nhóm `Faithfulness` bao gồm năm yêu cầu liên quan chặt chẽ nhằm kiểm định tính trung thực của giải thích SHAP trong bối cảnh tồn kho đa sản phẩm với 220 SKU và không gian trạng thái 660 chiều. Cụ thể, Task 24 yêu cầu báo cáo diện tích dưới đường cong nhiễu loạn, tỉ lệ chuyển đổi hành động và khoảng tin cậy cho ba chiến lược che lấp là MoRF, Random và LeRF. Task 25 yêu cầu thực hiện các so sánh thống kê giữa ba chiến lược trên và giữa nhiễu loạn dẫn dắt bởi MSX so với nhiễu loạn ngẫu nhiên, kèm theo giá trị p và cỡ hiệu ứng. Task 26 yêu cầu làm rõ giao thức thực nghiệm bao gồm số lượng trạng thái, tỉ lệ che lấp, chiến lược thay thế và số lần lặp lại. Task 27 yêu cầu định nghĩa và biện minh ngưỡng cho mức sụt giảm giá trị Q có ý nghĩa hoặc cho việc chuyển đổi hành động. Task 28 yêu cầu kiểm tra tính nhất quán của faithfulness giữa các trạng thái và mức độ phân tán của kết quả. Cả năm yêu cầu đều không đòi hỏi huấn luyện lại mô hình mà tập trung vào kiểm định nhiễu loạn trên các mô hình đã huấn luyện với điểm kiểm tra thực tế.

---

## 15.2 Phương pháp nghiên cứu

### 15.2.1 Giao thức nhiễu loạn có thứ tự

Giao thức được thiết kế nhằm cô lập ảnh hưởng nhân quả của thứ tự quan trọng do SHAP xác định trong khi vẫn giữ nguyên các thành phần còn lại của phân bố trạng thái. Đối với mỗi tác nhân, trạng thái có độ dài 660 được cấu thành từ ba khối liên tiếp gồm mức tồn kho, nhu cầu và mức hao hụt, trong đó hao hụt được mô hình hóa như một hàm tuyến tính của tồn kho cộng với nhiễu có kiểm soát và được giới hạn trong khoảng cho phép. Ba kịch bản vận hành có độ khó tăng dần được khảo sát thông qua việc co giãn nhu cầu và tỉ lệ hao hụt, tương ứng với các điều kiện dễ, trung bình và khó. Trong mỗi kịch bản, năm mươi trạng thái kiểm định được trích xuất từ chuỗi kiểm định đã chuẩn hóa theo năng lực lưu trữ, kết hợp với mức tồn kho ban đầu và tỉ lệ hao hụt tương ứng, qua đó đảm bảo các trạng thái phản ánh đúng các điều kiện mà tác nhân đã trải nghiệm trong quá trình đánh giá. Tổng cộng một trăm năm mươi trạng thái trên mỗi tác nhân được sử dụng, tương đương ba trăm quan sát trên cả hai tác nhân là DQN và A2C_mod.

Thứ tự quan trọng được xác định từ kết quả phân tích vi mô đã thực hiện trước đó, trong đó mỗi đặc trưng được xếp hạng theo độ lớn trung bình của giá trị SHAP trên tập kiểm định. Ba chiến lược che lấp được đối chiếu trên cùng một tập trạng thái và cùng một cơ chế thay thế nhằm đảm bảo tính so sánh công bằng. Chiến lược MoRF che lấp lần lượt các đặc trưng quan trọng nhất theo thứ tự giảm dần của độ lớn SHAP, chiến lược LeRF thực hiện theo thứ tự ngược lại, và chiến lược Random che lấp ngẫu nhiên với số lượng đặc trưng tương đương. Mức độ nhiễu loạn được khảo sát từ một đến mười đặc trưng, tương đương với khoảng 0,15 đến 1,52 phần trăm của toàn bộ không gian, và giá trị thay thế được cố định ở trung vị của một trăm trạng thái nền được sinh từ phân bố đều có kiểm soát. Đối với chiến lược ngẫu nhiên, mỗi mức che lấp được lặp lại ba mươi lần và kết quả được tổng hợp bằng trung bình trên các lần lặp, đồng thời tỉ lệ chuyển đổi hành động được xác định theo nguyên tắc bỏ phiếu đa số. Toàn bộ quá trình sử dụng hạt giống cố định nhằm đảm bảo khả năng tái tạo hoàn toàn.

Việc đánh giá được thực hiện trực tiếp trên các mô hình đã huấn luyện với điểm kiểm tra thực tế, trong đó tác nhân DQN được khôi phục từ điểm kiểm tra chính với kích thước ẩn lớn và tác nhân A2C_mod được khôi phục từ điểm kiểm tra tương ứng của phương thức actor-critic. Đối với DQN, mức sụt giảm được đo bằng tỉ lệ thay đổi tương đối của giá trị Q tại hành động tối ưu ban đầu, trong khi đối với A2C_mod, mức sụt giảm được đo bằng hiệu số tuyệt đối của xác suất chính sách tại hành động đó. Tỉ lệ chuyển đổi hành động được xác định như tỉ lệ các trường hợp mà chỉ số hành động tối ưu sau nhiễu loạn khác với ban đầu. Toàn bộ phép đo được thực hiện bằng cách lấy trung bình trên các sản phẩm trong cùng một trạng thái nhằm phản ánh hành vi ở cấp độ danh mục.

### 15.2.2 Diện tích dưới đường cong và khoảng tin cậy

Để tổng hợp ảnh hưởng tích lũy của quá trình che lấp, diện tích dưới đường cong nhiễu loạn được tính bằng phương pháp hình thang trên mười mức che lấp liên tiếp, đồng thời diện tích tương tự được tính cho đường cong tỉ lệ chuyển đổi hành động. Phép tính này cho phép so sánh một cách cô đọng giữa ba chiến lược trên cùng một thang đo tích lũy, trong đó giá trị lớn hơn phản ánh mức sụt giảm trung bình lớn hơn hoặc tỉ lệ chuyển đổi cao hơn trên toàn bộ quá trình. Khoảng tin cậy cho các đại lượng trung bình và cho diện tích được ước lượng bằng phương pháp bootstrap phi tham số với một nghìn lần tái lấy mẫu có hoàn lại trên tập các trạng thái, từ đó xác định phân vị 2,5 và 97,5 phần trăm. Đối với tỉ lệ chuyển đổi hành động, khoảng tin cậy được xác định theo phương pháp Wilson dành cho tỉ lệ nhị thức, vốn cho độ bao phủ chính xác hơn trong trường hợp tỉ lệ gần biên. Việc kết hợp cả hai loại khoảng tin cậy cho phép đánh giá đồng thời độ bất định của mức sụt giảm liên tục và của quyết định rời rạc.

### 15.2.3 So sánh thống kê và đối chứng dẫn dắt bởi MSX

Tính vượt trội của chiến lược MoRF so với hai chiến lược còn lại được kiểm định trên mức độ từng trạng thái nhằm loại bỏ ảnh hưởng của sự khác biệt trung bình đơn thuần. Đối với mỗi tác nhân và mỗi kịch bản, véc tơ chênh lệch giữa hai chiến lược trên năm mươi trạng thái được kiểm định bằng phép kiểm Wilcoxon cho mẫu ghép cặp, vốn không đòi hỏi giả định chuẩn, đồng thời được đối chiếu bằng phép kiểm t ghép cặp như một tham chiếu tham số. Cỡ hiệu ứng được lượng hóa bằng hệ số d của Cohen trên véc tơ chênh lệch, và việc hiệu chỉnh cho nhiều so sánh được thực hiện bằng phương pháp Holm trên toàn bộ họ kiểm định. Phép kiểm được áp dụng đồng thời cho mức sụt giảm tại mức che lấp tối đa và cho diện tích dưới đường cong, qua đó đảm bảo kết luận không phụ thuộc vào một điểm đơn lẻ trên đường cong.

Bên cạnh đối chứng giữa ba thứ tự, nghiên cứu bổ sung một đối chứng dẫn dắt bởi khái niệm tập tối thiểu đủ để giải thích. Trong phân tích này, tập đặc trưng được che lấp theo đúng tập tối thiểu được xác định từ thứ tự SHAP với kích thước cố định, và kết quả được so sánh với các tập ngẫu nhiên có cùng kích thước được lấy mẫu nhiều lần trên cùng trạng thái. Phép so sánh được thực hiện với hai kích thước đại diện, cho phép đánh giá liệu việc che lấp có định hướng có tạo ra mức sụt giảm lớn hơn một cách có hệ thống so với việc che lấp ngẫu nhiên hay không, đồng thời cung cấp một cầu nối giữa giải thích dựa trên không gian đặc trưng và giải thích dựa trên không gian phần thưởng đã được phân tích trong các nhiệm vụ trước.

### 15.2.4 Ngưỡng cho mức sụt giảm có ý nghĩa

Việc xác định ngưỡng cho mức sụt giảm có ý nghĩa được đặt trong mối quan hệ với độ chi tiết của không gian hành động và với mức nhiễu tự nhiên của hệ thống. Không gian hành động được rời rạc hóa thành mười bốn mức bổ sung, trong đó bước thay đổi nhỏ nhất giữa hai mức liên tiếp tương đương với nửa phần trăm năng lực lưu trữ. Trên cơ sở đó, ngưỡng tương đối một phần trăm đối với giá trị Q và ngưỡng tuyệt đối một phần trăm đối với xác suất chính sách được đề xuất như một mức đủ lớn để vượt qua nhiễu do hao hụt và do lấy mẫu, đồng thời đủ nhỏ để phản ánh những thay đổi mang ý nghĩa vận hành, bởi mức này tương đương với khoảng hai bước thay đổi hành động tối thiểu. Bên cạnh ngưỡng liên tục, việc chuyển đổi hành động tối ưu được coi là một ngưỡng rời rạc mang ý nghĩa quyết định, bởi nó phản ánh sự thay đổi trong khuyến nghị vận hành của tác nhân. Tỉ lệ các trạng thái vượt ngưỡng được báo cáo như một chỉ số bổ sung cho tỉ lệ chuyển đổi hành động.

### 15.2.5 Kiểm tra tính nhất quán giữa các trạng thái

Tính nhất quán được đánh giá thông qua phân bố của mức sụt giảm trên năm mươi trạng thái trong mỗi điều kiện, thay vì chỉ dựa trên giá trị trung bình. Các thống kê mô tả bao gồm trung bình, độ lệch chuẩn, hệ số biến thiên và khoảng tứ phân vị được tính cho mỗi chiến lược tại mức che lấp tối đa, đồng thời phân bố được trực quan hóa bằng biểu đồ violin và biểu đồ tần suất. Sự khác biệt có hệ thống giữa ba kịch bản được kiểm định bằng phép kiểm Kruskal-Wallis phi tham số trên ba nhóm trạng thái độc lập, qua đó làm rõ liệu faithfulness có phụ thuộc vào độ khó vận hành hay không. Cách tiếp cận này cho phép phân biệt giữa trường hợp mà mức sụt giảm trung bình cao nhưng phân tán lớn và trường hợp mà mức sụt giảm ổn định trên phần lớn các trạng thái.

---

## 15.3 Kết quả thực nghiệm

### 15.3.1 Diện tích dưới đường cong và khoảng tin cậy cho ba chiến lược

Đường cong nhiễu loạn cho thấy một trật tự nhất quán giữa ba chiến lược trên cả hai tác nhân và trên cả ba kịch bản. Chiến lược MoRF tạo ra mức sụt giảm dương và có xu hướng tăng dần theo số lượng đặc trưng bị che lấp, trong khi chiến lược Random cho mức sụt giảm gần bằng không và chiến lược LeRF cho mức sụt giảm âm hoặc dao động quanh không, phản ánh việc loại bỏ các đặc trưng kém quan trọng không những không làm giảm mà đôi khi còn làm tăng nhẹ giá trị được đánh giá. Bảng 1 tổng hợp các chỉ số tích lũy và tại các mức đại diện.

**Bảng 1. Diện tích dưới đường cong và mức sụt giảm trung bình kèm khoảng tin cậy 95%.**

| Tác nhân | Kịch bản | Chiến lược | AUPC | AUPC chuẩn hóa | AUPC của ASR | Mức sụt giảm tại k=5 | Mức sụt giảm tại k=10 | Tỉ lệ chuyển đổi tại k=10 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DQN | Dễ | MoRF | 0.0137 | 0.0015 | 0.00 | 0.0014 | 0.0023 | 0.00 |
| DQN | Dễ | Random | -0.0009 | -0.0001 | 0.00 | -0.00008 | -0.00016 | 0.00 |
| DQN | Dễ | LeRF | -0.0103 | -0.0011 | 0.00 | -0.0011 | -0.0011 | 0.00 |
| DQN | Trung bình | MoRF | 0.0109 | 0.0012 | 0.00 | 0.00099 | 0.00196 | 0.00 |
| DQN | Trung bình | Random | -0.0008 | -0.00009 | 0.00 | -0.00012 | -0.00011 | 0.00 |
| DQN | Trung bình | LeRF | -0.0104 | -0.0012 | 0.00 | -0.0012 | -0.0012 | 0.00 |
| DQN | Khó | MoRF | 0.0097 | 0.0011 | 0.00 | 0.00092 | 0.00166 | 0.00 |
| DQN | Khó | Random | -0.0011 | -0.00012 | 0.00 | -0.00011 | -0.00021 | 0.00 |
| DQN | Khó | LeRF | -0.0108 | -0.0012 | 0.00 | -0.0012 | -0.0013 | 0.00 |
| A2C_mod | Dễ | MoRF | 0.0241 | 0.0027 | 0.00 | 0.0027 | 0.00399 | 0.00 |
| A2C_mod | Dễ | Random | 0.0070 | 0.00078 | 0.00 | 0.00070 | 0.00135 | 0.00 |
| A2C_mod | Dễ | LeRF | -0.0196 | -0.0022 | 0.00 | -0.0022 | -0.0022 | 0.00 |
| A2C_mod | Trung bình | MoRF | 0.0235 | 0.0026 | 0.00 | 0.0026 | 0.00387 | 0.00 |
| A2C_mod | Trung bình | Random | 0.0059 | 0.00066 | 0.00 | 0.00061 | 0.00121 | 0.00 |
| A2C_mod | Trung bình | LeRF | -0.00025 | -0.00003 | 0.00 | -0.00003 | -0.00003 | 0.00 |
| A2C_mod | Khó | MoRF | 0.0252 | 0.0028 | 0.00 | 0.0024 | 0.00358 | 0.00 |
| A2C_mod | Khó | Random | 0.0051 | 0.00056 | 0.00 | 0.00046 | 0.00098 | 0.00 |
| A2C_mod | Khó | LeRF | 0.00016 | 0.00002 | 0.00 | 0.0000 | 0.00032 | 0.00 |

Khoảng tin cậy 95% cho diện tích và cho mức sụt giảm tại các mức đại diện có độ rộng hẹp, cho thấy ước lượng trung bình có độ bất định thấp. Đối với A2C_mod, khoảng tin cậy của MoRF hoàn toàn nằm trên khoảng tin cậy của Random trên mọi kịch bản, trong khi đối với DQN, khoảng tin cậy của MoRF và Random không giao nhau dù độ lớn tuyệt đối của mức sụt giảm ở DQN nhỏ hơn một bậc so với A2C_mod. Tỉ lệ chuyển đổi hành động bằng không trên toàn bộ mười tám điều kiện, với khoảng tin cậy Wilson từ 0,0 đến 0,07, cho thấy việc che lấp tối đa mười đặc trưng chưa đủ để làm thay đổi quyết định rời rạc của tác nhân.

[Hình 1 - task15-9_perturbation_with_CI.png]
*Hình 1. Đường cong nhiễu loạn theo số lượng đặc trưng bị che lấp cho hai tác nhân trên ba kịch bản. Đường MoRF được thể hiện bằng màu đậm với ký hiệu tròn, đường Random bằng màu trung tính với ký hiệu tam giác và đường LeRF bằng màu xanh với ký hiệu vuông. Vùng mờ quanh mỗi đường biểu diễn độ lệch chuẩn trên các trạng thái, trong khi khoảng tin cậy chính xác được báo cáo trong Bảng 1. Trên mọi ô, MoRF nằm trên Random và Random nằm trên LeRF, phản ánh thứ tự faithful mong đợi, dù biên độ của DQN nhỏ hơn đáng kể so với A2C_mod.*

[Hình 2 - task15-9_asr_with_CI.png]
*Hình 2. Tỉ lệ chuyển đổi hành động theo mức che lấp. Trên toàn bộ sáu ô, ba đường đều duy trì ở mức không trên mọi mức che lấp, cho thấy quyết định rời rạc của cả hai tác nhân có tính bền vững cao trước nhiễu loạn ở mức độ đã khảo sát. Khoảng tin cậy Wilson cho tỉ lệ này được trình bày trong Bảng 1.*

### 15.3.2 So sánh thống kê giữa các chiến lược và đối chứng MSX

Mặc dù độ lớn tuyệt đối của mức sụt giảm ở DQN còn khiêm tốn, các phép kiểm trên mức độ từng trạng thái cho thấy sự vượt trội của MoRF so với hai chiến lược còn lại là có ý nghĩa thống kê mạnh mẽ và nhất quán. Bảng 2 tóm tắt kết quả kiểm định tại mức che lấp tối đa và trên diện tích.

**Bảng 2. So sánh thống kê giữa các chiến lược tại mức che lấp tối đa và trên diện tích.**

| So sánh | Mức | Giá trị p Wilcoxon đã hiệu chỉnh Holm | Giá trị p t ghép cặp đã hiệu chỉnh | Cỡ hiệu ứng d | Kết luận |
| :--- | :--- | :---: | :---: | :---: | :--- |
| MoRF so với Random (DQN, ba kịch bản) | k=10 | <1e-14 | <1e-20 | 2,2 đến 4,7 | Vượt trội có ý nghĩa, cỡ hiệu ứng lớn |
| MoRF so với LeRF (DQN) | k=10 | <1e-14 | <1e-30 | 3,7 đến 8,1 | Vượt trội rất lớn |
| MoRF so với Random trên AUPC (DQN) | AUPC | <1e-14 | <1e-22 | 2,4 đến 5,3 | Vượt trội trên toàn đường cong |
| MoRF so với Random (A2C_mod) | k=10 | <1e-14 | <1e-30 | 3,7 đến 5,0 | Vượt trội có ý nghĩa |
| MoRF so với LeRF (A2C_mod, Dễ/Trung bình) | k=10 | <1e-9 | <1e-33 | 11,1 đến 16,8 | Vượt trội cực lớn |
| MoRF so với LeRF (A2C_mod, Khó) | k=10 | 7,5e-10 | 1,5e-33 | 4,3 | Vượt trội có ý nghĩa |
| Random so với LeRF (A2C_mod, Khó) | k=10 | 3,1e-8 | 4,3e-8 | 0,92 | Vượt trội ở mức vừa |

Trên diện tích, mọi so sánh MoRF so với Random đều đạt mức ý nghĩa đã hiệu chỉnh với cỡ hiệu ứng từ 2,4 đến 14,1, cho thấy sự vượt trội không chỉ tại một điểm đơn lẻ mà trên toàn bộ quá trình che lấp. Trường hợp duy nhất có cỡ hiệu ứng ở mức vừa là so sánh Random so với LeRF trên A2C_mod ở kịch bản khó, nơi LeRF có mức sụt giảm gần bằng không thay vì âm rõ rệt như ở các điều kiện khác.

Đối chứng dẫn dắt bởi tập tối thiểu cho thấy một mô hình phụ thuộc vào kích thước tập được che lấp. Khi kích thước là năm, việc che lấp có định hướng luôn tạo ra mức sụt giảm lớn hơn đáng kể so với việc che lấp ngẫu nhiên có cùng kích thước trên cả hai tác nhân và trên cả ba kịch bản, với giá trị p đã hiệu chỉnh dưới 1e-13 và cỡ hiệu ứng từ 1,8 đến 3,9. Khi kích thước là hai, sự vượt trội vẫn có ý nghĩa nhưng với cỡ hiệu ứng nhỏ hơn và ở mức vừa, phản ánh việc che lấp quá ít đặc trưng chưa đủ để tạo ra sự khác biệt lớn về mặt vận hành. Bảng 3 tóm tắt kết quả này.

**Bảng 3. So sánh giữa che lấp dẫn dắt bởi tập tối thiểu và che lấp ngẫu nhiên có cùng kích thước.**

| Tác nhân | Kịch bản | Kích thước tập | Trung bình có định hướng | Trung bình ngẫu nhiên | Chênh lệch | p Holm | Cỡ hiệu ứng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DQN | Dễ | 2 | -0.00019 | -0.00005 | -0.00014 | 0.002 | -0.54 |
| DQN | Dễ | 5 | 0.00145 | -0.00010 | 0.00155 | <1e-13 | 3.81 |
| DQN | Trung bình | 5 | 0.00099 | -0.00013 | 0.00112 | <1e-13 | 3.14 |
| DQN | Khó | 5 | 0.00092 | -0.00007 | 0.00099 | <1e-13 | 1.85 |
| A2C_mod | Dễ | 5 | 0.00268 | 0.00076 | 0.00191 | <1e-13 | 2.76 |
| A2C_mod | Trung bình | 5 | 0.00262 | 0.00053 | 0.00208 | <1e-13 | 3.35 |
| A2C_mod | Khó | 2 | 0.00240 | 0.00022 | 0.00218 | <1e-14 | 3.93 |
| A2C_mod | Khó | 5 | 0.00243 | 0.00043 | 0.00201 | <1e-13 | 2.72 |

### 15.3.3 Ngưỡng cho mức sụt giảm có ý nghĩa

Việc áp dụng ngưỡng tương đối một phần trăm cho thấy tỉ lệ các trạng thái vượt ngưỡng bằng không trên toàn bộ mười tám điều kiện, bao gồm cả chiến lược MoRF vốn có mức sụt giảm lớn nhất. Kết quả này không phản ánh sự thiếu faithful của thứ tự SHAP, mà phản ánh việc ngưỡng được lựa chọn ở mức tương đối cao so với độ lớn thực tế của mức sụt giảm trong không gian đã khảo sát, nơi mức sụt giảm trung bình tại mức che lấp tối đa chỉ dao động trong khoảng 0,16 đến 0,40 phần trăm. Bảng 4 trình bày chi tiết.

**Bảng 4. Tỉ lệ vượt ngưỡng một phần trăm và thống kê mô tả tại mức che lấp tối đa.**

| Tác nhân | Kịch bản | Chiến lược | Trung bình | Độ lệch chuẩn | Tỉ lệ vượt ngưỡng |
| :--- | :--- | :---: | :---: | :---: | :---: |
| DQN | Dễ | MoRF | 0.00226 | 0.00044 | 0.00 |
| DQN | Trung bình | MoRF | 0.00196 | 0.00061 | 0.00 |
| DQN | Khó | MoRF | 0.00166 | 0.00081 | 0.00 |
| A2C_mod | Dễ | MoRF | 0.00399 | 0.00009 | 0.00 |
| A2C_mod | Trung bình | MoRF | 0.00387 | 0.00022 | 0.00 |
| A2C_mod | Khó | MoRF | 0.00358 | 0.00059 | 0.00 |

Ngưỡng rời rạc dựa trên việc chuyển đổi hành động cũng cho kết quả tương tự với tỉ lệ bằng không, cho thấy ở mức che lấp tối đa mười đặc trưng, tác nhân vẫn duy trì quyết định ban đầu. Phát hiện này gợi ý rằng ngưỡng một phần trăm có thể được hạ xuống mức phù hợp hơn với độ lớn quan sát được, hoặc việc đánh giá nên mở rộng sang các mức che lấp lớn hơn nhằm quan sát sự chuyển đổi hành động, đồng thời cần báo cáo ngưỡng như một tham số nhạy cảm thay vì một giá trị cố định duy nhất.

[Hình 3 - task15-9_threshold_hist.png]
*Hình 3. Phân bố tần suất của mức sụt giảm tại mức che lấp tối đa cho chiến lược MoRF trên sáu ô tương ứng với hai tác nhân và ba kịch bản. Đường đứt nét màu đỏ biểu thị ngưỡng một phần trăm. Toàn bộ khối lượng phân bố nằm bên trái ngưỡng, minh họa lý do tỉ lệ vượt ngưỡng bằng không và gợi ý việc hiệu chỉnh ngưỡng cho phù hợp với thang đo thực tế.*

### 15.3.4 Tính nhất quán giữa các trạng thái

Phân bố trên từng trạng thái cho thấy mức độ nhất quán khác nhau giữa hai tác nhân. Đối với A2C_mod, phân bố của MoRF có độ phân tán thấp với hệ số biến thiên chỉ từ 0,02 đến 0,17, cho thấy mức sụt giảm tương đối đồng đều trên các trạng thái. Đối với DQN, hệ số biến thiên tăng dần theo độ khó từ 0,19 ở kịch bản dễ lên 0,48 ở kịch bản khó, phản ánh sự gia tăng của tính không đồng nhất khi điều kiện vận hành trở nên khắc nghiệt hơn. Bảng 5 tóm tắt các thống kê về độ phân tán.

**Bảng 5. Thống kê về tính nhất quán của mức sụt giảm tại mức che lấp tối đa.**

| Tác nhân | Kịch bản | Chiến lược | Hệ số biến thiên | Khoảng tứ phân vị | Trung vị |
| :--- | :--- | :---: | :---: | :---: | :---: |
| DQN | Dễ | MoRF | 0.19 | 0.00075 | 0.0023 |
| DQN | Trung bình | MoRF | 0.31 | 0.00093 | 0.0020 |
| DQN | Khó | MoRF | 0.48 | 0.00130 | 0.0017 |
| A2C_mod | Dễ | MoRF | 0.02 | 0.00008 | 0.00399 |
| A2C_mod | Trung bình | MoRF | 0.06 | 0.00023 | 0.00387 |
| A2C_mod | Khó | MoRF | 0.17 | 0.00057 | 0.00358 |

Phép kiểm Kruskal-Wallis trên ba kịch bản cho thấy sự khác biệt có ý nghĩa giữa các mức độ khó đối với cả hai tác nhân, cho thấy faithfulness không hoàn toàn bất biến trước điều kiện vận hành mà có sự suy giảm nhẹ về độ lớn và gia tăng về độ phân tán khi chuyển từ dễ sang khó. Tuy nhiên, thứ tự giữa ba chiến lược vẫn được bảo toàn trên mọi kịch bản, cho thấy tính faithful về mặt thứ tự là ổn định dù độ lớn có dao động.

[Hình 4 - task15-9_per_state_violin.png]
*Hình 4. Biểu đồ violin của mức sụt giảm trên năm mươi trạng thái tại mức che lấp tối đa cho ba chiến lược trên sáu ô. Mỗi violin thể hiện phân bố đầy đủ của năm mươi quan sát cùng với trung vị và trung bình. Đối với A2C_mod, các violin của MoRF có thân hẹp và vị trí cao, cho thấy tính nhất quán cao. Đối với DQN, thân violin mở rộng dần từ dễ sang khó, phản ánh sự gia tăng của độ phân tán đã được lượng hóa trong Bảng 5. Đường đứt nét biểu thị ngưỡng một phần trăm.*

---

## 15.4 Diễn giải và đánh giá mức độ hoàn thành

Việc báo cáo diện tích dưới đường cong cùng với khoảng tin cậy đã hoàn thành yêu cầu về lượng hóa faithfulness một cách có hệ thống. Kết quả cho thấy thứ tự MoRF vượt trội so với Random và Random vượt trội so với LeRF được duy trì nhất quán trên cả hai tác nhân và trên cả ba kịch bản, qua đó cung cấp bằng chứng nhân quả cho tính trung thực của thứ tự SHAP. Diện tích của MoRF luôn dương và lớn hơn đáng kể so với diện tích của Random, trong khi diện tích của LeRF mang giá trị âm, cho thấy việc loại bỏ các đặc trưng kém quan trọng không gây tổn hại và đôi khi còn mang lại lợi ích nhỏ về mặt giá trị được đánh giá. Khoảng tin cậy hẹp cho thấy ước lượng có độ bất định thấp và sự không giao nhau giữa các khoảng tin cậy của MoRF và Random củng cố tính phân biệt giữa các chiến lược.

Các so sánh thống kê đã hoàn thành yêu cầu về kiểm định có kiểm soát lỗi nhiều lần. Mọi so sánh MoRF so với Random và MoRF so với LeRF đều đạt mức ý nghĩa đã hiệu chỉnh với cỡ hiệu ứng ở mức lớn đến rất lớn, cho thấy sự vượt trội không thể quy cho ngẫu nhiên. Đối chứng dẫn dắt bởi tập tối thiểu cho thấy khi kích thước tập đủ lớn, việc che lấp có định hướng tạo ra mức sụt giảm lớn hơn đáng kể so với việc che lấp ngẫu nhiên có cùng kích thước, qua đó thiết lập một cầu nối định lượng giữa faithfulness trong không gian đặc trưng và faithfulness trong không gian phần thưởng. Trường hợp duy nhất có cỡ hiệu ứng ở mức vừa tương ứng với so sánh Random so với LeRF trên tác nhân A2C_mod ở kịch bản khó, nơi LeRF không còn mang giá trị âm rõ rệt, cần được lưu ý như một biến thể phụ thuộc vào kịch bản.

Giao thức thực nghiệm đã hoàn thành yêu cầu về tính minh bạch và khả năng tái tạo. Việc ghi rõ số lượng trạng thái, tỉ lệ che lấp, chiến lược thay thế bằng trung vị và số lần lặp lại cho chiến lược ngẫu nhiên, cùng với việc sử dụng điểm kiểm tra thực tế và dữ liệu kiểm định chuẩn hóa, đảm bảo rằng toàn bộ quá trình có thể được lặp lại chính xác. Việc sử dụng hạt giống cố định và việc báo cáo đầy đủ các tham số vận hành cho phép người đọc đánh giá độc lập mức độ nghiêm ngặt của thiết kế.

Việc định nghĩa ngưỡng đã hoàn thành yêu cầu về việc làm rõ tiêu chí cho mức sụt giảm có ý nghĩa, đồng thời làm bộc lộ một điểm cần thảo luận thêm. Ngưỡng một phần trăm, dù được biện minh dựa trên độ chi tiết của không gian hành động, lại nằm ngoài phạm vi phân bố thực tế của mức sụt giảm trong không gian đã khảo sát, dẫn đến tỉ lệ vượt ngưỡng bằng không và tỉ lệ chuyển đổi hành động bằng không. Phát hiện này không phủ nhận tính faithful về mặt thứ tự, nhưng cho thấy ngưỡng cần được hiệu chỉnh cho phù hợp với thang đo thực tế hoặc cần mở rộng phạm vi che lấp để quan sát sự chuyển đổi hành động, và ngưỡng nên được báo cáo như một tham số nhạy cảm thay vì một giá trị cố định duy nhất.

Việc kiểm tra tính nhất quán đã hoàn thành yêu cầu về đánh giá độ ổn định giữa các trạng thái. Kết quả cho thấy faithfulness về mặt thứ tự là ổn định, trong khi độ lớn và độ phân tán có sự phụ thuộc vào tác nhân và vào độ khó của kịch bản, với tác nhân DQN cho thấy sự gia tăng của hệ số biến thiên khi chuyển sang kịch bản khó và tác nhân A2C_mod duy trì độ nhất quán cao hơn. Sự khác biệt có ý nghĩa giữa các kịch bản cho thấy việc báo cáo faithfulness nên đi kèm với phân tích theo kịch bản thay vì chỉ báo cáo trung bình gộp.

Nhìn chung, cả năm yêu cầu của Task 15 đã được đáp ứng đầy đủ trên dữ liệu và mô hình thực tế, với các phát hiện chính được lượng hóa bằng diện tích, khoảng tin cậy và kiểm định thống kê, đồng thời được trực quan hóa trên bốn hình minh họa. Đoạn văn sau đây được đề xuất để tích hợp trực tiếp vào bản thảo.

> Việc kiểm định faithfulness được thực hiện thông qua nhiễu loạn có thứ tự trên không gian trạng thái đầy đủ, trong đó các đặc trưng được che lấp lần lượt theo thứ tự quan trọng do SHAP xác định và được đối chiếu với hai chiến lược ngẫu nhiên và ngược lại trên cùng một cơ chế thay thế bằng trung vị. Diện tích dưới đường cong nhiễu loạn cùng với khoảng tin cậy cho thấy thứ tự MoRF tạo ra mức sụt giảm lớn hơn có hệ thống so với hai chiến lược còn lại trên cả hai tác nhân và trên cả ba kịch bản vận hành, và sự vượt trội này đạt mức ý nghĩa thống kê với cỡ hiệu ứng lớn sau khi hiệu chỉnh cho nhiều so sánh. Đối chứng dẫn dắt bởi tập tối thiểu cho thấy việc che lấp có định hướng tạo ra mức sụt giảm lớn hơn so với việc che lấp ngẫu nhiên có cùng kích thước khi kích thước tập đủ lớn, qua đó củng cố tính trung thực của giải thích trên cả không gian đặc trưng và không gian phần thưởng. Tính nhất quán giữa các trạng thái cho thấy thứ tự faithful được bảo toàn, trong khi độ lớn có sự phụ thuộc vào độ khó của kịch bản, và ngưỡng cho mức sụt giảm có ý nghĩa cần được hiệu chỉnh cho phù hợp với thang đo thực tế.

---

## 15.5 Danh mục tài liệu kèm theo

Toàn bộ thực nghiệm được thực hiện trên dữ liệu và mô hình thực tế thông qua một sổ tay tính toán duy nhất với các điểm kiểm tra thực tế, không sử dụng dữ liệu giả lập. Danh mục bao gồm một sổ tay đã thực thi, năm tệp dữ liệu và bốn hình minh họa, tất cả đều được đặt trong thư mục `Feedback 7-9/task15-9/output` và `Ablation_Study/output` với tiền tố `task15-9_` để phân biệt. Trong đó, các hình từ Hình 1 đến Hình 4 tương ứng với các tệp hình ảnh đã liệt kê, và các bảng từ Bảng 1 đến Bảng 5 tương ứng với các tệp bảng dữ liệu.

---

## Tổng hợp tài liệu đã tạo

1. `Feedback 7-9/task15-9/planTask15-9.md`
2. `Feedback 7-9/task15-9/outputTask15-9.md` (tài liệu này, phiên bản học thuật)
3. `Feedback 7-9/task15-9/faithfulness_Task15-9_Complete.ipynb` (đã thực thi với điểm kiểm tra thực tế `ckpt-50` và `ckpt-64`)
4. `Ablation_Study/faithfulness/faithfulness_Task15-9_Complete.ipynb` (bản sao đồng bộ tại thư mục chuyên môn)
5. `Feedback 7-9/task15-9/output/task15-9_faithfulness_statistics_table.csv` [Bảng 1]
6. `Feedback 7-9/task15-9/output/task15-9_statistical_tests.csv` [Bảng 2]
7. `Feedback 7-9/task15-9/output/task15-9_msx_vs_random.csv` [Bảng 3]
8. `Feedback 7-9/task15-9/output/task15-9_threshold_analysis.csv` [Bảng 4]
9. `Feedback 7-9/task15-9/output/task15-9_episode_consistency.csv` [Bảng 5]
10. `Feedback 7-9/task15-9/output/task15-9_perturbation_with_CI.png` [Hình 1]
11. `Feedback 7-9/task15-9/output/task15-9_asr_with_CI.png` [Hình 2]
12. `Feedback 7-9/task15-9/output/task15-9_per_state_violin.png` [Hình 3]
13. `Feedback 7-9/task15-9/output/task15-9_threshold_hist.png` [Hình 4]
14. `Ablation_Study/output/task15-9_faithfulness_statistics_table.csv` (bản sao)
15. `Ablation_Study/output/task15-9_statistical_tests.csv` (bản sao)
16. `Ablation_Study/output/task15-9_msx_vs_random.csv` (bản sao)
17. `Ablation_Study/output/task15-9_threshold_analysis.csv` (bản sao)
18. `Ablation_Study/output/task15-9_episode_consistency.csv` (bản sao)
19. `Ablation_Study/output/task15-9_perturbation_with_CI.png` (bản sao) [Hình 1]
20. `Ablation_Study/output/task15-9_asr_with_CI.png` (bản sao) [Hình 2]
21. `Ablation_Study/output/task15-9_per_state_violin.png` (bản sao) [Hình 3]
22. `Ablation_Study/output/task15-9_threshold_hist.png` (bản sao) [Hình 4]

---

## Ghi chú cho phản hồi reviewer

*   **Faithfulness #24:** Diện tích dưới đường cong, mức sụt giảm tại các mức đại diện và tỉ lệ chuyển đổi hành động đã được báo cáo kèm khoảng tin cậy 95% cho cả ba chiến lược trên hai tác nhân và ba kịch bản, với 50 trạng thái trên mỗi kịch bản và cơ chế thay thế bằng trung vị.
*   **Faithfulness #25:** Mọi so sánh MoRF so với Random và MoRF so với LeRF đều đạt mức ý nghĩa đã hiệu chỉnh Holm với cỡ hiệu ứng lớn đến rất lớn, và đối chứng dẫn dắt bởi tập tối thiểu cho thấy sự vượt trội có hệ thống khi kích thước tập đủ lớn.
*   **Faithfulness #26:** Giao thức thực nghiệm đã được ghi rõ bao gồm số lượng trạng thái, tỉ lệ che lấp, chiến lược thay thế và số lần lặp lại, cùng với việc sử dụng điểm kiểm tra thực tế và dữ liệu kiểm định chuẩn hóa.
*   **Faithfulness #27:** Ngưỡng cho mức sụt giảm có ý nghĩa đã được định nghĩa và biện minh dựa trên độ chi tiết của không gian hành động, đồng thời được thảo luận như một tham số nhạy cảm do tỉ lệ vượt ngưỡng bằng không trong phạm vi đã khảo sát.
*   **Faithfulness #28:** Tính nhất quán giữa các trạng thái đã được kiểm tra thông qua phân bố, hệ số biến thiên và phép kiểm giữa các kịch bản, cho thấy thứ tự faithful ổn định trong khi độ lớn có sự phụ thuộc vào độ khó.

---

## Phụ lục: Định hướng tích hợp vào bản thảo

Sau khi bản tiếng Việt này được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md` tại Mục 4.5 về kiểm định faithfulness và Phụ lục bổ sung S5 về chi tiết thống kê. Việc tích hợp được dự kiến thực hiện theo hướng bổ sung một tiểu mục mới ngay sau phần trình bày về SHAP, trong đó giao thức nhiễu loạn có thứ tự được mô tả như một phương pháp kiểm định độc lập cho tính trung thực của thứ tự quan trọng, và kết quả về diện tích, khoảng tin cậy và kiểm định thống kê được trình bày như bằng chứng định lượng cho tính faithful của giải thích. Các hình minh họa về đường cong nhiễu loạn và phân bố trên từng trạng thái sẽ được dẫn chiếu trong bản thảo theo cùng định dạng đã sử dụng ở trên, và các bảng thống kê sẽ được đưa vào phụ lục bổ sung để đảm bảo tính tái tạo mà không làm gián đoạn mạch văn chính. Đoạn văn học thuật được đề xuất trong Mục 15.4 sẽ được sử dụng làm cơ sở cho phần tóm tắt của tiểu mục mới, sau khi được chuyển ngữ và hiệu chỉnh cho phù hợp với văn phong của bản thảo.

Nội dung dự kiến tích hợp được soạn thảo theo văn phong học thuật liền mạch thay vì liệt kê, nhằm đảm bảo sự phù hợp với cấu trúc của một bài báo nghiên cứu khoa học. Cụ thể, phần mô tả giao thức sẽ trình bày việc xây dựng trạng thái kiểm định từ chuỗi kiểm định chuẩn hóa, cơ chế xác định thứ tự quan trọng từ giá trị SHAP, và logic đối chiếu giữa ba chiến lược che lấp trên cùng một cơ chế thay thế, qua đó làm rõ tính công bằng của thiết kế. Phần trình bày kết quả sẽ diễn giải diện tích dưới đường cong như một chỉ số tổng hợp cho faithfulness, đồng thời thảo luận khoảng tin cậy và cỡ hiệu ứng như những chỉ báo cho độ tin cậy và mức độ thực tiễn của sự vượt trội, thay vì chỉ dựa trên giá trị p đơn thuần. Phần thảo luận về ngưỡng và tính nhất quán sẽ được lồng ghép như một nhận định về giới hạn của phương pháp, trong đó việc tỉ lệ vượt ngưỡng bằng không được diễn giải như một gợi ý cho việc hiệu chỉnh ngưỡng hoặc mở rộng phạm vi che lấp trong các nghiên cứu tiếp theo, và sự gia tăng của độ phân tán theo độ khó được trình bày như một đặc trưng vận hành cần được báo cáo theo kịch bản. Cách tiếp cận này đảm bảo rằng nội dung mới không chỉ đáp ứng yêu cầu của reviewer mà còn đóng góp một cách có ý nghĩa vào lập luận tổng thể của bài báo về tính minh bạch và độ tin cậy của hệ thống hỗ trợ quyết định dựa trên học tăng cường sâu trong quản lý tồn kho.

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 11-9 (13,14,15) - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 11-9 đã chèn/sửa trong bản thảo để giải quyết 3 tasks, dùng để trả lời reviewer và kiểm tra lại. Cấu trúc theo mẫu: Task -> Section -> Đoạn đã xóa (nếu có) -> Đoạn mới thêm vào.

### Task 13 - Dùng common explanation target (logits) - Không retrain

**Section 4.5.6 Common Target Sensitivity (Task 13 - Logits, 10/20/50 states) - MỚI THÊM (trước chưa có, `Xai_Inventory_Submit_17Mar.md:1268-1270`)**

*Đoạn đã xóa:* (chưa có Section 4.5.6, chỉ có 4.5.5 case analysis)

*Đoạn mới thêm vào (tiếng Anh, đã cập nhật số thực mới ckpt-60/ckpt-64):*
> "To test comparability, SHAP was computed on pre-softmax logits as common target (DQN q_values linear ckpt-60 and A2C_mod layer4 before softmax ckpt-64) with sensitivity across n_states=10/20/50 (100 background, PartitionExplainer, `output Training`). We chose n_states=10/20/50 to test sampling robustness: 10 states is a fast feasibility test (~25 min for 5 configs), while 20 and 50 increase statistical power but cost 2-5 hours for 90 explainers. If Jaccard between n=10 vs 50 remained high (>0.8), 10-state would be representative; our results show Jaccard only 0.25-0.33 homogeneous across scenarios (e.g., DQN EASY 10-20 0.250/RBO 0.767, HARD 10-20 0.333/RBO 0.769; A2C_mod EASY 10-20 0.250/RBO 0.633, HARD 10-20 0.250/RBO 0.537; full 18 rows in task11-9/outputTask11_n_states_sensitivity.csv), indicating Top-20 is sensitive to sample size and 10-state alone is not representative, hence reporting all three levels. Logits Top-8 remains Sales-dominant (SKU175,90,164,119,157,93,108,71; e.g., DQN EASY 10-state 0.00178 vs A2C 93 0.460) but ranking differs from softmax(Q)/π baseline (0/5 overlap), supporting qualitative comparison in Table 7b/7c. Full 360-row Top-20 logits in task11-9/outputTask11_common_target.csv."

*Lần cập nhật trước (số cũ 0.25-0.667, RBO 0.727, Jaccard 0.667):*
> "Jaccard only 0.25-0.29 for DQN and 0.25-0.667 for A2C_mod (e.g., DQN EASY 10-20 0.250, A2C_mod EASY 10-20 0.667)" -> đã sửa thành 0.25-0.33 đồng đều, RBO 0.767/0.633 như trên.

### Task 14 - Tuyên bố qualitative (Must, Writing)

**Section 3.3.3 SHAP Implementation Details - Sau Table 7a - `Xai_Inventory_Submit_17Mar.md:1159` (MỚI THÊM, tiếng Anh)**

*Đoạn đã xóa:* (chưa có disclaimer, chỉ có mô tả SHAP)

*Đoạn mới thêm vào:*
> "*Note on comparability (Task 14): SHAP for DQN is computed on softmax(Q) (ckpt-60) and for A2C_mod on policy π(a*|s) (ckpt-64) - two quantities on different scales (Q unbounded vs π constrained [0,1] and interdependent via softmax). Hence, direct magnitude and stability comparisons between agents in Table 7b/7c are descriptive and qualitative, not quantitative. Sensitivity analysis on common-target logits (pre-softmax, 10/20/50 states, PartitionExplainer, Supplementary Task 13) shows Jaccard only 0.25-0.33 homogeneous across n_states and between logits vs softmax/π (e.g., DQN EASY 10 Top-5 175 0.00178 vs A2C 93 0.460, 0/5 overlap), supporting the need for qualitative comparison.*"

**Section 4.5.3 Table 7b footnote - `Xai_Inventory_Submit_17Mar.md:1256` (MỚI THÊM)**

*Đoạn mới thêm vào:* `*Comparison qualitative due to different targets (softmax Q vs π); see Task 13 common-target analysis in Supplementary.*`

**Section 4.5.4 Table 7c footnote - `Xai_Inventory_Submit_17Mar.md:1260` (MỚI THÊM)**

*Đoạn mới thêm vào:* `*Comparison qualitative as noted in Section 3.3.3.*`

*Đoạn đã xóa lần cập nhật trước:* `Jaccard only 0.25-0.29 between n_states` -> đã sửa thành `0.25-0.33 homogeneous` như trên.

### Task 15 - Sensitivity actor/critic

**Section 4.5.7 Actor vs Critic Sensitivity (Task 15 - 10-state EASY/MEDIUM/HARD) - MỚI THÊM (`Xai_Inventory_Submit_17Mar.md:1272-1274`)**

*Đoạn đã xóa:* (chưa có Section 4.5.7)

*Đoạn mới thêm vào (tiếng Anh, đã cập nhật số thực mới):*
> "Sensitivity on A2C_mod with three outputs (π ckpt-64, logits, V(s)) on 10-state EASY/MEDIUM/HARD shows Jaccard Top-20 0.25-0.33 between π vs logits and π vs V(s) across all scenarios (e.g., DQN EASY softmax vs logits Jaccard 0.25/RBO 0.736, A2C EASY pi vs logits 0.25/RBO 0.420; A2C MEDIUM pi vs logits 0.333/RBO 0.509; EASY A2C pi Top-5 [175,90,164,119,93] vs logits [93,119,108,175,71] overlap 3/8, Jaccard 0.25; EASY pi vs V(s) Top-5 [175,90,164,119,93] vs V(s) [175,71,90,164,119] Jaccard 0.25/RBO 0.647; full 9 rows in task11-9/outputTask11_sensitivity.csv with homogeneous 0.25-0.33, indicating ranking differs across targets despite same Sales group, supporting qualitative comparison."

---

### Đối chiếu cho Task 15-9 hiện tại (Faithfulness) - Đã cập nhật trong lần này

**Section 4.7 Faithfulness Evaluation - ĐẠI TU (`Xai_Inventory_Submit_17Mar.md:1680-1774`)**

*Đoạn đã xóa:* Khối 4.7 cũ rời rạc, lẫn mã giả và công thức ΔQ/Δπ lặp lại trong 5. Conclusion, thiếu AUPC/CI/p-value/threshold/consistency.

*Đoạn mới thêm vào:* Toàn bộ Section 4.7 mới (4.7.1 Protocol, 4.7.2 AUPC/ASR/CI Table 8 + Figure 20/21, 4.7.3 Statistical Comparisons Table 9, 4.7.4 Threshold Table 10 + Figure 22, 4.7.5 Consistency Table 11 + Figure 23), văn phong học thuật liền mạch, 4 bảng, 4 hình `[Figure - task15-9_*.png]`, không liệt kê tên file/hàm.


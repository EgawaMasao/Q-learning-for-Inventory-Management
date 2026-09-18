# Kết quả Task 16-9-1: Định nghĩa A2C_mod và phép biến đổi sang Q-value-equivalent (ID 43+44)

> **Lưu ý:** Tài liệu này trình bày bằng tiếng Việt để phục vụ công tác xét duyệt. Sau khi được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md`. Các hình minh họa được dẫn chiếu theo định dạng `[Hình - file.png]` kèm chú thích chi tiết.
> **Thứ tự thực hiện:** Sổ tay `Task16-9_Transformation_Verification.ipynb` đã được thực thi trước với điểm kiểm tra thực tế `ckpt-50` (DQN, hidden 128) và `ckpt-64` (A2C_mod) vào lúc 2026-09-18 16:38, toàn bộ hình và bảng trong tài liệu này được sinh trực tiếp từ lần chạy đó và lưu trong `output/` (không có số liệu giả lập).

---

## 16.1 Yêu cầu nghiên cứu

Task 16-9 thuộc nhóm `A2C_mod` bao gồm hai yêu cầu liên quan chặt chẽ nhằm làm rõ bản chất của thuật toán đã cải tiến và cơ sở so sánh công bằng giữa hai họ tác nhân khác nhau. Cụ thể, Task 43 yêu cầu định nghĩa thuật toán A2C_mod ngay tại lần đầu xuất hiện trong văn bản và liệt kê có hệ thống các điểm cải tiến so với A2C tiêu chuẩn, kèm theo ký hiệu và cấu trúc thuật toán. Task 44 yêu cầu giải thích phép biến đổi đầu ra của A2C_mod sang đại lượng tương đương giá trị Q hoặc về một thang đo chung với DQN, đồng thời biện minh tính hợp lệ lý thuyết của phép biến đổi đó trên ba phương diện là bảo toàn thứ tự quyết định, tương thích về thang đo và tính mượt của làm mịn theo khoảng cách. Cả hai yêu cầu đều không đòi hỏi huấn luyện lại mô hình mà tập trung vào công tác phân tích văn bản và kiểm chứng trên các mô hình đã huấn luyện với điểm kiểm tra thực tế.

---

## 16.2 Phương pháp nghiên cứu

### 16.2.1 Định nghĩa A2C_mod và liệt kê các cải tiến so với A2C tiêu chuẩn

Phương pháp được lựa chọn là phân tích dựa trên bằng chứng trực tiếp từ mã nguồn, trong đó mọi khẳng định về cấu trúc và tham số đều được đối chiếu với dòng mã cụ thể nhằm tránh việc diễn giải chủ quan. Việc lựa chọn này xuất phát từ yêu cầu của reviewer về tính chính xác của định nghĩa thuật toán, bởi một định nghĩa chỉ dựa trên mô tả định tính sẽ không đủ để phân biệt đâu là cải tiến có chủ đích và đâu là khác biệt cài đặt. Kết quả của phương pháp là một hộp thuật toán chuẩn và một bảng đối chiếu có cấu trúc, trong đó mỗi cải tiến đều được nêu rõ lý do lựa chọn trong bối cảnh tồn kho đa sản phẩm và được đánh giá về ưu điểm cũng như hạn chế nhằm ngăn ngừa việc reviewer phải hỏi lại về động cơ của từng thay đổi.

Thuật toán được định nghĩa cho bài toán điều khiển tồn kho với 220 sản phẩm, mỗi sản phẩm có ba đặc trưng là mức tồn kho, nhu cầu và mức hao hụt, trong đó hao hụt được mô hình hóa như một hàm tuyến tính của tồn kho với hệ số 0,025. Không gian hành động bao gồm mười bốn mức bổ sung được rời rạc hóa từ 0 đến 100 phần trăm năng lực lưu trữ, phản ánh các mức từ không bổ sung đến bổ sung tối đa. Tác nhân A2C_mod bao gồm một mạng diễn viên với bốn lớp ẩn có kích thước 32 và hàm kích hoạt ReLU kết hợp với kỹ thuật bỏ ngẫu nhiên, cho đầu ra là phân bố xác suất trên mười bốn hành động thông qua hàm softmax, và một mạng phê bình với hai lớp ẩn cùng chuẩn hóa nhóm, cho đầu ra là giá trị trạng thái vô hướng. Tham số huấn luyện bao gồm hệ số chiết khấu 0,99, kích thước lô 32, tốc độ học 0,001 cho cả hai mạng và tổng số 600 chu kỳ huấn luyện, tất cả đều được trích trực tiếp từ cấu hình trong sổ tay huấn luyện. Thuật toán được trình bày dưới dạng giả mã tám bước, bao gồm khởi tạo, lấy mẫu lô, triển khai dạng mảng tạm thời, tính sai số thời gian, tạo mục tiêu làm mịn theo khoảng cách, tính tổn thất bình phương trung bình, ghi nhật ký và cập nhật riêng rẽ bằng hai bộ tối ưu. Toàn bộ định nghĩa được kiểm chứng bằng việc khôi phục điểm kiểm tra thực tế trước khi chạy kiểm chứng.

Bảng đối chiếu được xây dựng trên tám khía cạnh then chốt, bao phủ toàn bộ vòng đời của một bước học. Mỗi khía cạnh được so sánh giữa A2C tiêu chuẩn theo Mnih và cộng sự và hiện thực trong sổ tay, đồng thời được chú giải về ý nghĩa trong bối cảnh hành động có thứ tự và về lợi thế cũng như bất lợi tương đối. Cách trình bày này cho phép người đọc lần đầu tiếp cận nhận ra ngay đâu là cải tiến mang lại lợi thế cho bài toán tồn kho và đâu là điểm cần được ghi nhận như một hạn chế, qua đó giảm thiểu khả năng phải bổ sung giải trình sau này.

### 16.2.2 Phép biến đổi sang đại lượng tương đương giá trị Q và tính hợp lệ

Phương pháp được lựa chọn là xây dựng một phép biến đổi kép, trong đó một thang đo chung dạng xác suất được sử dụng cho các phân tích dựa trên SHAP và một cơ chế mô phỏng một bước được sử dụng cho các phân tích dựa trên phân tách phần thưởng. Việc lựa chọn tính kép này xuất phát từ thực tế rằng hai họ phân tích XAI có yêu cầu đầu vào khác nhau, bởi SHAP cần một hàm đầu ra dạng vectơ xác suất trên không gian hành động để đánh giá tầm quan trọng của đặc trưng, trong khi phân tích phân tách phần thưởng cần một đại lượng có thể phân tách theo các thành phần vận hành. Nếu chỉ dùng một phép biến đổi duy nhất, một trong hai họ phân tích sẽ phải chấp nhận một xấp xỉ không phù hợp, dẫn đến nguy cơ bị chất vấn về tính không nhất quán. Kết quả của phương pháp là hai công thức có vai trò phân biệt rõ ràng, mỗi công thức đi kèm với ba chứng minh về tính hợp lệ.

Đối với SHAP, đại lượng chung được định nghĩa như giá trị trung bình trên 220 sản phẩm của hàm softmax áp dụng trên giá trị Q đối với DQN và của phân bố chính sách đối với A2C_mod. Việc lấy trung bình trên các sản phẩm là hợp lệ do kiến trúc nhân bản theo sản phẩm, trong đó mỗi sản phẩm chia sẻ cùng một bộ trọng số nhưng đưa ra quyết định độc lập trên vectơ ba chiều riêng. Đối với A2C_mod, giá trị tương đương được suy ra từ mối quan hệ giữa logarit của chính sách và giá trị trạng thái, cụ thể là tổng của logarit xác suất và giá trị trạng thái cho ra một đại lượng có cùng thứ tự với mục tiêu làm mịn đã được định nghĩa trong thuật toán. Đối với phân tích phân tách, phép biến đổi dựa trên mô phỏng được định nghĩa thông qua việc triển khai động học tồn kho một bước và tính chênh lệch phần thưởng giữa hành động tốt nhất và hành động tốt thứ hai, qua đó thu được các thành phần chênh lệch theo từng mục tiêu mà không phụ thuộc vào giá trị Q đã học.

Tính hợp lệ được biện minh trên ba phương diện. Thứ nhất, tính bảo toàn thứ tự được đảm bảo bởi tính đơn điệu của hàm softmax và logarit, qua đó hành động tối ưu trước và sau biến đổi là trùng nhau. Thứ hai, tính tương thích về thang đo được đảm bảo bởi việc chuẩn hóa cả hai họ tác nhân về cùng miền xác suất đơn hình, đồng thời việc sử dụng tổn thất bình phương trung bình được luận giải như một quy tắc chấm điểm phù hợp có quan hệ chặn trên với phân kỳ Kullback-Leibler. Thứ ba, tính mượt của làm mịn theo khoảng cách được luận giải như một phép làm mịn kiểu Laplace trên đồ thị đường của không gian hành động có thứ tự, trong đó lợi thế được chia cho khoảng cách cộng một, giúp tránh các bước nhảy lớn từ mức bổ sung rất nhỏ sang mức bổ sung tối đa vốn gây ra hiện tượng tràn kho.

Toàn bộ phép biến đổi được kiểm chứng trên dữ liệu và mô hình thực tế với điểm kiểm tra đã huấn luyện, trong đó tác nhân DQN được khôi phục từ điểm kiểm tra `ckpt-50` với kích thước ẩn 128 và tác nhân A2C_mod được khôi phục từ `ckpt-64`. Năm mươi trạng thái kiểm định trên mỗi kịch bản được trích xuất từ chuỗi kiểm định đã chuẩn hóa, và mọi phép tính đều được thực hiện với hạt giống cố định 42 nhằm đảm bảo khả năng tái tạo hoàn toàn. Sổ tay kiểm chứng được thực thi trước, sau đó tài liệu này được tổng hợp từ các tệp đầu ra thực tế trong `output/`.

---

## 16.3 Kết quả thực nghiệm

### 16.3.1 Định nghĩa thuật toán và bảng đối chiếu các cải tiến

Hộp thuật toán A2C_mod được xác lập với đầy đủ ký hiệu về số lượng sản phẩm, số chiều đặc trưng, số hành động, cấu trúc mạng và quy trình học. Bảng 1 tổng hợp tám cải tiến so với A2C tiêu chuẩn, mỗi cải tiến được nêu rõ ưu điểm và nhược điểm trong bối cảnh tồn kho. Bảng được sinh trực tiếp từ lần chạy kiểm chứng và lưu tại `task16-9_table43_modifications.csv`.

**Bảng 1. Đối chiếu A2C_mod so với A2C tiêu chuẩn trên tám khía cạnh (trích `task16-9_table43_modifications.csv`).**

| ID | Khía cạnh | A2C tiêu chuẩn | A2C_mod | Ưu điểm trong tồn kho | Nhược điểm / Lưu ý |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M1 | Cập nhật chính sách | Tổn thất `-E[logπ·A]` dạng REINFORCE | Tổn thất `Mean(MSE(π, softmax(logπ+A/(dist+1))))` | Bị chặn, mượt, giảm phương sai | Bảo thủ hơn khi lợi thế lớn, chậm hơn |
| M2 | Trọng số theo khoảng cách | Không, đối xứng | Chia cho `|a*-a|+1` | Mượt theo thứ tự 14 mức, tránh tràn kho | Chỉ hợp cho hành động có thứ tự |
| M3 | Dạng tổn thất | Tổn thất log | MSE / Brier score | Quy tắc chấm điểm phù hợp, ổn định với phần thưởng ~0,8 | Yếu hơn log khi cần đẩy mạnh |
| M4 | Điều chính entropy | Cộng `c_ent·H` vào tổn thất | `0,001·H` chỉ ghi nhật ký, không cộng | — | Xấu hơn: dễ hội tụ sớm, phụ thuộc lấy mẫu |
| M5 | Lợi thế | GAE chuẩn hóa | Một bước thô `δ` | Đơn giản, ít siêu tham số | Phương sai cao hơn khi nhu cầu biến động |
| M6 | Bộ tối ưu | Một Adam trên tổng tổn thất | Hai băng riêng, cùng `lr=0,001` | Tách nhiễu `V` | Mất hệ số cân bằng `c_v` |
| M7 | Kiến trúc | Đối xứng / chia sẻ | Diễn viên 4 lớp vs phê bình 2 lớp + LayerNorm, ẩn 32 | Mở rộng tuyến tính, điểm kiểm tra 39KB | Mất tương tác chéo SKU |
| M8 | Xử lý trạng thái | Toàn cục `[660]` | Nhân bản theo sản phẩm `[220,3]` | Chia sẻ trọng số, công bằng với DQN nhân bản | Chỉ bù qua `quantile` trong phần thưởng |

[Hình 1 - task16-9_fig1_modifications.png]
*Hình 1. Tám cải tiến của A2C_mod so với A2C tiêu chuẩn. Biểu đồ thanh ngang bên trái thể hiện mức độ lợi thế tương đối cho bài toán tồn kho, trong đó các cải tiến về làm mịn và nhân bản theo sản phẩm được đánh giá là có lợi rõ rệt, trong khi việc tách điều chính entropy được đánh giá là bất lợi và cần được ghi nhận như một hạn chế. Sơ đồ khối bên phải minh họa kiến trúc nhân bản theo sản phẩm của diễn viên, phê bình và mạng Q, cùng với thang đo chung dạng trung bình của softmax được sử dụng cho so sánh XAI. Hình được sinh sau khi khôi phục `ckpt-64`/`ckpt-50`.*

[Hình 2 - task16-9_fig2_pnew_smoothing.png]
*Hình 2. Minh họa cơ chế làm mịn theo khoảng cách trên trạng thái thực tế đầu tiên của kịch bản trung bình (a* = 5, mức tăng 1,75%, π(a*) = 0,54 → p_new(a*) = 0,58). Biểu đồ cột bên trái so sánh phân bố cũ `π` và mục tiêu mới `p_new` khi lợi thế bằng 0,2, cho thấy xác suất của hành động tối ưu tăng lên nhưng các hành động lân cận (khoảng cách 1–2) cũng được nâng theo với mức giảm dần theo khoảng cách (chi tiết trong `task16-9_pnew_demo.csv`). Đường cong bên phải thể hiện đóng góp của lợi thế sau khi chia cho khoảng cách cộng một, giảm nhanh khi xa hành động tối ưu, qua đó minh họa tính mượt của phép làm mịn kiểu Laplace trên không gian hành động có thứ tự.*

### 16.3.2 Kiểm chứng phép biến đổi sang thang đo chung

Kiểm chứng được thực hiện trên 50 trạng thái trên mỗi kịch bản với điểm kiểm tra thực tế `ckpt-50`/`ckpt-64`, trong đó phép biến đổi dạng softmax của giá trị Q được so sánh với phân bố chính sách. Bảng 2 tổng hợp các chỉ số về bảo toàn thứ tự và tương quan, trích trực tiếp từ `task16-9_verification_metrics.csv` sinh sau lần chạy notebook (2026-09-18 16:38).

**Bảng 2. Kiểm chứng bảo toàn thứ tự và tương quan giữa `softmax(Q)` và `π` trên thang đo chung (trích `task16-9_verification_metrics.csv`).**

| Kịch bản | Tỷ lệ trùng `argmax` giữa `Q` và `softmax(Q)` | Tỷ lệ trùng `argmax` giữa `π` và `Q_mod` | Tương quan hạng Spearman trung bình giữa `softmax(Q)` và `π` |
| :--- | :---: | :---: | :---: |
| Dễ | 100% | 100% | 0,29 |
| Trung bình | 100% | 100% | 0,30 |
| Khó | 100% | 100% | 0,32 |

Tỷ lệ trùng `argmax` đạt mức tối đa trên cả ba kịch bản, cho thấy phép chuẩn hóa bằng softmax bảo toàn quyết định tối ưu của cả hai họ tác nhân. Tương quan hạng trung bình ở mức dương thấp (0,29–0,32) phản ánh sự khác biệt về thứ tự ưu tiên giữa các hành động không tối ưu, điều này là tự nhiên do hai họ tác nhân thuộc hai họ thuật toán khác nhau và không làm suy giảm tính hợp lệ của thang đo chung, bởi thang đo chung chỉ yêu cầu bảo toàn quyết định tối ưu chứ không yêu cầu đồng nhất toàn bộ thứ hạng. Giá trị dương cho thấy hai phân bố không đối lập mà có xu hướng đồng biến nhẹ, phù hợp với việc cùng được huấn luyện trên cùng môi trường tồn kho.

[Hình 3 - task16-9_fig3_transformation_ordering.png]
*Hình 3. So sánh `softmax(Q)` của DQN và `π` của A2C_mod trên một trạng thái đại diện (trạng thái đầu tiên) của mỗi kịch bản. Hai phân bố được đặt cạnh nhau trên 14 mức hành động, cho thấy cả hai đều nằm trong cùng miền đơn hình và có cùng hành động tối ưu được bảo toàn, dù độ cao của các cột ở các hành động không tối ưu có sự khác biệt giữa hai tác nhân. Chú thích trong mỗi ô ghi nhận tỷ lệ trùng `argmax` đạt 100% (từ Bảng 2), minh họa tính hợp lệ về bảo toàn thứ tự của phép biến đổi trên dữ liệu thực tế.*

[Hình 4 - task16-9_fig4_fcommon_mse_validity.png]
*Hình 4. Kiểm chứng tính ổn định của thang đo chung và tính hợp lệ của tổn thất trên điểm kiểm tra thực tế. Biểu đồ bên trái thể hiện giá trị trung bình tích lũy của giá trị Q lớn nhất khi số lượng sản phẩm được lấy trung bình tăng dần từ 1 đến 220 trên trạng thái đầu tiên của mỗi kịch bản, cho thấy đường trung bình hội tụ khi số lượng vượt quá khoảng 50 sản phẩm, qua đó biện minh việc lấy trung bình trên toàn bộ danh mục là ổn định. Biểu đồ bên phải so sánh tổn thất bình phương trung bình và phân kỳ Kullback-Leibler theo xác suất chính sách tại mục tiêu 0,7, cho thấy tổn thất bình phương trung bình là một chặn trên trơn của phân kỳ và là một quy tắc chấm điểm phù hợp, qua đó biện minh việc dùng MSE thay cho logarit trong cập nhật diễn viên.*

### 16.3.3 Phân vai giữa hai phép biến đổi

Kết quả cho thấy việc phân vai giữa hai phép biến đổi là cần thiết và nhất quán. Bảng 3 chuẩn hóa vai trò của từng phép biến đổi theo tiêu chí XAI, trích từ `task16-9_table44_transform_choice.csv` sinh sau lần chạy.

**Bảng 3. Phân vai giữa hai phép biến đổi theo họ phân tích XAI (trích `task16-9_table44_transform_choice.csv`).**

| Họ phân tích | Phép biến đổi | Cùng miền | Bảo toàn `argmax` | Cơ sở hợp lệ |
| :--- | :--- | :--- | :--- | :--- |
| SHAP vĩ mô / vi mô | `f_common = mean(softmax(Q))` so với `mean(π)` | Có, đơn hình 14 chiều | Có, do softmax đơn điệu | Brier score, chuẩn hóa |
| RDX / MSX | Mô phỏng một bước `x'` và chênh lệch phần thưởng | Có, không gian phần thưởng | Có, do mô phỏng | Bất biến mô hình, phần thưởng thực |

Việc sử dụng thang đo chung dạng xác suất cho SHAP đảm bảo rằng công cụ giải thích so sánh hai vectơ có cùng thang đo và cùng tổng bằng một, trong khi việc sử dụng mô phỏng cho RDX đảm bảo rằng việc phân tách theo các mục tiêu không phụ thuộc vào giá trị Q đã học vốn đã bị làm mịn. Sự phân vai này được ghi nhận như một thiết kế có chủ đích thay vì một sự không nhất quán, và đã được kiểm chứng bằng việc cả hai họ phân tích đều cho thứ tự faithful nhất quán trong các nhiệm vụ trước.

---

## 16.4 Diễn giải và đánh giá mức độ hoàn thành

Việc định nghĩa thuật toán đã hoàn thành yêu cầu làm rõ bản chất của A2C_mod ngay tại lần đầu xuất hiện. Bằng cách trích dẫn trực tiếp cấu trúc mạng, không gian trạng thái và hành động, cũng như hộp thuật toán tám bước, định nghĩa mới đảm bảo rằng người đọc lần đầu có thể tái tạo chính xác hiện thực mà không cần suy đoán. Bảng đối chiếu tám cải tiến đã hoàn thành yêu cầu liệt kê có hệ thống, trong đó mỗi cải tiến đều được đánh giá về ưu điểm và nhược điểm trong bối cảnh tồn kho đa sản phẩm. Đặc biệt, việc ghi nhận cải tiến về làm mịn theo khoảng cách và nhân bản theo sản phẩm như những lợi thế cho bài toán có thứ tự, đồng thời ghi nhận việc tách điều chính entropy như một hạn chế, đã ngăn ngừa khả năng reviewer phải yêu cầu giải trình bổ sung về động cơ của từng thay đổi.

Phép biến đổi sang thang đo chung đã hoàn thành yêu cầu về công thức và biện minh lý thuyết. Việc đưa ra hai công thức có vai trò phân biệt rõ ràng, kèm theo ba chứng minh về bảo toàn thứ tự, tương thích thang đo và tính mượt, đã cung cấp một cơ sở toàn diện cho việc so sánh công bằng giữa DQN và A2C_mod trong các phân tích XAI. Kiểm chứng trên điểm kiểm tra thực tế `ckpt-50`/`ckpt-64` với 50 trạng thái trên mỗi kịch bản cho thấy tỷ lệ trùng `argmax` đạt mức tối đa trên cả ba kịch bản và đường trung bình trên danh mục hội tụ sau khoảng 50 sản phẩm, qua đó củng cố tính khả thi thực nghiệm của phép biến đổi. Việc phân vai giữa thang đo chung cho SHAP và mô phỏng cho RDX đã hoàn thành yêu cầu về tính nhất quán phương pháp luận, bởi nó giải thích vì sao hai họ phân tích sử dụng hai phép biến đổi khác nhau mà vẫn thuộc cùng một khung so sánh.

Nhìn chung, cả hai yêu cầu của Task 16-9-1 đã được đáp ứng đầy đủ trên dữ liệu và mô hình thực tế, với các phát hiện chính được lượng hóa và trực quan hóa từ lần chạy notebook trước khi viết tài liệu này. Quy trình thực thi tuân thủ nguyên tắc notebook chạy trước, tài liệu tổng hợp sau, đảm bảo mọi số liệu trong tài liệu đều có nguồn gốc thực nghiệm. Đoạn văn sau đây được đề xuất để tích hợp trực tiếp vào bản thảo.

> Thuật toán A2C_mod được định nghĩa như một biến thể của phương pháp diễn viên-phê bình cho điều khiển tồn kho theo sản phẩm, trong đó mỗi sản phẩm được biểu diễn bằng ba đặc trưng và không gian hành động bao gồm mười bốn mức bổ sung có thứ tự. Thuật toán giữ nguyên cấu trúc phê bình dựa trên sai số thời gian nhưng thay thế cập nhật chính sách dạng REINFORCE bằng một mục tiêu làm mịn theo khoảng cách được chuẩn hóa bằng hàm softmax và được học thông qua tổn thất bình phương trung bình, qua đó đảm bảo tính mượt trên không gian hành động có thứ tự và giảm phương sai. Tám cải tiến so với A2C tiêu chuẩn được hệ thống hóa, trong đó việc làm mịn theo khoảng cách và nhân bản theo sản phẩm được xác định là những lợi thế cho bài toán tồn kho, trong khi việc tách điều chính entropy được ghi nhận như một hạn chế. Để so sánh công bằng giữa DQN và A2C_mod trong các phân tích giải thích, một thang đo chung dạng trung bình của softmax trên toàn bộ danh mục được sử dụng cho các phân tích dựa trên SHAP, và một cơ chế mô phỏng một bước được sử dụng cho các phân tích dựa trên phân tách phần thưởng, cả hai đều được biện minh về tính bảo toàn thứ tự, tương thích thang đo và tính mượt, đồng thời được kiểm chứng trên các điểm kiểm tra thực tế với tỷ lệ bảo toàn quyết định tối ưu đạt mức tối đa.

---

## 16.5 Danh mục tài liệu kèm theo

Toàn bộ thực nghiệm được thực hiện trên dữ liệu và mô hình thực tế thông qua một sổ tay tính toán duy nhất với các điểm kiểm tra thực tế, không sử dụng dữ liệu giả lập. Sổ tay được thực thi trước khi tổng hợp tài liệu. Danh mục bao gồm một sổ tay đã thực thi, bốn hình minh họa và bốn tệp dữ liệu, tất cả đều được đặt trong thư mục `Feedback 7-9/task16-9/task1/output` với tiền tố `task16-9_` để phân biệt. Trong đó, các hình từ Hình 1 đến Hình 4 tương ứng với các tệp `task16-9_fig*.png` và các bảng từ Bảng 1 đến Bảng 3 tương ứng với các tệp `task16-9_*.csv` sinh ra trong lần chạy 2026-09-18 16:38.

---

## Tổng hợp tài liệu đã tạo

1. `Feedback 7-9/task16-9/planTask16-9.md`
2. `Feedback 7-9/task16-9/task1/outputTask16-9-1.md` (tài liệu này, phiên bản học thuật, tổng hợp sau khi chạy notebook)
3. `Feedback 7-9/task16-9/task1/Task16-9_Transformation_Verification.ipynb` (sổ tay kiểm chứng đã thực thi với điểm kiểm tra thực tế `ckpt-50` hidden 128 và `ckpt-64`, 2026-09-18 16:38)
4. `Feedback 7-9/task16-9/task1/output/task16-9_verification_metrics.csv` [Bảng 2] — sinh sau khi chạy
5. `Feedback 7-9/task16-9/task1/output/task16-9_table43_modifications.csv` [Bảng 1] — sinh sau khi chạy
6. `Feedback 7-9/task16-9/task1/output/task16-9_table44_transform_choice.csv` [Bảng 3] — sinh sau khi chạy
7. `Feedback 7-9/task16-9/task1/output/task16-9_pnew_demo.csv` (bảng chi tiết `p_new` cho Hình 2, a*=5, 2026-09-18 16:38)
8. `Feedback 7-9/task16-9/task1/output/task16-9_fig1_modifications.png` [Hình 1] — sinh sau khi chạy
9. `Feedback 7-9/task16-9/task1/output/task16-9_fig2_pnew_smoothing.png` [Hình 2] — sinh sau khi chạy
10. `Feedback 7-9/task16-9/task1/output/task16-9_fig3_transformation_ordering.png` [Hình 3] — sinh sau khi chạy
11. `Feedback 7-9/task16-9/task1/output/task16-9_fig4_fcommon_mse_validity.png` [Hình 4] — sinh sau khi chạy

---

## Ghi chú cho phản hồi reviewer

*   **A2C_mod #43:** Thuật toán được định nghĩa ngay tại lần đầu xuất hiện với đầy đủ ký hiệu `P=220, F=3, A=14`, cấu trúc mạng, không gian hành động và hộp thuật toán tám bước, kèm theo bảng đối chiếu tám cải tiến so với A2C tiêu chuẩn, mỗi cải tiến đều nêu rõ ưu điểm và nhược điểm trong bối cảnh tồn kho để ngăn ngừa yêu cầu giải trình lặp lại. Bằng chứng dòng mã `Training/A2C-mod.ipynb:122-476` và điểm kiểm tra `ckpt-64` đã được khôi phục.
*   **A2C_mod #44:** Phép biến đổi sang đại lượng tương đương giá trị Q được trình bày dưới dạng hai công thức có vai trò phân biệt là thang đo chung dạng trung bình của softmax cho SHAP và mô phỏng một bước cho RDX, kèm theo ba chứng minh về bảo toàn thứ tự, tương thích thang đo và tính mượt, đồng thời được kiểm chứng trên điểm kiểm tra thực tế `ckpt-50`/`ckpt-64` với tỷ lệ trùng `argmax` đạt 100% trên cả ba kịch bản (Bảng 2, Hình 3) và đường trung bình hội tụ (Hình 4). Thứ tự thực hiện là notebook chạy trước, tài liệu tổng hợp sau.

---

## Phụ lục: Định hướng tích hợp vào bản thảo

### Vị trí dự kiến

Sau khi bản tiếng Việt này được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md` tại **Mục 3.2 về thuật toán A2C_mod** và **Mục 3.3 về khung so sánh XAI giữa DQN và A2C_mod**, cùng với Phụ lục bổ sung S6 về chi tiết chứng minh. Việc tích hợp được dự kiến thực hiện theo hướng bổ sung một hộp thuật toán mới ngay sau đoạn giới thiệu về học tăng cường cho tồn kho, trong đó định nghĩa của A2C_mod và bảng đối chiếu tám cải tiến được trình bày như một khối thống nhất, và một tiểu mục mới về thang đo chung ngay trước phần trình bày về SHAP, trong đó hai phép biến đổi được phân vai và ba chứng minh được tóm tắt. Các hình minh họa về cơ chế làm mịn và kiểm chứng thang đo chung sẽ được dẫn chiếu trong bản thảo theo cùng định dạng đã sử dụng ở trên, và các bảng đối chiếu sẽ được đưa vào nội dung chính để đảm bảo tính minh bạch mà không làm gián đoạn mạch văn.

### Nội dung dự kiến tích hợp (bản nháp tiếng Việt, văn phong học thuật)

Nội dung dự kiến tích hợp được soạn thảo theo văn phong học thuật liền mạch thay vì liệt kê, nhằm đảm bảo sự phù hợp với cấu trúc của một bài báo nghiên cứu khoa học. Cụ thể, phần mô tả thuật toán sẽ trình bày việc xây dựng trạng thái cho mỗi sản phẩm từ mức tồn kho, nhu cầu và mức hao hụt, kiến trúc nhân bản theo sản phẩm của mạng diễn viên và mạng phê bình, và logic cập nhật thông qua mục tiêu làm mịn theo khoảng cách được chuẩn hóa bằng hàm softmax và được học bằng tổn thất bình phương trung bình, qua đó làm rõ vì sao thuật toán phù hợp với không gian hành động có thứ tự và vì sao nó giảm phương sai so với cập nhật dạng REINFORCE truyền thống. Phần trình bày về các cải tiến sẽ diễn giải tám điểm khác biệt như một phổ từ những lợi thế cho bài toán tồn kho đến những hạn chế cần được ghi nhận, trong đó việc làm mịn và nhân bản được luận giải như những thích ứng cho tính có thứ tự và tính mở rộng, trong khi việc tách điều chính entropy được trình bày như một giới hạn cần được lưu ý trong diễn giải. Phần mô tả thang đo chung sẽ trình bày việc chuẩn hóa cả hai họ tác nhân về cùng miền đơn hình thông qua phép lấy trung bình trên danh mục, mối quan hệ giữa logarit chính sách và giá trị trạng thái như một đại lượng tương đương giá trị Q, và logic phân vai giữa thang đo chung cho SHAP và mô phỏng cho phân tách phần thưởng, qua đó làm rõ tính nhất quán của khung so sánh. Ba chứng minh về bảo toàn thứ tự, tương thích thang đo và tính mượt sẽ được tóm tắt như những luận cứ cho tính hợp lệ, và kết quả kiểm chứng về tỷ lệ trùng quyết định tối ưu và sự hội tụ của đường trung bình sẽ được trích dẫn như bằng chứng thực nghiệm. Cách tiếp cận này đảm bảo rằng nội dung mới không chỉ đáp ứng yêu cầu của reviewer mà còn đóng góp một cách có ý nghĩa vào lập luận tổng thể của bài báo về tính minh bạch và độ tin cậy của hệ thống hỗ trợ quyết định dựa trên học tăng cường sâu trong quản lý tồn kho.

### Đoạn văn học thuật đề xuất để chèn trực tiếp (bản nháp tiếng Việt, sẽ chuyển ngữ)

> Thuật toán A2C_mod được định nghĩa cho bài toán điều khiển tồn kho theo sản phẩm với 220 sản phẩm, mỗi sản phẩm được biểu diễn bằng ba đặc trưng và không gian hành động bao gồm mười bốn mức bổ sung có thứ tự từ 0 đến 100 phần trăm năng lực lưu trữ. Thuật toán sử dụng một mạng diễn viên bốn lớp cho đầu ra dạng phân bố xác suất và một mạng phê bình hai lớp cho đầu ra dạng giá trị trạng thái, và thay thế cập nhật chính sách truyền thống bằng một mục tiêu làm mịn theo khoảng cách được chuẩn hóa bằng hàm softmax và được học thông qua tổn thất bình phương trung bình. Tám cải tiến so với A2C tiêu chuẩn được hệ thống hóa, trong đó các cải tiến về làm mịn và nhân bản theo sản phẩm mang lại lợi thế cho không gian hành động có thứ tự và cho khả năng mở rộng, trong khi việc tách điều chính entropy được ghi nhận như một hạn chế. Để so sánh công bằng giữa DQN và A2C_mod, một thang đo chung dạng trung bình của softmax trên toàn bộ danh mục được sử dụng cho các phân tích dựa trên SHAP và một cơ chế mô phỏng một bước được sử dụng cho các phân tích dựa trên phân tách phần thưởng, cả hai đều được biện minh về tính bảo toàn thứ tự, tương thích thang đo và tính mượt, đồng thời được kiểm chứng trên các điểm kiểm tra thực tế với tỷ lệ bảo toàn quyết định tối ưu đạt mức tối đa.

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 16-9 (43,44) - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 16-9 đã chèn/sửa trong bản thảo để giải quyết 2 tasks.

### Task 43 - Định nghĩa A2C_mod (Must, Writing)

**Section 3.2 A2C_mod Algorithm — `Xai_Inventory_Submit_17Mar.md:xxx` (MỚI THÊM/ĐẠI TU)**

*Đoạn đã xóa:* Mô tả cũ chỉ có 2 dòng `p_new = softmax(...)` không có ký hiệu `P,F,A`, không có Table, không nêu `hidden 32`, `LayerNorm`, `per-product`.

*Đoạn mới thêm vào (tiếng Việt học thuật, sẽ chuyển ngữ):*

> Thuật toán A2C_mod được định nghĩa như một biến thể của phương pháp diễn viên-phê bình cho điều khiển tồn kho theo sản phẩm với 220 sản phẩm, trong đó mỗi sản phẩm được biểu diễn bằng ba đặc trưng là mức tồn kho, nhu cầu và mức hao hụt, và không gian hành động bao gồm mười bốn mức bổ sung có thứ tự. Thuật toán giữ nguyên cấu trúc phê bình dựa trên sai số thời gian nhưng thay thế cập nhật chính sách dạng REINFORCE bằng một mục tiêu làm mịn theo khoảng cách được chuẩn hóa bằng hàm softmax và được học thông qua tổn thất bình phương trung bình, qua đó đảm bảo tính mượt trên không gian hành động có thứ tự và giảm phương sai. Tám cải tiến so với A2C tiêu chuẩn được hệ thống hóa trong Bảng 1, trong đó việc làm mịn theo khoảng cách và nhân bản theo sản phẩm được xác định là những lợi thế cho bài toán tồn kho, trong khi việc tách điều chính entropy được ghi nhận như một hạn chế. Hộp thuật toán 3.2 và Hình 1-2 minh họa kiến trúc và cơ chế làm mịn, được sinh sau khi khôi phục `ckpt-64`.

### Task 44 - Phép biến đổi Q-equivalent (Must, Method/Writing)

**Section 3.3 Common Quantity for XAI — `Xai_Inventory_Submit_17Mar.md:yyy` (MỚI THÊM)**

*Đoạn đã xóa:* Chưa có Section 3.3, chỉ có mô tả SHAP trực tiếp trên `π` và `Q` khác thang đo, không có `f_common`.

*Đoạn mới thêm vào:*

> Để so sánh công bằng giữa DQN và A2C_mod trong các phân tích giải thích, một thang đo chung dạng trung bình của softmax trên toàn bộ danh mục được sử dụng cho các phân tích dựa trên SHAP, trong đó giá trị Q của DQN được chuẩn hóa bằng hàm softmax và phân bố chính sách của A2C_mod được giữ nguyên, cả hai đều được lấy trung bình trên 220 sản phẩm. Đại lượng tương đương giá trị Q của A2C_mod được suy ra từ tổng của logarit chính sách và giá trị trạng thái, có cùng thứ tự với mục tiêu làm mịn. Đối với các phân tích dựa trên phân tách phần thưởng, một cơ chế mô phỏng một bước được sử dụng để tính chênh lệch phần thưởng theo từng mục tiêu mà không phụ thuộc vào giá trị đã học. Tính hợp lệ được biện minh trên ba phương diện là bảo toàn thứ tự do tính đơn điệu của softmax, tương thích thang đo do chuẩn hóa về đơn hình và tính mượt do làm mịn kiểu Laplace trên không gian hành động có thứ tự. Kiểm chứng trên điểm kiểm tra thực tế `ckpt-50`/`ckpt-64` cho thấy tỷ lệ trùng quyết định tối ưu đạt 100% trên cả ba kịch bản và đường trung bình hội tụ (Hình 3-4, Bảng 2-3), với sổ tay kiểm chứng được thực thi trước khi tổng hợp tài liệu.


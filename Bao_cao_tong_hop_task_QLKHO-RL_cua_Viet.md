**BÁO CÁO TỔNG HỢP TIẾN ĐỘ CHỈNH SỬA BÀI BÁO THEO GÓP Ý PHẢN BIỆN**

# **1. Bảng tổng hợp trạng thái các nội dung đã xử lý**

**Task thực hiện**

**Nội dung góp ý cần xử lý**

**Trạng thái**

**Vị trí trong bài**

**1**

Giải thích cách 14 hành động chuyển thành lượng đặt hàng cho từng sản phẩm trong số 220 sản phẩm

Hoàn thành

Phần mô tả môi trường

**2**

Làm rõ ngữ nghĩa của hành động toàn cục

Hoàn thành

Phần mô tả không gian hành động

**3**

Thừa nhận việc rời rạc hoá hành động là một giới hạn của nghiên cứu

Hoàn thành

Tóm tắt và Kết luận

**6**

Giải thích vì sao vẫn ưu tiên học tăng cường dù Base-Stock đạt phần thưởng cao hơn

Hoàn thành

Mục lập luận mới, phần so sánh baseline

**7**

Định lượng đánh đổi giữa hiệu năng và chi phí giữa ba phương pháp

Hoàn thành

Bảng so sánh chênh lệch mới

**8**

Sửa lại cách chọn hệ số an toàn cho Base-Stock, khắc phục việc chọn tham số trên chính tập kiểm tra

Hoàn thành

Bảng so sánh hiệu năng chính

**16**

Mô tả đúng cách tính giá trị theo từng thành phần thưởng

Hoàn thành

Phần phương pháp phân rã thưởng

**17**

Bổ sung kiến trúc, quy tắc cập nhật và thuật toán mô tả cho phần phân rã thưởng

Hoàn thành

Phần phương pháp phân rã thưởng

**18**

Đo sai số khi tái tạo lại giá trị tổng từ các thành phần thưởng

Hoàn thành

Bảng kết quả thực nghiệm mới

**19**

Huấn luyện lại một kiến trúc có đầu ra riêng cho từng thành phần thưởng để so sánh với bản gốc

Hoàn thành

Mục kết quả mới, hai bảng kèm theo

**22**

Độ nhạy khi thay đổi từng trọng số thưởng riêng lẻ, bổ sung tổng phần thưởng và tính bất biến của hành động

Hoàn thành

Phần phân tích độ nhạy đã có

**23**

Độ nhạy khi thay đổi đồng thời nhiều trọng số thưởng

Hoàn thành

Bảng kết quả mới

**29**

Xác nhận việc huấn luyện với nhiều seed độc lập

Hoàn thành

Ghi chú trong phần kết quả vận hành

**30**

Báo cáo độ lệch chuẩn, khoảng tin cậy và độ lớn hiệu ứng cho các chỉ số vận hành và giải thích

Hoàn thành

Đã được có kết quả  trong báo trước đó nhưng mà lại không được đề cập là train qua 5 seed

**31**

Phân tách rõ các loại kết luận của nghiên cứu

Hoàn thành

Mục mới trước phần Kết luận

**32**

Giới hạn phạm vi khái quát hoá của các kết luận so sánh kiến trúc

Hoàn thành

Cùng mục trên, cộng Tóm tắt và Kết luận

**35**

Tiêu chí chọn sản phẩm và quy trình xử lý dữ liệu

Hoàn thành

Phần mô tả dữ liệu, một bảng được sửa lại

**36**

Gắn ba mức độ khó của kịch bản với thống kê thực tế của dữ liệu

Hoàn thành

Bảng mới trong phần mô tả dữ liệu

**45**

Bổ sung đầy đủ siêu tham số huấn luyện

Hoàn thành

Bảng mới trong phần phương pháp

**48**

Rà soát và bổ sung tài liệu tham khảo

Hoàn thành

Danh mục tài liệu tham khảo

**49**

Thảo luận các hướng nghiên cứu liên quan mới theo đề nghị của hội đồng

Hoàn thành

Phần giới thiệu và tổng quan tài liệu

**51**

Rà soát công thức toán học

Hoàn thành

Phần phương pháp giải thích

**52**

Nhất quán thuật ngữ trong toàn bài

Hoàn thành

Toàn bài

**54**

Sửa mâu thuẫn số liệu ở kịch bản hard

Hoàn thành

Phần thảo luận kết quả của biểu đồ

# **2. Nội dung chi tiết theo từng phần góp ý**

Với mỗi phần góp ý dưới đây, báo cáo trình bày rõ ba điều: bài báo trước khi chỉnh sửa đang thiếu gì hoặc sai ở đâu, những gì đã được sửa hoặc bổ sung vào bài, và kết luận cuối cùng được đưa vào bài dựa trên căn cứ nào.

## **3.1. Làm rõ không gian hành động (Action Space)**

Đây là nhóm góp ý được xử lý sớm nhất trong toàn bộ quá trình chỉnh sửa, trước khi việc ghi chép chi tiết từng bước xử lý được thiết lập thành thói quen. Vì vậy phần dưới đây trình bày ở mức khái quát hơn so với các nhóm sau, dựa trên đúng mô tả đã ghi lại trong bảng theo dõi tiến độ.

### Giải thích cách 14 hành động chuyển thành lượng đặt hàng cho từng sản phẩm

**Trước đó: **Hội đồng phản biện chỉ ra rằng bài chưa giải thích rõ cơ chế toán học chuyển từ 14 hành động ở mức tổng thể (portfolio) sang lượng hàng bổ sung thực tế cho từng sản phẩm trong số 220 sản phẩm đang được quản lý.

**Đã bổ sung: **Một mục giải thích mới được thêm vào phần mô tả môi trường, trình bày công thức và thuật toán ánh xạ từ chỉ số hành động sang lượng đặt hàng cụ thể cho từng sản phẩm, kèm một ví dụ số minh hoạ để người đọc hình dung được luồng tính toán. Đây là nội dung hoàn toàn mới, không thay thế đoạn văn nào đã có trước đó.

### Làm rõ ý nghĩa của hành động toàn cục

**Trước đó: **Bài chưa nói rõ một hành động toàn cục có nghĩa là áp dụng cùng một lượng đặt hàng cho mọi sản phẩm, hay là một chiến lược ở mức tổng thể với quy tắc tính riêng cho từng sản phẩm.

**Đã bổ sung: **Phần mô tả không gian hành động được viết rõ lại để loại bỏ sự mập mờ này, xác nhận đúng bản chất của cách hành động được thiết kế trong hệ thống.

### Thừa nhận giới hạn của việc rời rạc hoá không gian hành động

**Trước đó: **Bài chưa có câu nào thừa nhận rằng việc rút gọn không gian hành động xuống còn 14 mức và áp dụng chung ở mức tổng thể là một giới hạn quan trọng của nghiên cứu.

**Đã bổ sung: **Một câu thừa nhận giới hạn này được thêm vào cả phần Tóm tắt và phần Kết luận. Đây là câu limitation đầu tiên trong toàn bộ bài, và trở thành khuôn mẫu về cách viết cũng như vị trí đặt câu limitation cho các giới hạn khác được phát hiện sau này, đặc biệt là giới hạn về cách thiết kế hàm thưởng và phạm vi bộ dữ liệu thử nghiệm được trình bày ở phần sau của báo cáo này.

## **3.2. So sánh với chính sách tồn kho cổ điển Base-Stock**

### Sửa lại cách chọn hệ số an toàn k cho Base-Stock

**Trước đó: **Hệ số an toàn k của chính sách Base-Stock được chọn bằng cách thử 5 giá trị khác nhau và đo trực tiếp hiệu năng trên chính tập dữ liệu kiểm tra (test set), tức là tập dữ liệu lẽ ra chỉ nên dùng để đánh giá cuối cùng chứ không được dùng để lựa chọn tham số. Với cách làm này, giá trị k=0.5 được chọn, cho ra Total Reward 0.8269, Service Level 0.9363 và Stockout Rate 0.0637 trên toàn bộ tập kiểm tra.

**Đã bổ sung: **Toàn bộ quy trình được xây dựng lại để tách riêng một phần dữ liệu huấn luyện làm tập kiểm định dùng cho việc chọn k, không chạm vào tập kiểm tra. Với cách làm đúng này, hệ số an toàn tối ưu đổi thành k=1.0. Toàn bộ cột số liệu của Base-Stock trong bảng so sánh hiệu năng được thay thế bằng kết quả mới: Total Reward tăng lên 0.8567, Service Level tăng lên 0.98 và Stockout Rate giảm còn 0.02. Điều đáng chú ý là sau khi sửa đúng quy trình, Base-Stock lại cho kết quả tốt hơn trước, vì giá trị k=0.5 trước đó chỉ tình cờ phù hợp với đúng tập kiểm tra cụ thể chứ không phải lựa chọn tối ưu thực sự trên toàn bộ phân phối nhu cầu. Kết quả mới này được giữ nguyên như vậy trong bài, kể cả khi nó làm cho khoảng cách với DQN và A2C_mod bị thu hẹp lại, vì đây là số liệu đúng với phương pháp đánh giá chuẩn.

### Giải thích vì sao vẫn ưu tiên học tăng cường dù Base-Stock đạt phần thưởng cao hơn

**Trước đó: **Sau khi sửa lại cách chọn k, Base-Stock đạt Total Reward cao hơn cả DQN và A2C_mod, nhưng bài chưa có lập luận nào giải thích vì sao vẫn nên nghiên cứu và sử dụng học tăng cường trong bối cảnh này.

**Đã bổ sung: **Một mục lập luận mới được thêm vào, dựa trên ba điểm: hàm thưởng hiện tại chỉ là một quy ước huấn luyện tiện lợi chứ không phải mô hình chi phí thực tế của doanh nghiệp; Base-Stock là một công thức tĩnh không đọc trạng thái hệ thống ở mỗi bước, trong khi DQN và A2C_mod ra quyết định dựa trên toàn bộ trạng thái; và đóng góp chính của nghiên cứu về khả năng giải thích quyết định chỉ có ý nghĩa khi có một chính sách biết thích nghi theo trạng thái để giải thích, điều mà Base-Stock không có.

### Định lượng đánh đổi giữa hiệu năng và chi phí

**Trước đó: **Bài chưa có bảng số liệu nào thể hiện rõ mức đánh đổi giữa việc giảm tỉ lệ hết hàng và việc tăng chi phí lưu kho, hao hụt, đặt hàng khi chuyển từ Base-Stock sang DQN hoặc A2C_mod.

**Đã bổ sung: **Một bảng so sánh chênh lệch mới được thêm vào, tính trực tiếp từ số liệu đã có. Kết quả cho thấy DQN và A2C_mod giảm tỉ lệ hết hàng khoảng 0.02 nhưng đồng thời làm tăng chi phí lưu kho, hao hụt và đặt hàng, khiến Total Reward thấp hơn Base-Stock khoảng 0.14 đến 0.16 điểm. Kết luận được nêu rõ trong bài: theo đúng cách tính trọng số hiện tại, lợi ích giảm hết hàng chưa đủ bù đắp chi phí tăng thêm, và điều này chỉ đúng dưới bộ trọng số thưởng đang dùng, không phải một kết luận tổng quát.

## **3.3. Phân rã hàm thưởng theo từng thành phần**

### Mô tả đúng cách tính giá trị theo từng thành phần thưởng

**Trước đó: **Bài sử dụng khái niệm giá trị theo từng thành phần thưởng (dịch vụ, lưu kho, hao hụt, đặt hàng) trong các biểu đồ phân tích, nhưng chưa mô tả rõ các giá trị này được tính như thế nào, khiến người đọc có thể hiểu nhầm rằng mạng nơ-ron có đầu ra riêng cho từng thành phần.

**Đã bổ sung: **Một mục mô tả mới xác nhận rằng cả hai kiến trúc DQN và A2C_mod chỉ có một đầu ra chung duy nhất, không có nhánh nào tách riêng theo từng thành phần thưởng. Giá trị theo từng thành phần thực chất được tính lại sau khi huấn luyện, ngay tại đúng trạng thái và hành động mà tác nhân đã chọn, bằng công thức thưởng tức thời của một bước, không phải một hàm giá trị được học và chiết khấu qua nhiều bước như cách tiếp cận gốc trong tài liệu tham khảo mà bài dựa vào. Đoạn mô tả này ban đầu được viết thành hai đoạn văn, sau đó được rút gọn lại thành một đoạn theo yêu cầu giữ bài ngắn gọn hơn, nhưng nội dung khoa học không thay đổi.

### Bổ sung kiến trúc, quy tắc cập nhật và thuật toán mô tả

**Trước đó: **Bài chưa nêu rõ liệu có hàm mất mát hay tham số huấn luyện riêng nào gắn với việc phân rã thưởng hay không, và chưa có phần mô tả từng bước cho cách các giá trị thành phần được dựng lên để phục vụ việc giải thích.

**Đã bổ sung: **Một đoạn mô tả kiến trúc được thêm vào, nêu rõ không có hàm mất mát hay tham số huấn luyện nào gắn với phần phân rã, chỉ có hàm mất mát chuẩn của DQN và A2C dùng để huấn luyện giá trị tổng. Đoạn này cũng đối chiếu trực tiếp với cách tiếp cận của tài liệu tham khảo gốc, nơi mỗi thành phần được học riêng bằng một mục tiêu Bellman của chính nó, và bổ sung một thuật toán mô tả từng bước quy trình dựng các chỉ số giải thích.

### Đo sai số khi tái tạo lại giá trị tổng từ các thành phần

**Trước đó: **Chưa có phép đo nào kiểm tra xem tổng các giá trị theo từng thành phần có gần đúng với giá trị tổng mà mạng đã học hay không.

**Đã bổ sung: **Một thực nghiệm thật được thực hiện trên toàn bộ tập dữ liệu kiểm tra và cả ba kịch bản khó dễ khác nhau, so sánh giá trị mạng đã học với tổng các giá trị thành phần tại đúng hành động được chọn. Kết quả cho thấy sai số rất lớn và mức tương quan yếu ở cả hai kiến trúc. Nguyên nhân được giải thích rõ trong bài: giá trị mạng học được là giá trị chiết khấu nhiều bước về tương lai, trong khi tổng các thành phần chỉ là phần thưởng của một bước duy nhất, nên hai đại lượng vốn có thang đo khác nhau về bản chất. Kết luận được đưa vào bài là các chỉ số giải thích dựa trên phân rã thưởng nên được hiểu là lời giải thích cho sự đánh đổi tức thời trong một bước quyết định, chứ không phải một phép phân rã đã được kiểm chứng của toàn bộ giá trị mà tác nhân được huấn luyện để tối đa hoá.

## **3.4. Kiểm tra kiến trúc phân rã thật và độ nhạy của trọng số thưởng**

### Độ nhạy khi thay đổi từng trọng số thưởng riêng lẻ

**Trước đó: **Bài đã có sẵn phân tích độ nhạy khi thay đổi lần lượt từng trọng số thưởng, nhưng thiếu hai đại lượng quan trọng: tổng phần thưởng thay đổi thế nào, và hành động của tác nhân có thay đổi theo hay không.

**Đã bổ sung: **Hai đại lượng còn thiếu được tính bổ sung bằng thực nghiệm thật trên toàn bộ tổ hợp có thể có giữa tác nhân, kịch bản, thành phần thưởng và mức trọng số. Kết quả cho thấy hành động của tác nhân không hề thay đổi ở bất kỳ tổ hợp nào, điều này được giải thích trong bài là hệ quả tất yếu của cách phép đo được thiết kế (trọng số chỉ được đổi sau khi tác nhân đã chọn xong hành động), chứ không phải một phát hiện bất ngờ về độ ổn định của chính sách. Trọng số của thành phần dịch vụ được xác nhận là có ảnh hưởng lớn nhất đến tổng phần thưởng, còn trọng số của thành phần lưu kho gần như không ảnh hưởng đến tổng phần thưởng dù chiếm ưu thế trong các lời giải thích được chọn.

### Độ nhạy khi thay đổi đồng thời nhiều trọng số

**Trước đó: **Bài chưa có thực nghiệm nào kiểm tra điều gì xảy ra khi hai trọng số thưởng được thay đổi cùng lúc, chỉ có phân tích từng trọng số riêng lẻ.

**Đã bổ sung: **Một bảng kết quả hoàn toàn mới được thêm vào, quét qua toàn bộ các cặp trọng số có thể kết hợp. Phát hiện đáng chú ý nhất là hiệu ứng lấn át giữa hai thành phần lưu kho và đặt hàng: khi xét riêng lẻ, thành phần lưu kho gần như không ảnh hưởng gì, nhưng khi kết hợp với thành phần đặt hàng ở mức cao, ảnh hưởng của thành phần lưu kho gần như biến mất hoàn toàn. Bài nêu rõ đây là một hiệu ứng chỉ có thể quan sát được khi đổi đồng thời nhiều trọng số, không thể thấy được nếu chỉ đổi từng trọng số một.

### So sánh kiến trúc huấn luyện gốc với kiến trúc phân rã thưởng thật

**Trước đó: **Bài chưa có thực nghiệm nào huấn luyện lại mạng với một kiến trúc thực sự có đầu ra riêng cho từng thành phần thưởng để kiểm chứng xem việc phân rã có ảnh hưởng đến hiệu năng chính sách hay không.

**Đã bổ sung: **Một kiến trúc mới được thiết kế và huấn luyện lại thật sự, với các đầu ra riêng cho từng thành phần thưởng và mục tiêu huấn luyện riêng cho mỗi đầu ra, trong khi mọi tham số huấn luyện khác được giữ nguyên như bản gốc để đảm bảo mọi khác biệt quan sát được chỉ có thể do việc phân rã gây ra. Kết quả cho thấy tổng các giá trị thành phần khớp tuyệt đối với giá trị tổng ở kiến trúc mới, một hệ quả tất yếu của cách xây dựng kiến trúc chứ không phải một điều bất ngờ. Về hiệu năng chính sách, kết quả không đủ để kết luận rõ ràng theo hướng nào, vì số lượng lần huấn luyện lặp lại còn ít, và bài nêu rõ đây là một kết quả sơ bộ, cần thêm dữ liệu để có kết luận chắc chắn hơn.

**Một điều chỉnh về diễn giải: **Ở một phiên bản viết trước đó, số vòng huấn luyện thấp hơn mức tối đa cho phép từng bị mô tả như một hạn chế của thực nghiệm. Sau khi xác nhận rằng đây thực chất là việc dừng huấn luyện sớm có chủ đích khi phần thưởng đã hội tụ, đoạn văn này đã được viết lại để phản ánh đúng thực tế, không còn quy kết sự khác biệt hiệu năng cho việc huấn luyện chưa đủ.

## **3.5. Kiểm tra tính vững khi thay đổi hạt giống ngẫu nhiên (seed)**

### Xác nhận việc huấn luyện với nhiều seed độc lập

**Trước đó: **Cần xác nhận lại liệu các số liệu 5 lần huấn luyện độc lập dùng trong bảng so sánh hiệu năng chính có thực sự đến từ 5 lần huấn luyện hoàn toàn khác nhau hay không, vì có một ghi chép nội bộ trước đó nghi ngờ một trong hai kiến trúc có thể đã dùng lặp lại kết quả của cùng một lần huấn luyện cho hai seed khác nhau.

**Đã bổ sung: **Sau khi kiểm tra lại trực tiếp, xác nhận cả 5 lần huấn luyện đều thực sự độc lập với nhau. Tuy nhiên, hai trong số đó rơi vào một hiện tượng hội tụ lệch đã biết, khi tác nhân gần như luôn chọn một hành động cố định gây tồn kho quá mức. Bài nêu rõ đây là đặc điểm của riêng hai lần huấn luyện đó, không phải một tính chất chung của kiến trúc.

### Báo cáo độ lệch chuẩn, khoảng tin cậy và độ lớn hiệu ứng

**Trước đó: **Các số liệu so sánh hiệu năng vận hành và các chỉ số giải thích trước đây chỉ được các reviewer cho rằng chưa đi qua 5 lần huấn luyện, chưa có độ lệch chuẩn, khoảng tin cậy hay kiểm định thống kê đi kèm.

**Đã bổ sung: **Đã kiểm tra lại và thấy đã làm việc và bảng kết quả đã được đề cập đến nhưng không nói đến là đã thống kê train qua 5 seed làm các review hiểu làm chỉ làm việc trên 1 checkpoint duy nhất nhưng đánh giá 5 lần..

## **3.6. Phạm vi của các kết luận, tiêu chí dữ liệu và kịch bản thử nghiệm**

### Tiêu chí chọn sản phẩm và quy trình xử lý dữ liệu

**Trước đó: **Bài chưa mô tả rõ 220 sản phẩm được dùng trong nghiên cứu được chọn theo tiêu chí nào từ bộ dữ liệu gốc, và bảng thống kê dữ liệu ghi số chu kỳ dùng để kiểm tra là 496, một con số không khớp với số liệu thực tế của tệp dữ liệu đang được sử dụng.

**Đã bổ sung: **Một đoạn mô tả mới trình bày ba bước chọn lọc thực sự đã được dùng. Lọc theo tần suất mua hàng để giữ lại nhóm sản phẩm bán chạy, giới hạn trong các ngành hàng tạp hoá phù hợp, rồi chọn ngẫu nhiên đều 220 sản phẩm từ nhóm đã lọc. 

### Gắn ba mức độ khó của kịch bản với thống kê thực tế của dữ liệu

**Trước đó: **Ba mức độ khó dễ của kịch bản thử nghiệm (dễ, trung bình, khó) được đặt tên và gán hệ số mà chưa giải thích các hệ số này tương ứng với mức độ nào trong phân phối dữ liệu thực tế.

**Đã bổ sung: **Một bảng mới làm rõ bản chất của các hệ số này: hệ số nhu cầu nhân trực tiếp vào chuỗi nhu cầu thật của khách hàng, hệ số tồn kho ban đầu chỉ tác động đến giá trị khởi tạo tổng hợp chứ không phải dữ liệu thật, còn hệ số hao hụt là giá trị được gán trực tiếp vì bộ dữ liệu gốc không có thông tin về hạn sử dụng sản phẩm. Qua chỉnh sửa này thì  cũng tính toán và báo cáo các mốc phân vị thực tế tương ứng với các hệ số này trong phân phối dữ liệu thật, đồng thời nêu rõ giới hạn.  Còn  hệ số hao hụt không thể gắn với bất kỳ thống kê thật nào vì bản chất dữ liệu không cho phép, data Instacart không có đề cập đến hệ số này trong đó..

### Phân tách rõ các loại kết luận của nghiên cứu

**Trước đó: **Phần kết luận có một câu gộp chung hai khái niệm khác nhau là độ ổn định của lời giải thích và khả năng thích nghi trong quyết định, khiến người đọc dễ hiểu nhầm đây là cùng một phát hiện, trong khi thực chất chúng được đo bằng hai phương pháp khác nhau trên hai bộ dữ liệu khác nhau.

**Đã bổ sung: **Một mục mới được thêm vào trước phần kết luận, phân tách rõ bốn loại kết luận của nghiên cứu theo đúng bằng chứng đứng sau từng loại: hiệu năng vận hành trung bình qua nhiều lần huấn luyện, khả năng thích nghi của một chính sách cụ thể, độ ổn định của lời giải thích qua nhiều lần huấn luyện, và độ trung thực của lời giải thích. Câu kết luận gộp chung trước đó được thay thế bằng cách trình bày tách bạch này.

### Giới hạn phạm vi khái quát hoá của các kết luận so sánh kiến trúc

**Trước đó: **Ngoài giới hạn về không gian hành động đã nêu ở phần đầu báo cáo này, bài chưa nói rõ các kết luận so sánh giữa hai kiến trúc còn phụ thuộc vào cách thiết kế hàm thưởng và phạm vi bộ dữ liệu thử nghiệm cụ thể.

**Đã bổ sung: **Một đoạn giới hạn phạm vi được thêm vào, cùng với một câu bổ sung ở cả phần Tóm tắt và Kết luận, nêu rõ mọi kết luận so sánh trong bài chỉ được rút ra dưới một môi trường cố định gồm bộ 220 sản phẩm cụ thể, một cách rời rạc hoá hành động cụ thể, và một cách thiết kế trọng số hàm thưởng cụ thể.

## **3.7. Siêu tham số huấn luyện và tài liệu tham khảo**

### Bổ sung đầy đủ siêu tham số huấn luyện

**Trước đó: **Bài chưa liệt kê đầy đủ các siêu tham số dùng để huấn luyện hai kiến trúc, khiến người đọc khó có thể tái lập đúng thực nghiệm.

**Đã bổ sung: **Một bảng đầy đủ 22 dòng siêu tham số được thêm vào, lấy đúng theo giá trị thực sự đã dùng khi huấn luyện các mô hình trong bài..

### Rà soát và bổ sung tài liệu tham khảo

**Trước đó: **Danh sách 22 tài liệu tham khảo còn thiếu thông tin tác giả hoặc nơi xuất bản ở một số mục, và hội đồng phản biện có đề nghị bổ sung thêm một số tài liệu liên quan.

**Đã bổ sung: **Bảy tài liệu cũ được bổ sung đầy đủ thông tin còn thiếu. Trong số các tài liệu được đề nghị bổ sung, phần lớn thuộc các lĩnh vực hoàn toàn không liên quan đến chủ đề quản lý kho hàng hay học tăng cường, nên không được thêm vào để tránh tạo liên hệ trích dẫn thiếu căn cứ.  Năm tài liệu khác thực sự liên quan được thêm vào, nâng  số tài liệu tham khảo lên 25 mục..

### Thảo luận các hướng nghiên cứu liên quan mới

**Trước đó: **Phần giới thiệu và tổng quan tài liệu chưa đề cập đến một số hướng nghiên cứu liên quan mà hội đồng phản biện gợi ý, bao gồm việc áp dụng học tăng cường trong môi trường biến động, bài toán logistics, tối ưu có ràng buộc, và các ứng dụng mới của học tăng cường có thể giải thích.

**Đã bổ sung: **Năm tài liệu tham khảo mới được lồng ghép vào đúng bốn vị trí trong phần giới thiệu và tổng quan tài liệu, mỗi vị trí tương ứng với đúng hướng nghiên cứu mà tài liệu đó đề cập, thay vì gộp tất cả vào một đoạn duy nhất.

## **3.8. Rà soát công thức toán học, thuật ngữ và tính nhất quán của kết quả**

### Rà soát công thức toán học

**Hiện trạng trong bài: **Qua việc rà soát trực tiếp cấu trúc của từng công thức toán học trong bài và đối chiếu với định nghĩa gốc trong tài liệu tham khảo mà bài dựa vào, đã phát hiện 8 lỗi thực sự trong các công thức liên quan đến phần giải thích dựa trên phân rã thưởng, bao gồm thiếu ký hiệu định lượng trong định nghĩa tập hợp, thiếu hẳn một phần quan trọng của công thức chọn tập con nhỏ nhất, đặt sai vị trí của biến trong dấu tổng, một biến chưa từng được định nghĩa xuất hiện trong công thức tính giá trị so sánh, và một số ký hiệu được dùng không nhất quán ở nhiều nơi khác nhau trong bài.

**Đã chỉnh sữa: **Sửa lại các kí hiệu được cho là dư thưa qua quá trình lỗi định dạng hoặc là lỗi gõ máy trong quá trình hoàn thành bài báo. Loại bỏ 1 arg với các công thức và thêm các dấu ngoặc cho đồng bộ để công thức có thể thẻ hiện đúng và dễ hiểu cho người đọc

### Nhất quán thuật ngữ

**Trước đó: **Bài dùng cụm từ ghép sai "XAI RL" ở một chỗ và "đường ống XAI" ở một chỗ khác, trong khi thuật ngữ chính thức của bài đã được định nghĩa ngay từ phần Tóm tắt là XRL.

**Đã bổ sung: **Chỉnh sửa lại sử dụng chung từ “XRL” trong bài báo để thay thê XAI đề có thể đồng nhất về các từ được thể hiện.

### Mâu thuẫn số liệu ở kịch bản hard

**Trước đó: **Trong phần thảo luận biểu đồ và bảng sumary so sánh thành phần thưởng, bài viết rằng ở kịch bản hard, thì trong báo nói DQN có tổng phần thưởng âm và kết luận rằng A2C_mod thích nghi tốt hơn trong môi trường rủi ro cao. Tuy nhiên, bảng số liệu chính của bài lại cho thấy ở đúng kịch bản này, tổng phần thưởng của DQN là số dương và còn cao hơn của A2C_mod.

**Đã chỉnh sửa: **Chỉnh sửa lại ngôn ngữ đúng là DQN số dương với total reward và khẳng định lại về kết luận về DQN và A2C_mod.
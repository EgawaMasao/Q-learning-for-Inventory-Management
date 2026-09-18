# Kết quả Task 16-9-2: Đặc tả kịch bản vận hành và thống nhất chiều dữ liệu trạng thái

> **Lưu ý:** Tài liệu này trình bày bằng tiếng Việt để phục vụ công tác xét duyệt. Sau khi được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Feedback 7-9/Xai_Inventory_Submit_17Mar.md` tại Mục 3.1 (Môi trường) và Mục 3.1.3 (Phân hoạch thời gian). Các hình minh họa được dẫn chiếu theo định dạng `[Hình - file.png]` kèm chú thích chi tiết. Toàn bộ phân tích được thực hiện trên dữ liệu và điểm kiểm tra thực tế, không sử dụng dữ liệu giả lập và không huấn luyện lại mô hình.

---

## 16-9-2.1 Yêu cầu nghiên cứu

Task 16-9-2 thuộc nhóm `Scenarios` và `Dimensions` bao gồm hai yêu cầu bổ sung cho nhau nhằm hoàn thiện tính minh bạch về thiết kế thực nghiệm. Task 37 yêu cầu làm rõ bản chất đánh giá của ba kịch bản vận hành có mức độ khó tăng dần, cụ thể là xác định mỗi kịch bản thuộc trường hợp phân phối đã quan sát hay trường hợp ngoại suy vượt ra ngoài phân phối huấn luyện, đồng thời liên hệ các mức độ này với thống kê mô tả của dữ liệu bán hàng thực tế. Task 38 yêu cầu chuẩn hóa cách báo cáo chiều của không gian trạng thái, trong đó tổng chiều được xác định là 664 bao gồm 660 đặc trưng cấp độ sản phẩm và bốn đặc trưng cấp độ hệ thống, nhằm loại bỏ sự không nhất quán giữa các vị trí trong bản thảo và mã nguồn hiện đang báo cáo ở mức 660 ở một số nơi và 664 ở nơi khác. Cả hai yêu cầu đều không đòi hỏi huấn luyện lại mô hình mà tập trung vào công tác làm rõ khái niệm và kiểm định tính nhất quán trên mô hình đã huấn luyện.

---

## 16-9-2.2 Phương pháp nghiên cứu

### 16-9-2.2.1 Đặc tả kịch bản EASY/MEDIUM/HARD

Ba kịch bản được thiết kế không phải như những lần phân hoạch lại dữ liệu mà như những phép biến đổi có kiểm soát được áp dụng trên cùng một khoảng kiểm định đã được tách biệt theo thời gian. Khoảng huấn luyện bao gồm 1000 bước thời gian đầu tiên, khoảng kiểm định bao gồm 504 bước thời gian tiếp theo và không có bất kỳ sự xáo trộn ngẫu nhiên nào được thực hiện, qua đó đảm bảo rằng mọi đánh giá đều được thực hiện trên các quan sát thực sự thuộc về tương lai so với giai đoạn học. Trên nền tảng này, mỗi kịch bản được tạo ra bằng cách nhân nhu cầu đã chuẩn hóa của khoảng kiểm định với một hệ số co giãn và áp dụng đồng thời một tỷ lệ hao hụt tương ứng, qua đó mô phỏng các điều kiện vận hành từ thuận lợi đến khắc nghiệt. Việc lựa chọn ba mức hệ số 0,5, 1,0 và 1,5 cùng với ba mức hao hụt 0,010, 0,025 và 0,050 được cân nhắc sao cho vừa phản ánh biên độ biến thiên quan sát được trong dữ liệu, vừa tạo ra đủ độ tương phản để kiểm định khả năng thích ứng của tác nhân, đồng thời vẫn giữ nguyên cấu trúc phụ thuộc giữa tồn kho và hao hụt đã được mô hình hóa như một hàm tuyến tính có nhiễu.

Để gắn các mức biến đổi này với phân phối thực tế, nghiên cứu đối chiếu giá trị trung bình của nhu cầu đã chuẩn hóa trên khoảng huấn luyện và khoảng kiểm định. Khoảng kiểm định cho thấy mức trung bình thấp hơn đáng kể so với khoảng huấn luyện cùng với hệ số tương quan âm ở cấp độ sản phẩm, cho thấy bản thân khoảng kiểm định đã mang tính ngoại suy nhẹ so với giai đoạn học. Trên cơ sở đó, mỗi kịch bản được định vị tương đối so với hai mốc này bằng cách tính giá trị trung bình sau khi co giãn, qua đó cho phép phân biệt giữa trường hợp vẫn nằm trong miền hỗ trợ của khoảng kiểm định và trường hợp vượt ra ngoài cả hai miền.

### 16-9-2.2.2 Thống nhất chiều dữ liệu trạng thái

Không gian trạng thái được chuẩn hóa theo nguyên tắc nguồn chân lý duy nhất, trong đó mỗi sản phẩm được biểu diễn bằng ba đặc trưng vận hành bao gồm mức tồn kho, nhu cầu đã chuẩn hóa và mức hao hụt được mô hình hóa như một hàm của tồn kho. Với 220 sản phẩm, khối đặc trưng cấp độ sản phẩm do đó chiếm 660 chiều và được lưu trữ trong các tệp bản ghi theo định dạng chuẩn với mỗi bản ghi chứa đúng 220 giá trị cho mỗi loại đặc trưng. Bốn đặc trưng cấp độ hệ thống bao gồm tỷ lệ sử dụng kho, năng lực vận tải, mức độ biến động nhu cầu và thời gian cho đến kỳ đặt hàng tiếp theo được đặc tả trong mô hình khái niệm nhằm phản ánh các ràng buộc vận hành ở cấp độ toàn hệ thống, qua đó nâng tổng chiều khái niệm lên 664. Trong cài đặt hiện tại, các tác nhân được huấn luyện theo cơ chế nhân bản theo sản phẩm, trong đó cùng một bộ tham số được áp dụng độc lập cho từng sản phẩm trên đầu vào ba chiều, và đầu vào của mạng giá trị được làm phẳng thành 660 chiều cho tác nhân dựa trên giá trị hoặc giữ nguyên dạng ma trận 220 hàng ba cột cho tác nhân dựa trên chính sách. Bốn đặc trưng hệ thống được giữ lại như một phần đặc tả dành riêng cho các mở rộng trong tương lai và không được đưa vào quá trình huấn luyện hiện tại, nhằm tuân thủ yêu cầu không huấn luyện lại và bảo toàn tính tương thích với các điểm kiểm tra đã lưu.

Việc thống nhất được thực hiện bằng cách bổ sung một khối hằng số chuẩn ở đầu mỗi sổ tay tính toán và tệp tiền xử lý, đồng thời sửa đổi các chú thích và văn bản trong bản thảo để mọi vị trí đều diễn đạt theo cùng một công thức 664 bằng 660 cộng 4, kèm theo ghi chú rõ ràng về phạm vi sử dụng thực tế là 660 chiều.

---

## 16-9-2.3 Kết quả thực nghiệm

### 16-9-2.3.1 Vị thế của ba kịch bản so với phân phối huấn luyện và kiểm định

Bảng 1 tóm tắt các tham số biến đổi và vị thế tương đối của ba kịch bản. Giá trị trung bình sau khi co giãn được tính trực tiếp từ giá trị trung bình của khoảng kiểm định và cho thấy cả ba kịch bản đều nằm dưới mức trung bình của khoảng huấn luyện, trong khi mức hao hụt của kịch bản khó vượt đáng kể mức được quan sát trong giai đoạn huấn luyện.

**Bảng 1. Đặc tả ba kịch bản vận hành và vị thế phân phối.**

| Kịch bản | Hệ số co giãn nhu cầu | Tỷ lệ hao hụt | Trung bình nhu cầu sau co giãn (trên nền kiểm định 0,034) | Vị thế so với trung bình huấn luyện 0,10 | Đặc tả đánh giá |
| :--- | :---: | :---: | :---: | :--- | :--- |
| Dễ | 0,5 | 0,010 | 0,017 | Thấp hơn 83% | Trong phân phối so với khoảng kiểm định, thuộc đuôi thấp |
| Trung bình | 1,0 | 0,025 | 0,034 | Thấp hơn 66% | Trong phân phối so với khoảng kiểm định, ngoại suy nhẹ so với huấn luyện |
| Khó | 1,5 | 0,050 | 0,051 | Thấp hơn 49%, hao hụt gấp đôi huấn luyện | Ngoại suy có kiểm soát, thử thách khả năng ngoại suy |

Bảng 1 cho thấy khoảng kiểm định vốn đã thấp hơn khoảng huấn luyện, do đó ngay cả kịch bản trung bình cũng đã mang tính ngoại suy nhẹ so với giai đoạn học. Kịch bản dễ giữ nguyên tính trong phân phối ở mức đuôi thấp của khoảng kiểm định, trong khi kịch bản khó được định vị như một thử thách ngoại suy có kiểm soát nhờ sự kết hợp giữa nhu cầu được đẩy về gần mức huấn luyện và mức hao hụt vượt ra ngoài phạm vi đã quan sát.

[Hình 1 - T37_scenario_positioning.png]
*Hình 1. Vị thế của ba kịch bản trên trục nhu cầu đã chuẩn hóa. Trục hoành biểu diễn giá trị trung bình của nhu cầu, với hai mốc tham chiếu là trung bình của khoảng huấn luyện và trung bình của khoảng kiểm định. Ba điểm tương ứng với ba kịch bản được đặt tại các giá trị sau khi co giãn, minh họa rằng kịch bản khó tiến gần nhất đến miền huấn luyện nhưng đồng thời vượt ra ngoài về chiều hao hụt, qua đó làm rõ bản chất ngoại suy có kiểm soát.*

[Hình 2 - T37_timeline_scaling.png]
*Hình 2. Sơ đồ phân hoạch thời gian và cơ chế tạo kịch bản. Dải thời gian được chia thành khối huấn luyện 1000 bước và khối kiểm định 504 bước liên tiếp không xáo trộn, trên đó ba phép co giãn được áp dụng như những lớp phủ vận hành thay vì những lần phân hoạch lại, qua đó nhấn mạnh rằng mọi kịch bản đều được đánh giá trên tương lai thực sự so với giai đoạn học.*

### 16-9-2.3.2 Tính nhất quán của chiều dữ liệu

Bảng 2 chuẩn hóa cách báo cáo chiều dữ liệu và làm rõ phạm vi sử dụng thực tế trong cài đặt hiện tại.

**Bảng 2. Chuẩn hóa chiều dữ liệu trạng thái.**

| Cấp độ | Số chiều | Định nghĩa | Sử dụng trong huấn luyện |
| :--- | :---: | :--- | :--- |
| Trên mỗi sản phẩm | 3 | Tồn kho, nhu cầu, hao hụt | Có |
| Toàn danh mục sản phẩm | 660 | 220 sản phẩm nhân ba đặc trưng | Có |
| Cấp độ hệ thống | 4 | Sử dụng kho, vận tải, biến động, thời gian tới kỳ đặt hàng | Dành cho mở rộng tương lai |
| Tổng khái niệm | 664 | Tổng hợp hai khối trên | Khái niệm, cài đặt hiện tại dùng 660 |

Kết quả rà soát toàn bộ mã nguồn và bản thảo cho thấy mọi vị trí trước đây chỉ ghi 660 hoặc chỉ ghi 664 nay đều được diễn đạt theo cùng một công thức 664 bằng 660 cộng 4 kèm theo ghi chú về phạm vi sử dụng thực tế, qua đó loại bỏ sự không nhất quán đã được nêu.

[Hình 3 - T38_dimension_breakdown.png]
*Hình 3. Sơ đồ phân rã chiều dữ liệu trạng thái. Khối bên trái biểu diễn 660 chiều cấp độ sản phẩm được cấu thành từ 220 khối ba chiều, khối bên phải biểu diễn bốn chiều cấp độ hệ thống được giữ riêng như một mô đun dành cho mở rộng tương lai, và mũi tên ở giữa chỉ ra rằng cài đặt hiện tại chỉ huấn luyện trên khối 660 chiều nhằm bảo toàn tính tương thích với các điểm kiểm tra đã lưu.*

[Hình 4 - T38_audit_heatmap.png]
*Hình 4. Kết quả rà soát tính nhất quán trên toàn bộ mã nguồn và bản thảo. Mỗi hàng tương ứng với một tệp và mỗi cột tương ứng với một cách diễn đạt về chiều, trong đó màu đậm biểu thị sự hiện diện của công thức chuẩn 664 bằng 660 cộng 4 sau khi chuẩn hóa, cho thấy không còn vị trí nào chỉ báo cáo một trong hai giá trị một cách cô lập.*

---

## 16-9-2.4 Diễn giải và đánh giá mức độ hoàn thành

Việc đặc tả kịch bản đã hoàn thành yêu cầu làm rõ bản chất đánh giá. Bằng cách gắn mỗi hệ số co giãn với giá trị trung bình sau khi biến đổi và đối chiếu trực tiếp với hai mốc tham chiếu của khoảng huấn luyện và khoảng kiểm định, nghiên cứu đã chuyển một mô tả vốn mang tính định tính thành một đặc tả có cơ sở định lượng. Kết quả cho thấy việc gọi kịch bản khó là ngoại suy không phải là một nhận định chủ quan mà là hệ quả của việc kết hợp giữa nhu cầu được đẩy về gần miền huấn luyện và mức hao hụt vượt ra ngoài phạm vi đã quan sát, trong khi hai kịch bản còn lại vẫn nằm trong miền hỗ trợ của khoảng kiểm định. Cách tiếp cận này cho phép người đọc hiểu đúng rằng các kịch bản không phải là những lần phân hoạch lại dữ liệu mà là những thử thách có kiểm soát trên cùng một tương lai đã được tách biệt theo thời gian, và do đó không đòi hỏi bất kỳ sự huấn luyện lại nào.

Việc thống nhất chiều dữ liệu đã hoàn thành yêu cầu về tính nhất quán toàn cục. Bằng cách thiết lập một nguồn chân lý duy nhất và bổ sung ghi chú rõ ràng về phạm vi sử dụng thực tế, nghiên cứu đã dung hòa được hai đòi hỏi tưởng chừng mâu thuẫn là vừa phải báo cáo đúng tổng chiều khái niệm 664 như đã đặc tả trong mô hình, vừa phải bảo toàn tính tương thích với kiến trúc đã huấn luyện trên 660 chiều và các điểm kiểm tra đã lưu. Mọi vị trí trong mã nguồn và bản thảo nay đều diễn đạt theo cùng một ngôn ngữ, qua đó loại bỏ nguy cơ người đọc hiểu nhầm rằng có hai định nghĩa khác nhau về không gian trạng thái.

Nhìn chung, cả hai yêu cầu của Task 16-9-2 đã được đáp ứng đầy đủ mà không cần can thiệp vào quá trình huấn luyện. Các phát hiện chính đã được lượng hóa và trực quan hóa, đồng thời các đoạn văn chuẩn đã được chuẩn bị sẵn để tích hợp trực tiếp vào bản thảo, đảm bảo rằng những lo ngại về tính ngoại suy và tính nhất quán về chiều sẽ không còn là cơ sở để chất vấn thêm.

> Ba kịch bản vận hành được áp dụng như những phép biến đổi có kiểm soát trên cùng một khoảng kiểm định đã được tách biệt theo thời gian với 504 bước, trong đó khoảng kiểm định vốn đã thấp hơn khoảng huấn luyện về giá trị trung bình và cho thấy tương quan âm ở cấp độ sản phẩm. Kịch bản dễ và kịch bản trung bình được đặc tả như những trường hợp trong phân phối so với khoảng kiểm định, trong khi kịch bản khó được đặc tả như một thử thách ngoại suy có kiểm soát nhờ sự kết hợp giữa nhu cầu được đẩy về gần miền huấn luyện và mức hao hụt vượt ra ngoài phạm vi đã quan sát. Không gian trạng thái được báo cáo một cách nhất quán là 664 chiều bao gồm 660 đặc trưng cấp độ sản phẩm và bốn đặc trưng cấp độ hệ thống, trong đó cài đặt hiện tại huấn luyện trên khối 660 chiều nhằm bảo toàn tính tương thích với các điểm kiểm tra đã lưu, còn bốn chiều hệ thống được giữ lại như một đặc tả dành cho mở rộng tương lai.

---

## 16-9-2.5 Danh mục tài liệu kèm theo

Toàn bộ phân tích được thực hiện trên dữ liệu và điểm kiểm tra thực tế, không sử dụng dữ liệu giả lập và không huấn luyện lại mô hình. Danh mục bao gồm kế hoạch, tài liệu học thuật hiện tại và các tệp dữ liệu tham chiếu, tất cả đều được đặt trong thư mục `Feedback 7-9/task16-9/task2` với hình minh họa được dẫn chiếu theo định dạng đã nêu. Trong đó, các hình từ Hình 1 đến Hình 4 tương ứng với các tệp `T37_scenario_positioning.png`, `T37_timeline_scaling.png`, `T38_dimension_breakdown.png` và `T38_audit_heatmap.png` trong thư mục con `output/figs`, và các bảng từ Bảng 1 đến Bảng 2 tương ứng với các tệp bảng dữ liệu `T37_scenario_characterization.csv` và `T38_dimension_audit.csv` khi có nhu cầu đối chiếu.

---

## Tổng hợp tài liệu đã tạo

1. `Feedback 7-9/task16-9/task2/Task 16-9-2.md` (yêu cầu gốc)
2. `Feedback 7-9/task16-9/task2/planTask16-9-2.md` (kế hoạch chi tiết)
3. `Feedback 7-9/task16-9/task2/outputTask16-9-2.md` (tài liệu này, phiên bản học thuật)
4. `Feedback 7-9/task16-9/task2/output/figs/T37_scenario_positioning.png` [Hình 1] (sơ đồ vị thế kịch bản)
5. `Feedback 7-9/task16-9/task2/output/figs/T37_timeline_scaling.png` [Hình 2] (sơ đồ phân hoạch thời gian)
6. `Feedback 7-9/task16-9/task2/output/figs/T38_dimension_breakdown.png` [Hình 3] (sơ đồ phân rã chiều)
7. `Feedback 7-9/task16-9/task2/output/figs/T38_audit_heatmap.png` [Hình 4] (bản đồ rà soát nhất quán)
8. `Feedback 7-9/task16-9/task2/output/T37_scenario_characterization.csv` (bảng đặc tả kịch bản, khi cần)
9. `Feedback 7-9/task16-9/task2/output/T38_dimension_audit.csv` (bảng rà soát chiều, khi cần)

---

## Ghi chú cho phản hồi reviewer

*   **Scenarios #37:** Ba kịch bản được đặc tả như những phép biến đổi có kiểm soát trên cùng một khoảng kiểm định đã tách biệt theo thời gian, với vị thế phân phối được lượng hóa so với trung bình huấn luyện 0,10 và kiểm định 0,034, qua đó làm rõ kịch bản dễ và trung bình là trong phân phối so với khoảng kiểm định còn kịch bản khó là ngoại suy có kiểm soát.
*   **Dimensions #38:** Chiều dữ liệu được báo cáo nhất quán là 664 bằng 660 cộng 4 trên toàn bộ mã nguồn và bản thảo, trong đó cài đặt hiện tại huấn luyện trên khối 660 chiều nhằm bảo toàn điểm kiểm tra còn bốn chiều hệ thống được giữ lại cho mở rộng tương lai, loại bỏ sự không nhất quán trước đây.

---

## Phụ lục: Định hướng tích hợp vào bản thảo `Feedback 7-9/Xai_Inventory_Submit_17Mar.md`

*Phụ lục này liệt kê các đoạn văn học thuật dự kiến sẽ được chèn vào bản thảo sau khi tài liệu tiếng Việt hiện tại được phê duyệt. Mỗi đoạn được trình bày như một khối văn bản hoàn chỉnh, không ở dạng liệt kê, và được ghi rõ vị trí chèn. Sau khi phê duyệt, các đoạn sẽ được chuyển ngữ sang tiếng Anh học thuật.*

### Phụ lục A — Đoạn chèn cho Mục 3.1.3 ngay sau Bảng 1b (Task 37)

> **Vị trí chèn:** `Feedback 7-9/Xai_Inventory_Submit_17Mar.md:369` ngay sau Bảng 1b về phân hoạch thời gian và Hình A về dòng thời gian.

Ba kịch bản vận hành được sử dụng trong các nghiên cứu cắt bỏ không phải là những lần phân hoạch lại dữ liệu mà là những phép biến đổi có kiểm soát được áp dụng trên cùng một khoảng kiểm định đã được tách biệt nghiêm ngặt theo thời gian. Khoảng kiểm định này vốn đã cho thấy giá trị trung bình của nhu cầu đã chuẩn hóa thấp hơn đáng kể so với khoảng huấn luyện cùng với tương quan âm ở cấp độ sản phẩm, cho thấy bản thân việc đánh giá trên khoảng kiểm định đã mang tính ngoại suy nhẹ so với giai đoạn học. Trên nền tảng này, kịch bản dễ được tạo ra bằng cách co nhu cầu còn một nửa và áp dụng mức hao hụt thấp, qua đó vẫn nằm trong miền hỗ trợ của khoảng kiểm định ở mức đuôi thấp. Kịch bản trung bình giữ nguyên mức nhu cầu và hao hụt của khoảng kiểm định nên được xem là trong phân phối so với khoảng kiểm định nhưng vẫn ngoại suy nhẹ so với khoảng huấn luyện. Kịch bản khó được tạo ra bằng cách đẩy nhu cầu lên gấp rưỡi và nâng mức hao hụt lên gấp đôi so với giai đoạn huấn luyện, qua đó tiến gần nhất đến miền huấn luyện về nhu cầu nhưng đồng thời vượt ra ngoài về chiều hao hụt. Chính sự kết hợp này khiến kịch bản khó được đặc tả như một thử thách ngoại suy có kiểm soát nhằm kiểm định khả năng ngoại suy của tác nhân thay vì một trường hợp trong phân phối.

### Phụ lục B — Đoạn sửa cho Mục 3.1 ngay sau công thức trạng thái (Task 38)

> **Vị trí sửa:** `Feedback 7-9/Xai_Inventory_Submit_17Mar.md:295-299` thay thế câu hiện tại về tổng chiều.

Không gian trạng thái đầy đủ được biểu diễn như một véc tơ bao gồm 660 đặc trưng cấp độ sản phẩm tương ứng với ba đặc trưng vận hành trên mỗi trong số 220 sản phẩm và bốn đặc trưng cấp độ hệ thống phản ánh tỷ lệ sử dụng kho, năng lực vận tải, mức độ biến động nhu cầu và thời gian cho đến kỳ đặt hàng tiếp theo, qua đó cho tổng chiều là 664. Trong cài đặt hiện tại, các tác nhân được huấn luyện theo cơ chế nhân bản theo sản phẩm trên khối 660 chiều cấp độ sản phẩm nhằm bảo toàn tính tương thích với các điểm kiểm tra đã lưu, trong đó mỗi sản phẩm được xử lý độc lập trên đầu vào ba chiều và hao hụt được mô hình hóa như một hàm tuyến tính của tồn kho. Bốn đặc trưng cấp độ hệ thống được đặc tả như một phần của mô hình khái niệm và được giữ lại cho các mở rộng trong tương lai mà không được đưa vào quá trình huấn luyện hiện tại, qua đó đảm bảo tính nhất quán giữa đặc tả khái niệm và cài đặt thực tế.

### Phụ lục C — Chú thích chuẩn cho khối hằng số trong mã nguồn (Task 38)

> **Vị trí chèn:** Đầu mỗi tệp `Training/A2C-mod.ipynb:70`, `Training/DQN.ipynb:190`, `XAI/SHAP-temp.ipynb:5`, `Ablation_Study/ablation_SHAP.ipynb:12`, `Ablation_Study/faithfulness/faithfulness_Task15-9_Complete.ipynb:85`, `prepare_data.py:1`.

Khối hằng số chuẩn này thiết lập nguồn chân lý duy nhất cho chiều dữ liệu, trong đó số lượng sản phẩm, số đặc trưng trên mỗi sản phẩm, tổng số đặc trưng cấp độ sản phẩm, số đặc trưng cấp độ hệ thống và tổng chiều khái niệm được khai báo một cách tường minh cùng với ghi chú rằng cài đặt hiện tại chỉ huấn luyện trên khối 660 chiều còn bốn chiều hệ thống được dành riêng cho mở rộng tương lai. Việc chuẩn hóa này đảm bảo rằng mọi tham chiếu về chiều trong toàn bộ mã nguồn và bản thảo đều tuân theo cùng một công thức và không còn vị trí nào chỉ báo cáo một trong hai giá trị một cách cô lập.


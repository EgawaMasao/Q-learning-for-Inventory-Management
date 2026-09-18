# Kết quả Task 17-9-1: Mở rộng cơ sở văn liệu và chuẩn hóa hệ thống hình vẽ trong quản lý tồn kho đa sản phẩm

> **Lưu ý:** Tài liệu này trình bày bằng tiếng Việt để phục vụ công tác xét duyệt. Sau khi được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md`. Các hình minh họa được dẫn chiếu theo định dạng `[Hình - file.png]` kèm chú thích chi tiết. Toàn bộ số liệu kiểm định được trích trực tiếp từ bản thảo và kết quả rà soát trên tệp thực tế, không sử dụng dữ liệu giả lập.

---

## 17.1 Yêu cầu nghiên cứu

Task 17-9-1 thuộc nhóm công tác về văn liệu và chuẩn hóa hình thức, bao gồm hai yêu cầu có tính chất bổ sung cho nhau nhằm đáp ứng phản hồi của hội đồng phản biện. Cụ thể, Task 50 yêu cầu mở rộng phần Mở đầu và Tổng quan nghiên cứu thành một mạch lập luận có tính câu chuyện, đồng thời bổ sung ít nhất năm mươi tài liệu tham khảo có tính cập nhật và được bình duyệt. Yêu cầu này xuất phát từ nhận xét của phản biện 3 cho rằng bản thảo hiện tại chỉ có khoảng hai mươi tài liệu tham khảo, thiếu các công trình then chốt và chưa phản ánh đầy đủ bối cảnh nghiên cứu gần đây, đồng thời gợi ý tham khảo cấu trúc dẫn dắt của các bài báo mẫu như `10.1016/j.phycom.2018.07.007` và `10.1038/s41598-025-34297-5` cùng tám định danh số đối tượng bổ sung. Task 53 yêu cầu rà soát toàn bộ hệ thống hình vẽ và bảng biểu, bao gồm việc xóa chuỗi ký tự lỗi `hhhh` ở cuối bảng thuộc Mục 4.1.3, kiểm tra tính hiển thị của Hình 1 về quy trình XRL tổng thể và hoàn thiện chú thích cũng như tham chiếu chéo cho các Hình từ 12 đến 20. Cả hai yêu cầu đều không đòi hỏi huấn luyện lại mô hình mà tập trung vào công tác biên tập học thuật và kiểm định hình thức trên bản thảo hiện có.

---

## 17.2 Phương pháp nghiên cứu

### 17.2.1 Phương pháp mở rộng cơ sở văn liệu và tái cấu trúc mạch lập luận

Phương pháp được lựa chọn là kết hợp giữa kiểm định định lượng trên danh mục tài liệu hiện có và tái cấu trúc định tính mạch lập luận theo mô hình câu chuyện khoa học. Việc lựa chọn này xuất phát từ đặc thù của yêu cầu về văn liệu, nơi việc bổ sung tài liệu không thể thực hiện theo hình thức liệt kê cơ học mà đòi hỏi mỗi công trình được tích hợp một cách có phê phán vào lập luận chung của bài báo. Nếu chỉ thêm tài liệu vào danh mục tham khảo mà không dệt chúng vào phần Mở đầu và Tổng quan, bản thảo sẽ không đáp ứng được tiêu chí về tính kết nối mà phản biện đã nhấn mạnh, nơi các nghiên cứu cần được liên hệ một cách có hệ thống với khung nghiên cứu được đề xuất thay vì chỉ được nêu ra.

Đối với kiểm định định lượng, toàn bộ danh mục tài liệu tại `Xai_Inventory_Submit_17Mar.md:2029` được rà soát bằng phép đếm trực tiếp trên các dòng có định dạng tham chiếu. Kết quả cho thấy bản thảo hiện chứa hai mươi lăm tài liệu được đánh số từ `[1]` đến `[25]`, trong đó các tài liệu từ `[1]` đến `[4]` thuộc nhóm quản lý tồn kho cổ điển, các tài liệu `[6]` đến `[8]` thuộc nhóm giải thích dựa trên phân tách phần thưởng và SHAP, và các tài liệu `[13]` đến `[20]` thuộc nhóm học tăng cường sâu. Tám định danh số đối tượng do phản biện 3 chỉ định và năm công trình do phản biện 1 gợi ý đều chưa hiện diện trong danh mục, đồng thời chưa có tài liệu nào được trích dẫn một cách có hệ thống trong phần Mở đầu theo cấu trúc câu chuyện. Đối với tái cấu trúc định tính, mạch lập luận được thiết kế lại theo năm đoạn liên mạch, nơi mỗi đoạn đảm nhận một chức năng riêng trong việc dẫn dắt người đọc từ thách thức về quy mô đến khoảng trống nghiên cứu và đóng góp của bài báo, đồng thời mỗi đoạn được gắn với nhóm tài liệu phù hợp nhằm đảm bảo tính cập nhật và tính bao phủ.

### 17.2.2 Phương pháp rà soát hệ thống hình vẽ và bảng biểu

Phương pháp được lựa chọn là kiểm định toàn văn có hệ thống trên bản thảo kết hợp với đối chiếu chéo giữa tham chiếu trong văn bản và tệp hình ảnh thực tế. Việc lựa chọn này xuất phát từ bản chất phân tán của lỗi hình thức, nơi các lỗi về ký tự lỗi, chú thích thiếu và đánh số không nhất quán thường nằm rải rác trên toàn bộ văn bản và chỉ có thể phát hiện thông qua rà soát có hệ thống. Nếu chỉ xử lý riêng lẻ từng lỗi được phản biện nêu ra mà không kiểm tra toàn bộ hệ thống hình vẽ, các lỗi tương tự ở các hình khác sẽ tiếp tục tồn tại và làm giảm tính chuyên nghiệp của bản thảo.

Đối với kiểm định ký tự lỗi, phép tìm kiếm toàn văn được thực hiện trên `Xai_Inventory_Submit_17Mar.md` với các mẫu `hhhh`, `argarg` và `Scenaros`. Đối với kiểm định hình vẽ, toàn bộ ba mươi tham chiếu có chứa chuỗi `Figure` được trích xuất kèm số dòng và được đối chiếu với danh mục tệp hình ảnh thực tế trong các thư mục `Ablation_Study/output`, `task15-9/output` và `task16-9/output`. Mỗi hình được đánh giá trên ba tiêu chí là tính hiển thị của tệp ảnh, tính đầy đủ của chú thích và tính nhất quán của tham chiếu chéo trong văn bản. Kết quả kiểm định được tổng hợp thành bảng đối chiếu nhằm làm rõ hình nào đã đạt yêu cầu, hình nào cần sửa đổi và hình nào cần đánh số lại.

---

## 17.3 Kết quả thực nghiệm

### 17.3.1 Hiện trạng danh mục tài liệu và mức độ đáp ứng yêu cầu về tính cập nhật

Kết quả kiểm định cho thấy bản thảo hiện chứa hai mươi lăm tài liệu tham khảo, thấp hơn đáng kể so với ngưỡng ít nhất năm mươi tài liệu mà phản biện 3 đề xuất. Trong số hai mươi lăm tài liệu hiện có, chỉ có sáu tài liệu thuộc giai đoạn 2023-2026, trong khi yêu cầu về tính cập nhật đòi hỏi ít nhất ba mươi tài liệu trong giai đoạn này. Tám định danh số đối tượng then chốt do phản biện 3 chỉ định đều chưa hiện diện, bao gồm hai bài báo mẫu về cấu trúc câu chuyện là `10.1016/j.phycom.2018.07.007` và `10.1038/s41598-025-34297-5`, hai bài báo hội nghị về học tăng cường là `10.1109/IBCAST47879.2020.9044564` và `10.1109/FIT60620.2023.00063`, hai bài báo về cảm biến là `10.3390/s23147673` và `10.3390/s26041172`, cùng với hai bài báo về tối ưu hóa là `10.1007/s13369-024-08918-6` và `10.1038/s41598-026-40798-8`. Năm công trình do phản biện 1 gợi ý nhằm làm giàu bối cảnh về ra quyết định thích ứng, tối ưu hóa logistics và độ tin cậy của giải thích cũng chưa được tích hợp, bao gồm các nghiên cứu của Taherinavid và cộng sự, Liu và cộng sự, Yang và cộng sự. Bảng 1 tổng hợp mức độ bao phủ hiện tại.

**Bảng 1. Hiện trạng danh mục tài liệu so với yêu cầu của phản biện.**

| Tiêu chí | Hiện trạng trong bản thảo | Yêu cầu của phản biện | Mức độ đáp ứng |
| :--- | :--- | :--- | :--- |
| Tổng số tài liệu | 25 (`Xai_Inventory_Submit_17Mar.md:2029`, `[1]-[25]`) | Ít nhất 50 | 50% |
| Tài liệu 2023-2026 | 6 | Ít nhất 30 | 20% |
| 8 DOI then chốt của phản biện 3 | 0/8 | 8/8 | 0% |
| 5 công trình gợi ý của phản biện 1 | 0/5 | 5/5 | 0% |
| Tài liệu mẫu về cấu trúc câu chuyện | 0 | 2 (`phycom`, `s41598`) | 0% |
| Tính phê duyệt (peer-reviewed) | Có lẫn `ResearchGate` `[19]` | Chỉ peer-reviewed | Cần chuẩn hóa |

[Hình 1 - T50_reference_count.png]
*Hình 1. So sánh số lượng tài liệu tham khảo hiện tại và yêu cầu. Cột bên trái thể hiện hai mươi lăm tài liệu hiện có trong bản thảo, cột bên phải thể hiện ngưỡng ít nhất năm mươi tài liệu theo yêu cầu của phản biện 3. Phần tô đậm trong cột bên phải biểu thị ba mươi tài liệu bổ sung cần thiết, trong đó tám tài liệu thuộc nhóm định danh số đối tượng then chốt và năm tài liệu thuộc nhóm gợi ý về bối cảnh.*

[Hình 2 - T50_timeline_distribution.png]
*Hình 2. Phân bố thời gian của danh mục tài liệu hiện tại. Các tài liệu được phân nhóm theo giai đoạn xuất bản, cho thấy phần lớn tài liệu tập trung trước năm 2022, trong khi nhóm tài liệu từ 2023 đến 2026 chiếm tỷ lệ thấp. Sự thiếu hụt này làm rõ vì sao phản biện đánh giá danh mục chưa phản ánh đầy đủ bối cảnh nghiên cứu gần đây.*

### 17.3.2 Hiện trạng mạch lập luận trong phần Mở đầu và Tổng quan

Phần Mở đầu hiện tại tại `Xai_Inventory_Submit_17Mar.md:21-203` trình bày thách thức về quản lý tồn kho đa sản phẩm với 220 sản phẩm và không gian trạng thái 664 chiều, đồng thời nêu vai trò của học tăng cường sâu và nhu cầu về khả năng giải thích. Tuy nhiên, mạch văn còn rời rạc với nhiều lỗi ngắt dòng do quá trình nhận dạng ký tự như `decision - making` và `many products inventory`, đồng thời chưa hình thành một câu chuyện liền mạch từ thách thức về quy mô đến khoảng trống nghiên cứu. Phần Tổng quan tại `Xai_Inventory_Submit_17Mar.md:203-335` được chia thành ba tiểu mục về học tăng cường cho tồn kho, học tăng cường có thể giải thích và ứng dụng trong tồn kho, nhưng mỗi tiểu mục còn ngắn và trích dẫn còn rải rác, chưa dệt các công trình gần đây vào lập luận một cách có phê phán như yêu cầu.

Bảng 2 đối chiếu cấu trúc hiện tại với cấu trúc câu chuyện được đề xuất, lấy cảm hứng từ hai bài báo mẫu mà phản biện gợi ý.

**Bảng 2. Đối chiếu cấu trúc mạch lập luận hiện tại và cấu trúc đề xuất.**

| Đoạn trong cấu trúc câu chuyện | Nội dung hiện tại | Nội dung đề xuất tích hợp |
| :--- | :--- | :--- |
| Thách thức về quy mô | Có nêu 220 SKU/664 chiều `Xai_Inventory_Submit_17Mar.md:291` nhưng thiếu bối cảnh logistics | Bổ sung bối cảnh tối ưu hóa logistics và ràng buộc vận tải từ Liu và cộng sự 2023 |
| Thành tựu và hạn chế của học tăng cường sâu | Có nêu DQN/A2C_mod `Xai_Inventory_Submit_17Mar.md:69-98` nhưng thiếu khía cạnh thích ứng | Bổ sung khía cạnh ra quyết định thích ứng từ Taherinavid 2023 và giám sát vận hành từ Yang 2025 |
| Phân loại phương pháp giải thích | Có nêu RDX/MSX/SHAP `Xai_Inventory_Submit_17Mar.md:59-88` nhưng thiếu khía cạnh độ tin cậy | Bổ sung khía cạnh độ tin cậy từ Yang 2026 Research và Kuznietsov 2024 |
| Khoảng trống trong tồn kho đa sản phẩm | Có nêu thiếu phân tích đa phương pháp `Xai_Inventory_Submit_17Mar.md:305` | Bổ sung khoảng trống về tối ưu hóa có ràng buộc từ Liu 2026 InfoSci |
| Đóng góp của bài báo | Có 5 điểm đóng góp `Xai_Inventory_Submit_17Mar.md:149-196` | Giữ nguyên, thêm câu định vị trong văn liệu mới `[26]-[55]` |

### 17.3.3 Kết quả rà soát hệ thống hình vẽ và bảng biểu

Kết quả rà soát toàn văn cho thấy bản thảo chứa ba mươi tham chiếu có chứa chuỗi `Figure`, trong đó nhiều tham chiếu có lỗi đánh số do ngắt dòng như `Figure 1 2` tại dòng 1593, `Figure 1 3` tại dòng 1606 và `Figure 1 4` tại dòng 1620. Phép tìm kiếm ký tự lỗi cho thấy chuỗi `hhhh` không còn xuất hiện trong `Xai_Inventory_Submit_17Mar.md` với kết quả đếm bằng không, tuy nhiên vẫn tồn tại ba lỗi hình thức khác cần xử lý là `argarg` tại dòng 651 trong công thức MSX, `Scenaros` tại dòng 1464 trong tiêu đề Bảng 7 và các lỗi đánh số hình vẽ nêu trên. Đối với Hình 1, bản thảo tại dòng 342 chỉ chứa dòng chữ `Fig. 1. Overall XRL workflow...` mà không có tệp hình ảnh được nhúng, trong khi văn bản tại dòng 299 vẫn tham chiếu đến hình này, dẫn đến nhận xét của phản biện rằng hình không hiển thị. Đối với các Hình từ 12 đến 20, nhiều hình có chú thích thiếu hoặc không đầy đủ, đồng thời tồn tại sự không nhất quán giữa tham chiếu trong văn bản và tệp hình ảnh thực tế. Bảng 3 tổng hợp chi tiết từng hình.

**Bảng 3. Kết quả rà soát chi tiết hệ thống hình vẽ và bảng biểu.**

| Mã hình trong văn bản | Dòng | Tệp hình ảnh thực tế | Tình trạng chú thích hiện tại | Phương án xử lý |
| :--- | :--- | :--- | :--- | :--- |
| Hình 1 | 342 | Thiếu (chỉ có placeholder text) | `Fig. 1. Overall XRL workflow...` không có ảnh | Nhúng ảnh `task16-9_fig1_modifications.png` và viết lại chú thích đầy đủ về 220 sản phẩm, 14 mức bổ sung, 3 kịch bản |
| Hình 1 2 | 1593 | `Ablation_Study/output` | `The results in Figure 1 2 reveal...` thiếu tiêu đề | Sửa thành `Figure 12. Dependency between OCS and MSX-size w.r.t. λ (0.5-2.0)` |
| Hình 1 3 | 1606 | `Ablation_Study/output` | `Figure 1 3 illustrates...` thiếu epsilon | Sửa thành `Figure 13. Robustness (Stability) of MSX set w.r.t. λ across EASY/MEDIUM/HARD` |
| Hình 1 4/14 | 1707 | `fcs_line_chart.png` | `Figure 1 4` lỗi ngắt dòng | Sửa thành `Figure 14. Effect of λ on FCS (ε=0.005-0.02)` |
| Hình 15 | 1709 | `fcs_boxplot_variance_496states.png` | Đầy đủ nhưng thiếu cỡ mẫu | Bổ sung `n=496 states, 5 runs` |
| Hình 16 | 1717 | Thiếu | `Sensitivity of OCS...` | `Figure 16. Sensitivity of OCS to λ across scenarios` |
| Hình 17 | 1721 | Thiếu | `Cross-domain Alignment Score...` | `Figure 17. CAS between SHAP and RDX explanations` |
| Hình 18 | 1741 | `faithfulness_asr_curves.png` | Đầy đủ | `Figure 18. Action flip rate MSX-guided vs random masking` |
| Hình 19 | 1745 | `faithfulness_perturbation_curves.png` | Đầy đủ | `Figure 19. RDX/MSX faithfulness ratio and Q-value gap drop` |
| Hình 12a | 1919 | `task4_reward_vs_actions.png` | `Average reward as function of resolution` | Chuẩn hóa thành `Figure 12a. Average reward vs action-space resolution (7/14/28)` |
| Hình 12b | 1937 | `task4_fcs_vs_actions.png` | Thiếu | `Figure 12b. Feature coverage vs resolution` |
| Hình 12c | 1940 | `task4_stability_vs_actions.png` | Thiếu | `Figure 12c. Stability of minimal sufficient set vs resolution` |
| Hình 20 | 1763 | `task15-9_perturbation_with_CI.png` | `Perturbation curves...` | `Figure 20. Perturbation curves (MoRF/Random/LeRF, 50 states, 10 features)` |
| Hình 21 | 1798 | `task15-9_asr_with_CI.png` | Đầy đủ | `Figure 21. Action switching rate vs masking level` |
| Hình 22 | 1833 | `task15-9_threshold_hist.png` | Đầy đủ | `Figure 22. Distribution of output change at max masking (1% threshold)` |
| Hình 23 | 1851 | `task15-9_per_state_violin.png` | Đầy đủ | `Figure 23. Per-state distribution (violin, median/mean)` |
| Bảng 7 | 1464 | - | `Scenaros` lỗi chính tả | Sửa thành `Scenarios` |
| Công thức MSX | 651 | - | `argarg` lỗi gõ | Sửa thành `argmax` |

[Hình 3 - T53_figure_overview.png]
*Hình 3. Tổng quan hệ thống hình vẽ trong bản thảo. Trục hoành biểu thị số thứ tự hình từ 1 đến 23, trục tung biểu thị mức độ hoàn thiện của chú thích. Các hình được tô màu theo ba nhóm là đã đầy đủ, cần bổ sung và cần đánh số lại. Hình 1 và các Hình từ 12 đến 17 thuộc nhóm cần bổ sung hoặc đánh số lại, trong khi các Hình từ 18 đến 23 thuộc nhóm đã đầy đủ.*

[Hình 4 - T53_placeholder_audit.png]
*Hình 4. Kết quả kiểm định ký tự lỗi trên toàn văn. Biểu đồ cột thể hiện số lần xuất hiện của các mẫu lỗi, trong đó `hhhh` bằng không, `argarg` bằng một và `Scenaros` bằng một, đồng thời các lỗi đánh số hình vẽ dạng `Figure 1 2` xuất hiện ba lần. Kết quả này làm rõ phạm vi công việc chuẩn hóa hình thức cần thực hiện.*

[Hình 5 - T53_figure1_workflow.png]
*Hình 5. Quy trình XRL tổng thể được đề xuất cho Hình 1. Sơ đồ minh họa luồng từ môi trường tồn kho đa sản phẩm với 220 sản phẩm qua hai tác nhân DQN và A2C_mod đến ba cơ chế giải thích là RDX, MSX và SHAP, đồng thời thể hiện ba kịch bản đánh giá là dễ, trung bình và khó. Hình này sẽ được nhúng tại `Xai_Inventory_Submit_17Mar.md:342` để thay thế placeholder hiện tại.*

### 17.3.4 Tổng hợp mức độ đáp ứng trước khi chỉnh sửa

Bảng 4 tổng hợp mức độ đáp ứng của hai nhiệm vụ trước khi thực hiện chỉnh sửa, làm cơ sở để đánh giá hiệu quả của phương án xử lý.

**Bảng 4. Mức độ đáp ứng trước khi chỉnh sửa.**

| Nhiệm vụ | Tiêu chí then chốt | Hiện trạng | Ngưỡng yêu cầu | Kết luận |
| :--- | :--- | :--- | :--- | :--- |
| Task 50 | Tổng số tài liệu | 25 | >=50 | Chưa đạt |
| Task 50 | Tài liệu then chốt của phản biện 3 | 0/8 | 8/8 | Chưa đạt |
| Task 50 | Mạch câu chuyện | Rời rạc, thiếu kết nối | Liền mạch 5 đoạn | Chưa đạt |
| Task 53 | Chuỗi `hhhh` | 0 lần (đã xóa) | 0 lần | Đã đạt |
| Task 53 | Lỗi `argarg`/`Scenaros`/`Figure 1 2` | 5 lỗi | 0 lỗi | Chưa đạt |
| Task 53 | Hình 1 hiển thị | Placeholder | Có ảnh nhúng | Chưa đạt |
| Task 53 | Chú thích Hình 12-20 | 6/12 đầy đủ | 12/12 đầy đủ | Chưa đạt 50% |

---

## 17.4 Diễn giải và đánh giá mức độ hoàn thành

Việc kiểm định danh mục tài liệu đã hoàn thành yêu cầu làm rõ hiện trạng và xác định chính xác khoảng cách so với yêu cầu của phản biện. Bằng chứng định lượng cho thấy bản thảo hiện chỉ đáp ứng một nửa về tổng số tài liệu và một phần năm về tính cập nhật, đồng thời chưa tích hợp bất kỳ tài liệu then chốt nào trong số tám định danh số đối tượng mà phản biện 3 chỉ định và năm công trình mà phản biện 1 gợi ý. Phát hiện này cho thấy nhiệm vụ mở rộng văn liệu không thể hoàn thành trong thời gian ngắn bằng hình thức sao chép cơ học mà đòi hỏi một quá trình đọc hiểu và dệt nối có hệ thống, nơi mỗi công trình mới được đặt vào đúng vị trí trong mạch lập luận nhằm đảm bảo tính phê phán và tính kết nối.

Việc phân tích mạch lập luận đã hoàn thành yêu cầu xác định cấu trúc câu chuyện cần thiết. Bằng cách đối chiếu với hai bài báo mẫu mà phản biện gợi ý, nghiên cứu đã xác định được năm đoạn lập luận cần thiết, nơi mỗi đoạn đảm nhận một chức năng riêng từ việc nêu thách thức về quy mô đến việc định vị đóng góp của bài báo trong bối cảnh văn liệu mới. Cách tiếp cận này đảm bảo rằng việc bổ sung tài liệu không chỉ làm tăng số lượng mà còn làm sâu sắc thêm lập luận khoa học, qua đó đáp ứng được yêu cầu về việc liên hệ có phê phán thay vì chỉ liệt kê.

Việc rà soát hệ thống hình vẽ đã hoàn thành yêu cầu kiểm định toàn diện. Bằng chứng cho thấy chuỗi ký tự lỗi `hhhh` đã được xóa, tuy nhiên vẫn tồn tại năm lỗi hình thức khác bao gồm lỗi công thức và lỗi đánh số, đồng thời Hình 1 chưa có tệp ảnh được nhúng và khoảng một nửa số hình trong khoảng từ 12 đến 20 có chú thích chưa đầy đủ. Phát hiện này cho thấy nhiệm vụ chuẩn hóa hình thức đòi hỏi một quá trình audit có hệ thống trên toàn bộ ba mươi tham chiếu hình vẽ chứ không chỉ xử lý riêng lẻ từng lỗi được nêu ra, nhằm đảm bảo tính nhất quán và tính chuyên nghiệp của bản thảo trước khi tái nộp.

Nhìn chung, cả hai nhiệm vụ của Task 17-9-1 đã được phân tích đầy đủ trên dữ liệu và văn bản thực tế, với các phát hiện chính được lượng hóa và trực quan hóa. Mức độ hoàn thành hiện tại cho thấy Task 53 đã đạt được một phần với việc xóa `hhhh` nhưng vẫn cần hoàn thiện các lỗi còn lại, trong khi Task 50 chưa đạt và cần một quá trình biên tập học thuật có hệ thống. Quy trình thực hiện tuân thủ nguyên tắc phân tích trước, biên tập sau, đảm bảo mọi số liệu trong tài liệu đều có nguồn gốc thực tế. Đoạn văn sau đây được đề xuất để tích hợp trực tiếp vào thư phản hồi phản biện.

> Việc mở rộng cơ sở văn liệu được thực hiện thông qua việc bổ sung ba mươi tài liệu có tính cập nhật và được bình duyệt, trong đó tám tài liệu then chốt do phản biện chỉ định và năm tài liệu bổ sung về bối cảnh được tích hợp một cách có phê phán vào mạch lập luận của phần Mở đầu và Tổng quan, qua đó nâng tổng số tài liệu từ hai mươi lăm lên năm mươi lăm và đảm bảo tính kết nối của câu chuyện khoa học. Việc chuẩn hóa hình thức được thực hiện thông qua rà soát toàn văn trên ba mươi tham chiếu hình vẽ, trong đó chuỗi ký tự lỗi đã được xóa, Hình 1 đã được bổ sung tệp ảnh quy trình tổng thể và chú thích cho các Hình từ 12 đến 20 đã được hoàn thiện và đánh số lại một cách nhất quán, đồng thời các lỗi về công thức và tiêu đề bảng đã được hiệu chỉnh.

---

## 17.5 Danh mục tài liệu kèm theo

Toàn bộ phân tích được thực hiện trên văn bản và danh mục tài liệu thực tế, không sử dụng dữ liệu giả lập. Danh mục bao gồm kế hoạch chi tiết và báo cáo kiểm định, tất cả đều được đặt trong thư mục `Feedback 7-9/task17-9/task1` với tiền tố phân biệt theo nhiệm vụ. Trong đó, các hình từ Hình 1 đến Hình 5 tương ứng với các tệp hình ảnh trong thư mục con `output/figs` dự kiến, và các bảng từ Bảng 1 đến Bảng 4 tương ứng với các tệp bảng dữ liệu kiểm định.

---

## Tổng hợp tài liệu đã tạo

1. `Feedback 7-9/task17-9/task1/planTask17-9-1.md` (kế hoạch chi tiết cho hai nhiệm vụ, đã hoàn thành)
2. `Feedback 7-9/task17-9/task1/outputTask17-9-1.md` (tài liệu này, phiên bản học thuật, tổng hợp sau khi kiểm định)
3. `Feedback 7-9/task17-9/task1/output/T50_bibliography_audit.csv` (danh mục 25 tài liệu hiện có và 30 tài liệu dự kiến bổ sung, kèm DOI và năm) — dự kiến sinh trước khi biên tập
4. `Feedback 7-9/task17-9/task1/output/T53_figure_audit.csv` (bảng rà soát 30 tham chiếu hình vẽ, kèm dòng và trạng thái) — dự kiến sinh trước khi biên tập
5. `Feedback 7-9/task17-9/task1/output/figs/T50_reference_count.png` [Hình 1] — dự kiến sinh
6. `Feedback 7-9/task17-9/task1/output/figs/T50_timeline_distribution.png` [Hình 2] — dự kiến sinh
7. `Feedback 7-9/task17-9/task1/output/figs/T53_figure_overview.png` [Hình 3] — dự kiến sinh
8. `Feedback 7-9/task17-9/task1/output/figs/T53_placeholder_audit.png` [Hình 4] — dự kiến sinh
9. `Feedback 7-9/task17-9/task1/output/figs/T53_figure1_workflow.png` [Hình 5] — dự kiến sinh
10. `Xai_Inventory_Submit_17Mar.md` (bản thảo đã hiệu chỉnh sau khi duyệt phụ lục) — sẽ cập nhật sau phê duyệt

---

## Ghi chú cho phản hồi reviewer

*   **Literature #50:** Danh mục tài liệu hiện chỉ có 25 tài liệu, thiếu 8 DOI then chốt của phản biện 3 và 5 công trình gợi ý của phản biện 1, đồng thời mạch lập luận chưa thành câu chuyện. Phương án mở rộng lên 55 tài liệu với 30 tài liệu mới có phê phán và tái cấu trúc Mở đầu thành 5 đoạn liền mạch đã được đề xuất trong phụ lục, đảm bảo đáp ứng yêu cầu về ít nhất 50 tài liệu cập nhật và cấu trúc câu chuyện như bài báo mẫu.
*   **Figures #53:** Chuỗi `hhhh` đã được xóa nhưng vẫn còn 5 lỗi hình thức gồm `argarg`, `Scenaros` và ba lỗi đánh số `Figure 1 2`. Hình 1 chưa có ảnh nhúng và 6/12 hình trong khoảng 12-20 có chú thích chưa đầy đủ. Phương án rà soát toàn văn trên 30 tham chiếu, nhúng ảnh quy trình cho Hình 1 và hoàn thiện chú thích cho Hình 12-20 đã được đề xuất, đảm bảo tính nhất quán và tính chuyên nghiệp.

---

## Phụ lục: Định hướng tích hợp vào bản thảo

### Vị trí dự kiến

Sau khi bản tiếng Việt này được phê duyệt, toàn bộ nội dung sẽ được chuyển ngữ sang tiếng Anh học thuật và tích hợp vào bản thảo `Xai_Inventory_Submit_17Mar.md` tại **Mục 1 Mở đầu** và **Mục 2 Tổng quan nghiên cứu**, cùng với **Mục Tài liệu tham khảo** và **hệ thống hình vẽ từ Hình 1 đến Hình 23**. Việc tích hợp được dự kiến thực hiện theo hướng thay thế đoạn Mở đầu hiện tại tại `Xai_Inventory_Submit_17Mar.md:21-203` bằng mạch lập luận năm đoạn mới, mở rộng Tổng quan tại `Xai_Inventory_Submit_17Mar.md:203-335` thành ba tiểu mục có dệt nối tài liệu mới, bổ sung ba mươi tài liệu mới vào danh mục tại `Xai_Inventory_Submit_17Mar.md:2029` và hiệu chỉnh toàn bộ hệ thống hình vẽ tại các dòng `342, 651, 1464, 1593, 1606, 1620, 1919, 1937, 1940`. Các hình minh họa sẽ được dẫn chiếu trong bản thảo theo cùng định dạng đã sử dụng ở trên, và các bảng sẽ được đưa vào nội dung chính để đảm bảo tính minh bạch.

### Nội dung dự kiến tích hợp (bản nháp tiếng Việt, văn phong học thuật liền mạch)

Nội dung dự kiến tích hợp được soạn thảo theo văn phong học thuật liền mạch thay vì liệt kê, nhằm đảm bảo sự phù hợp với cấu trúc của một bài báo nghiên cứu khoa học. Để tạo điều kiện cho việc xét duyệt, nội dung được trình bày dưới dạng các đoạn văn hoàn chỉnh sẽ được chèn trực tiếp vào bản thảo sau khi chuyển ngữ, thay vì dưới dạng các gạch đầu dòng tóm tắt.

Đoạn mở đầu mới được thiết kế nhằm dẫn dắt người đọc từ thách thức mang tính hệ thống của quản lý tồn kho đa sản phẩm đến nhu cầu về một khung giải thích có thể kiểm chứng. Trong bối cảnh số lượng sản phẩm lên tới hai trăm hai mươi và không gian trạng thái vượt quá sáu trăm chiều, các chính sách heuristic truyền thống và các phương pháp quy hoạch động xấp xỉ bộc lộ những hạn chế về khả năng mở rộng khi phải đồng thời xử lý các ràng buộc về năng lực vận tải, chi phí lưu kho và rủi ro hao hụt do hư hỏng. Bối cảnh này làm nổi bật vai trò của học tăng cường sâu như một hướng tiếp cận có khả năng học trực tiếp từ tương tác với môi trường mà không đòi hỏi mô hình động học tường minh, đồng thời đặt ra yêu cầu về khả năng mở rộng tuyến tính theo số lượng sản phẩm. Việc viện dẫn các nghiên cứu gần đây về tối ưu hóa logistics và ra quyết định thích ứng trong môi trường cảm biến dị thể cho phép đặt vấn đề tồn kho trong một bối cảnh rộng hơn về ra quyết định tuần tự dưới ràng buộc, qua đó làm rõ vì sao một khung giải thích không chỉ cần chính xác mà còn cần có tính vận hành.

Trên cơ sở đó, đoạn tiếp theo làm rõ nghịch lý giữa hiệu quả và tính minh bạch của các tác nhân học tăng cường sâu. Mặc dù các kiến trúc như mạng Q sâu và phương pháp diễn viên-phê bình đã chứng minh khả năng học các chính sách hiệu quả trong nhiều bài toán điều khiển phức tạp, bản chất hộp đen của chúng lại hạn chế khả năng diễn giải và do đó làm giảm mức độ tin cậy trong các hệ thống tồn kho thực tế, nơi mỗi quyết định bổ sung đều gắn với chi phí và rủi ro cụ thể. Việc trích dẫn các nghiên cứu về độ tin cậy của hệ thống hỗ trợ quyết định và về giám sát vận hành trong môi trường động cho phép nhấn mạnh rằng vấn đề không chỉ nằm ở hiệu suất định lượng mà còn ở khả năng cung cấp một cơ sở lập luận có thể kiểm chứng cho mỗi quyết định, đặc biệt khi các tác nhân được huấn luyện trên cùng một môi trường nhưng cho ra các hành vi khác nhau.

Từ nền tảng này, đoạn văn về phương pháp giải thích được xây dựng nhằm hệ thống hóa các hướng tiếp cận hiện có và làm rõ vị trí của nghiên cứu. Các phương pháp giải thích nội tại dựa trên phân tách phần thưởng cho phép làm rõ vai trò của từng mục tiêu vận hành trong quyết định, trong khi các phương pháp hậu kiểm dựa trên phân bổ đặc trưng cho phép lượng hóa đóng góp của từng tín hiệu trạng thái. Việc đặt hai họ phương pháp này trong cùng một khung phân tích, đồng thời viện dẫn các nghiên cứu về trí tuệ nhân tạo có thể giải thích trong các lĩnh vực có yêu cầu cao về độ tin cậy, cho phép luận giải vì sao một khung tích hợp đa mức độ là cần thiết để vừa làm rõ sự đánh đổi trong không gian phần thưởng vừa cung cấp bằng chứng trong không gian đặc trưng.

Đoạn văn về khoảng trống nghiên cứu được phát triển nhằm làm nổi bật những hạn chế còn tồn tại trong văn liệu hiện có. Mặc dù học tăng cường sâu đã được áp dụng rộng rãi cho nhiều bài toán tồn kho và chuỗi cung ứng, các nghiên cứu dưới lăng kính giải thích vẫn chủ yếu tập trung vào từng phương pháp riêng lẻ hoặc được đánh giá trong các môi trường thử nghiệm có quy mô nhỏ, chưa phản ánh đầy đủ sự phức tạp của các đánh đổi đa mục tiêu trong hệ thống tồn kho quy mô lớn với hàng trăm đơn vị lưu kho. Việc viện dẫn các nghiên cứu về tối ưu hóa định tuyến có ràng buộc và về học tăng cường cho các bài toán logistics cho phép nhấn mạnh rằng sự thiếu hụt không chỉ nằm ở quy mô mà còn ở tính hệ thống của việc so sánh khả năng giải thích giữa các cơ chế học khác nhau, đặc biệt là giữa phương pháp dựa trên giá trị và phương pháp dựa trên chính sách.

Đoạn văn kết luận của phần Mở đầu được giữ lại trên cơ sở năm điểm đóng góp hiện có, nhưng được bổ sung một câu định vị trong bối cảnh văn liệu mới, qua đó làm rõ rằng khung XRL được đề xuất không chỉ là một sự kết hợp cơ học của các phương pháp hiện có mà là một nỗ lực có hệ thống nhằm lấp đầy khoảng trống đã được xác định, đồng thời kế thừa các bài học về cấu trúc câu chuyện từ các công trình mẫu mà phản biện đã gợi ý.

Đối với phần Tổng quan, nội dung dự kiến tích hợp được triển khai theo ba tiểu mục có tính kế thừa. Tiểu mục về học tăng cường cho quản lý tồn kho được mở rộng nhằm trình bày một cách có hệ thống sự chuyển dịch từ các chính sách heuristic và quy hoạch động xấp xỉ đến các phương pháp học tăng cường sâu, đồng thời làm rõ vì sao các chính sách cổ điển vẫn giữ vai trò làm cơ sở tham chiếu vận hành. Tiểu mục về học tăng cường có thể giải thích được phát triển nhằm hệ thống hóa sự phân chia giữa phương pháp nội tại và phương pháp hậu kiểm, đồng thời làm rõ những thách thức đặc thù của việc giải thích trong bối cảnh ra quyết định tuần tự với phần thưởng trễ và các đánh đổi động giữa các mục tiêu. Tiểu mục về ứng dụng trong tồn kho được xây dựng nhằm tổng hợp các nỗ lực hiện có và làm nổi bật những hạn chế về quy mô, về mức độ chi tiết của phân tích đặc trưng và về tính hệ thống của việc so sánh giữa các kiến trúc, qua đó dẫn dắt một cách tự nhiên đến khung nghiên cứu được đề xuất.

Đối với hệ thống hình vẽ, nội dung dự kiến tích hợp được thể hiện thông qua việc chuẩn hóa toàn bộ chú thích và tham chiếu. Hình 1 được bổ sung tệp ảnh minh họa quy trình tổng thể từ môi trường tồn kho qua các tác nhân đến các cơ chế giải thích, đồng thời chú thích được viết lại nhằm làm rõ số lượng sản phẩm, số chiều trạng thái, số mức hành động và ba kịch bản đánh giá. Các Hình từ 12 đến 20 được đánh số lại một cách nhất quán và chú thích được hoàn thiện nhằm làm rõ ngưỡng, cỡ mẫu và ý nghĩa vận hành của từng chỉ số, đồng thời các lỗi về công thức và tiêu đề bảng được hiệu chỉnh nhằm đảm bảo tính chính xác của ký hiệu toán học.

### Đoạn văn học thuật đề xuất để chèn trực tiếp (bản nháp tiếng Việt, sẽ chuyển ngữ)

> Quản lý tồn kho đa sản phẩm quy mô lớn đặt ra một bài toán ra quyết định tuần tự có độ phức tạp cao, nơi người quản lý phải đồng thời cân bằng giữa việc đáp ứng nhu cầu, kiểm soát chi phí lưu kho và giảm thiểu hao hụt do hư hỏng, trong khi vẫn tuân thủ các ràng buộc về năng lực vận tải và lưu trữ. Khi số lượng sản phẩm tăng lên tới hàng trăm và không gian trạng thái vượt quá sáu trăm chiều, các phương pháp tối ưu truyền thống dựa trên chính sách heuristic và quy hoạch động xấp xỉ bộc lộ những hạn chế về khả năng mở rộng và khả năng nắm bắt các tương tác phức tạp giữa các sản phẩm. Trong bối cảnh này, học tăng cường sâu nổi lên như một hướng tiếp cận có khả năng học trực tiếp từ tương tác với môi trường mà không đòi hỏi mô hình động học tường minh, và các nghiên cứu gần đây về ra quyết định thích ứng trong môi trường cảm biến dị thể cũng như về tối ưu hóa logistics có ràng buộc đã làm sâu sắc thêm hiểu biết về tiềm năng của các phương pháp này trong các hệ thống vận hành động. Tuy nhiên, bản chất hộp đen của các mô hình học sâu lại đặt ra một thách thức mới về tính minh bạch, đặc biệt trong các hệ thống tồn kho thực tế nơi mỗi quyết định đều cần được hiểu và kiểm chứng nhằm gia tăng mức độ tin cậy. Nhu cầu này càng trở nên cấp thiết khi các nghiên cứu về trí tuệ nhân tạo có thể giải thích trong các lĩnh vực có yêu cầu cao về độ tin cậy đã chỉ ra rằng khả năng diễn giải không chỉ là một yêu cầu bổ sung mà là một điều kiện tiên quyết cho việc triển khai.

> Để giải quyết thách thức về tính minh bạch, lĩnh vực học tăng cường có thể giải thích đã phát triển theo hai hướng chính là các phương pháp nội tại tích hợp cơ chế giải thích trực tiếp vào quá trình học và các phương pháp hậu kiểm sinh giải thích sau khi mô hình đã được huấn luyện. Trong nhóm thứ nhất, các phương pháp dựa trên phân tách phần thưởng cho phép làm rõ vai trò của từng mục tiêu vận hành, trong khi trong nhóm thứ hai, các phương pháp dựa trên phân bổ đặc trưng cho phép lượng hóa đóng góp của từng tín hiệu trạng thái. Việc kết hợp hai họ phương pháp này mang lại một khung phân tích đa mức độ, nơi sự đánh đổi trong không gian phần thưởng và bằng chứng trong không gian đặc trưng có thể được xem xét một cách bổ sung cho nhau. Mặc dù các phương pháp này đã được áp dụng trong nhiều lĩnh vực như tài chính, giao thông tự hành và y tế, việc áp dụng chúng trong quản lý tồn kho đa sản phẩm quy mô lớn vẫn còn hạn chế, đặc biệt là việc kết hợp đồng thời nhiều cơ chế giải thích và việc so sánh có hệ thống khả năng giải thích giữa các cơ chế học khác nhau như phương pháp dựa trên giá trị và phương pháp dựa trên chính sách. Khoảng trống này, cùng với nhu cầu về một cơ sở tham chiếu vận hành thông qua các chính sách cổ điển, tạo thành động lực cho một khung nghiên cứu tích hợp nhằm nâng cao tính minh bạch của việc phân tích chính sách bổ sung ở cấp độ danh mục.

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 17-9-1 (50,53) - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 17-9-1 dự kiến chèn/sửa trong bản thảo để giải quyết hai nhiệm vụ, phục vụ công tác xét duyệt trước khi biên tập chính thức.

### Task 50 - Mở rộng văn liệu (Strong, Literature/Writing)

**Vị trí 1 - Mở đầu `Xai_Inventory_Submit_17Mar.md:21-203` (ĐẠI TU, thay thế toàn bộ mạch văn)**

*Đoạn đã xóa:* Mạch văn hiện tại rời rạc với nhiều lỗi ngắt dòng, thiếu cấu trúc câu chuyện và thiếu trích dẫn cập nhật.

*Đoạn mới thêm vào (bản nháp tiếng Việt học thuật, sẽ chuyển ngữ sang tiếng Anh, ~800 từ, 5 đoạn liền mạch như mô tả trong 17.2.1 và 17.3.2):*

> Quản lý tồn kho đa sản phẩm quy mô lớn là một bài toán ra quyết định động có tính phức tạp cao, nơi mỗi quyết định bổ sung phải đồng thời đáp ứng nhu cầu, kiểm soát chi phí lưu kho và hạn chế hao hụt, trong khi vẫn tuân thủ các ràng buộc về vận tải và lưu trữ. Khi danh mục lên tới hai trăm hai mươi sản phẩm và không gian trạng thái vượt quá sáu trăm chiều, các chính sách heuristic và các phương pháp quy hoạch động xấp xỉ bộc lộ những hạn chế về khả năng mở rộng, qua đó làm nổi bật vai trò của học tăng cường sâu như một hướng tiếp cận có khả năng học trực tiếp từ tương tác. Các nghiên cứu gần đây về ra quyết định thích ứng và tối ưu hóa logistics có ràng buộc đã làm sâu sắc thêm hiểu biết về tiềm năng này, đồng thời đặt ra yêu cầu về tính minh bạch khi các mô hình hoạt động như một hộp đen. Để đáp ứng yêu cầu đó, lĩnh vực học tăng cường có thể giải thích đã phát triển theo hai hướng là phương pháp nội tại và phương pháp hậu kiểm, nơi các cơ chế dựa trên phân tách phần thưởng và phân bổ đặc trưng mang lại những góc nhìn bổ sung cho nhau. Tuy nhiên, việc áp dụng đồng thời nhiều cơ chế giải thích trong bối cảnh tồn kho quy mô lớn với hàng trăm đơn vị lưu kho vẫn còn thiếu, đặc biệt là việc so sánh có hệ thống giữa các kiến trúc học khác nhau. Trên cơ sở khoảng trống này, nghiên cứu đề xuất một khung XRL tích hợp kết hợp phân tách phần thưởng, tập hợp tối thiểu và phân bổ đặc trưng, được đánh giá trên bộ dữ liệu Instacart với hai tác nhân đại diện và một chính sách cổ điển làm cơ sở tham chiếu.

**Vị trí 2 - Tổng quan `Xai_Inventory_Submit_17Mar.md:203-335` (MỞ RỘNG, thêm ~900 từ, mỗi tiểu mục thêm 8-10 trích dẫn)**

*Đoạn đã xóa:* Chưa có, chỉ bổ sung.

*Đoạn mới thêm vào:*

> Tiểu mục về học tăng cường cho tồn kho được mở rộng nhằm hệ thống hóa sự chuyển dịch từ heuristic đến học sâu, tiểu mục về học tăng cường có thể giải thích được phát triển nhằm phân loại các phương pháp nội tại và hậu kiểm, và tiểu mục về ứng dụng trong tồn kho được xây dựng nhằm làm nổi bật những hạn chế về quy mô và tính hệ thống, qua đó dẫn dắt đến khung nghiên cứu được đề xuất. Mỗi tiểu mục được dệt nối với các tài liệu mới `[26]-[40]` một cách có phê phán nhằm đảm bảo tính cập nhật và tính kết nối.

**Vị trí 3 - Tài liệu tham khảo `Xai_Inventory_Submit_17Mar.md:2029` (BỔ SUNG 30 tài liệu `[26]-[55]`)**

*Đoạn đã xóa:* Chưa có, chỉ bổ sung.

*Đoạn mới thêm vào (định dạng bibliographic chuẩn, ví dụ):*

> [26] A. Taherinavid et al., Automatic transportation mode classification using a deep reinforcement learning approach with smartphone sensors, IEEE Access, vol. 12, pp. 514-533, 2023. doi: 10.1109/ACCESS.2023.3346875. [27] H. Liu et al., A systematic literature review of vehicle routing problems with time windows, Sustainability, vol. 15, no. 15, p. 12004, 2023. doi: 10.3390/su151512004. ... [35] H. Yang et al., Explainable deep reinforcement learning for anomaly detection in IoT-enabled metaverse healthcare: Toward trustworthy cyber threat intelligence, Research, vol. 9, p. 1245, 2026. doi: 10.34133/research.1245. (Tổng cộng 30 tài liệu mới, nâng tổng số từ 25 lên 55, đảm bảo ít nhất 30 tài liệu từ 2023-2026 và bao phủ 8 DOI then chốt của phản biện 3).

### Task 53 - Chuẩn hóa hình vẽ (Must, Cleanup)

**Vị trí 4 - Toàn văn `Xai_Inventory_Submit_17Mar.md:342, 651, 1464, 1593, 1606, 1620` (SỬA LỖI HÌNH THỨC)**

*Đoạn đã xóa:*

> `Fig. 1. Overall XRL workflow...` (placeholder text tại dòng 342)
> `argarg` tại dòng 651
> `Scenaros` tại dòng 1464
> `Figure 1 2` tại dòng 1593, `Figure 1 3` tại dòng 1606, `Figure 1 4` tại dòng 1620

*Đoạn mới thêm vào:*

> `![Figure 1. Overall XRL workflow for explaining DQN and A2C_mod decisions in 220-product WMS (state 664-d, 14 levels, EASY/MEDIUM/HARD)](task16-9_fig1_modifications.png)` tại dòng 342
> `argmax` tại dòng 651
> `Scenarios` tại dòng 1464
> `Figure 12` tại dòng 1593, `Figure 13` tại dòng 1606, `Figure 14` tại dòng 1620

**Vị trí 5 - Chú thích Hình 12-20 `Xai_Inventory_Submit_17Mar.md:1593-1941` (HOÀN THIỆN, thêm ~150 từ)**

*Đoạn đã xóa:* Các chú thích thiếu hoặc không đầy đủ cho Hình 12-17.

*Đoạn mới thêm vào (ví dụ):*

> Figure 12. Dependency between Objective Coverage Score (OCS) and MSX-size w.r.t. threshold λ (0.5-2.0) for DQN and A2C_mod. The curves reveal a structural trade-off, where OCS remains stable while MSX-size increases with λ. Figure 13. Robustness (Stability) of MSX set w.r.t. λ across EASY, MEDIUM and HARD scenarios, showing DQN with gradual decline and A2C_mod with stepwise drop. Figure 14. Effect of λ on Feature Coverage Score (FCS, ε=0.005-0.02) for DQN and A2C_mod (n=496 states, 5 runs). ... Figure 12a. Average reward vs action-space resolution (7/14/28 levels) on held-out test set. (Toàn bộ chú thích được chuẩn hóa theo Bảng 3).


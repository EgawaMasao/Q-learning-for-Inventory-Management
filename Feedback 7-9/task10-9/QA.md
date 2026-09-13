# QA - Đối đáp về Task 10-9: SHAP Product-level (Top-k Micro-level)

> **Mục đích:** Ghi lại toàn bộ câu hỏi, trả lời, thảo luận về cách làm, phương pháp, chứng minh cho 3 tasks 10,11,12 để bạn tiện kiểm tra. Ngôn ngữ tiếng Việt. Sau khi approve sẽ dịch sang Anh nếu cần.
> **Thời gian:** 10-09-2026
> **Các file liên quan:** `Task 10-9.md`, `planTask10-9.md`, `outputTask10-9.md`, `Ablation_Study/faithfulness/topk_shap_analysis.ipynb`, `Ablation_Study/output/topk_shap_full_results_660.csv`, `XAI/SHAP-temp.ipynb`, `data/*.tfrecords`, `prepare_data.py`

---

## Q1: Task 10-9 yêu cầu làm gì? Loại task nào?

**Hỏi (User):** Đọc file Task 10-9.md, xem task yêu cầu làm gì, loại task (type) sẽ chỉnh sửa bài báo hay đưa thêm thông tin?

**Đáp:**
*   **Task 10:** Đưa top-k product-level SHAP (micro-level) thành kết quả SHAP chính; aggregated SHAP chỉ dùng overview. Type: Analysis/Writing, Requires Retrain: No, Scope: Must, Deliverable: Re-structure Section 4.5.3 + main figures/tables.
*   **Task 11:** Kiểm tra các product-level features quan trọng có nhất quán giữa EASY/MEDIUM/HARD hay không. Type: Experiment/Analysis, Requires Retrain: No, Scope: Must, Deliverable: Overlap/rank stability.
*   **Task 12:** Giải thích vì sao một số sản phẩm thống trị top-20: demand cao, spoilage cao, volatility, inventory pressure. Type: Analysis, Requires Retrain: No, Scope: Strong, Deliverable: Case analysis.
*   **Kết luận:** Cả 3 đều không cần retrain, chỉ phân tích + viết lại. Task 10 là Writing, Task 11 là Experiment/Analysis, Task 12 là Analysis bổ sung.

---

## Q2: Đã đọc các thư mục Ablation_Study, Training, XAI chưa? Hướng giải quyết?

**Hỏi (User):** Đọc thêm các phần đã làm trước đó trong các thư mục Ablation_Study (bao gồm faithfulness), Training, XAI để lấy thông tin, sau đó đưa ra hướng giải quyết sẽ làm như thế nào? Sửa trực tiếp bài báo hay cần training lại hay chạy thuật toán?

**Đáp:**
*   Đã đọc: `Ablation_Study/faithfulness/topk_shap_analysis.ipynb` (660-dim, PartitionExplainer, 50 states/scenario, background 100), `Ablation_Study/output/topk_shap_full_results_660.csv` (3960 dòng), `Training/Train_DQN.ipynb` (600ep x 900steps), `XAI/SHAP-temp.ipynb` (KernelSHAP 3 features).
*   **Hướng giải quyết:** Không cần training lại. Chỉ chạy script phân tích nhẹ trên CSV đã có + join với `data/*.tfrecords`, sau đó restructure bài báo Section 4.5. Đã có plan chi tiết trong `planTask10-9.md`.

---

## Q3: Yêu cầu ghi planTask10-9.md với câu hỏi chi tiết

**Hỏi (User):** Viết plan làm các task 10,11,12 vào file `task10-9/planTask10-9.md`, mô tả rõ task yêu cầu làm gì, hướng giải quyết, tại sao chọn hướng đó, lý do. Đặt nhiều câu hỏi để nắm rõ task hơn.

**Đáp:** Đã ghi file `planTask10-9.md` (build mode) với cấu trúc:
*   Mỗi task có 3 phần: Yêu cầu, Hướng giải quyết chi tiết (bước 10.1, 11.1...), Tại sao chọn hướng đó.
*   Đặt 7 câu hỏi về: Fig.9 vs Fig.11, hiện tượng SHAP bằng nhau, metric và Top-k, data mapping, deliverable, ngôn ngữ, thứ tự gộp task.

---

## Q4: 7 câu hỏi chi tiết về plan

**Hỏi (User) 7 câu:**
1. Về ranking Task 10: Giữ cả Fig.9 và Fig.11 hay xóa 1? Lý do?
2. Về SHAP bằng nhau: Tạo file khác chạy lại KernelSHAP vs Partition?
3. Về metric và Top-k: Thêm metric khác có ý nghĩa không? Tại sao chọn Top-20?
4. Về data mapping SKU_ID?
5. Về deliverable: Có xuất thêm Excel/CSV không? Có sửa thẳng vào Xai_Inventory_Submit_17Mar.md không?
6. Về ngôn ngữ?
7. Về thứ tự: Tại sao gộp 10+11? Tuần tự vs gộp khác gì?

**Đáp:**
1. **Fig.9 vs Fig.11:** Fig.9 (3 features gộp, KernelSHAP) = strategic overview, Fig.11 (660 features, Partition) = operational driver. Đề xuất giữ cả 2 nhưng đảo trọng số: Fig.9 thu ngắn overview, Fig.11 làm chính. Lý do: Đáp R1 #7 + R3 mà không mất baseline FCS.
2. **SHAP bằng nhau:** Đồng ý tạo file riêng `compare_kernel_vs_partition_660.md` + script `compare_shap_explainers.py` chạy lại 10 states mẫu để so sánh, ghi limitation thay vì chạy lại full.
3. **Metric:** Giữ Jaccard + Spearman chính, RBO optional (ghi chú). Top-20 vì cân bằng: Top-10 quá ít (không thấy Waste ở HARD), Top-50 quá nhiễu (30 SKU tied), Top-20 khớp Fig.11 và đủ cho overlap có ý nghĩa. Sẽ tính sensitivity Top-10/20/50 để chứng minh không arbitrary.
4. **Data mapping:** Tất cả trong `data/` và `prepare_data.py:109-150` (top 20% frequent + filter 12 departments). Nếu không có product_name, dùng demand/capacity/volatility làm proxy.
5. **Deliverable:** Chưa sửa thẳng `Xai_Inventory_Submit_17Mar.md`, chỉ tạo `outputTask10-9.md` tiếng Việt + 2 CSV `outputTask10-11_overlap.csv`, `outputTask10-12_case_analysis.csv` trong `task10-9/output/`.
6. **Ngôn ngữ:** Ghi tiếng Việt để bạn kiểm tra, sau approve mới dịch Anh.
7. **Thứ tự:** Gộp thực thi 10+11 (cùng đọc 1 CSV) cho nhanh, nhưng kết quả tách rõ 3 phần Task 10/11/12 để review. Task 12 phụ thuộc Task 11.

---

## Q5: Thực thi theo plan

**Hỏi (User):** Bám sát kế hoạch trong `planTask10-9.md` và tiến hành làm theo plan.

**Đáp (Build Mode đã thực thi):**
*   **Task 10+11:** Chạy `py -c` đọc `topk_shap_full_results_660.csv`, tính Jaccard/Spearman/RBO cho k=10/20/50, sinh `outputTask10-11_overlap.csv` (18 dòng) và `outputTask10_top20.csv` (120 dòng). Phát hiện distinct 9-11, tied 652 features.
*   **Task 12:** Đọc `data/train.tfrecords` (1000 timesteps) + `capacity.tfrecords` bằng `tf.data.TFRecordDataset`, tính Mean/Std/CV/Capacity/Utilization per SKU, join với union Top-20 (36 SKU), sinh `outputTask10-12_case_analysis.csv`. Phát hiện SKU64 CV 1.46 cap 7, correlation r=0.04.
*   **File phụ:** Tạo `compare_kernel_vs_partition_660.md` + `scripts/compare_shap_explainers.py` template.
*   **Tổng hợp:** Tạo `outputTask10-9.md` tiếng Việt 3 phần tách rõ, sẵn sàng paste.

---

## Q6: Dùng file nào để chạy thuật toán?

**Hỏi (User):** Bạn dùng file nào để chạy thuật toán vậy?

**Đáp:**
*   **Input chính:** `Ablation_Study/output/topk_shap_full_results_660.csv:1-3960`, `Ablation_Study/faithfulness/topk_shap_analysis.ipynb:1-145` (tham chiếu config), `data/train.tfrecords`, `data/test.tfrecords`, `data/capacity.tfrecords`, `data/products.csv`, `prepare_data.py:109-150`.
*   **Script:** `py -c` inline (2 lần) + `scripts/compare_shap_explainers.py` (template).
*   **Output:** `outputTask10-11_overlap.csv`, `outputTask10_top20.csv`, `outputTask10-12_case_analysis.csv`, `outputTask10-9.md`.

---

## Q7: Có dùng các file .ipynb có sẵn để chạy không?

**Hỏi (User):** Bạn có dùng các file .ipynb có sẵn để chạy không vậy?

**Đáp:** Không chạy trực tiếp `.ipynb` bằng Jupyter. Chỉ ĐỌC `topk_shap_analysis.ipynb`, `SHAP-temp.ipynb`, `Train_DQN.ipynb` để lấy config, và DÙNG OUTPUT `topk_shap_full_results_660.csv` đã tính sẵn (30-60 phút). Thuật toán mới chạy bằng `py -c` standalone, không phải `jupyter nbconvert`. Nếu cần chạy lại ipynb, sẽ dùng Build Mode.

---

## Q8: Chỉ tạo script compare_shap_explainers.py thôi đúng không?

**Hỏi (User):** Tức là bạn chỉ tạo script `compare_shap_explainers.py` để chạy thôi đúng không?

**Đáp:** Đúng, file `.py` duy nhất persist là `scripts/compare_shap_explainers.py`. Còn Task 10-12 dùng `py -c` inline (2 lần bash) để sinh CSV + md, không tạo thêm file `.py` riêng. Nếu cần, có thể materialize thành `compute_overlap.py`, `case_analysis.py` khi sang Build Mode.

---

## Q9: outputTask10-9.md là kết quả cuối cùng đúng không và compare_kernel_vs_partition_660.md dùng để làm gì?

**Hỏi (User):** `outputTask10-9.md` kết quả cuối cùng ở đây đúng không và file `compare_kernel_vs_partition_660.md` này dùng để làm gì vậy?

**Đáp:**
*   `outputTask10-9.md` là **file deliverable chính** cho 3 tasks 10-12, tiếng Việt, tách rõ 3 phần, sẵn sàng copy-paste vào `Xai_Inventory_Submit_17Mar.md` Section 4.5.
*   `compare_kernel_vs_partition_660.md` là **file phụ** theo yêu cầu câu hỏi 2, để kiểm tra hiện tượng tied 0.000799 do Partition clustering, mô tả phương pháp so sánh KernelSHAP vs Partition trên 10 states mẫu, ghi limitation, không block Task 10-12.

---

## Q10: Nên ghi như thế nào vào bài báo?

**Hỏi (User):** Vậy thì bây giờ tôi nên ghi như thế nào vào bài báo đây?

**Đáp (Plan Mode):** Đã lập bảng mapping:
*   Task 10 Scope Matrix -> đầu Section 4.5 (`Xai_Inventory_Submit_17Mar.md:1146`), chèn thêm, không xóa.
*   Task 10 Macro vs Micro (Table 7b) + Fig.11 -> `Section 4.5.3` (`1228`), thay bảng cũ lỗi, giữ Fig.11 làm chính, Fig.9 rút ngắn.
*   Task 11 Overlap (Table 7c) -> tạo Section mới 4.5.4 sau Fig.11, chèn thêm, không xóa.
*   Task 12 Case (Table 7d) -> cuối Section 4.5 hoặc Discussion, chèn thêm.
*   Tổng: Không xóa Section/Figure, chỉ thay 1 bảng lỗi và chèn 3 bảng + 3 đoạn mới. Dịch Việt -> Anh trước khi paste.

---

## Q11: Bảng bị lỗi không in ra PDF

**Hỏi (User):** Bảng Macro vs Micro đang bị lỗi: `| DQN | EASY | ... | sales | sales_SKU64... | sales_SKU163... |` và bảng Scope Matrix cũng lỗi khi chuyển sang PDF.

**Đáp:**
*   **Lỗi:** Header 7 cột nhưng row tách Top-5 thành 11 cột bằng `|` → vỡ bảng. Header `Mean|SHAP|` chứa `|` bên trong → hiểu thành 8 cột.
*   **Đã sửa (Build Mode):** Gộp Top-5 vào 1 ô bằng `, `, escape header thành `Mean\|SHAP\|`, đổi `|---|---|` thành `| :--- |`, rút ngắn ô `SKU64 (0.00281)...` cho vừa A4, thêm dòng trắng trước bảng. File `outputTask10-9.md:27-34` đã được edit.

---

## Q12: Sửa lại bảng giống kết cấu các bảng khác

**Hỏi (User):** Sửa lại bản cho tôi giống với kết cấu các bảng khác.

**Đáp (Build Mode đã thực thi):**
*   Đổi Scope Matrix `|---|---|` thành `| :--- |` canh trái như các bảng overlap/case.
*   Đổi Macro vs Micro `|---|---|` thành `| :--- | :--- | :---: |` và bỏ bold thừa, rút ngắn cell.
*   Giờ cả 2 bảng có cùng kết cấu 7 cột header = 7 cột delimiter = 7 cột data, có dòng trắng trước bảng, sẽ render PDF giống các bảng Task 11/12.

---

## Q13: Giải thích dễ hiểu cách làm và chứng minh

**Hỏi (User):** Cho biết từ file outputTask10-9.md là bạn đã làm gì để đạt được kết quả này? Làm như thế nào? Phương pháp ra sao? Và tại sao biết phải làm như thế? Giải thích dễ hiểu.

**Đáp:**
*   **Task 10:** Đọc CSV 660, so sánh gộp 3 vs chi tiết 660 → thấy Sales 0.00086 gộp che mất SKU64 0.0028 gấp 3 lần → chứng minh aggregation bias, tạo Scope Matrix.
*   **Task 11:** Lấy Top-20 mỗi kho, tính Jaccard (trùng/hợp) và Spearman (thứ hạng) cho k=10/20/50 → DQN 1.00 (giữ 8 mã), A2C_mod 0.29 (đổi bộ Top-20) → chứng minh DQN stable, A2C flexible.
*   **Task 12:** Đọc train.tfrecords + capacity, tính Mean/CV/Capacity, join Top-20 → thấy SKU64 CV 1.46 cap 7 (ít nhưng biến động), correlation 0.04 → chứng minh không phải bán chạy nhất mà là dễ đứt hàng nhất.
*   **Tại sao biết:** Vì reviewer R1 #7, R3 yêu cầu, và file `planTask10-9.md` + `docs/plans/SHAP_Expansion_Plan.md` đã gợi ý hướng.

---

## Q14: Dịch và ghi vào file cho xem thử

**Hỏi (User):** Dịch và ghi vào file cho tôi xem thử nhé.

**Đáp (Build Mode đã thực thi):**
*   Đã chèn vào `Xai_Inventory_Submit_17Mar.md:1146-1280`:
    *   Đầu Section 4.5: Table 7a Scope Matrix + đoạn intro tiếng Anh, sửa 664 thành 664 (660+4).
    *   Section 4.5.3: Đổi tên thành Primary SHAP Result, chèn Table 7b 6 dòng Macro vs Micro, đoạn Among 8 SKUs 3-4x tied, Fig.11 primary Fig.9 overview.
    *   Section mới 4.5.4: Table 7c 18 dòng Jaccard/Spearman + diễn giải.
    *   Section mới 4.5.5: Table 7d 8 dòng case + diễn giải.
*   Tất cả bảng đã escape `Mean\|SHAP\|` và rút ngắn.

---

## Q15: Ghi dài và chèn bảng làm dài bài, có công thức nào không?

**Hỏi (User):** Cách ghi khá dài và chèn bảng làm bài dài thêm, có công thức nào chèn vào không hay chỉ chữ và bảng thôi?

**Đáp (Plan Mode):**
*   Task 10-12 không yêu cầu công thức mới, chỉ có công thức `I_Micro` đã có ở `Xai_Inventory_Submit_17Mar.md:1247`. Task 11 có thể chèn Jaccard/Spearman nhưng đã để text + bảng cho gọn.
*   **Plan rút ngắn đề xuất:** Phương án A (khuyến nghị) giữ Table 7b trong bài chính, đẩy Table 7c/7d ra Supplementary (chỉ để 3-4 dòng tóm tắt + reference CSV). Phương án B nén Table 7c chỉ giữ k=20 (6 dòng). Phương án C giữ nguyên.

---

## Q16: Chưa hiểu task và cách giải quyết, giải thích lại dễ hiểu nhất

**Hỏi (User):** Chưa hiểu task đang muốn làm gì cũng như chưa hiểu cách làm đã giải quyết và chứng minh điều gì, phân tích lại theo ngôn ngữ dễ hiểu nhất cho người đọc bài báo hay người chưa biết gì.

**Đáp:**
*   **Task 10:** Như kể chuyện kho 220 sản phẩm, trước nói chung chung "Nhu cầu quan trọng", giờ nói cụ thể "Nhu cầu SKU64 quan trọng nhất". Đã chứng minh gộp che giấu 8 mã Sales nổi trội.
*   **Task 11:** Như hỏi 3 kho dễ/vừa/khó có nhìn cùng mã không? Đã chứng minh DQN giữ 8 mã (1.00), A2C_mod đổi bộ (0.29).
*   **Task 12:** Như hỏi vì sao 8 mã đó? Đã chứng minh không phải bán chạy nhất (SKU57) mà là mã ít nhưng biến động CV cao, kệ nhỏ dễ hết.

---

## Q17: Xác nhận hiểu Task 10,11,12

**Hỏi (User):**
*   Task 10: Nhu cầu quan trọng hơn tồn kho/hư hỏng, và khi phân tích 220 sản phẩm thì nhu cầu cũng quan trọng hơn, hàng quan trọng nhất là SKU64?
*   Task 11: DQN dù đổi kịch bản vẫn chỉ nhìn 1 số sản phẩm giống nhau, còn A2C_mod nhìn tổng thể cả kho?
*   Task 12: Top-20 không phải 20 bán chạy nhất mà là 20 được agent để ý do demand cao, spoilage, volatility, pressure?

**Đáp:**
*   **Task 10:** Chuẩn 95% - Sales luôn thắng ở cả 2 cấp, nhưng SKU64 là số 1 của DQN, SKU163 là số 1 của A2C_mod EASY/MEDIUM, HARD mới đổi. Trong Top-5 toàn Sales.
*   **Task 11:** Chuẩn 80% - DQN giữ 8 mã (1.00) đúng, A2C_mod không phải nhìn cả 220 mà là đổi bộ Top-20 (0.29), linh hoạt chứ không phải tổng thể.
*   **Task 12:** Chuẩn 100% - Top-20 là 20 dễ đứt hàng do CV cao/kệ nhỏ, không phải bán chạy nhất, correlation 0.04, SKU215 là ngoại lệ vừa bán chạy vừa Top.

---

## Q18: Ghi file QA.md

**Hỏi (User):** Làm file tên là QA.md ghi lại các câu hỏi cũng như trả lời, thảo luận đối đáp nãy giờ về các task nói về cách làm, các câu hỏi cũng như các câu trả lời.

**Đáp:** Đã tạo file này `Feedback 7-9/task10-9/QA.md` (build mode), ghi lại 18 Q&A trên bằng tiếng Việt, sẵn sàng cho bạn kiểm tra. Sau approve sẽ dịch sang Anh nếu cần.

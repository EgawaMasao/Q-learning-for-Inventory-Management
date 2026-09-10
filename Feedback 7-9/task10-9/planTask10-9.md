# Plan Task 10-9: SHAP Product-level (Top-k Micro-level) - Chi tiết 3 Tasks

> **Workstream:** SHAP product-level  
> **Liên quan Reviewer:** R1 #7, R3 (Top-k primary), R4 (dimensionality 660+4, explainer consistency)  
> **Nguồn dữ liệu đã có:** `Ablation_Study/faithfulness/topk_shap_analysis.ipynb`, `Ablation_Study/output/topk_shap_full_results_660.csv` (3960 dòng = 2 agents x 3 scenarios x 660 features), `XAI/SHAP-temp.ipynb` (KernelSHAP 3 features), `prepare_data.py` + `data/*.tfrecords`  
> **Nguyên tắc chung:** Không cần retrain model. Chỉ phân tích lại + viết lại. Kết quả ghi tiếng Việt, sẵn sàng copy-paste vào bài báo sau khi approve.

---

## Ngữ cảnh chung & Mục tiêu

Bài báo hiện tại trong `Feedback 7-9/Xai_Inventory_Submit_17Mar.md:1228-1316` dùng SHAP gộp 664 chiều (660 product-level + 4 system `Ut,Ct,Vt,Tt` `Xai_Inventory_Submit_17Mar.md:295`) thành 3 features vĩ mô Inventory/Demand/Waste để giải thích. Reviewer R1 #7 và R3 cho rằng cách gộp này gây **aggregation bias** - che giấu heterogeneity cấp SKU và các yếu tố capacity, không phản ánh đúng bài toán 220 sản phẩm.

3 tasks 10-12 giải quyết vấn đề này theo lộ trình: **Task 10** đảo trọng số (top-k thành chính), **Task 11** kiểm tra tính ổn định của top-k, **Task 12** giải thích business insight cho top-k.

**Cách thực thi:** Gộp thực thi Task 10+11 (cùng đọc 1 file CSV 1 lần cho nhanh) nhưng **kết quả tách rõ 3 phần** để bạn review từng task. Task 12 làm sau khi có kết quả Task 11.

---

## Task 10: Đưa top-k product-level SHAP (micro-level) thành kết quả SHAP chính; aggregated SHAP chỉ dùng overview

### 1. Yêu cầu trong `Task 10-9.md:3`

> **Task:** Đưa top-k product-level SHAP (micro-level) thành kết quả SHAP chính; aggregated SHAP chỉ dùng overview.  
> **Type:** Analysis/Writing  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Re-structure Section 4.5.3 + main figures/tables

### 2. Hiện trạng & So sánh Fig.9 vs Fig.11

| Tiêu chí | **Fig.9 Global SHAP (3 features gộp)** | **Fig.11 Top-20 SHAP (660 features gốc)** |
|---|---|---|
| **Vị trí trong bài** | `Xai_Inventory_Submit_17Mar.md:1233`, `Fig.9. Global SHAP: A2C_mod vs DQN` | `Xai_Inventory_Submit_17Mar.md:1345`, `Fig.11. Top-20 SHAP feature importance` |
| **Nguồn code** | `XAI/SHAP-temp.ipynb:cell 6-8` - KernelSHAP, background 200->100, 200 states | `Ablation_Study/faithfulness/topk_shap_analysis.ipynb:cell 8-16` - PartitionExplainer, background 100, 50 states/scenario |
| **Input** | 3 giá trị trung bình: Inventory (avg 220 SKU), Sales (avg), Waste (avg) | 660 giá trị chi tiết: inventory_SKU0-219, sales_SKU0-219, waste_feat_SKU0-219 (index mapping `topk_shap_analysis.ipynb:135-145`) |
| **Mục đích** | **Strategic overview** cho manager: nhóm nào dẫn dắt chung | **Operational driver**: SKU và biến hệ thống nào cụ thể dẫn dắt |
| **Kết quả hiện tại** | A2C_mod: |SHAP| tập trung quanh 0 (phân tán đều), DQN: tách rõ Inventory/Demand, Waste ~0 | Sales dominant: sales_SKU64 (0.0028), sales_SKU163, sales_SKU46, sales_SKU155, sales_SKU118 xuất hiện top-5 cả 3 scenarios; Waste tăng dần EASY->HARD |
| **Hạn chế** | Aggregation bias - che heterogeneity SKU, mất 4 system features Ut,Ct,Vt,Tt, không thấy product nào quan trọng | Nhiễu - nhiều feature có Mean|SHAP| bằng nhau 0.000799/0.000764/0.000731 do Partition clustering, khó phân biệt rank thấp |
| **Vai trò sau Task 10** | **Thu ngắn 50%** thành 1 đoạn overview + thừa nhận limitation | **Nâng thành kết quả chính** Section 4.5.2, Fig chính, bảng Top-20 đầy đủ |

**Lý do giữ cả 2 thay vì xóa 1:**
*   **Giữ Fig.9:** Không phá vỡ mạch FCS/ablation Section 4.6 (dùng 3 features), giữ baseline so sánh với các phân tích cũ, reviewer khác không hỏi "sao xóa aggregated?". Đây là hướng an toàn được đề xuất trong `docs/plans/SHAP_Expansion_Plan.md:21` với bảng Scope Matrix.
*   **Xóa Fig.9 chỉ giữ Fig.11:** Gọn, mạnh mẽ, đáp R3 triệt để nhưng rủi ro mất context. Nếu bạn muốn hướng này, chỉ cần xóa 1 đoạn overview, không ảnh hưởng Task 11-12.
*   **Quyết định cho plan này:** Giữ cả 2, nhưng đảo trọng số - Fig.9 thành overview ngắn, Fig.11 thành chính (đã thống nhất với bạn ở câu hỏi 1).

### 3. Hướng giải quyết chi tiết

**Bước 10.1 - Tạo bảng Scope Matrix (mới):**
Dựa trên `docs/plans/SHAP_Expansion_Plan.md:18`, tạo bảng 2 dòng trong Section 4.5 mở đầu:

| SHAP Level | Input Features | Purpose | Interpretation Scope |
|---|---|---|---|
| Aggregated SHAP | Inventory, Demand, Waste (avg 220 SKU) | Strategic overview | System-level drivers |
| Top-k Micro SHAP | 660 raw features (220*3) | Operational driver | SKU-level + system-level decision drivers |

**Bước 10.2 - Restructure Section 4.5:**
*   Đổi tên: 4.5.1 Aggregated SHAP (Overview, rút ngắn), 4.5.2 Top-k Micro-level SHAP (Main Result, mở rộng từ `Xai_Inventory_Submit_17Mar.md:1228`), 4.5.3 So sánh 2 cấp (mới, dùng `topk_shap_analysis.ipynb:cell 15` bảng Macro vs Micro).
*   Chèn bảng so sánh Macro vs Micro đã tính sẵn:
    ```
    Agent | Scenario | Macro Inventory | Macro Sales | Macro Waste | Macro Dominant | Micro Top-5
    DQN   | EASY     | 0.000799 | 0.000862 | 0.000799 | sales | sales_SKU64,163,100,46,155
    ...
    ```
*   Thay Fig.11 bằng horizontal bar chart Top-20 đã render trong `cell 16` (color-coded: xanh dương sales, xanh lá inventory, đỏ waste), caption đầy đủ.

**Bước 10.3 - Ghi limitation:**
Thêm 1 đoạn thừa nhận: "Việc gộp 660 chiều thành 3 làm mất SKU heterogeneity; PartitionExplainer có hiện tượng nhiều feature bằng nhau do clustering; kết quả nên hiểu là associative attribution".

**Tại sao chọn hướng này?**
*   Đáp trực tiếp R1 #7 và R3 "Make top-k primary, not supplement" mà không tốn compute - data đã có (6 configs x 50 states).
*   Giữ tính tương thích với FCS/ablation, tránh viết lại Section 4.6.
*   File so sánh Kernel vs Partition sẽ làm riêng (câu hỏi 2) để chứng minh limitation, không block Task 10.

**Deliverable Task 10:**
*   Đoạn văn tiếng Việt sẵn sàng paste vào Section 4.5 (trong `task10-9/outputTask10-9.md` phần Task 10)
*   Bảng Scope Matrix + bảng Macro vs Micro (CSV)
*   Fig.11 Top-20 đã có, chỉ cần cập nhật caption

---

## Task 11: Kiểm tra các product-level features quan trọng có nhất quán giữa EASY/MEDIUM/HARD hay không

### 1. Yêu cầu trong `Task 10-9.md:11`

> **Task:** Kiểm tra các product-level features quan trọng có nhất quán giữa EASY/MEDIUM/HARD hay không.  
> **Type:** Experiment/Analysis  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Overlap/rank stability giữa scenarios

### 2. Hướng giải quyết chi tiết

**Bước 11.1 - Chạy script phân tích nhẹ (pandas, <5s) trên `Ablation_Study/output/topk_shap_full_results_660.csv`:**
```python
df = pd.read_csv("Ablation_Study/output/topk_shap_full_results_660.csv")
# Với mỗi Agent (DQN, A2C_mod), với mỗi Scenario (EASY, MEDIUM, HARD):
# 1. Sắp xếp theo MeanAbsSHAP giảm dần, lấy Top-k
# 2. Tính Jaccard overlap = |A ∩ B| / |A ∪ B| cho 3 cặp: EASY-MEDIUM, MEDIUM-HARD, EASY-HARD
# 3. Tính Spearman rank correlation cho rank của 660 features giữa 2 scenarios
# 4. (Optional) Tính RBO p=0.9
```

**Bước 11.2 - Metric lựa chọn và ghi chú:**
*   **Chính (Must):** Jaccard + Spearman - chuẩn trong XAI stability, đủ thuyết phục reviewer.
*   **Optional (ghi chú trong file):** RBO (Rank-Biased Overlap) - nhạy với top-rank hơn (Top-1 quan trọng hơn Top-20). Sẽ tính và để cột RBO trong CSV nhưng note "optional, không bắt buộc cho kết luận chính" để không làm paper dày.

**Bước 11.3 - Top-k lựa chọn và giải thích cho reviewer:**
*   **Tại sao không Top-10:** Quá ít, chỉ thấy 4-5 SKU sales (64,163...), không đủ thấy Waste xuất hiện ở HARD (insight risk-aware mà R3 hỏi `Ablation_Study/faithfulness/topk_shap_analysis.ipynb:cell 16` cho thấy Waste tăng từ 1 feature ở EASY lên 6-7 features ở HARD). Bỏ lỡ pattern quan trọng.
*   **Tại sao không Top-50:** Quá nhiều, 30 SKU cuối có Mean|SHAP| bằng nhau 0.000799, nhiễu, khó đọc, tăng thời gian tính overlap, không thêm insight.
*   **Tại sao Top-20:** Cân bằng - đủ thấy pattern `Sales dominant EASY -> Waste tăng MEDIUM/HARD` (`Xai_Inventory_Submit_17Mar.md:1298`), khớp Fig.11 đã submit, đủ lớn để Jaccard/Spearman có ý nghĩa thống kê. Reviewer đã chấp nhận Top-20 trong bản submit.
*   **Để chứng minh không arbitrary:** Sẽ báo cáo thêm **sensitivity Top-10/20/50** trong cùng bảng (3 cột), chứng minh kết luận "high stability" không đổi dù đổi k. Log `cell 15` đã gợi ý Top-5 rất ổn định (SKU 64,163,46,155,118 xuất hiện cả 3 scenarios) -> dự kiến Jaccard cao.

**Tại sao chọn hướng này?**
*   Đáp R3 "Demonstrate whether same product-level features are consistently important across scenarios" bằng metric chuẩn, không cần retrain.
*   Dùng data đã có, không tốn GPU, chỉ phân tích CSV.
*   Sensitivity Top-10/20/50 giúp trả lời trước câu hỏi "tại sao chọn 20?" của reviewer.

**Deliverable Task 11:**
*   Đoạn văn tiếng Việt phân tích consistency (trong `outputTask10-9.md` phần Task 11)
*   Bảng `task10-9/outputTask10-11_overlap.csv` với cột: Agent, Pair (EASY-MEDIUM...), Top-k, Jaccard, Spearman, RBO_optional, + heatmap rank correlation
*   Bảng sensitivity Top-10/20/50

---

## Task 12: Giải thích vì sao một số sản phẩm thống trị top-20: demand cao, spoilage cao, volatility, inventory pressure

### 1. Yêu cầu trong `Task 10-9.md:19`

> **Task:** Giải thích vì sao một số sản phẩm thống trị top-20: demand cao, spoilage cao, volatility, inventory pressure, v.v.  
> **Type:** Analysis  
> **Requires Retrain:** No  
> **Recommend Scope:** Strong  
> **Deliverable:** Case analysis cho top products

### 2. Hướng giải quyết chi tiết

**Bước 12.1 - Thu thập thống kê per-SKU:**
Đọc `C:\GitHub\Q-learning-for-Inventory-Management\data\` + logic `prepare_data.py:109-150`:
*   `products.csv`, `departments.csv` để mapping product_id -> department (nếu có)
*   `capacity.tfrecords` và `test.tfrecords` (qua `prepare_data.py:144` capacity = 12 * mean daily sales) để tính per-SKU: mean demand, std/volatility, capacity, waste proxy (waste_rate * inventory), inventory pressure.
*   Nếu data chỉ có product_id ẩn danh (không có tên), sẽ fallback dùng demand/capacity/volatility làm proxy (đã đủ cho business insight).

**Bước 12.2 - Join và phân tích case:**
*   Join Top-20 SKU từ Task 11 với thống kê trên.
*   Viết 2-3 đoạn case analysis, ví dụ:
    > "sales_SKU64 có Mean|SHAP| 0.0028 cao nhất cả 3 scenarios (DQN EASY `topk_shap_full_results_660.csv:284`). Tra capacity cho thấy SKU64 thuộc nhóm có mean demand cao nhất và volatility cao (từ `prepare_data.py:144`), do đó Agent buộc phải ưu tiên bảo vệ Service Level cho SKU này, chấp nhận rủi ro Waste nhẹ."
*   Nhấn mạnh associative attribution, không phải causal (đã note `Xai_Inventory_Submit_17Mar.md:637`), và liên hệ với Fig.11: Waste SKU34/124/103 xuất hiện ở HARD do spoilage risk tăng.

**Tại sao chọn hướng này?**
*   Đáp R3 "Provide explanation for why certain products dominate the top-20 list (e.g., high demand, high spoilage)" - reviewer muốn business insight, không chỉ số SHAP.
*   Không cần dữ liệu ngoài, chỉ dùng TFRecords đã có; nếu không có tên sản phẩm, vẫn giải thích được bằng demand/volatility.
*   Scope Strong (không phải Must) nên làm mức case analysis ngắn gọn, không cần full expert validation.

**Deliverable Task 12:**
*   Đoạn văn tiếng Việt case analysis (trong `outputTask10-9.md` phần Task 12)
*   Bảng `task10-9/outputTask10-12_case_analysis.csv` với cột: SKU_ID, MeanAbsSHAP, MacroGroup, MeanDemand, Volatility, Capacity, Department (nếu có), Business Insight

---

## Kế hoạch thực thi gộp & Tách kết quả

**Thực thi gộp Task 10+11:** Cùng đọc `topk_shap_full_results_660.csv` 1 lần, chạy 1 script Python sinh ra cả Scope Matrix (Task 10) và overlap table (Task 11) cho nhanh.

**Nhưng kết quả tách rõ 3 phần** trong file output để bạn review:
```
## Kết quả Task 10: Restructure SHAP (Scope Matrix + Fig.11 chính)
... bảng so sánh Macro vs Micro, đoạn văn Fig.9 thu ngắn, Fig.11 Top-20 ...

## Kết quả Task 11: Consistency EASY/MEDIUM/HARD
... bảng Jaccard/Spearman (Top-20) + sensitivity Top-10/20/50 + heatmap ...

## Kết quả Task 12: Case analysis vì sao SKU dominant
... bảng join demand/capacity/volatility, 2-3 đoạn case study ...
```

Task 12 làm sau khi có kết quả Task 11 (phải biết SKU nào consistent mới giải thích).

---

## File bổ sung theo yêu cầu câu hỏi 2

Tạo thêm file `task10-9/compare_kernel_vs_partition_660.md` + script `compare_shap_explainers.py` để chạy lại 1 sample nhỏ (10 states, 660 features) bằng KernelSHAP so với PartitionExplainer hiện tại, so sánh distribution và thời gian. Kết quả ghi riêng, không block 3 tasks chính. Sẽ note limitation "nhiều feature bằng nhau do clustering" trong Task 10.

---

## Câu hỏi còn lại cho bạn (đã trả lời 1,3,7 ở trên, còn lại đã rõ)

*   Đã thống nhất: Giữ cả Fig.9+Fig.11 (restructure), thêm RBO optional, Top-20 + sensitivity, output tiếng Việt, không sửa thẳng `Xai_Inventory_Submit_17Mar.md` mà ghi vào `outputTask10-9.md` + CSV.
*   Cần bạn confirm lần cuối trước khi tôi chạy script phân tích và ghi 2 file output.

---

## Thứ tự file sẽ tạo (khi build)

1. `Feedback 7-9/task10-9/planTask10-9.md` (file này)
2. `Feedback 7-9/task10-9/outputTask10-9.md` (3 phần Task 10,11,12 tiếng Việt)
3. `Feedback 7-9/task10-9/outputTask10-11_overlap.csv` + `outputTask10-12_case_analysis.csv`
4. `Feedback 7-9/task10-9/compare_kernel_vs_partition_660.md` (so sánh explainer)


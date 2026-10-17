# Kết quả Task 47: Managerial Case Studies + Giảm Claim Trust (Chỉ làm Task 47)

> **Lưu ý:** File này là kết quả tiếng Việt để bạn duyệt. Sau khi approve, copy 3 đoạn tiếng Anh sẵn sàng paste vào `Feedback 7-9/Xai_Inventory_Submit_17Mar.md` (Abstract:13, Conclusion:1945-1995). Không sửa thẳng file chính ở phase này. Không cần expert ngoài, chỉ dùng số liệu đã có.

---

## 1. Tóm tắt Task 47

Reviewer R1 #13: *"Perturbation chứng minh faithfulness với model, chưa chứng minh manager hiểu được. Cần expert evaluation hoặc ít nhất case studies + giảm claim trust"*.

Vì Task 46 (hỏi chuyên gia) là Optional và tốn thời gian, ta chọn **Task 47 (Strong)**: thêm 2-3 case studies bằng tiếng quản lý và thêm đoạn Limitations.

---

## 2. Ba Case Studies bằng tiếng quản lý (dịch từ RDX/MSX/SHAP)

### Case 1: RDX - Đánh đổi giữa phục vụ khách và tốn tiền lưu kho (EASY vs HARD)

**Tình huống kho:**
- **EASY:** Tồn kho 30%, nhu cầu 20%, hỏng 1% - kho trống, ít rủi ro `Table 5:1105` `Xai_Inventory_Submit_17Mar.md:1105`
- **HARD:** Tồn kho 90%, nhu cầu 80%, hỏng 15% - kho chật, nhu cầu cao nhưng dễ hỏng `Table 5:1113`

**AI chọn gì (Table 4:1039):**
- EASY: Cả DQN và A2C_mod đều đặt hàng mạnh (aggressive), service ~1.0, stockout ~0.
- HARD: DQN vẫn đặt mạnh (service 0.9997, reward 0.58), A2C_mod giảm đặt (service 0.9994, reward 0.546) nhưng holding cost thấp hơn (0.083 vs 0.107).

**RDX nói gì (Fig 2-4:993-1017):**
- EASY: RDX của 2 agents gần như giống nhau, service component áp đảo, chi phí không đáng kể.
- HARD: A2C_mod giảm holding/ordering cost rõ, DQN vẫn ưu tiên service dù total reward âm.

**Quản lý hiểu và làm gì:**
> *"Khi kho trống và ít hỏng (EASY), AI khuyên đặt mạnh để không hụt hàng - chi phí lưu kho không đáng lo. Khi kho chật và dễ hỏng (HARD), A2C_mod khuyên bớt đặt lại để tiết kiệm tiền kho/hỏng, chấp nhận rủi ro hụt hàng nhỏ; còn DQN thì 'cố đấm ăn xôi' giữ service cao dù tốn tiền. Quản lý có thể chọn A2C_mod nếu muốn tiết kiệm, chọn DQN nếu ưu tiên không để khách chờ, và RDX cho biết chính xác trade-off này."*

**Bài học cho trust:** RDX không bảo "AI đúng", mà cho manager thấy **đánh đổi nào đang diễn ra**, để tự quyết.

---

### Case 2: MSX - Lý do tối thiểu đủ để ra quyết định (tại sao chỉ 1 lý do là đủ?)

**Tình huống:**
- Cùng 3 scenarios EASY/MEDIUM/HARD, hỏi: "Cần bao nhiêu lý do để biện minh cho quyết định đặt hàng?"

**MSX nói gì (Fig 5:1129, Fig 7-8:1171-1205):**
- **A2C_mod:** EASY/MEDIUM chỉ cần 1 lý do là **service** (phục vụ khách). Sang HARD, lý do cốt lõi đổi thành **holding** (tiền lưu kho).
- **DQN:** Mọi scenario chỉ cần 1 lý do là **holding** - rất ổn định nhưng ít linh hoạt.

**Quản lý hiểu và làm gì:**
> *"Nếu sếp hỏi 'tại sao đặt ít đi trong tuần này?', với A2C_mod ở kho chật (HARD), câu trả lời MSX là: 'Vì tiền lưu kho đang là vấn đề chính, không phải vì sợ hết hàng'. Còn DQN thì lúc nào cũng trả lời 'vì tiền lưu kho', dù kho trống hay đầy. Manager sẽ thấy A2C_mod 'biết điều chỉnh lý do theo tình hình', còn DQN 'cứng nhắc'. MSX giúp trả lời ngắn gọn 1 câu thay vì 4 lý do dài dòng."*

**Bài học cho trust:** MSX cho **lý do ngắn nhất đủ thuyết phục**, không phải liệt kê hết. Manager không cần hiểu toán, chỉ cần 1 câu.

---

### Case 3: SHAP - SKU nào cần theo dõi nhất (không phải best-seller)

**Tình huống:**
- Kho có 220 SKU, không thể theo dõi hết. SHAP micro-level trên 660 chiều gốc (3*220) cho biết SKU nào AI nhạy nhất `Fig 11:1490` `outputTask10-12_case_analysis.csv:2-9`.

**SHAP nói gì:**
- Top-5 DQN: SKU64 (Mean|SHAP| 0.00281), SKU163 (0.00278), SKU46,155,118 - đều là **sales features**.
- Đặc điểm chung của 5 SKU này (`outputTask10-12_case_analysis.csv:2-7`):
  - **Không phải best-seller:** Best-seller thực sự là SKU57 (demand 22.6), SKU81 (17.2) - không nằm trong Top-8.
  - **Là low-volume nhưng biến động cao:** SKU64 mean 0.66, CV 1.46, capacity 7; SKU163 mean 1.22, CV 1.27, capacity 12. Tương quan demand-SHAP chỉ r=0.04.
  - **DQN ổn định (Jaccard 1.0), A2C_mod linh hoạt (Jaccard 0.29)** giữa 3 scenarios `outputTask10-11_overlap.csv`.

**Quản lý hiểu và làm gì:**
> *"AI không quan tâm nhất đến món bán chạy nhất, mà quan tâm nhất đến món 'ít bán nhưng lúc bán lúc không, kệ lại nhỏ' (SKU64,163). Vì món này dễ hết hàng đột ngột, nếu không đặt trước sẽ hụt. Quản lý nên đặt cảnh báo tồn kho riêng cho nhóm SKU biến động cao, kệ nhỏ này, thay vì chỉ nhìn top doanh số. SHAP giúp khoanh vùng 5-8 SKU cần giám sát chặt, không phải 220."*

**Bài học cho trust:** SHAP không bảo manager phải tin AI, mà chỉ **chỉ tay vào 5 SKU cần nhìn**, manager tự kiểm tra lại có đúng volatile không (và đúng là vậy theo CSV).

---

## 3. Giảm Claim Trust - 3 vị trí sửa trong bài báo

### 3.1 Vị trí 1: Abstract dòng 13

**Câu cũ (`Xai_Inventory_Submit_17Mar.md:13`):**
> `While Deep Reinforcement Learning (DRL) optimizes these systems, its black-box nature limits interpretability and operational trust.`

**Câu mới đề xuất (tiếng Anh sẵn paste):**
> `While Deep Reinforcement Learning (DRL) optimizes these systems, its black-box nature limits interpretability; without expert validation, operational trust remains to be established.`

**Lý do:** Thêm điều kiện "chưa có expert thì chưa nói trust".

### 3.2 Vị trí 2: Abstract cuối

**Câu cũ:**
> `Overall, this multi-level XRL framework enhances the transparency of portfolio-level DRL replenishment analysis.`

**Câu mới:**
> `Overall, this multi-level XRL framework enhances transparency of portfolio-level DRL replenishment analysis; its usefulness for decision support requires future expert validation (see Limitations).`

### 3.3 Vị trí 3: Conclusion Section 5 (`Xai_Inventory_Submit_17Mar.md:1945-1995`)

**Đoạn cũ đã có nhưng chưa đủ nhấn:**
> `Third, systematic validation by domain experts is needed to assess the alignment between the identified trade-offs and real-world managerial reasoning.`

**Đoạn mới đề xuất thay thế (mở rộng, 6 dòng):**
> `Third, systematic validation by domain experts is needed to assess whether the identified trade-offs and feature attributions align with real-world managerial reasoning. Perturbation-based faithfulness (Section 4.7) establishes that explanations are faithful to model behavior, but not that they are useful or understandable to managers. Claims about trust are therefore moderated to transparency. Future work will conduct structured expert evaluation of clarity, usefulness, and consistency, and translate the present case studies into an interactive decision-support interface.`

**Lý do:** Nói rõ **faithfulness ≠ usefulness**, và **transparency ≠ trust**.

---

## 4. Bảng tóm tắt cases (để paste vào Supplementary)

| Case | Nguồn XRL | Scenario | Phát hiện kỹ thuật | Diễn giải quản lý | Hành động gợi ý |
|---|---|---|---|---|---|
| 1 | RDX | EASY vs HARD | A2C_mod linh hoạt giảm holding cost ở HARD, DQN giữ service | EASY nên đặt mạnh, HARD cân nhắc tiết kiệm vs phục vụ | Chọn policy theo ưu tiên chi phí/dịch vụ |
| 2 | MSX | EASY->HARD | A2C_mod đổi core reason service->holding, DQN cố định holding | Lý do quyết định thay đổi theo áp lực kho | Dùng MSX để trả lời sếp 1 câu ngắn gọn |
| 3 | SHAP micro | All (50 states) | Top SKU là low-volume high-CV (SKU64 CV1.46), không phải best-seller | Theo dõi 5-8 SKU biến động cao, kệ nhỏ | Đặt cảnh báo riêng cho nhóm volatile |

File CSV chi tiết: `outputTask47_cases.csv`

---

## 5. Đoạn Limitations hoàn chỉnh để paste vào Section 5 (tiếng Anh)

> **Limitations (Human interpretability).** This study demonstrates explanation faithfulness via perturbation tests (AUPC, ASR, bootstrap CI, Wilcoxon) and provides managerial case studies that translate RDX/MSX/SHAP into operational language. However, no domain-expert evaluation was conducted. Faithfulness to model outputs does not imply usefulness or understandability for inventory managers. The case studies are illustrative translations of statistical patterns, not validated decision aids. Future work will involve inventory experts to rate clarity, usefulness, and consistency, and will test whether managers make better or faster decisions with the explanations.

---

## 6. File đính kèm Task 47

- `planTask17-9.md` (kế hoạch)
- `outputTask47.md` (file này - 3 cases + 3 câu sửa)
- `outputTask47_cases.csv` (bảng 3 cases)
- `generate_cases_task47.ipynb` (notebook đọc CSV có sẵn, in bảng, <10s)

**Ghi chú cho Response Letter (sẵn paste):**
> *R1 #13 (trust): We have not conducted expert evaluation (Task 46 optional, deferred). Instead, we added three managerial case studies (Section 4.10) that translate RDX, MSX, and SHAP into operational language, and moderated trust claims in Abstract and Conclusion to transparency, adding a Limitations paragraph clarifying faithfulness ≠ usefulness. Expert validation is listed as future work.*

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 17-9 (47) - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 47 đã chèn/sửa trong bản thảo để giải quyết Task 47 (Managerial cases + giảm claim trust), dùng để trả lời reviewer và kiểm tra lại. Cấu trúc theo mẫu Task 11-9: Task -> Section -> Đoạn đã xóa -> Đoạn mới thêm vào. Văn phong trong bản thảo là học thuật, không liệt kê tên file/hàm; các chú thích figure dạng `[Figure - file.png]` để tiện theo dõi.

### Task 47 - Managerial case studies (Strong, Writing) - MỚI THÊM Section 4.10

**Section 4.10 Managerial Case Studies and Interpretation - MỚI THÊM hoàn toàn (chèn sau `Xai_Inventory_Submit_17Mar.md:1943` trước `5. Conclusion`, trước chưa có)**

*Đoạn đã xóa:* (chưa có Section 4.10, chỉ kết thúc 4.9 tại dòng `...finer control comes at the cost of more complex justifications.`)

*Đoạn mới thêm vào (tiếng Anh, học thuật, 2 bảng gọn 4 cột + 2 figure placeholder):*
> "### 4.10 Managerial Case Studies and Interpretation [toàn bộ 5 đoạn + Table 12 (Case/Lens/Context/Question, 3 dòng) + Figure 13 `[Figure - task47_case_overview.png]` + Case 1 RDX (stable vs pressured, actor-critic giảm holding còn value-based giữ service) + Case 2 MSX ({service}->{holding} vs {holding} ổn định) + Case 3 SHAP + Table 13 (Product/Mean demand/Variability/Capacity, 5 dòng: SKU64 0.66/1.46/7, SKU163 1.22/1.27/12, SKU46 2.82/1.13/28, SKU155 1.60/1.15/15, SKU57 ref 22.6/0.45/208) + Figure 14 `[Figure - task47_sku_profile.png]` + đoạn kết faithfulness ≠ usefulness, trust moderated to transparency]. Chi tiết bảng gọn được tổng hợp từ `outputTask47_cases.csv` (3 cases) và `outputTask10-12_case_analysis.csv` (36 SKU) nhưng trong bản thảo chỉ giữ 4 cột, không liệt kê tên file/hàm và không liệt kê số rời rạc trong câu."

*Lý do chọn phương pháp:* Expert evaluation là chuẩn nhưng cần tiếp cận chuyên gia và protocol có cấu trúc, không khả thi trong phạm vi hiện tại (Task 46 Optional) nên chọn case study dịch sang ngôn ngữ quản lý làm cầu nối interim, phù hợp với thực hành XAI decision-support. Ba cases được chọn để bao phủ 3 lens bổ sung (reward trade-off, minimal justification, feature attribution) và được diễn giải thành hành động cụ thể (chọn policy, câu trả lời 1 câu, cảnh báo SKU).

### Task 47 - Giảm claim trust - Abstract

**Section Abstract - Dòng 13 `Xai_Inventory_Submit_17Mar.md:13`**

*Đoạn đã xóa:*
> `While Deep Reinforcement Learning (DRL) optimizes these systems, its black-box nature limits interpretability and operational trust.`

*Đoạn mới thêm vào:*
> `While Deep Reinforcement Learning (DRL) optimizes these systems, its black-box nature limits interpretability; without expert validation, operational trust remains to be established.`

**Section Abstract - Dòng cuối `Xai_Inventory_Submit_17Mar.md:13`**

*Đoạn đã xóa:*
> `Overall, this multi-level XRL framework enhances the transparency of portfolio-level DRL replenishment analysis.`

*Đoạn mới thêm vào:*
> `Overall, this multi-level XRL framework enhances transparency of portfolio-level DRL replenishment analysis; its usefulness for decision support requires future expert validation (see Limitations).`

### Task 47 - Giảm claim trust - Conclusion

**Section 5. Conclusion - Đoạn Limitations `Xai_Inventory_Submit_17Mar.md:1997`**

*Đoạn đã xóa (khối lỗi lẫn Fig 20 + 3 limitations rời rạc, thiếu phân biệt faithfulness/usefulness):*
> `Although the proposed framework provides structured Fig 20. Perturbation curves... 𝐷𝑟𝑜𝑝... Second, future studies should extend SHAP... Third, systematic validation by domain experts is needed to assess the alignment between the identified trade-offs and feature allocations and real-world managerial reasoning. In addition to methodological extensions...`

*Đoạn mới thêm vào (học thuật, liền mạch, 1 đoạn):*
> `Although the proposed framework provides structured insights into reasoning in both reward space and feature space, several limitations point to directions for future research. First, explanation robustness should be evaluated across multiple random seeds using dedicated stability metrics... Second, future studies should extend SHAP analysis beyond aggregated features to more fine-grained product-level representations... Third, systematic validation by domain experts is needed to assess the alignment between the identified trade-offs and feature attributions and real-world managerial reasoning. Perturbation-based faithfulness establishes that explanations are faithful to model behavior, but not that they are useful or understandable to managers; claims about trust are therefore moderated to transparency. The managerial case studies presented in Section 4.10 serve as an interim, illustrative translation in the absence of such expert evaluation, and future work will conduct structured expert assessment of clarity, usefulness, and consistency and translate the present cases into an interactive decision-support interface. In addition to methodological extensions, applying the framework to other DRL algorithms such as PPO or SAC...`

*Đoạn đã xóa sau CRediT (rác):*
> `change. Overall, the perturbation-based results provide empirical evidence...` -> đã xóa và gộp gọn thành `Dac Hoang Nguyen: Conceptualization of the review article, Literature search, Writing the manuscript as the main author...` một đoạn duy nhất.

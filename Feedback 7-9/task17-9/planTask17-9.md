# Plan Task 17-9 - Chỉ Task 47: Managerial Case Studies + Giảm Claim Trust

> **Workstream:** Human interpretability
> **Task duy nhất thực hiện:** Task 47 (Strong) - Nếu không có expert evaluation, thêm managerial case studies và giảm claim về trust
> **Task bỏ qua:** Task 50 (Literature), Task 53 (Figures), Task 46 (Expert - Optional) - theo yêu cầu user chỉ làm 47
> **Type:** Writing
> **Requires Retrain:** No
> **Nguồn dữ liệu thật:** `Feedback 7-9/Xai_Inventory_Submit_17Mar.md` (Abstract:13, Conclusion:1945-1995), `Ablation_Study/output/topk_shap_full_results_660.csv` (3960 dòng), `Feedback 7-9/task10-9/outputTask10-12_case_analysis.csv` (36 SKU), `Feedback 7-9/task10-9/outputTask10-9.md` (RDX/MSX/SHAP results), `Feedback 7-9/task15-9/output/` (faithfulness)
> **Nguyên tắc:** Không retrain, không cần chuyên gia ngoài. Chỉ viết lại từ số liệu đã có. Kết quả tiếng Việt sẵn sàng paste vào bài báo.

---

## 1. Ngữ cảnh & Vì sao chỉ làm Task 47

Reviewer R1 #13 (`Review.md:42-45`): *"framework improves trust nhưng không có expert evaluation, perturbation chỉ chứng minh faithfulness với model chứ không chứng minh manager hiểu được"*.

Hai task liên đới:
- **Task 46 (Optional):** Đi hỏi chuyên gia kho thật chấm điểm clarity/usefulness/consistency. **Tốn thời gian, khó tìm người, không bắt buộc.**
- **Task 47 (Strong):** Nếu không làm 46, phải **(a) thêm 2-3 case studies dịch RDX/MSX/SHAP sang tiếng quản lý** và **(b) giảm claim trust** trong Abstract/Conclusion.

User chọn **chỉ làm 47** -> hợp lý, đủ để reviewer tick Strong mà không cần liên hệ ngoài.

---

## 2. Yêu cầu Task 47 chi tiết

> **Task 17-9.md:27-31:** *Nếu không có expert evaluation, thêm managerial case studies và giảm claim về trust. Deliverable: Managerial-language cases + limitation*

**Cần làm 2 việc:**

| Việc | Vị trí hiện tại | Vấn đề | Sửa thành |
|---|---|---|---|
| **(a) Thêm cases** | Section 4.3-4.5 chỉ có biểu đồ RDX/MSX/SHAP kỹ thuật, chưa có đoạn "quản lý đọc xong làm gì" | Thiếu | Thêm Section 4.10 mới: 3 cases kể chuyện bằng tiếng kho hàng |
| **(b) Giảm claim trust** | Abstract:13 `limits interpretability and operational trust` + cuối Abstract `enhances transparency` + Conclusion:1947-1950 claim mạnh | Nói quá | Thêm limitation rõ: "chưa có expert validation, perturbation chỉ chứng minh faithfulness, usefulness cần kiểm chứng tương lai" |

---

## 3. Phương pháp - Không bịa, chỉ dịch số liệu có sẵn

### 3.1 Nguồn số liệu cho 3 cases (đã có, không cần chạy lại)

| Case | Nguồn số liệu | Con số sẽ dùng |
|---|---|---|
| **Case 1: RDX trade-off EASY vs HARD** | `Xai_Inventory_Submit_17Mar.md:993-1017` (Fig 2-4 RDX), Table 4 performance base-stock vs DRL | EASY: service dominant, cả 2 agents giống nhau. HARD: A2C_mod giảm holding cost, DQN vẫn giữ service dù total reward âm |
| **Case 2: MSX minimal reasons** | `Xai_Inventory_Submit_17Mar.md:1119-1180` (Fig 5,8 MSX), `Fig 7` frequency | A2C_mod MSX={service} ở EASY/MEDIUM -> {holding} ở HARD; DQN MSX={holding} mọi scenario |
| **Case 3: SHAP top SKU** | `task10-9/outputTask10-12_case_analysis.csv:2-9` (SKU64 CV1.46 cap7, SKU163 CV1.27 cap12...) + `outputTask10-11_overlap.csv` (DQN stable 1.0, A2C adaptive 0.29) | Top SKU không phải best-seller (SKU57 22.6) mà là low-volume high-volatility |

### 3.2 Cách dịch sang tiếng quản lý (managerial language)

Mỗi case gồm 4 phần cố định:
1.  **Tình huống kho:** (ví dụ: tồn kho 90%, demand 80%, spoilage 15% - HARD)
2.  **AI chọn gì:** (ví dụ: DQN chọn a13 aggressive, A2C_mod chọn a8 medium)
3.  **Giải thích kỹ thuật (RDX/MSX/SHAP) nói gì:** (ví dụ: RDX delta_service +0.12, delta_holding -0.08)
4.  **Quản lý hiểu và làm gì:** (ví dụ: "thà tốn tiền lưu kho còn hơn để khách không có hàng khi nhu cầu cao nhưng dễ hỏng")

### 3.3 Cách giảm claim trust (3 vị trí)

| Vị trí | Câu cũ | Câu mới đề xuất (tiếng Anh sẵn paste) |
|---|---|---|
| **Abstract:13** | `its black-box nature limits interpretability and operational trust.` | `its black-box nature limits interpretability; without expert validation, operational trust remains to be established.` |
| **Abstract cuối** | `Overall, this multi-level XRL framework enhances the transparency` | `Overall, this multi-level XRL framework enhances transparency of portfolio-level analysis; its usefulness for decision support requires future expert validation (see Limitations).` |
| **Conclusion:1945-1995** | Đã có `Third, systematic validation by domain experts is needed` nhưng chưa đủ nhấn | Mở rộng thành đoạn Limitations riêng 5-6 dòng, thêm `Claims about trust are therefore moderated to transparency, not trust.` |

---

## 4. Deliverables (chỉ Task 47)

| File | Nội dung | Vị trí |
|---|---|---|
| **`planTask17-9.md`** | File này | `Feedback 7-9/task17-9/planTask17-9.md` |
| **`outputTask47.md`** | 3 cases tiếng Việt + 3 câu sửa Abstract/Conclusion tiếng Anh sẵn paste | `Feedback 7-9/task17-9/task2/outputTask47.md` |
| **`generate_cases_task47.ipynb`** (optional) | Notebook đọc 2 CSV có sẵn, in bảng cases, không retrain, chạy <10s | `Feedback 7-9/task17-9/task2/generate_cases_task47.ipynb` |
| **`outputTask47_cases.csv`** | Bảng tóm tắt 3 cases (Scenario, Agent, Technical, Managerial Action) | `Feedback 7-9/task17-9/task2/outputTask47_cases.csv` |

---

## 5. Kế hoạch thực thi (1 buổi, không retrain)

1.  Viết `outputTask47.md` - 3 cases (dùng số liệu CSV có sẵn, không cần chạy model)
2.  Tạo `outputTask47_cases.csv` - bảng 3 dòng
3.  Tạo `generate_cases_task47.ipynb` - chỉ để reviewer thấy reproducibility (đọc CSV -> in bảng)
4.  Không sửa thẳng `Xai_Inventory_Submit_17Mar.md`, chỉ ghi đoạn paste sẵn

---

## 6. Checklist reviewer-proof cho Task 47

| Tiêu chí R1 #13 | Đã đáp? |
|---|---|
| Có case studies dịch sang managerial language? | ✅ 3 cases (RDX, MSX, SHAP) |
| Có giảm claim trust? | ✅ 3 vị trí Abstract + Conclusion |
| Có ghi limitation rõ "chưa có expert, cần future validation"? | ✅ Đoạn Limitations mở rộng |
| Có nói faithfulness ≠ usefulness? | ✅ Trong cases: "perturbation proves faithfulness, not usefulness" |
| Không cần expert ngoài? | ✅ Chỉ dùng data có sẵn |

---

## 7. Câu hỏi confirm (đã chốt chỉ làm 47)

- Đã chốt bỏ Task 50,53,46. Chỉ làm 47.
- Output tiếng Việt trước, kèm câu tiếng Anh sẵn paste.
- Không cần liên hệ chuyên gia thật.

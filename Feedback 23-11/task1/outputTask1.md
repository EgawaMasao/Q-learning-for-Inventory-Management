# Output Task 1 — Độ ổn định SHAP qua 5 seed độc lập (`A2C_mod`)

## Tóm tắt công việc & Kết quả chính
- **Thực nghiệm đã chạy:** [task1_shap_seed_stability.ipynb](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task1/task1_shap_seed_stability.ipynb) trên **5 seed độc lập** của `A2C_mod` (`42, 123, 256, 512, 1024`), 3 kịch bản (`EASY, MEDIUM, HARD`), 2 mức (**Macro 3 đặc trưng**, **Micro 660 đặc trưng**), 2 mục tiêu (`π` và `logits`), trên cùng 50 test states và 100 background states cố định (`seed = 42`). Dữ liệu lưu tại `Feedback 23-11/task1/output/`.
- **Kết luận khoa học (ổn định có điều kiện):**
  1. **Macro (3 đặc trưng):** `logits` ổn định tuyệt đối (`Demand` #1 ở cả 5 seed, Jaccard = `1.000`, Spearman = `1.000`). Với `π`, **top-1 không ổn định** (Jaccard = `0.600` all seeds / `0.500` bỏ 256 vì 4 seed xếp `Inventory` #1 còn seed 512 xếp `Demand` #1), nhưng **top-2 ổn định** (Jaccard = `1.000`).
  2. **Micro (660 đặc trưng):** Nhóm **top-8 sales SKU** tái lập tốt (`logits` Jaccard = `1.000`; `π` k=20 Jaccard = `0.861` all / `0.896` bỏ 256). Tuy nhiên, `648–651/660` đặc trưng bị đồng hạng (tie) gần 0 nên các chỉ số micro chỉ mang tính định hướng cho nhóm top-8 sales.
  3. **Seed-256 bị collapse (báo cáo riêng):** chọn `action 13` ở `87.7%–88.7%` quyết định (entropy `0.0093–0.0096`, xác nhận lặp lại ở `ckpt-76` với `86.6%`). Seed 256 làm giảm nhẹ độ ổn định micro của `π` (Jaccard k=20 với 4 seed còn lại = `0.808`), nhưng không phải nguyên nhân duy nhất gây bất ổn top-1 macro của `π`.

---

## Ghi vào bài báo

### Chỉnh sửa 1 — Chèn tiểu mục 3.13.1 kèm 2 bảng gọn (Table 19 & Table 20)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **Results and Discussion** → **Section 3.13 *Reliability and Stability of the Explanations***, đặt **ngay sau Table 18** (sau `P437`, trước `P440 Conclusion`). *(Không tạo lại heading 3.13; hình chi tiết và bảng 660 feature đầy đủ để ở Supplementary Material.)*

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

#### 3.13.1. Multi-Seed Stability of SHAP Feature Importance and Seed-256 Collapse

To verify whether SHAP attributions are reproducible across independent training runs, we evaluated all five `A2C_mod` checkpoints (seeds 42, 123, 256, 512, and 1024; 530 episodes for seed 42 and 600 for the remaining seeds) on the same 50 held-out states per scenario (EASY, MEDIUM, HARD) using a fixed 100-state synthetic background (seed 42). Both macro-level (3 features, KernelSHAP) and micro-level (660 features, PartitionExplainer) attributions were computed for the policy probability target $\pi$ and pre-softmax `logits`. Across-seed stability was measured via pairwise top-$k$ Jaccard overlap and full-ranking Spearman correlation (Table 19), alongside feature-wise mean $\pm$ SD of $|\text{SHAP}|$ (Table 20).

At the macro level, pre-softmax `logits` are fully stable across all five seeds and scenarios (`Demand` ranks first in every run; Jaccard $= 1.000$, Spearman $= 1.000$). Under the policy probability target $\pi$, top-1 ranking is not stable (mean Jaccard $= 0.600$ across all seeds and $0.500$ excluding seed 256), because four seeds (42, 123, 256, and 1024) rank `Inventory` first whereas seed 512 ranks `Demand` first; however, the top-2 set (`Inventory` and `Demand`) is invariant across all runs (Jaccard $= 1.000$, Spearman $= 0.800$). At the micro level, the same eight sales-related SKUs (`sales_sku128, 66, 14, 71, 162, 65, 36, 4`) consistently occupy the top ranks (`logits` Jaccard $= 1.000$; $\pi$ at $k = 20$ achieves mean Jaccard $= 0.861$ across all seeds and $0.896$ excluding seed 256, with Spearman $= 0.804$ and $0.934$, respectively). Because PartitionExplainer on the SKU-averaged policy yields only 9–12 distinct importance values with 648–651 features tied near zero, micro-level top-$k$ metrics primarily reflect this stable top-8 sales subset and should be interpreted as indicative (consistent with Section 2.3.4.2). Table 20 shows that `logits` exhibit lower relative dispersion across seeds (CV $= 0.14$–$0.34$ for top micro features) than $\pi$ (CV $= 0.59$–$1.08$), reflecting policy saturation.

**Separate analysis of collapsed seed 256.** Diagnostics confirm that seed 256 undergoes severe policy collapse: across EASY, MEDIUM, and HARD, 88.5%, 88.7%, and 87.7% of SKU decisions select action 13, with mean policy entropy $0.0093$–$0.0096$ and mean maximum probability $0.994$–$0.995$ (persisting at checkpoint 76 with 86.6% action 13 and entropy $0.0070$). Seed 256 is therefore reported separately in Table 19. Relative to the other four seeds, seed 256 reduces micro-level $\pi$ agreement (mean Jaccard $= 0.808$ at $k = 20$, Spearman $= 0.609$) while preserving the same dominant sales SKU set (`logits` Jaccard $= 1.000$). Importantly, excluding seed 256 does not eliminate macro $\pi$ top-1 instability (Jaccard remains $0.500$ because seed 512 ranks `Demand` first while seeds 42, 123, and 1024 rank `Inventory` first). Overall, multi-seed SHAP stability for `A2C_mod` is conditional: macro `logits`, macro $\pi$ top-2, and micro top-8 sales SKUs are reproducible across independent runs, whereas macro $\pi$ top-1 rankings are seed-sensitive. Full per-scenario tables, the 660-feature table, and stability figures are provided in the Supplementary Material.

**Table 19. Stability of SHAP feature importance across five independent `A2C_mod` training runs (means over EASY, MEDIUM, and HARD).**

| Level | Target | $k$ | All 5 seeds (Jaccard / Spearman) | Excl. seed 256 (Jaccard / Spearman) | Seed 256 vs. others (Jaccard / Spearman) |
|---|---|---:|---:|---:|---:|
| Macro | $\pi$ | 1 | 0.600 / 0.800* | 0.500 / 0.750* | 0.750 / 0.875* |
| Macro | $\pi$ | 2 | 1.000 / 0.800* | 1.000 / 0.750* | 1.000 / 0.875* |
| Macro | `logits` | 1 | 1.000 / 1.000* | 1.000 / 1.000* | 1.000 / 1.000* |
| Macro | `logits` | 2 | 1.000 / 1.000* | 1.000 / 1.000* | 1.000 / 1.000* |
| Micro† | $\pi$ | 10 | 0.819 / 0.804 | 0.909 / 0.934 | 0.683 / 0.609 |
| Micro† | $\pi$ | 20 | 0.861 / 0.804 | 0.896 / 0.934 | 0.808 / 0.609 |
| Micro† | $\pi$ | 50 | 0.870 / 0.804 | 0.928 / 0.934 | 0.782 / 0.609 |
| Micro† | `logits` | 10, 20, 50 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

*\*Full 3-feature Spearman correlation (top-1 Spearman is undefined).*  
*†Indicative: 648–651 of 660 features are tied near zero; overlap primarily reflects the stable top-8 sales SKUs.*

**Table 20. Mean $\pm$ SD of $|\text{SHAP}|$ across five `A2C_mod` seeds in MEDIUM (macro features and top-5 micro features; averaged over 50 states and 14 actions).**

| Scope / Target | Feature | Mean $\pm$ SD | CV |
|---|---|---:|---:|
| Macro (`logits`) | Demand | $25.092 \pm 7.574$ | 0.302 |
| Macro (`logits`) | Inventory | $7.006 \pm 4.450$ | 0.635 |
| Macro (`logits`) | Waste | $0.054 \pm 0.019$ | 0.353 |
| Macro ($\pi$) | Demand | $0.0250 \pm 0.0267$ | 1.067 |
| Macro ($\pi$) | Inventory | $0.0221 \pm 0.0091$ | 0.412 |
| Macro ($\pi$) | Waste | $0.00020 \pm 0.00020$ | 1.041 |
| Micro top-5 (`logits`) | `sales_sku128` / `66` / `14` / `71` / `162` | $0.145 \pm 0.021$ to $0.126 \pm 0.042$ | 0.144–0.329 |
| Micro top-5 ($\pi$) | `sales_sku162` / `66` / `36` / `71` / `14` | $0.000185 \pm 0.000110$ to $0.000150 \pm 0.000121$ | 0.594–1.006 |

*(Ghi chú: bảng đầy đủ 660 feature nằm trong Supplementary Table S2.)*

- **Bản Tiếng Việt (Để theo dõi):**

#### 3.13.1. Độ ổn định đa seed của tầm quan trọng đặc trưng SHAP và hiện tượng collapse ở seed 256

Để kiểm chứng khả năng tái lập của SHAP qua các lần huấn luyện độc lập, chúng tôi đánh giá cả 5 checkpoint của `A2C_mod` (seed 42, 123, 256, 512, 1024; 530 episode cho seed 42 và 600 cho các seed còn lại) trên cùng 50 trạng thái kiểm định mỗi kịch bản (EASY, MEDIUM, HARD) với tập nền tổng hợp cố định 100 trạng thái (seed 42). Cả SHAP vĩ mô (3 đặc trưng, KernelSHAP) và vi mô (660 đặc trưng, PartitionExplainer) được tính cho hai mục tiêu là xác suất chính sách $\pi$ và `logits` trước softmax, định lượng bằng độ trùng lặp Jaccard top-$k$, tương quan hạng Spearman (Bảng 19) và trung bình $\pm$ độ lệch chuẩn $|\text{SHAP}|$ (Bảng 20).

Ở cấp vĩ mô, `logits` ổn định tuyệt đối trên cả 5 seed và mọi kịch bản (`Demand` đứng #1 ở mọi lần chạy; Jaccard $= 1.000$, Spearman $= 1.000$). Với mục tiêu xác suất $\pi$, thứ hạng top-1 không ổn định (Jaccard $= 0.600$ trên cả 5 seed và $0.500$ khi loại seed 256) vì 4 seed (42, 123, 256, 1024) xếp `Inventory` #1 còn seed 512 xếp `Demand` #1; tuy nhiên tập top-2 (`Inventory` và `Demand`) bất biến ở mọi lần chạy (Jaccard $= 1.000$, Spearman $= 0.800$). Ở cấp vi mô, cùng nhóm 8 SKU bán hàng (`sales_sku128, 66, 14, 71, 162, 65, 36, 4`) liên tục chiếm các vị trí đầu (`logits` Jaccard $= 1.000$; $\pi$ ở $k=20$ đạt Jaccard $= 0.861$ trên cả 5 seed và $0.896$ khi loại seed 256, Spearman $= 0.804$ và $0.934$). Vì PartitionExplainer trên chính sách trung bình theo SKU chỉ sinh 9–12 giá trị phân biệt với 648–651 đặc trưng bị tie gần 0, các chỉ số vi mô chủ yếu phản ánh nhóm top-8 sales ổn định này và mang tính định hướng (§2.3.4.2). Bảng 20 cho thấy `logits` có hệ số biến thiên thấp hơn nhiều (CV $= 0.14$–$0.34$) so với $\pi$ (CV $= 0.59$–$1.08$) do chính sách bị bão hòa.

**Phân tích riêng seed 256 bị collapse.** Chẩn đoán xác nhận seed 256 rơi vào trạng thái collapse nặng: ở EASY, MEDIUM và HARD, lần lượt 88.5%, 88.7% và 87.7% quyết định SKU chọn `action 13`, với entropy chính sách $0.0093$–$0.0096$ và xác suất cực đại $0.994$–$0.995$ (vẫn tiếp diễn ở checkpoint 76 với 86.6% `action 13`, entropy $0.0070$). Do đó seed 256 được tách báo cáo riêng trong Bảng 19. So với 4 seed còn lại, seed 256 làm giảm độ trùng lặp micro của $\pi$ (Jaccard $= 0.808$ ở $k=20$, Spearman $= 0.609$) nhưng vẫn giữ nguyên nhóm SKU sales chủ đạo (`logits` Jaccard $= 1.000$). Quan trọng hơn, loại seed 256 không làm hết bất ổn top-1 macro của $\pi$ (Jaccard vẫn là $0.500$ do seed 512 xếp `Demand` #1 còn 3 seed 42, 123, 1024 xếp `Inventory` #1). Tóm lại, độ ổn định đa seed của SHAP cho `A2C_mod` mang tính có điều kiện: `logits` vĩ mô, top-2 $\pi$ vĩ mô và top-8 sales vi mô tái lập tốt, còn thứ hạng top-1 $\pi$ vĩ mô nhạy theo seed. Các bảng chi tiết từng kịch bản, bảng 660 đặc trưng và hình minh họa nằm trong Supplementary Material.

---

### Chỉnh sửa 2 — Nối mạch tại `Table 18` (1 hàng) và `Study Limitations` (1 câu)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx):
  1. **Table 18** (`P437`, nhóm *Stability*): thêm 1 hàng.
  2. **Study Limitations and Future Work** (`P444`, câu về multi-seed ở limitation đầu tiên): cập nhật 1 câu.

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**
  - *Thêm 1 hàng vào **Table 18**:*

| Criterion | Metric | DQN | A2C_mod | Reported in |
|---|---|---|---|---|
| Reproducibility (independent runs) | 5-seed SHAP top-$k$ Jaccard / Spearman | Not measured | Conditional: `logits` & top-8 sales stable (1.00); $\pi$ micro $k=20$ Jaccard 0.861 (0.896 excl. collapsed seed 256); macro $\pi$ top-1 unstable (0.600) | Section 3.13.1 |

  - *Cập nhật 1 câu tại `P444` (Study Limitations):*
    > While multi-seed SHAP reproducibility for `A2C_mod` is evaluated across five runs in Section 3.13.1, extending this multi-seed audit to DQN, matched training budgets, and larger run counts remains future work.

- **Bản Tiếng Việt (Để theo dõi):**
  - *Thêm 1 hàng vào **Bảng 18**:* Tiêu chí `Khả năng tái lập (5 lần chạy độc lập)` | Chỉ số `Jaccard / Spearman top-k SHAP qua 5 seed` | DQN `Chưa đo` | `A2C_mod` `Ổn định có điều kiện: logits & top-8 sales ổn định (1.00); micro π k=20 Jaccard 0.861 (0.896 khi loại seed 256 collapse); macro π top-1 không ổn định (0.600)` | Mục `Section 3.13.1`.
  - *Cập nhật 1 câu tại `P444`:* Mặc dù khả năng tái lập đa seed của SHAP cho `A2C_mod` đã được đánh giá qua 5 lần chạy tại §3.13.1, việc mở rộng kiểm chứng này sang DQN, đồng bộ ngân sách huấn luyện và tăng số lần chạy vẫn dành cho nghiên cứu tiếp theo.

---

## Checklist kiểm chứng hoàn thành Task 1

- [x] Đủ 5 seed (`42, 123, 256, 512, 1024`), cả **Macro (3-dim)** và **Top-k Micro (660-dim)**.
- [x] Báo cáo đầy đủ **Jaccard / Spearman top-k** (Table 19) và **`mean ± SD |SHAP|`** (Table 20), kèm cảnh báo tie minh bạch.
- [x] **Seed-256 collapse** được chẩn đoán, tách khỏi aggregate và báo cáo riêng (đã sửa lỗi tự mâu thuẫn ở bản cũ).
- [x] Rút gọn từ 714 dòng xuống bản súc tích chuẩn bài báo khoa học (1 tiểu mục 3.13.1 + 2 bảng gọn + 2 điểm nối mạch), đồng bộ hoàn toàn format với `outputTask2–5.md`.

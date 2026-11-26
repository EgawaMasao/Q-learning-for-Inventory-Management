# Output Task 1 — Độ ổn định của SHAP qua các lần huấn luyện độc lập

**Tài liệu:** Báo cáo nội bộ + bản thảo sẵn sàng đưa vào bài báo  
**Task:** Task 1 — chứng minh/kiểm chứng feature importance ổn định qua các independent training runs  
**Dữ liệu và mã:** `Feedback 23-11/task1/`  
**Trạng thái:** Task 1 đã được xử lý về mặt thực nghiệm; kết luận là **ổn định có điều kiện**, không phải “mọi thứ đều ổn định”.  
**Vị trí đề xuất trong bài báo (bản mới `Xai_Inventory_Revision_27Sep (1).docx`):** **Section 3.13 “Reliability and Stability of the Explanations”** (ngay trước Conclusion). Mục này hiện đã có đoạn dẫn + **Table 18 (scorecard)**; Task 1 được chèn thành hai mục con **3.13.1** và **3.13.2** ngay **sau Table 18**.

> **Cách đọc tài liệu:** Phần 1–5 là báo cáo nội bộ bằng tiếng Việt. Phần 6–9 là bản thảo đưa vào bài báo, trong đó **bản tiếng Việt đặt trước** để anh/chị đọc và duyệt, sau đó là **bản tiếng Anh** để dán vào Word.

---

## 1. Tóm tắt điều hành

- Task 1 đặt câu hỏi: **feature importance của SHAP có ổn định qua 5 lần huấn luyện độc lập** (seed 42, 123, 256, 512, 1024) hay không.
- Đã chạy **60 cấu hình SHAP**: 5 seed × 3 scenario (EASY/MEDIUM/HARD) × 2 target (`π`, `logits`) × 2 mức (macro 3 feature, micro 660 feature), trên 50 state mỗi scenario, với background cố định 100 state.
- **Kết quả chính:**
  - **Micro:** Nhóm **top-8 sales feature** ổn định qua các seed. Với `logits`, Jaccard top-k = **1.000** và Spearman = **1.000** ở mọi k và mọi scenario; với `π`, Jaccard k=20 trung bình **0.861** (all seeds), **0.896** (bỏ seed 256), **1.000** (cặp 42–512); Spearman tương ứng **0.804 / 0.934 / 1.000**. Tuy nhiên, mỗi file micro chỉ có **9–12 giá trị importance khác nhau** trên 660 feature, với **648–651 feature bị tie**, nên các chỉ số top-k chủ yếu phản ánh nhóm sales feature và phải được đọc là indicative (xem Mục 4.3).
  - **Macro:** ổn định có điều kiện. Với `logits`, Demand là top-1 ở cả 5 seed, Jaccard = **1.000**, Spearman = **1.000**. Với `π`, top-1 **không ổn định**: Jaccard k=1 = **0.600** (all seeds), **0.500** (bỏ 256), **0.000** (cặp 42–512), vì 4 seed chọn Inventory còn seed 512 chọn Demand; tuy nhiên top-2 vẫn trùng **1.000**.
  - **Seed 256 bị collapse:** 87.7–88.7% quyết định chọn action 13, entropy 0.0093–0.0096, max prob 0.994–0.995. Checkpoint `ckpt-76` cũng tương tự (86.6% action 13, entropy 0.0070), nên đây không phải hiện tượng nhất thời. Seed 256 được báo cáo riêng.
- **Kết luận:** Task 1 **đã được giải quyết** với tư cách một thực nghiệm kiểm chứng. Câu trả lời đúng là: **top-8 sales feature ổn định; 651 feature còn lại bị tie nên không thể kết luận per-feature; macro ổn định có điều kiện; macro `π` top-1 không ổn định; seed 256 là outlier đã collapse và phải báo cáo riêng.** Không nên viết trong bài báo rằng “A2C_mod có explanation ổn định” mà không kèm điều kiện.

---

## 2. Vấn đề của Task 1

Bài báo hiện phân tích SHAP trên **một checkpoint được chọn** cho mỗi kiến trúc. Vì vậy, chưa có bằng chứng rằng thứ hạng feature quan trọng là **tái lập được qua các lần huấn luyện độc lập**. Đây là khoảng trống có thật, và chính phần **Study Limitations** của bài báo đã ghi: cần đánh giá độ ổn định của explanation qua nhiều seed bằng các metric chuyên dụng.

Vấn đề này quan trọng vì:

1. **Huấn luyện có tính ngẫu nhiên.** Hai seed khác nhau có thể học các biểu diễn khác nhau nhưng vẫn đạt reward tương đương; khi đó SHAP có thể chỉ ra các feature khác nhau.
2. **Policy bị bão hòa.** Nếu `π` gần one-hot, SHAP trên `π` có thể rất nhỏ và dễ nhiễu; cần thêm `logits` để thấy cấu trúc ưu tiên bên trong.
3. **Không thể suy từ một seed.** Nếu kết luận “feature X quan trọng” chỉ đúng ở một seed, thì đó là đặc điểm của lần huấn luyện đó, không phải của kiến trúc.

Task 1 yêu cầu cụ thể:

- Chạy SHAP **macro + top-k micro** trên checkpoint của **cả 5 seed**.
- Báo cáo **Jaccard/Spearman top-k** giữa các seed.
- Báo cáo **mean ± SD |SHAP|** theo từng feature.
- Nếu **seed 256 bị collapse** thì **báo cáo riêng**.

Task không yêu cầu kết quả phải “tích cực”; task yêu cầu **kiểm chứng và báo cáo trung thực**. Một kết luận có điều kiện vẫn là một lời giải hợp lệ, miễn là bằng chứng đầy đủ và diễn giải đúng.

---

## 3. Hướng tiếp cận

### 3.1. Thiết kế thực nghiệm

| Seed | Checkpoint | Episode | Ghi chú |
|---:|---|---:|---|
| 42 | `ckpt-53` | 530 | checkpoint cuối của seed 42 |
| 123 | `ckpt-60` | 600 | checkpoint cuối |
| 256 | `ckpt-60` | 600 | seed bị collapse, báo cáo riêng |
| 512 | `ckpt-60` | 600 | checkpoint cuối |
| 1024 | `ckpt-60` | 600 | checkpoint cuối |

- **Scenario:** EASY (`sales_scale=0.5`, `waste_rate=0.010`), MEDIUM (`1.0`, `0.025`), HARD (`1.5`, `0.050`).
- **State:** 50 state held-out mỗi scenario, lấy từ chuỗi test; giữ **cùng state** cho cả 5 seed.
- **Background:** 100 state synthetic, seed 42, **cố định cho mọi seed**. Mục đích là chỉ thay đổi trọng số mô hình, không thay đổi tập giải thích.
- **Actor deterministic:** dropout = 0; chỉ restore actor với `expect_partial()`; không retrain.
- **Không dùng checkpoint cũ** của `output Training` vì không thuộc 5 seed này.

### 3.2. Hai mức giải thích

- **Macro (3 feature):** gộp 660 feature thành 3 nhóm:
  - Inventory = trung bình tồn kho 220 SKU.
  - Demand = trung bình sales/demand 220 SKU.
  - Waste = trung bình waste 220 SKU.
- **Micro (660 feature):** 220 SKU × 3 biến (inventory, sales, waste), giữ nguyên cấu trúc sản phẩm.

Macro trả lời câu hỏi chiến lược (“nhóm yếu tố nào dẫn dắt quyết định?”). Micro trả lời câu hỏi sản phẩm (“SKU nào quan trọng?”). Task yêu cầu cả hai, nên báo cáo tách riêng và không trộn kết luận.

### 3.3. Hai target: `π` và `logits`

- **`π` (policy probability):** đầu ra quyết định thực tế; trả lời “feature này làm tăng/giảm xác suất chọn action bao nhiêu?”. Đây là target chính của bài báo.
- **`logits` (pre-softmax):** điểm ưu tiên trước chuẩn hóa; trả lời “feature này làm thay đổi xu hướng ưu tiên của mạng bao nhiêu?”. Khi `π` bão hòa, `logits` vẫn giữ cấu trúc xếp hạng.
- **Vì sao cần cả hai:** cả 5 seed đều gần deterministic (xem Mục 4.1). Nếu chỉ dùng `π`, SHAP có thể gần 0 và dễ nhiễu; nếu chỉ dùng `logits`, ta mất góc nhìn quyết định. Báo cáo cả hai giúp tách “ổn định ở mức quyết định” khỏi “ổn định ở mức biểu diễn”.

### 3.4. SHAP estimator

- **Macro:** `shap.KernelExplainer` trên 3 feature, background 100 state.
- **Micro:** `shap.PartitionExplainer` trên 660 feature, masker Partition với background 100 state.
- **Giá trị importance:** mean absolute SHAP trên 50 state và 14 action cho từng feature (áp dụng cho cả macro và micro).
- **Lưu ý:** SHAP là **associative**, không phải causal; kết quả diễn giải trong điều kiện phân phối nền đã chọn.

### 3.5. Metric

- **Jaccard top-k:** phần giao của tập top-k giữa hai seed chia cho phần hợp.
  - Micro: k = 10, 20, 50.
  - Macro: k = 1, 2 (vì chỉ có 3 feature).
- **Spearman:**
  - `spearman_all`: tương quan hạng trên toàn bộ feature.
  - `spearman_topk`: tương quan hạng trên hợp của hai tập top-k.
  - Với macro k=1, Spearman top-1 không xác định; báo cáo Spearman trên cả 3 feature.
- **Mean ± SD |SHAP|:** trung bình và độ lệch chuẩn qua 5 seed cho từng feature; kèm hệ số biến thiên CV = SD/mean.
- **Các tập con:**
  - `all`: cả 5 seed.
  - `excl_256`: 4 seed còn lại.
  - `stable_pair`: cặp 42–512 (hai seed có profile action gần nhau nhất).
- **Cross-target:** Jaccard/Spearman giữa `π` và `logits` trên cùng seed.

### 3.6. Chẩn đoán collapse

- Action distribution, entropy, max probability trên 50 state × 220 SKU.
- Seed 256 được kiểm tra thêm ở `ckpt-76` để xác nhận collapse không phải nhất thời.
- Seed 256 được tách khỏi aggregate chính và phân tích riêng.

### 3.7. Tính tái lập

- Notebook: `task1_shap_seed_stability.ipynb`.
- 60 file raw NPZ trong `output/raw/`.
- Các bảng CSV trong `output/`.
- Hình trong `output/figs/`.
- Notebook có cơ chế resume: nếu file raw đã có thì bỏ qua, không chạy lại SHAP.

---

## 4. Kết quả

### 4.1. Bão hòa policy và collapse của seed 256

Bảng dưới đây là MEDIUM; EASY/HARD cho xu hướng tương tự.

| Seed | Action chiếm ưu thế | Tỷ lệ action 13 | Entropy | Max prob |
|---:|---|---:|---:|---:|
| 42 | action 11 (81.6%), action 0 (18.4%) | 0.0% | 0.0008 | 0.9996 |
| 123 | action 13 (82.9%), action 2 (17.1%) | 82.9% | 0.0107 | 0.9960 |
| **256** | **action 13 (88.7%), action 8 (11.3%)** | **88.7%** | **0.0095** | **0.9944** |
| 512 | action 9 (85.6%), action 12 (14.4%) | 0.0% | 0.0025 | 0.9992 |
| 1024 | action 13 (66.7%), action 5 (33.3%) | 66.7% | 0.0140 | 0.9949 |

- **Tất cả 5 seed đều gần deterministic**, không chỉ seed 256. Seed 42 và 512 có action-13 = 0 nhưng lại bão hòa vào action khác.
- **Seed 256 là seed tập trung mạnh nhất vào một action** (action 13), với entropy thấp và max prob cao.
- Ở `ckpt-76`, seed 256 vẫn collapse: MEDIUM 86.6% action 13, entropy 0.0070, max prob 0.9973. Vì vậy không thể coi đây là dao động nhất thời của một checkpoint.
- **Kết luận:** điều kiện “seed 256 bị collapse” đã xảy ra; phải báo cáo riêng.

Hình minh họa: `fig_task1_seed256_collapse.png`.

### 4.2. Độ ổn định macro

Bảng dưới đây là trung bình qua 3 scenario; chi tiết từng scenario nằm trong `stability_summary.csv`.

| Target | k | Jaccard (all) | Jaccard (excl. 256) | Jaccard (42–512) | Spearman (all) | Spearman (excl. 256) | Spearman (42–512) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `π` | 1 | 0.600 | 0.500 | 0.000 | 0.800 | 0.750 | 0.500 |
| `π` | 2 | 1.000 | 1.000 | 1.000 | 0.800 | 0.750 | 0.500 |
| `logits` | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `logits` | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

- **`logits` rất ổn định:** Demand là top-1 ở cả 5 seed, mọi scenario.
- **`π` top-1 không ổn định:** 4 seed chọn Inventory, seed 512 chọn Demand.
  - MEDIUM top-1: seed 42 Inventory (0.0130), 123 Inventory (0.0274), 256 Inventory (0.0323), 512 Demand (0.0717), 1024 Inventory (0.0257).
- **Top-2 vẫn ổn định:** Jaccard k=2 = 1.000; cả 5 seed đều có Inventory và Demand trong top-2.
- **Diễn giải:** với macro, nên báo cáo `logits` cho stability; với `π`, chỉ nên khẳng định top-2, không khẳng định top-1.

Hình minh họa: `fig_task1_stability_summary.png` (hai panel macro).

### 4.3. Độ ổn định micro

| Target | k | Jaccard (all) | Jaccard (excl. 256) | Jaccard (42–512) | Spearman (all) | Spearman (excl. 256) | Spearman (42–512) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `π` | 10 | 0.819 | 0.909 | 1.000 | 0.804 | 0.934 | 1.000 |
| `π` | 20 | 0.861 | 0.896 | 1.000 | 0.804 | 0.934 | 1.000 |
| `π` | 50 | 0.870 | 0.928 | 1.000 | 0.804 | 0.934 | 1.000 |
| `logits` | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `logits` | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `logits` | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Cảnh báo quan trọng về tie.** Mỗi file micro SHAP chỉ chứa **9–12 giá trị importance khác nhau** trên 660 feature; **648–651 feature bị tie** ở mức gần 0. Nguyên nhân là PartitionExplainer được chạy trên policy trung bình qua 220 SKU, nên các feature không được phân biệt ở cấp SKU. Hệ quả: top-k Jaccard/Spearman bị chi phối bởi tie-breaking, không phải bởi thứ hạng per-feature. Chỉ **top-8 sales feature** là thực sự phân biệt được; các feature còn lại (inventory, waste và các sales ngoài top-8) không thể xếp hạng. Vì vậy, các chỉ số micro trong mục này và Bảng 18 nên được đọc là **indicative**, nhất quán với caveat ở Section 2.3.4.2 của bài báo.

- **`logits` cho Jaccard/Spearman = 1.000 ở mọi k và scenario**, nhưng con số này bị inflated bởi tie; không nên đọc là mọi feature đều ổn định.
- **Top-8 ổn định:** `sales_sku128, sales_sku66, sales_sku14, sales_sku71, sales_sku162, sales_sku65, sales_sku36, sales_sku4` (trong đó `sales_sku4` và `sales_sku59` đồng hạng ở vị trí 8–9).
- **`π` vẫn ổn định ở top-8 sales feature nhưng thấp hơn:** Jaccard k=20 trung bình 0.861 (range 0.834–0.891), 0.896 khi bỏ seed 256, 1.000 cho cặp 42–512.
- **Seed 256 làm giảm nhẹ stability của `π`**, nhưng không làm thay đổi kết luận ở mức top-8 sales: cùng nhóm SKU sales vẫn dẫn đầu; phần còn lại bị tie nên không thể kết luận.
- **Lưu ý kỹ thuật:** `spearman_topk` ở k=10 có thể thấp do nhiều feature gần 0 và ties; vì vậy báo cáo chính dùng `spearman_all` + Jaccard, còn `spearman_topk` để trong Supplementary.

Hình minh họa: `fig_task1_stability_summary.png` (hai panel micro); heatmap chi tiết: `fig_task1_micro_jaccard_k20_heatmaps.png`.

### 4.4. Mean ± SD |SHAP| theo feature

**Macro (MEDIUM):** Giá trị |SHAP| được lấy trung bình trên 50 state và 14 action.

| Target | Feature | Mean | SD | CV |
|---|---|---:|---:|---:|
| `logits` | Inventory | 7.006 | 4.450 | 0.635 |
| `logits` | Demand | 25.092 | 7.574 | 0.302 |
| `logits` | Waste | 0.054 | 0.019 | 0.353 |
| `π` | Inventory | 0.0221 | 0.0091 | 0.412 |
| `π` | Demand | 0.0250 | 0.0267 | 1.067 |
| `π` | Waste | 0.000196 | 0.000204 | 1.041 |

**Micro top-10 `logits` (MEDIUM):**

| Feature | Mean | SD | CV |
|---|---:|---:|---:|
| `sales_sku128` | 0.1452 | 0.0214 | 0.147 |
| `sales_sku66` | 0.1440 | 0.0207 | 0.144 |
| `sales_sku14` | 0.1435 | 0.0353 | 0.246 |
| `sales_sku71` | 0.1343 | 0.0294 | 0.219 |
| `sales_sku162` | 0.1262 | 0.0415 | 0.329 |
| `sales_sku65` | 0.1235 | 0.0420 | 0.340 |
| `sales_sku36` | 0.1148 | 0.0382 | 0.333 |
| `sales_sku4` | 0.1144 | 0.0248 | 0.216 |
| `sales_sku59` | 0.1144 | 0.0248 | 0.216 |
| `waste_sku188` | 0.0369 | 0.0090 | 0.245 |

**Micro top-10 `π` (MEDIUM):**

| Feature | Mean | SD | CV |
|---|---:|---:|---:|
| `sales_sku162` | 0.000185 | 0.000110 | 0.594 |
| `sales_sku66` | 0.000165 | 0.000129 | 0.781 |
| `sales_sku36` | 0.000155 | 0.000156 | 1.006 |
| `sales_sku71` | 0.000150 | 0.000104 | 0.692 |
| `sales_sku14` | 0.000150 | 0.000121 | 0.809 |
| `sales_sku128` | 0.000145 | 0.000132 | 0.909 |
| `sales_sku59` | 0.000109 | 0.000111 | 1.019 |
| `sales_sku4` | 0.000102 | 0.000110 | 1.079 |
| `sales_sku65` | 0.000082 | 0.000055 | 0.672 |
| `sales_sku137` | 0.000038 | 0.000035 | 0.909 |

- **Lưu ý tie:** Bảng top-10 micro ở trên chỉ có 8–9 sales feature là phân biệt được; `sales_sku4` và `sales_sku59` trùng giá trị, còn `waste_sku188` trùng với 648–651 feature khác. Vì vậy, thứ hạng của các feature ngoài top-8 sales không xác định.
- Bảng đầy đủ 660 feature: `mean_sd_micro.csv`.
- Bảng đầy đủ macro: `mean_sd_macro.csv`.
- **Diễn giải:** `logits` có magnitude lớn hơn và CV thấp hơn; `π` có magnitude nhỏ hơn nhiều và CV cao hơn, phù hợp với policy bão hòa.

### 4.5. Phân tích riêng seed 256

**So sánh seed 256 với 4 seed còn lại (MEDIUM):**

| Level | Target | k | Jaccard (256 vs others) | Spearman (256 vs others) |
|---|---|---:|---:|---:|
| Macro | `π` | 1 | 0.750 | 0.875 |
| Macro | `π` | 2 | 1.000 | 0.875 |
| Macro | `logits` | 1 | 1.000 | 1.000 |
| Macro | `logits` | 2 | 1.000 | 1.000 |
| Micro | `π` | 10 | 0.635 | 0.537 |
| Micro | `π` | 20 | 0.780 | 0.537 |
| Micro | `π` | 50 | 0.770 | 0.537 |
| Micro | `logits` | 10 | 1.000 | 1.000 |
| Micro | `logits` | 20 | 1.000 | 1.000 |
| Micro | `logits` | 50 | 1.000 | 1.000 |

- Với `π` micro, seed 256 thấp hơn rõ: Jaccard k=20 = 0.780 ở EASY/MEDIUM, 0.863 ở HARD; Spearman = 0.537 / 0.537 / 0.752.
- Với `logits` micro, seed 256 vẫn trùng hoàn toàn top-k (Jaccard = 1.000), nhưng con số này bị inflated bởi tie; chỉ top-8 sales feature là phân biệt được.
- **Macro `π`:** seed 256 xếp Inventory #1, trong khi 4 seed còn lại xếp Demand #1. Đây là khác biệt định tính.
- **Macro `logits`:** seed 256 vẫn xếp Demand #1, giống 4 seed còn lại.

**Top feature của seed 256 so với 4 seed còn lại (MEDIUM):**

| Level | Target | Feature | Seed 256 mean | Other 4 mean ± SD | Rank in other 4 |
|---|---|---|---:|---:|---:|
| Macro | `π` | Inventory | 0.0323 | 0.0195 ± 0.0082 | 2 |
| Macro | `π` | Demand | 0.0119 | 0.0283 ± 0.0297 | 1 |
| Macro | `logits` | Demand | 19.591 | 26.468 ± 7.992 | 1 |
| Macro | `logits` | Inventory | 3.529 | 7.876 ± 4.622 | 2 |
| Micro | `logits` | `sales_sku66` | 0.1175 | 0.1506 ± 0.0167 | 3 |
| Micro | `logits` | `sales_sku128` | 0.1158 | 0.1525 ± 0.0159 | 1 |
| Micro | `logits` | `sales_sku14` | 0.1125 | 0.1513 ± 0.0355 | 2 |
| Micro | `π` | `sales_sku36` | 0.000172 | 0.000151 ± 0.000180 | 6 |
| Micro | `π` | `sales_sku162` | 0.000164 | 0.000190 ± 0.000126 | 2 |

- **Micro `logits`:** seed 256 xếp `sales_sku66` #1, còn 4 seed còn lại xếp `sales_sku128` #1; tập top vẫn giống nhau.
- **Micro `π`:** seed 256 xếp `sales_sku36` #1, còn 4 seed còn lại xếp `sales_sku162` #1; thứ hạng chi tiết nhạy hơn.
- **Quan trọng:** seed 256 **không phải nguyên nhân duy nhất** làm macro `π` mất ổn định. Kể cả khi bỏ seed 256, Jaccard top-1 vẫn chỉ 0.500, và seed 512 vẫn chọn Demand #1. Vì vậy không thể quy toàn bộ bất ổn macro `π` cho seed 256.

### 4.6. So sánh `π` và `logits`

| Level | k | Jaccard (`π` vs `logits`) | Spearman |
|---|---:|---:|---:|
| Micro | 20 | 0.928 | 0.899 |
| Macro | 1 | 0.200 | 0.600 |
| Macro | 2 | 1.000 | 0.600 |

- **Micro:** hai target đồng thuận cao ở mức SKU, nhưng chủ yếu ở top-8 sales feature; phần còn lại bị tie nên không nên over-interpret Jaccard 0.928.
- **Macro:** top-1 khác nhau, nhưng top-2 đồng thuận hoàn toàn.
- **Diễn giải:** target sensitivity chủ yếu ảnh hưởng đến macro top-1; micro ranking vững hơn.

### 4.7. Hình và file chính

**Hình cho bài báo (chọn lọc):**
- `fig_task1_stability_summary.png` — tóm tắt Jaccard/Spearman theo top-k cho micro/macro, `π`/`logits`, all/excl-256/42–512.
- `fig_task1_seed256_collapse.png` — action-13 share và entropy theo seed, làm nổi bật seed 256.

**Hình bổ sung:**
- `fig_task1_micro_jaccard_k20_heatmaps.png` — heatmap Jaccard top-20 micro cho `π`/`logits` × EASY/MEDIUM/HARD.
- `micro_top20_logits_MEDIUM.png`, `micro_top20_pi_MEDIUM.png`, `macro_mean_sd.png`, `action_distribution_MEDIUM.png` — các hình cũ có thể dùng cho Supplementary.

**Bảng/CSV chính:**
- `stability_summary.csv`, `stability_pairwise.csv`.
- `mean_sd_macro.csv`, `mean_sd_micro.csv`.
- `seed256_pairwise_stability.csv`, `seed256_top_macro.csv`, `seed256_top_micro.csv`.
- `collapse_diagnostics.csv`, `extra_checkpoint_diagnostics.csv`.
- `cross_target_stability.csv`.

---

## 5. Đánh giá: Task 1 đã được giải quyết chưa?

| Yêu cầu của Task 1 | Trạng thái | Bằng chứng |
|---|---|---|
| Chạy SHAP macro + top-k micro trên **cả 5 seed** | **ĐẠT** | 60 raw NPZ; 5 seed × 3 scenario × 2 target × 2 mức |
| Báo cáo **Jaccard/Spearman top-k** giữa các seed | **ĐẠT** | `stability_pairwise.csv` (510 dòng), `stability_summary.csv` (90 dòng); báo cáo có cả `spearman_all` và `spearman_topk` |
| Báo cáo **mean ± SD \|SHAP\|** theo feature | **ĐẠT** | `mean_sd_macro.csv` (18 dòng), `mean_sd_micro.csv` (3.960 dòng); n = 5 seed |
| **Seed 256 collapse → báo cáo riêng** | **ĐẠT** | Mục 4.1 + 4.5; `seed256_pairwise_stability.csv`, `seed256_top_macro.csv`, `seed256_top_micro.csv`; seed 256 tách khỏi aggregate |

**Kết luận:** Task 1 **đã được giải quyết** với tư cách một thực nghiệm kiểm chứng và báo cáo. Kết quả không phải “mọi thứ ổn định”, mà là:

- **Micro:** top-8 sales feature ổn định, đặc biệt với `logits`; 651 feature còn lại bị tie nên không thể kết luận per-feature.
- **Macro:** ổn định có điều kiện; `logits` ổn định, `π` top-1 không ổn định, `π` top-2 ổn định.
- **Seed 256:** collapse rõ, là outlier, phải báo cáo riêng.

**Những gì bài báo KHÔNG nên khẳng định:**
- Không nên nói “A2C_mod có explanation ổn định” mà không nêu điều kiện.
- Không nên nói “micro ổn định trên toàn bộ 660 feature” — dữ liệu chỉ phân biệt được top-8 sales feature; 648–651 feature còn lại bị tie.
- Không nên nói “macro `π` top-1 ổn định” — dữ liệu không ủng hộ.
- Không nên suy rộng cho DQN (audit này chỉ chạy A2C_mod).
- Không nên suy rộng cho mọi background/state; audit dùng 50 state và một background cố định.
- Không nên nói đã khử hoàn toàn confound episode: seed 42 là 530 episode, các seed khác 600. Đây là hạn chế đã ghi, không phải lỗi.

**Vì sao đây vẫn là lời giải hợp lệ:** Task 1 yêu cầu “chạy và báo cáo”, không yêu cầu “phải chứng minh là ổn định”. Một kết quả có điều kiện, có bằng chứng định lượng, có tách seed 256, và có metric rõ ràng là một lời giải khoa học đầy đủ. Nó cũng trực tiếp trả lời khoảng trống mà phần Study Limitations của bài báo đã nêu.

---

## 6. Vị trí trình bày trong bài báo (bản mới `Xai_Inventory_Revision_27Sep (1).docx`)

### 6.1. Vị trí chèn

Trong file **`Xai_Inventory_Revision_27Sep (1).docx`**, mục cần chèn là:

> **Section 3.13 — Reliability and Stability of the Explanations**
> (nằm ngay trước **Conclusion**, sau **Managerial Case Studies and Interpretation**).

Lưu ý quan trọng so với bản cũ:
- Bản mới **đã đánh số lại**: Results là **Section 3** (bản cũ là 4). Mục reliability/stability giờ là **3.13**, không còn là 4.12.
- Mục 3.13 **không còn trống**: đã có một đoạn dẫn và **Table 18 — Interpretability scorecard**.
- Vì vậy Task 1 phải được chèn **sau Table 18**, thành hai mục con mới **3.13.1** và **3.13.2**; **không** thay thế scorecard và **không** tạo lại heading 3.13.

Vị trí này phù hợp vì:
- Đúng chủ đề “reliability and stability”.
- Bài báo đã có tham chiếu nội bộ “Table 18 (Section 3.13)”.
- Đoạn dẫn 3.13 đã liệt kê tiêu chí “reproducibility across independently trained agents” nhưng Table 18 chưa có hàng này; Task 1 chính là phần bổ sung đó.
- Phần **Study Limitations** nêu khoảng trống multi-seed; 3.13.1 trực tiếp lấp khoảng trống.

### 6.2. Tên mục con cần tạo (sau Table 18, trước Conclusion)

- **3.13.1 Stability of SHAP Feature Importance Across Independent Training Runs**
  - Tiếng Việt: **3.13.1 Độ ổn định của tầm quan trọng đặc trưng SHAP qua các lần huấn luyện độc lập**
- **3.13.2 Seed-256 Policy Collapse and Separate Explanation Analysis**
  - Tiếng Việt: **3.13.2 Collapse chính sách của seed 256 và phân tích giải thích riêng**

Nếu Word tự đánh số khác, giữ nguyên tên mục và để Word renumber; vị trí vật lý vẫn là sau Table 18, ngay trước Conclusion.

### 6.3. Nối với scorecard hiện có (khuyến nghị)

Section 3.13 hiện có câu “Table 18 collects them into a single scorecard … reproducibility across independently trained agents …”, nhưng **Table 18 chưa có hàng nào cho “reproducibility across independently trained agents”**. Task 1 chính là phần đó. Đề xuất:

1. Thêm một hàng vào **Table 18** (đặt trong nhóm *Stability*, ngay trước *Consistency*):
   - Criterion: **Reproducibility (independent runs)**
   - Metric: Top-k Jaccard / Spearman of SHAP importance across five A2C_mod runs
   - DQN: Not measured
   - A2C_mod: top-8 sales stable (logits Jaccard 1.00; `π` k = 20 Jaccard 0.861); macro `π` top-1 unstable (0.60)
   - Reported in: **Section 3.13.1**
2. Thêm một câu nối vào cuối đoạn dẫn 3.13 (trước Table 18):
   - EN: “A dedicated multi-seed SHAP audit for A2C_mod is reported in Section 3.13.1, and the collapsed seed 256 is analyzed separately in Section 3.13.2.”
   - VI: “Một kiểm chứng SHAP đa seed riêng cho A2C_mod được báo cáo tại Mục 3.13.1, và seed 256 bị collapse được phân tích riêng tại Mục 3.13.2.”

### 6.4. Chỉnh cross-reference ở mục khác

1. **Section 3.6 (SHapley Additive exPlanations):** thêm một câu:
   - EN: “A dedicated audit of SHAP stability across the five independent A2C_mod training runs is reported in Section 3.13.1.”
   - VI: “Một kiểm chứng riêng về độ ổn định của SHAP qua 5 lần huấn luyện A2C_mod độc lập được báo cáo tại Mục 3.13.1.”

2. **Section 3.9 (Sensitivity to Action Space Resolution):** cross-ref hiện có “the seed-256 collapse noted in Section 3.3.4”.
   - **Đây KHÔNG phải lỗi:** Section 3.3.4 (Ablation Validation) thực sự có bàn seed-256 (A2C_mod collapsed ở seed 123 và 256). Có thể giữ nguyên 3.3.4.
   - Nếu muốn trỏ nơi phân tích chuyên sâu, đổi thành: “... the seed-256 collapse noted in Section 3.3.4 and analyzed in Section 3.13.2.”
   - VI: “... seed-256 đã ghi nhận tại Mục 3.3.4 và được phân tích tại Mục 3.13.2.”

3. **Section 3.10 (Scope and Interpretation of the Study's Claims):** thêm một câu:
   - EN: “Stability claims in this section refer to the selected checkpoint; multi-seed SHAP stability is quantified separately in Section 3.13.1.”
   - VI: “Các khẳng định về stability trong mục này gắn với checkpoint được chọn; độ ổn định SHAP qua nhiều seed được định lượng riêng tại Mục 3.13.1.”

4. **Section 5 (Study Limitations and Future Work):** cập nhật limitation đầu tiên:
   - Hiện tại: “explanation robustness should be evaluated across multiple random seeds using dedicated stability metrics...”
   - Đề xuất: “The multi-seed robustness of the A2C_mod SHAP explanations is evaluated in Section 3.13.1. Extending the audit to DQN, additional backgrounds and state samples, matched training budgets, and a formal SHAP noise-floor analysis remains future work.”
   - VI: “Độ vững của SHAP cho A2C_mod qua nhiều seed đã được đánh giá tại Mục 3.13.1. Việc mở rộng audit sang DQN, background/state khác, cùng số episode, và phân tích noise floor của SHAP vẫn là hướng tiếp theo.”

5. **Sửa số cũ còn sót:** bảng “Trained runs (seeds) used in each analysis” vẫn ghi “Sections 3.3 - 4.5 and 3.7.3”; **“4.5” phải là “3.5”** (Sensitivity Analysis).

6. **Mâu thuẫn cần xử lý (ngoài phạm vi Task 1):** Conclusion nói “MSX Stability varies widely across five independently trained runs”, nhưng bảng “Trained runs” ghi Stability = **one run/agent**, và Table 18 không có hàng five-run MSX. Cần bổ sung kết quả 5-seed MSX hoặc sửa câu Conclusion.

7. **Supplementary Material:** thêm một mục ngắn:
   - Full per-scenario Jaccard/Spearman tables.
   - Full 660-feature mean ± SD table.
   - Pairwise heatmaps và top-20 figures.
   - Seed 256 separate tables.

### 6.5. Đánh số bảng/hình

Bản mới hiện có tối đa **Table 18** và **Figure 21**. Do đó Task 1 dùng:
- **Table 19** — Stability of SHAP top-k across five A2C_mod seeds.
- **Table 20** — Mean ± SD |SHAP| (MEDIUM).
- **Table 21** — Seed-256 collapse and separate analysis.
- **Figure 22** — Stability summary (`fig_task1_stability_summary.png`).
- **Figure 23** — Seed-256 collapse (`fig_task1_seed256_collapse.png`).
- **Supplementary Figure S1** — Micro Jaccard heatmaps (`fig_task1_micro_jaccard_k20_heatmaps.png`).

Nếu Word đánh số tự động khác, chỉ cần giữ caption và để Word cập nhật.

---

## 7. Bản thảo tiếng Việt (đọc trước)

> Bản này để anh/chị đọc và duyệt nội dung. Bản tiếng Anh ở Mục 8 là bản dùng để dán vào bài báo.
> **Không tạo lại heading 3.13** (đã có). Chỉ chèn hai mục con dưới đây **sau Table 18**.

#### 3.13.1. Độ ổn định của tầm quan trọng đặc trưng SHAP qua các lần huấn luyện độc lập

**Thiết kế thực nghiệm.** Mỗi seed được đánh giá trên cùng 50 state held-out cho mỗi scenario, với background cố định gồm 100 state synthetic (seed 42), để khác biệt về attribution có thể quy cho tham số mô hình thay vì tập giải thích. Attribution macro dùng KernelSHAP trên 3 feature tổng hợp (inventory, demand, waste); attribution micro dùng PartitionExplainer trên 660 feature sản phẩm. Với mỗi target, scenario và seed, chúng tôi tính |SHAP| trung bình theo từng feature trên 50 state. Độ ổn định được định lượng bằng (i) độ trùng lặp Jaccard của top-k feature giữa mọi cặp seed (k = 10, 20, 50 cho micro; k = 1, 2 cho macro) và (ii) tương quan hạng Spearman trên toàn bộ feature và trên hợp của hai tập top-k. Chúng tôi báo cáo kết quả cho cả 5 seed, cho 4 seed sau khi loại seed 256, và cho cặp 42–512, là hai seed có profile action gần nhau nhất. Giá trị trong Bảng 19 là trung bình qua ba scenario; giá trị từng scenario và Spearman top-k được báo cáo trong Bảng S1 của Supplementary.

**Mức macro.** `logits` trước softmax rất ổn định: Demand là feature top-1 ở cả 5 seed trong mọi scenario, với Jaccard = 1,00 và Spearman = 1,00 ở k = 1 và k = 2. Target xác suất chính sách kém ổn định hơn ở top-1: Jaccard trung bình là 0,60 trên tất cả cặp seed, 0,50 khi loại seed 256, và 0,00 cho cặp 42–512, vì bốn seed xếp Inventory đầu tiên còn seed 512 xếp Demand đầu tiên (Bảng 19). Tuy nhiên, top-2 vẫn trùng hoàn toàn (Jaccard = 1,00), và Spearman trên cả ba feature là 0,80 (0,75 khi loại seed 256). Kết luận macro rằng demand chi phối là vững đối với `logits` và đối với tập top-2, nhưng không vững đối với thứ hạng top-1 của `π`.

**Mức micro.** Target `logits` gần như bất biến qua các seed: Jaccard = 1,00 và Spearman = 1,00 ở k = 10, 20, 50 trong mọi scenario. Cùng một nhóm SKU liên quan đến sales chiếm các vị trí đầu: `sales_sku128`, `sales_sku66`, `sales_sku14`, `sales_sku71`, `sales_sku162`, `sales_sku65`, `sales_sku36` và `sales_sku4` tạo thành top-8 ổn định. Target `π` cũng ổn định nhưng thấp hơn: ở k = 20, Jaccard trung bình là 0,861 trên tất cả seed (range 0,834–0,891 qua các scenario), 0,896 khi loại seed 256, và 1,000 cho cặp 42–512; Spearman tương ứng là 0,804, 0,934 và 1,000. Giá trị thấp hơn của `π` phù hợp với policy gần one-hot: khi `π` bão hòa, thay đổi nhỏ trong chính sách có thể tạo thay đổi lớn hơn trong thứ hạng của các feature ít quan trọng.

**Cảnh báo về tie.** PartitionExplainer trên policy trung bình qua 220 SKU chỉ tạo ra 9–12 giá trị importance khác nhau trên 660 feature; 648–651 feature bị tie ở mức gần 0. Vì vậy, các chỉ số top-k chủ yếu phản ánh nhóm sales feature (top-8); thứ hạng của các feature còn lại không được xác định. Các chỉ số micro nên được đọc là indicative, nhất quán với caveat ở Section 2.3.4.2.

**Trung bình ± độ lệch chuẩn của |SHAP|.** Bảng 20 báo cáo trung bình ± độ lệch chuẩn của |SHAP| trung bình qua các seed, trên 50 state và 14 action, cho các feature macro và các feature micro hàng đầu trong scenario MEDIUM. Với `logits`, các feature micro hàng đầu có |SHAP| trung bình từ 0,114 đến 0,145, với hệ số biến thiên 0,14–0,34; cùng các feature này được mọi seed xếp vào top. Với `π`, magnitude nhỏ hơn hai đến ba bậc và hệ số biến thiên lớn hơn (0,59–1,08), phản ánh policy output bị bão hòa. Bảng đầy đủ 660 feature nằm trong Bảng S2 của Supplementary.

**So sánh hai target.** Hai target đồng thuận ở mức micro (Jaccard trung bình 0,928 và Spearman 0,899 ở k = 20) nhưng khác nhau ở top-1 macro (Jaccard 0,20; 1,00 ở k = 2). Độ nhạy target này giải thích vì sao lựa chọn target quan trọng với diễn giải macro nhưng không ảnh hưởng nhiều đến thứ hạng micro.

**Diễn giải.** Kết quả ủng hộ một kết luận có điều kiện. Nhóm top-8 sales feature tái lập được qua các lần huấn luyện A2C_mod độc lập, đặc biệt với `logits`; 651 feature còn lại bị tie nên không thể khẳng định thứ hạng per-feature cho toàn bộ 660 feature. `logits` macro ổn định; `π` macro ổn định ở tập top-2 nhưng không ổn định ở top-1. Vì vậy, nên dùng SHAP micro cho diễn giải cấp sản phẩm ở mức top-8 sales và `logits` cho các khẳng định ổn định ở cấp macro, đồng thời tránh các khẳng định top-1 macro dựa trên `π`.

**Hạn chế của kiểm chứng.** Kiểm chứng dùng 50 state mỗi scenario và một background synthetic cố định; chỉ bao gồm A2C_mod, không bao gồm DQN; seed 42 được đánh giá ở 530 episode trong khi các seed khác ở 600 episode; và chưa định lượng noise floor của estimator SHAP. Ngoài ra, PartitionExplainer trên policy trung bình tạo ra tie lan rộng (648–651/660 feature), nên chỉ top-8 sales feature là xếp hạng được; một explainer per-SKU hoặc tie-aware là cần thiết để xác định thứ hạng đầy đủ. Những lựa chọn này giới hạn phạm vi khẳng định nhưng không làm thay đổi kết luận định tính: top-8 sales feature ổn định, còn thứ hạng top-1 macro của `π` thì không. Kiểm tra matched-500-episode và phân tích noise floor là các mở rộng tự nhiên.

#### 3.13.2. Collapse chính sách của seed 256 và phân tích giải thích riêng

Seed 256 là một outlier rõ rệt. Trên 50 state mỗi scenario, 88,5%, 88,7% và 87,7% quyết định ở mức sản phẩm chọn action 13 lần lượt ở EASY, MEDIUM và HARD; entropy chính sách trung bình là 0,0093–0,0096 và xác suất lớn nhất trung bình là 0,994–0,995 (Bảng 21). Cùng pattern xuất hiện ở checkpoint 76 (86,6% action 13 ở MEDIUM, entropy 0,0070), nên collapse không phải hiện tượng nhất thời của checkpoint được chọn. Vì policy gần one-hot làm nén target `π`, seed 256 được loại khỏi aggregate ổn định chính và báo cáo riêng.

Ở mức macro, seed 256 xếp Inventory đầu tiên cho `π` (|SHAP| trung bình 0,032) và Demand đầu tiên cho `logits` (19,59). Thứ hạng `π` khác với bốn seed còn lại, vốn xếp Demand đầu tiên (trung bình 0,028 ± 0,030) và Inventory thứ hai (0,020 ± 0,008); thứ hạng `logits` lại đồng thuận với bốn seed còn lại. Ở mức micro, các feature hàng đầu của seed 256 trùng với nhóm feature phổ biến, nhưng thứ tự khác: với `logits`, `sales_sku66` đứng đầu ở seed 256 trong khi `sales_sku128` đứng đầu ở bốn seed còn lại; tuy vậy Jaccard với các seed khác vẫn là 1,00 ở k = 20. Với `π`, mức trùng thấp hơn (Jaccard trung bình 0,780 ở k = 20 cho EASY và MEDIUM, 0,863 cho HARD; Spearman 0,537 và 0,752). Collapse vì vậy ảnh hưởng đến magnitude và thứ tự chi tiết của attribution `π`, nhưng không làm đảo ngược kết luận micro chính: cùng nhóm SKU liên quan đến sales vẫn chiếm ưu thế.

Quan trọng hơn, seed 256 không phải nguyên nhân duy nhất gây mất ổn định macro `π`. Kể cả khi loại seed 256, Jaccard top-1 vẫn là 0,50, và seed 512 vẫn xếp Demand đầu tiên. Đây là lý do chúng tôi báo cáo kết quả top-1 macro `π` là phụ thuộc target và seed, trong khi tập top-2 và thứ hạng `logits` ổn định. Seed 256 nên được xem là một lần huấn luyện bị collapse và được báo cáo riêng, không dùng một mình để ủng hộ hay bác bỏ tính ổn định của các seed khác.

---

## 8. English manuscript version (dán vào bài báo)

> **Không tạo lại heading 3.13** (đã có trong bài). Chỉ chèn hai mục con dưới đây **sau Table 18** và **trước Conclusion**. Nếu muốn, thêm câu nối ở Mục 6.3 vào cuối đoạn dẫn 3.13.

#### 3.13.1. Stability of SHAP Feature Importance Across Independent Training Runs

**Scope and link to the scorecard.** Table 18 summarizes the reliability evidence collected in this section. This subsection supplies the one scorecard criterion that the table lists but does not yet quantify, namely reproducibility across independently trained agents, for the five A2C_mod runs used in this study (seeds 42, 123, 256, 512, and 1024). The audit covers the macro-level three-feature representation and the micro-level 660-dimensional representation, the EASY, MEDIUM, and HARD scenarios, and two explanation targets: the policy probability `π` and the pre-softmax logits. It is intended to establish the reproducibility of the feature-space explanations, not to re-evaluate their faithfulness, which is assessed in Section 3.8.

**Experimental design.** Each seed was evaluated on the same 50 held-out states per scenario, with a fixed background of 100 synthetic states (seed 42), so that differences in attributions can be attributed to the trained parameters rather than to the explanation sample. Macro attributions use KernelSHAP on the three aggregated features (inventory, demand, and waste); micro attributions use PartitionExplainer on the 660 product-level features. For each target, scenario, and seed, we compute the mean absolute SHAP value per feature over the 50 states. Stability is quantified by (i) the Jaccard overlap of the top-k features between every pair of seeds (k = 10, 20, and 50 for micro; k = 1 and 2 for macro) and (ii) the Spearman rank correlation over all features and over the union of the top-k sets. We report results for all five seeds, for the four seeds excluding seed 256, and for the pair of seeds 42 and 512, which have the most similar action profiles. Values in Table 19 are means over the three scenarios; per-scenario values and top-k Spearman correlations are reported in Supplementary Table S1.

**Macro level.** The pre-softmax logits are highly stable: Demand is the top feature for all five seeds in every scenario, with Jaccard = 1.00 and Spearman = 1.00 at k = 1 and k = 2. The policy-probability target is less stable at the top-1 level: the mean Jaccard is 0.60 across all seed pairs, 0.50 when seed 256 is excluded, and 0.00 for the 42–512 pair, because four seeds rank Inventory first while seed 512 ranks Demand first (Table 19). The top-2 overlap is nevertheless 1.00, and the Spearman correlation over the three features is 0.80 (0.75 excluding seed 256). The macro-level conclusion that demand dominates is therefore robust for logits and for the top-2 set, but not for the top-1 ranking of `π`.

**Micro level.** The logits target is essentially invariant across seeds: Jaccard = 1.00 and Spearman = 1.00 for k = 10, 20, and 50 in all scenarios. The same sales-related SKUs occupy the top ranks, with `sales_sku128`, `sales_sku66`, `sales_sku14`, `sales_sku71`, `sales_sku162`, `sales_sku65`, `sales_sku36`, and `sales_sku4` forming the stable top-8. The `π` target is also stable but less so: for k = 20, the mean Jaccard is 0.861 across all seeds (range 0.834–0.891 across scenarios), 0.896 when seed 256 is excluded, and 1.000 for the 42–512 pair; the corresponding Spearman correlations are 0.804, 0.934, and 1.000. The lower values for `π` are consistent with the near-one-hot policies: when `π` is saturated, small changes in the policy can produce larger changes in the ranking of low-importance features. **Tie caveat.** The PartitionExplainer on the SKU-averaged policy produces only 9–12 distinct importance values across the 660 features; 648–651 features are tied near zero. The top-k Jaccard and Spearman values are therefore dominated by tie-breaking and should be read as indicative rather than as per-feature stability. Only the top-8 sales-related SKUs are distinguishable; the remaining inventory, waste, and non-top sales features cannot be ranked. This is consistent with the paper's earlier observation that micro SHAP concentrates on 5–8 SKUs (Section 3.6.3) and with the caveat in Section 2.3.4.2 that micro rankings are indicative.

**Mean ± SD of |SHAP|.** Table 20 reports the mean ± SD of mean absolute SHAP values across seeds, averaged over the 50 states and 14 actions, for the macro features and the top micro features in the MEDIUM scenario. For logits, the top micro features have mean |SHAP| between 0.114 and 0.145, with coefficients of variation of 0.14–0.34; the same features are ranked in the top positions by every seed. For `π`, the magnitudes are two to three orders of magnitude smaller and the coefficients of variation are larger (0.59–1.08), reflecting the saturation of the policy output. The full 660-feature mean ± SD table is provided in Supplementary Table S2.

**Cross-target comparison.** The two targets agree at the micro level (mean Jaccard 0.928 and Spearman 0.899 at k = 20) but diverge at the macro top-1 level (Jaccard 0.20; 1.00 at k = 2). This target sensitivity explains why the choice of target matters for the macro narrative but not for the micro ranking. In the 50-state setting used here, the micro-level agreement is higher than in the 10-state sensitivity analysis of Section 3.6.6, consistent with the sample-size sensitivity reported there.

**Interpretation.** The results support a conditional conclusion. The top-8 sales-related micro features are reproducible across independent A2C_mod runs, especially for logits; the remaining 651 features are tied and cannot be ranked, so the micro metrics should be read as indicative. Macro-level logits are stable; macro-level `π` is stable in its top-2 set but not in its top-1 ranking. The audit therefore supports the use of micro-level SHAP for product-level interpretation at the level of the top-8 sales features and of logits for macro-level stability claims, while cautioning against top-1 macro claims based on `π`. This across-seed stability should be distinguished from the across-scenario sensitivity reported in Section 3.6.4: the same seed changes its ranking when the demand/waste regime changes, but different seeds trained on the same regime produce similar rankings.

**Limitations of the audit.** The audit uses 50 states per scenario and one fixed synthetic background; it covers A2C_mod but not DQN; seed 42 is evaluated at 530 episodes while the other seeds are evaluated at 600; and the SHAP estimator noise floor is not quantified. In addition, the PartitionExplainer on the SKU-averaged policy produces extensive ties (648–651 of 660 features), so only the top-8 sales features can be ranked; a per-SKU or tie-aware explainer is needed to rank the remaining features. These choices bound the scope of the claims but do not change the qualitative conclusion: the top-8 sales features are stable, and the macro-level `π` top-1 ranking is not. A matched-500-episode check and a noise-floor analysis are natural extensions.

#### 3.13.2. Seed-256 Policy Collapse and Separate Explanation Analysis

Seed 256 is a clear outlier. Across the 50 states per scenario, 88.5%, 88.7%, and 87.7% of product-level decisions select action 13 in EASY, MEDIUM, and HARD, respectively; the mean policy entropy is 0.0093–0.0096 and the mean maximum probability is 0.994–0.995 (Table 21). The same pattern is present at checkpoint 76 (86.6% action 13 in MEDIUM, entropy 0.0070), so the collapse is not a transient artifact of the selected checkpoint. Because a near-one-hot policy compresses the `π` target, seed 256 is excluded from the primary stability aggregate and reported separately.

At the macro level, seed 256 ranks Inventory first for `π` (mean |SHAP| 0.032) and Demand first for logits (19.59). The `π` ranking differs from the other four seeds, which rank Demand first (mean 0.028 ± 0.030) and Inventory second (0.020 ± 0.008); the logits ranking agrees with the other seeds. At the micro level, the top features of seed 256 overlap with the common top features, but the ordering differs: for logits, `sales_sku66` is first for seed 256 while `sales_sku128` is first for the other four seeds; the Jaccard overlap with the other seeds is nevertheless 1.00 at k = 20. For `π`, the overlap is lower (mean Jaccard 0.780 at k = 20 in EASY and MEDIUM, 0.863 in HARD; Spearman 0.537 and 0.752, respectively). The collapse therefore affects the magnitude and fine-grained ordering of `π` attributions, but it does not overturn the main micro-level conclusion: the same sales-related SKUs remain dominant.

Importantly, seed 256 is not the sole source of macro-level `π` instability. Even after excluding it, the top-1 Jaccard is 0.50, and seed 512 still ranks Demand first. This is why we report the macro `π` top-1 result as target- and seed-dependent, while the top-2 set and the logits ranking are stable. Seed 256 should therefore be treated as a collapsed run and reported separately, not used on its own to support or refute the stability of the other runs.

---

## 9. Bảng, hình và caption song ngữ

> Số bảng/hình đã cập nhật theo bản mới: Table 18 đã là scorecard, nên Task 1 dùng **Table 19–21** và **Figure 22–23**.

### 9.1. Table 19 — Stability of SHAP top-k across five A2C_mod seeds

**Caption EN:** Table 19. Stability of SHAP feature importance across five independent A2C_mod training runs. Values are means over EASY, MEDIUM, and HARD; Jaccard is the overlap of top-k sets; Spearman is the full-ranking correlation. For macro k = 1, top-1 Spearman is undefined, so the table reports Spearman over the three macro features. Full per-scenario values and top-k Spearman are in Supplementary Table S1. Micro-level values are indicative because 648–651 of the 660 features are tied; the metrics mainly reflect the stable top-8 sales features.

**Caption VI:** Bảng 19. Độ ổn định của tầm quan trọng đặc trưng SHAP qua 5 lần huấn luyện A2C_mod độc lập. Giá trị là trung bình qua EASY, MEDIUM, HARD; Jaccard là độ trùng lặp tập top-k; Spearman là tương quan hạng trên toàn bộ feature. Với macro k = 1, Spearman top-1 không xác định, nên bảng báo cáo Spearman trên cả ba feature macro. Giá trị từng scenario và Spearman top-k nằm trong Bảng S1 của Supplementary. Các giá trị micro là indicative vì 648–651 trong 660 feature bị tie; các chỉ số chủ yếu phản ánh top-8 sales feature ổn định.

| Level | Target | k | Jaccard (all) | Jaccard (excl. 256) | Jaccard (42–512) | Spearman (all) | Spearman (excl. 256) | Spearman (42–512) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Macro | `π` | 1 | 0.600 | 0.500 | 0.000 | 0.800 | 0.750 | 0.500 |
| Macro | `π` | 2 | 1.000 | 1.000 | 1.000 | 0.800 | 0.750 | 0.500 |
| Macro | `logits` | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Macro | `logits` | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Micro | `π` | 10 | 0.819 | 0.909 | 1.000 | 0.804 | 0.934 | 1.000 |
| Micro | `π` | 20 | 0.861 | 0.896 | 1.000 | 0.804 | 0.934 | 1.000 |
| Micro | `π` | 50 | 0.870 | 0.928 | 1.000 | 0.804 | 0.934 | 1.000 |
| Micro | `logits` | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Micro | `logits` | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Micro | `logits` | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 9.2. Table 20 — Mean ± SD |SHAP| (MEDIUM)

**Caption EN:** Table 20. Mean ± SD of mean absolute SHAP values across five A2C_mod seeds in the MEDIUM scenario: (A) macro features; (B) top-10 micro features for logits; (C) top-10 micro features for `π`. CV = SD/mean. Values are averaged over the 50 states and 14 actions. Only the top-8 sales features are distinguishable; the remaining 652 features are tied at near-zero importance. The full 660-feature table is in Supplementary Table S2.

**Caption VI:** Bảng 20. Trung bình ± độ lệch chuẩn của |SHAP| trung bình qua 5 seed A2C_mod trong scenario MEDIUM: (A) feature macro; (B) top-10 feature micro cho `logits`; (C) top-10 feature micro cho `π`. CV = SD/mean. Giá trị được lấy trung bình trên 50 state và 14 action. Chỉ top-8 sales feature là phân biệt được; 652 feature còn lại bị tie ở mức gần 0. Bảng đầy đủ 660 feature nằm trong Bảng S2 của Supplementary.

**(A) Macro (MEDIUM)**

| Target | Feature | Mean | SD | CV |
|---|---|---:|---:|---:|
| `logits` | Inventory | 7.006 | 4.450 | 0.635 |
| `logits` | Demand | 25.092 | 7.574 | 0.302 |
| `logits` | Waste | 0.054 | 0.019 | 0.353 |
| `π` | Inventory | 0.0221 | 0.0091 | 0.412 |
| `π` | Demand | 0.0250 | 0.0267 | 1.067 |
| `π` | Waste | 0.000196 | 0.000204 | 1.041 |

**(B) Micro top-10 `logits` (MEDIUM)**

| Feature | Mean | SD | CV |
|---|---:|---:|---:|
| `sales_sku128` | 0.1452 | 0.0214 | 0.147 |
| `sales_sku66` | 0.1440 | 0.0207 | 0.144 |
| `sales_sku14` | 0.1435 | 0.0353 | 0.246 |
| `sales_sku71` | 0.1343 | 0.0294 | 0.219 |
| `sales_sku162` | 0.1262 | 0.0415 | 0.329 |
| `sales_sku65` | 0.1235 | 0.0420 | 0.340 |
| `sales_sku36` | 0.1148 | 0.0382 | 0.333 |
| `sales_sku4` | 0.1144 | 0.0248 | 0.216 |
| `sales_sku59` | 0.1144 | 0.0248 | 0.216 |
| `waste_sku188` | 0.0369 | 0.0090 | 0.245 |

**(C) Micro top-10 `π` (MEDIUM)**

| Feature | Mean | SD | CV |
|---|---:|---:|---:|
| `sales_sku162` | 0.000185 | 0.000110 | 0.594 |
| `sales_sku66` | 0.000165 | 0.000129 | 0.781 |
| `sales_sku36` | 0.000155 | 0.000156 | 1.006 |
| `sales_sku71` | 0.000150 | 0.000104 | 0.692 |
| `sales_sku14` | 0.000150 | 0.000121 | 0.809 |
| `sales_sku128` | 0.000145 | 0.000132 | 0.909 |
| `sales_sku59` | 0.000109 | 0.000111 | 1.019 |
| `sales_sku4` | 0.000102 | 0.000110 | 1.079 |
| `sales_sku65` | 0.000082 | 0.000055 | 0.672 |
| `sales_sku137` | 0.000038 | 0.000035 | 0.909 |

> Để bài báo gọn, có thể đưa **A + B top-5** vào main paper, phần còn lại vào Supplementary.

### 9.3. Table 21 — Seed-256 collapse and separate analysis

**Caption EN:** Table 21. Seed-256 policy collapse and separate explanation analysis. (A) Collapse diagnostics in EASY, MEDIUM, and HARD, and at checkpoint 76. (B) Stability of seed 256 against the other four seeds in MEDIUM. (C) Top macro features of seed 256 versus the other four seeds. Seed 256 is excluded from the primary stability aggregate.

**Caption VI:** Bảng 21. Collapse chính sách của seed 256 và phân tích giải thích riêng. (A) Chẩn đoán collapse ở EASY, MEDIUM, HARD và tại checkpoint 76. (B) Độ ổn định của seed 256 so với 4 seed còn lại trong MEDIUM. (C) Top feature macro của seed 256 so với 4 seed còn lại. Seed 256 được loại khỏi aggregate ổn định chính.

**(A) Collapse diagnostics (MEDIUM; EASY/HARD tương tự)**

| Seed | Action chiếm ưu thế | Tỷ lệ action 13 | Entropy | Max prob |
|---|---:|---:|---:|---:|
| 42 | action 11 (81.6%), action 0 (18.4%) | 0.0% | 0.0008 | 0.9996 |
| 123 | action 13 (82.9%), action 2 (17.1%) | 82.9% | 0.0107 | 0.9960 |
| **256** | **action 13 (88.7%), action 8 (11.3%)** | **88.7%** | **0.0095** | **0.9944** |
| 512 | action 9 (85.6%), action 12 (14.4%) | 0.0% | 0.0025 | 0.9992 |
| 1024 | action 13 (66.7%), action 5 (33.3%) | 66.7% | 0.0140 | 0.9949 |

> Ở EASY/MEDIUM/HARD, seed 256 chọn action 13 với tỷ lệ 88.5% / 88.7% / 87.7%. Tại checkpoint 76 (MEDIUM): 86.6% action 13, entropy 0.0070.

**(B) Seed 256 vs 4 seed còn lại (MEDIUM)**

| Level | Target | k | Jaccard (256 vs others) | Spearman (256 vs others) |
|---|---|---:|---:|---:|
| Macro | `π` | 1 | 0.750 | 0.875 |
| Macro | `π` | 2 | 1.000 | 0.875 |
| Macro | `logits` | 1 | 1.000 | 1.000 |
| Macro | `logits` | 2 | 1.000 | 1.000 |
| Micro | `π` | 10 | 0.635 | 0.537 |
| Micro | `π` | 20 | 0.780 | 0.537 |
| Micro | `π` | 50 | 0.770 | 0.537 |
| Micro | `logits` | 10 | 1.000 | 1.000 |
| Micro | `logits` | 20 | 1.000 | 1.000 |
| Micro | `logits` | 50 | 1.000 | 1.000 |

**(C) Top macro: seed 256 vs 4 seed còn lại (MEDIUM)**

| Target | Feature | Seed 256 mean | Other 4 mean ± SD | Rank in other 4 |
|---|---|---:|---:|---:|
| `π` | Inventory | 0.0323 | 0.0195 ± 0.0082 | 2 |
| `π` | Demand | 0.0119 | 0.0283 ± 0.0297 | 1 |
| `logits` | Demand | 19.591 | 26.468 ± 7.992 | 1 |
| `logits` | Inventory | 3.529 | 7.876 ± 4.622 | 2 |

### 9.4. Figure 22 — Stability summary

**File:** `output/figs/fig_task1_stability_summary.png`

**Caption EN:** Fig. 22. Stability of SHAP feature importance across five independent A2C_mod training runs. Panels show Jaccard overlap and full-ranking Spearman correlation as a function of top-k for micro (k = 10, 20, 50) and macro (k = 1, 2) explanations. Lines compare `π` across all seeds, `π` excluding seed 256, and `logits`; the dotted line in the macro panels shows the 42–512 pair. Shaded bands indicate the range across EASY, MEDIUM, and HARD. The micro panels are indicative because 648–651 of the 660 features are tied; they mainly reflect the stable top-8 sales features.

**Caption VI:** Hình 22. Độ ổn định của tầm quan trọng đặc trưng SHAP qua 5 lần huấn luyện A2C_mod độc lập. Các panel biểu diễn Jaccard và Spearman toàn bộ feature theo top-k cho micro (k = 10, 20, 50) và macro (k = 1, 2). Các đường so sánh `π` trên tất cả seed, `π` loại seed 256, và `logits`; đường chấm trong panel macro là cặp 42–512. Dải tô thể hiện range qua EASY, MEDIUM, HARD. Các panel micro là indicative vì 648–651 trong 660 feature bị tie; chúng chủ yếu phản ánh top-8 sales feature ổn định.

### 9.5. Figure 23 — Seed-256 collapse

**File:** `output/figs/fig_task1_seed256_collapse.png`

**Caption EN:** Fig. 23. Policy saturation across the five A2C_mod seeds. (a) Share of product-level decisions selecting action 13, averaged over the three scenarios; (b) mean policy entropy. Seed 256 is highlighted. Seeds 42 and 512 have zero action-13 share but are also near-deterministic on other actions, as shown by their low entropy.

**Caption VI:** Hình 23. Mức độ bão hòa chính sách của 5 seed A2C_mod. (a) Tỷ lệ quyết định ở mức sản phẩm chọn action 13, trung bình qua ba scenario; (b) entropy chính sách trung bình. Seed 256 được làm nổi bật. Seed 42 và 512 có action-13 = 0 nhưng cũng gần deterministic ở action khác, thể hiện qua entropy thấp.

### 9.6. Supplementary Figure S1 — Micro Jaccard heatmaps

**File:** `output/figs/fig_task1_micro_jaccard_k20_heatmaps.png`

**Caption EN:** Fig. S1. Pairwise Jaccard overlap of top-20 micro features across the five A2C_mod seeds for `π` (top row) and `logits` (bottom row) in EASY, MEDIUM, and HARD. Seed 256 shows lower overlap for `π`; the `logits` overlap is 1.00 for every pair. The heatmaps mainly reflect the stable top-8 sales features; 648–651 features are tied.

**Caption VI:** Hình S1. Jaccard top-20 feature micro giữa 5 seed A2C_mod cho `π` (hàng trên) và `logits` (hàng dưới) trong EASY, MEDIUM, HARD. Seed 256 có mức trùng thấp hơn với `π`; với `logits`, mọi cặp đều trùng 1,00. Các heatmap chủ yếu phản ánh top-8 sales feature ổn định; 648–651 feature bị tie.

---

## 10. Hướng dẫn chèn vào Word (bản mới `(1).docx`)

1. Mở **`Xai_Inventory_Revision_27Sep (1).docx`**.
2. Tìm **Section 3.13 “Reliability and Stability of the Explanations”** (ngay trước **Conclusion**). Mục này đã có đoạn dẫn + **Table 18 (scorecard)**.
3. **Không tạo lại heading 3.13.** Đặt con trỏ **sau Table 18**, trước “Conclusion”, rồi tạo hai heading con:
   - **3.13.1 Stability of SHAP Feature Importance Across Independent Training Runs**
   - **3.13.2 Seed-256 Policy Collapse and Separate Explanation Analysis**
4. Dán phần **English manuscript version** ở Mục 8 vào hai mục con tương ứng.
5. Chèn **Table 19** sau đoạn “Experimental design” (hoặc sau “Macro level”), nơi có callout “(Table 19)”.
6. Chèn **Table 20** sau đoạn “Mean ± SD of |SHAP|”.
7. Chèn **Table 21** trong mục 3.13.2, sau đoạn mở đầu về seed 256.
8. Chèn **Figure 22** (`output/figs/fig_task1_stability_summary.png`) sau đoạn “Micro level” hoặc ở cuối 3.13.1; chèn **Figure 23** (`output/figs/fig_task1_seed256_collapse.png`) trong 3.13.2.
9. Chèn **Figure S1** (`output/figs/fig_task1_micro_jaccard_k20_heatmaps.png`) trong Supplementary Material; có thể thêm `micro_top20_logits_MEDIUM.png`, `micro_top20_pi_MEDIUM.png`, `macro_mean_sd.png`, `action_distribution_MEDIUM.png`.
10. Cập nhật cross-reference (chi tiết ở Mục 6.3–6.4):
    - Thêm hàng **Reproducibility (independent runs)** vào Table 18, trỏ Section 3.13.1.
    - Section 3.6: thêm câu trỏ 3.13.1.
    - Section 3.9: giữ “Section 3.3.4” (hợp lệ) hoặc đổi thành “noted in Section 3.3.4 and analyzed in Section 3.13.2”.
    - Section 3.10: thêm câu phân biệt stability của checkpoint được chọn và multi-seed stability.
    - Section 5: cập nhật limitation đầu tiên như Mục 6.4.
    - Sửa số cũ còn sót: “Sections 3.3 - 4.5 and 3.7.3” → “Sections 3.3 - 3.5 and 3.7.3”.
11. Kiểm tra lại numbering của Table/Figure; nếu Word tự đánh số khác, để Word renumber và chỉnh callout tương ứng.
12. Thêm một câu trong Supplementary Material: “Full per-scenario Jaccard/Spearman tables, the 660-feature mean ± SD table, pairwise heatmaps, and seed-256-specific tables are provided in the reproducibility package.”

### 10.1. Hình ảnh cần thiết (tóm tắt)

| Mục đích | File | Dùng ở |
|---|---|---|
| **Bắt buộc** — stability summary | `output/figs/fig_task1_stability_summary.png` | Figure 22, cuối 3.13.1 |
| **Bắt buộc** — seed-256 collapse | `output/figs/fig_task1_seed256_collapse.png` | Figure 23, trong 3.13.2 |
| Khuyến nghị — heatmap micro | `output/figs/fig_task1_micro_jaccard_k20_heatmaps.png` | Figure S1, Supplementary |
| Tùy chọn — top features | `output/figs/micro_top20_logits_MEDIUM.png`, `output/figs/micro_top20_pi_MEDIUM.png` | Supplementary |
| Tùy chọn — macro mean/SD | `output/figs/macro_mean_sd.png` | Supplementary |
| Tùy chọn — action distribution | `output/figs/action_distribution_MEDIUM.png` | Supplementary |

### 10.2. Bảng cần thiết (tóm tắt)

| Bảng | Nội dung | Dữ liệu nguồn |
|---|---|---|
| **Table 19** | Stability of SHAP top-k across 5 seeds | `stability_summary.csv` (đã tóm tắt ở Mục 9.1) |
| **Table 20** | Mean ± SD \|SHAP\| (MEDIUM) | `mean_sd_macro.csv`, `mean_sd_micro.csv` (Mục 9.2) |
| **Table 21** | Seed-256 collapse + separate analysis | `collapse_diagnostics.csv`, `seed256_pairwise_stability.csv`, `seed256_top_macro.csv` (Mục 9.3) |

---

## 11. Checklist và bản đồ file

### 11.1. Checklist yêu cầu Task 1

- [x] 5 seed: 42, 123, 256, 512, 1024.
- [x] Macro SHAP: 3 feature, cả 3 scenario, cả 2 target.
- [x] Micro SHAP: 660 feature, top-k k = 10/20/50 (nhưng 648–651 feature bị tie; chỉ top-8 sales phân biệt được).
- [x] Jaccard top-k giữa các seed.
- [x] Spearman (all + top-k) giữa các seed.
- [x] Mean ± SD |SHAP| theo feature (macro + micro).
- [x] Seed 256 collapse: chẩn đoán, tách aggregate, phân tích riêng.
- [x] Báo cáo có kết luận rõ ràng: ổn định có điều kiện.
- [x] Chỉ ra vị trí chèn trong bài báo: Section 3.13.1–3.13.2 (sau Table 18).
- [x] Hình/bảng chọn lọc, có caption song ngữ.

### 11.2. Bản đồ file

| File | Nội dung |
|---|---|
| `outputTask1.md` | Tài liệu này |
| `output/Task1_Report.md` | Báo cáo gốc chi tiết |
| `task1_shap_seed_stability.ipynb` | Notebook thực nghiệm |
| `planTask1.md` | Plan Task 1 |
| `output/raw/*.npz` | 60 raw SHAP arrays |
| `output/stability_pairwise.csv` | Pairwise Jaccard/Spearman |
| `output/stability_summary.csv` | Aggregate stability |
| `output/mean_sd_macro.csv` | Mean ± SD macro |
| `output/mean_sd_micro.csv` | Mean ± SD 660 micro |
| `output/seed256_pairwise_stability.csv` | Seed 256 vs others |
| `output/seed256_top_macro.csv` | Seed 256 top macro |
| `output/seed256_top_micro.csv` | Seed 256 top micro |
| `output/collapse_diagnostics.csv` | Action/entropy diagnostics |
| `output/extra_checkpoint_diagnostics.csv` | ckpt-76/ckpt-64 checks |
| `output/cross_target_stability.csv` | `π` vs `logits` |
| `output/figs/fig_task1_stability_summary.png` | Figure 22 |
| `output/figs/fig_task1_seed256_collapse.png` | Figure 23 |
| `output/figs/fig_task1_micro_jaccard_k20_heatmaps.png` | Figure S1 |
| `output/figs/micro_top20_logits_MEDIUM.png` | Supplementary top features |
| `output/figs/micro_top20_pi_MEDIUM.png` | Supplementary top features |
| `output/figs/macro_mean_sd.png` | Supplementary macro |
| `output/figs/action_distribution_MEDIUM.png` | Supplementary action distribution |

---

## 12. Kết luận cuối cùng

Task 1 đã được xử lý đầy đủ về mặt thực nghiệm:

- Có đủ 5 seed, macro + micro, 3 scenario, 2 target.
- Có Jaccard/Spearman top-k, mean ± SD |SHAP|, và phân tích riêng seed 256.
- Kết luận khoa học là **ổn định có điều kiện**: top-8 sales feature ổn định (đặc biệt `logits`); 651 feature còn lại bị tie nên không thể kết luận per-feature; macro `logits` ổn định; macro `π` top-1 không ổn định nhưng top-2 ổn định; seed 256 là outlier đã collapse và được báo cáo riêng.

Vị trí trình bày trong bài báo: **Section 3.13 “Reliability and Stability of the Explanations”** trong bản mới `Xai_Inventory_Revision_27Sep (1).docx`. Mục này đã có đoạn dẫn + **Table 18 (scorecard)**, nên Task 1 được chèn thành **3.13.1** và **3.13.2** ngay **sau Table 18**, trước Conclusion. Đây là vị trí trực tiếp giải quyết khoảng trống multi-seed mà phần **Study Limitations** đã nêu, đồng thời bổ sung đúng tiêu chí “reproducibility across independently trained agents” mà scorecard đang thiếu.

# Plan Task 14-9: SHAP Implementation (Background, Validity, Explainer) - Chi tiết 4 Tasks

> **Workstream:** SHAP implementation  
> **Liên quan Reviewer:** R1 #6 (KernelSHAP background 200→100), R4 (KernelSHAP vs Partition inconsistency)  
> **Nguồn dữ liệu đã có:** `XAI/SHAP-temp.ipynb:463` `generate_background_data(200)` + `shap.sample(...,100)`, `Ablation_Study/ablation_SHAP.ipynb:428`, `Ablation_Study/faithfulness/topk_shap_analysis.ipynb:450` `generate_background_660(100)`, `data/train.tfrecords:1000x220` (`prepare_data.py:157`), `docs/responses/result2.md:107-118`  
> **Nguyên tắc chung:** Không cần retrain. Chỉ viết lại + thí nghiệm nhỏ trên SHAP background. Kết quả ghi tiếng Việt, sẵn sàng copy-paste vào bài báo sau khi approve. Không chèn vào `Review.md`.

---

## Ngữ cảnh chung & Mục tiêu

Bài báo hiện tại trong `Xai_Inventory_Submit_17Mar.md:591` (Section 3.3.3) và `Section 4.5` dùng SHAP với background 200→100 cho KernelSHAP (3 features) và 100 cho PartitionExplainer (660 features), nhưng **cách tạo background, cách chọn 100 representative, và sự khác biệt giữa 2 Explainer chưa được mô tả rõ**, reviewer R1 #6 và R4 yêu cầu làm rõ và thêm sensitivity.

4 tasks 39-42 giải quyết theo lộ trình: **Task 39** mô tả cách tạo 200 và chọn 100, **Task 40** so sánh synthetic vs trajectory và sensitivity 50/100/200, **Task 41** kiểm tra perturbed states có thỏa constraints không, **Task 42** giải thích rõ KernelSHAP vs Partition.

**Cách thực thi:** Task 39 và 42 là Writing (không cần experiment), Task 40 và 41 là Experiment (cần chạy SHAP với background khác nhau và validity check), nhưng tất cả đều không cần retrain.

---

## Task 39: Mô tả cách tạo 200 KernelSHAP background states và cách chọn còn 100 representative states

### 1. Yêu cầu trong `Task 14-9.md:3`

> **Task:** Mô tả cách tạo 200 KernelSHAP background states và cách chọn còn 100 representative states.  
> **Type:** Clarification  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Background construction procedure

### 2. Hiện trạng

*   **Cách tạo 200 background:** `XAI/SHAP-temp.ipynb:463` `generate_background_data(num_samples=200)` với `inventory ~U(0,1)`, `sales ~U(0,1)`, `waste = 0.025*inventory + N(0,0.005)` clip `[0,0.1]`, `np.random.seed(42)` trước khi tạo `XAI/SHAP-temp.ipynb:483`.
*   **Cách chọn 100:** `XAI/SHAP-temp.ipynb:554` `sampled_background = shap.sample(background_data.numpy(), 100)` - random uniform subsampling không replacement, seeded `42` trước khi gọi `XAI/SHAP-temp.ipynb:554`.
*   **Chưa mô tả trong paper:** `Xai_Inventory_Submit_17Mar.md:591` Section 3.3.3 chỉ nói `200 generated, subsampled to 100 representative` mà chưa nói công thức và `shap.sample`.

### 3. Hướng giải quyết chi tiết (Writing)

**Bước 39.1 - Thêm subsection vào `Xai_Inventory_Submit_17Mar.md:591` Section 3.3.3:**

> "Background distribution for KernelSHAP (macro-level, 3 aggregated features) was constructed by generating 200 synthetic states: inventory and sales independently sampled from Uniform(0,1), waste sampled as 0.025*inventory + Normal(0,0.005) clipped to [0,0.1] to preserve inventory-waste correlation. With seed 42, `shap.sample(background, 100)` randomly subsampled 100 representative states from 200 to balance expectation convergence vs cost (see `XAI/SHAP-temp.ipynb:463` and `docs/responses/result2.md:107`). Masking via marginalization integrates out missing features by sampling from the 100 background, not zero-masking."

**Bước 39.2 - Bảng tham số:**

| Tham số | Giá trị | Lý do |
| :--- | :--- | :--- |
| `num_samples` background | 200 → 100 | 200 đủ đa dạng, 100 cân bằng cost (KernelSHAP ~5-10 phút/config) |
| `seed` | 42 | Đảm bảo reproducible |
| `sampling` | `shap.sample` uniform | Không clustering, đơn giản |

**Tại sao chọn hướng này?**
*   Must scope, chỉ Writing, không cần chạy code, đủ để reviewer hiểu background không phải chọn bừa.

**Deliverable Task 39:**
*   Đoạn văn tiếng Việt + bảng tham số trong `task14-9/outputTask14-9.md` phần Task 39
*   Không cần CSV/script, chỉ Writing

---

## Task 40: So sánh synthetic background với training-trajectory states; sensitivity theo background size/sampling strategy

### 1. Yêu cầu trong `Task 14-9.md:10`

> **Task:** So sánh synthetic background với training-trajectory states; sensitivity theo background size/sampling strategy.  
> **Type:** Experiment  
> **Requires Retrain:** No  
> **Recommend Scope:** Strong  
> **Deliverable:** Background sensitivity results

### 2. Hiện trạng

*   **Synthetic background:** 100 states tự tạo như Task 39 `XAI/SHAP-temp.ipynb:463` - Uniform, không giống phân phối thực `MeanDemand SKU57 22.6` vs synthetic 0.5.
*   **Trajectory states:** 1000 states thực từ `data/train.tfrecords:1000x220` (`prepare_data.py:157` loop `range(0,1000)`), 220 SKU từ top 20% frequent `prepare_data.py:105`.
*   **Chưa có sensitivity:** Chưa chạy `background size 50/100/200` hay `sampling random vs k-means`.

### 3. Hướng giải quyết chi tiết (Gộp 50/100/200 + synthetic vs trajectory - Phương án B mạnh nhất như bạn yêu cầu)

**Phương án đã chọn theo yêu cầu của bạn:** Thêm sensitivity `50/100/200` (3 mức) **rồi** so sánh synthetic vs trajectory - gộp cả 2 vế Task 40.

**Bước 40.1 - Tạo 2 loại background 100:**
*   **Synthetic 100:** Như Task 39 `generate_background_data(100)` (đã có).
*   **Trajectory 100:** Sample 100 states thực từ `data/train.tfrecords:1000` bằng `np.random.choice(1000, 100, replace=False)` với `seed 42`, mỗi state là `sales/capacity` 220 + `capacity` như `topk_shap_analysis.ipynb:373`.

**Bước 40.2 - Sensitivity 50/100/200 cho mỗi loại:**
*   Chạy SHAP trên 10 states mẫu (như Task 11) với 6 configs: `synthetic 50/100/200` + `trajectory 50/100/200` (mỗi config ~5-10 phút với KernelSHAP 3 features, tổng 6 configs ~30-60 phút, nhanh hơn Task 13 660-dim).
*   Tính `Mean|SHAP|` Top-5 Jaccard giữa các sizes và giữa synthetic vs trajectory.

**Bước 40.3 - Đánh giá:**
*   Nếu Top-5 vẫn `SKU64,163...` dù đổi synthetic/trajectory hay 50/100/200 → nền robust.
*   Nếu khác → báo cáo và chọn trajectory.

**Tại sao chọn hướng này?**
*   Gộp cả 2 vế Task 40, trả lời cùng lúc "synthetic có giống thực không?" và "cần bao nhiêu nền thì đủ?" - như Task 11 đã làm với `k=10/20/50`.
*   Thời gian 30-60 phút cho Strong scope, chấp nhận được.

**Deliverable Task 40:**
*   Đoạn văn tiếng Việt + bảng Jaccard giữa synthetic vs trajectory và giữa 50/100/200 trong `outputTask14-9.md` phần Task 40
*   File `task14-9/output/sensitivity_background_50_100_200.csv` (Jaccard) + `synthetic_vs_trajectory.csv`
*   Script `task14-9/scripts/background_sensitivity.py` (tạo 2 loại background, chạy 6 configs)

---

## Task 41: Kiểm tra perturbed states vẫn thỏa inventory/capacity constraints

### 1. Yêu cầu trong `Task 14-9.md:19`

> **Task:** Kiểm tra perturbed states vẫn thỏa inventory/capacity constraints.  
> **Type:** Experiment/Check  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Validity-rate/constraint check

### 2. Hiện trạng

*   **Chưa có check:** `grep capacity` chỉ ở `topk_shap_analysis.ipynb:363` load, không validate perturbed. `XAI/SHAP-temp.ipynb:572` marginal sampling rút từ `sampled_background` 100, có `clip(0,0.1)` cho waste nhưng không check `x+u≤1` hay `waste=0.025*x` joint.
*   **Training constraints:** `Training/Train_DQN.ipynb:552` `overstock = max(0,(x+u)-1)`, `prepare_data.py:144` capacity - không áp dụng cho SHAP.
*   **Hiện tại:** `docs/responses/result2.md:117` đã cấy correlation `waste=0.025*inventory` khi tạo background, nên perturbed `≈100%` hợp lệ do clip, nhưng chưa báo cáo.

### 3. Hướng giải quyết chi tiết (Giữ Phương án A và làm thêm Enforce - theo yêu cầu của bạn, 2 kết quả trong cùng file)

**Theo yêu cầu của bạn: Giữ Phương án A và làm thêm enforce hard constraint (loại bỏ perturbed không hợp lệ) để chặt chẽ hơn.**

**Bước 41.1 - Tính validity-rate và enforce trong cùng `validity_check.csv` (thêm cột `enforced_removed`):**
*   Với 100 background x 200 perturbed (mỗi lần SHAP che 1 feature, thay bằng giá trị từ background), tính:
    ```
    validity_rate = #perturbed ∈ [0,1] ∩ waste∈[0,0.1] / #perturbed  (báo cáo)
    enforced_removed = #perturbed bị loại bỏ khi enforce x>1 hoặc waste>0.1
    valid_after_enforce = #perturbed còn lại sau enforce
    ```
*   Hiện `clip(0,0.1)` đã đảm bảo `≈100%` hợp lệ, nhưng vẫn enforce để chặt chẽ: loại bỏ perturbed `x>1` hoặc `waste>0.1` trước khi tính SHAP, báo cáo số lượng bị loại (ví dụ 2/200).

**Bước 41.2 - Báo cáo trong cùng file:**
*   Trong `outputTask14-9.md` phần Task 41: `validity-rate = 99.2% (198/200), enforced_removed = 2, valid_after_enforce = 198` - đủ cho Must và thêm chặt chẽ.
*   File `task14-9/output/validity_check.csv` sẽ có 5 cột: `Scenario, validity_rate, enforced_removed, valid_after_enforce, note` cho 3 scenarios EASY/MEDIUM/HARD (giống bạn yêu cầu Task 41 có 2 kết quả trong cùng file).

**Tại sao chọn A + Enforce?**
*   Vừa báo cáo (A) vừa enforce (B) trong cùng file, reviewer thấy bạn đã kiểm tra và đã chặt chẽ loại bỏ, không hỏi lại. Thời gian 30 phút + 10 phút enforce.

**Deliverable Task 41:**
*   Đoạn văn tiếng Việt + `validity-rate` và `enforced_removed` trong `outputTask14-9.md` phần Task 41
*   File `task14-9/output/validity_check.csv` (5 cột: Scenario, validity_rate, enforced_removed, valid_after_enforce, note) - 1 file duy nhất
*   Script `task14-9/scripts/validity_check.py` (vừa tính validity-rate vừa enforce)

---

## Task 42: Giải thích rõ KernelSHAP dùng ở macro-level và Partition Explainer dùng ở micro-level; nêu config từng cái

### 1. Yêu cầu trong `Task 14-9.md:27`

> **Task:** Giải thích rõ KernelSHAP dùng ở macro-level và Partition Explainer dùng ở micro-level; nêu config từng cái.  
> **Type:** Writing  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Consistent SHAP methods subsection

### 2. Hiện trạng

*   **Section 3.3.3 mô tả KernelSHAP:** `XAI/SHAP-temp.ipynb:572` `shap.KernelExplainer(...,100)` với `200→100` cho 3 features, nhưng micro-level lại dùng Partition `topk_shap_analysis.ipynb:629` `shap.maskers.Partition + PartitionExplainer` cho 660 features - reviewer thấy lẫn lộn `Review.md:128`.

### 3. Hướng giải quyết chi tiết (Writing)

**Bước 42.1 - Thêm subsection vào `Xai_Inventory_Submit_17Mar.md:591` Section 3.3.3:**

> Tạo bảng so sánh:

|  | KernelSHAP (macro, 3 features) | PartitionExplainer (micro, 660 features) |
| :--- | :--- | :--- |
| Background | 200→100 `shap.sample` | 100 trực tiếp `generate_background_660(100)` |
| Input dim | 3 aggregated (inventory, sales, waste) | 660 raw (220×3) |
| Test states | 200 (background itself) | 50/scenario |
| Target | DQN `softmax(Q)` / A2C `π` | DQN `q_mean` / A2C `probs_mean` |
| Config | `shap.KernelExplainer(predict_fn, sampled_background)` `nsamples≈2048` | `shap.maskers.Partition(background_660) + PartitionExplainer` hierarchical clustering, `max_samples=100` |
| Lý do | Model-agnostic, chính xác ở low dim | Nhanh ~10x ở high dim, dùng clustering tự nhiên 3 nhóm |

**Bước 42.2 - Note associative + tied-value `0.000799` limitation như `topk:26`.**

**Tại sao chọn hướng này?**
*   Must scope, chỉ Writing, 30 phút, đủ để reviewer thấy bạn không nhầm lẫn.

**Deliverable Task 42:**
*   Đoạn văn tiếng Việt + bảng so sánh trong `outputTask14-9.md` phần Task 42
*   Không cần CSV/script, chỉ Writing

---

## Kế hoạch thực thi gộp & Tách kết quả

**Thực thi gộp Task 40 + 41:** Cùng background 100, cùng 10 states mẫu, chạy 6 configs synthetic 50/100/200 + 6 configs trajectory 50/100/200 = 12 configs cho Task 40 (~1.5-2.5 tiếng), và validity check 100x200 perturbed cho Task 41 (~30 phút) trong cùng script `background_sensitivity.py`.

**Nhưng kết quả tách rõ 4 phần** trong file output để bạn review:

```
## Kết quả Task 39: Background 200→100 (Writing)
... mô tả công thức + bảng tham số ...

## Kết quả Task 40: Synthetic vs Trajectory + 50/100/200 (Experiment)
... bảng Jaccard synthetic vs trajectory và giữa 50/100/200 ...

## Kết quả Task 41: Validity-rate (Check)
... validity-rate 99% ...

## Kết quả Task 42: KernelSHAP vs Partition (Writing)
... bảng so sánh macro vs micro ...
```

Task 39 và 42 là Writing độc lập, có thể làm song song với 40+41.

---

## File sẽ tạo (khi build)

1. `Feedback 7-9/task14-9/planTask14-9.md` (file này)
2. `Feedback 7-9/task14-9/outputTask14-9.md` (4 phần Task 39/40/41/42 tiếng Việt, tách rõ)
3. `Feedback 7-9/task14-9/output/sensitivity_background_50_100_200.csv` (Jaccard giữa 50/100/200, Task 40)
4. `Feedback 7-9/task14-9/output/synthetic_vs_trajectory.csv` (Jaccard synthetic vs trajectory, Task 40)
5. `Feedback 7-9/task14-9/output/validity_check.csv` (validity-rate, Task 41)
6. `Feedback 7-9/task14-9/scripts/background_sensitivity.py` (tạo 2 loại background, chạy 12 configs)
7. `Feedback 7-9/task14-9/scripts/validity_check.py` (tính validity-rate)

---

## Thứ tự thực hiện

1. Viết `planTask14-9.md` (file này) - đã xong
2. Viết `outputTask14-9.md` 4 phần tiếng Việt (Task 39/42 Writing trước, 30 phút)
3. Chạy `background_sensitivity.py` cho Task 40 (12 configs, 1.5-2.5 tiếng) + `validity_check.py` cho Task 41 (30 phút)
4. Khi bạn approve bản Việt, dịch sang Anh và chèn vào `Xai_Inventory_Submit_17Mar.md:591` (Section 3.3.3) cho Task 39/42, và Supplementary cho Task 40/41

# Plan Task 11-9: SHAP Comparability (Common Target & Sensitivity) - Chi tiết 3 Tasks

> **Workstream:** SHAP comparability  
> **Liên quan Reviewer:** R1 #4 (A2C_mod definition & Q-equivalence), R1 #5 (SHAP target Q vs π, unfair magnitude/stability comparison)  
> **Nguồn dữ liệu đã có:** `Training/Train_DQN.ipynb:132-153` (DQN `q_values` linear 14), `Training/A2C-mod.ipynb:122-162` (Actor `layer4 32→14` logits + softmax `π`, Critic `V(s)`), `XAI/SHAP-temp.ipynb:297-308` (wrapper `softmax(Q)` vs `π` hiện tại), `Ablation_Study/faithfulness/topk_shap_analysis.ipynb:541-584` (wrapper 660-dim `reduce_mean`), `Ablation_Study/output/topk_shap_full_results_660.csv` (3960 dòng baseline), `Ablation_Study/ablation_SHAP.ipynb:478-513`  
> **Nguyên tắc chung:** Ưu tiên **không retrain** (Task 13A logits). Chỉ phân tích lại + viết lại. Kết quả ghi tiếng Việt, sẵn sàng copy-paste vào bài báo sau khi approve. Không chèn vào `Review.md`.

---

## Ngữ cảnh chung & Mục tiêu

Bài báo hiện tại trong `Xai_Inventory_Submit_17Mar.md:1185-1255` (Section 3.3.3 và 4.5.2) giải thích SHAP cho DQN trên **Q(s,a*)** (thậm chí `softmax(Q)` trong `XAI/SHAP-temp.ipynb:303`) và cho A2C_mod trên **π(a*|s)** (softmax policy `Training/A2C-mod.ipynb:145`). R1 #5指出 đây là hai đại lượng khác scale: Q unbounded còn π constrained [0,1] qua softmax và phụ thuộc tất cả actions qua normalization, nên so sánh trực tiếp magnitude/stability giữa DQN và A2C_mod là **không công bằng** - khác biệt có thể do target chứ không phải kiến trúc học.

3 tasks 13-15 giải quyết theo lộ trình: **Task 13** thử common target nếu khả thi (13A logits không retrain), **Task 14** tuyên bố rõ nếu không đổi target thì comparison chỉ qualitative, **Task 15** thêm sensitivity trên outputs khác của A2C_mod để kiểm tra robustness.

**Cách thực thi:** Task 13A và Task 15 gộp thực thi (cùng wrapper logits, cùng 10 states mẫu, cùng background 100) cho nhanh, nhưng **kết quả tách rõ 3 phần** để bạn review từng task. Task 14 là Writing độc lập, không phụ thuộc experiment.

---

## Task 13: Dùng common explanation target cho DQN và A2C_mod (advantage, pre-softmax logits hoặc normalized value difference) nếu khả thi

### 1. Yêu cầu trong `Task 11-9.md:3`

> **Task:** Dùng common explanation target cho DQN và A2C_mod (advantage, pre-softmax logits hoặc normalized value difference) nếu khả thi.  
> **Type:** Implementation/Experiment  
> **Requires Retrain:** Maybe  
> **Recommend Scope:** Strong  
> **Deliverable:** SHAP trên target tương đương giữa 2 model

### 2. Hiện trạng

*   **Wrapper hiện tại (không tương đương):**
    *   DQN: `dqn_predict(x) = softmax(q_values)` `XAI/SHAP-temp.ipynb:303-308` (q_values linear 14 `Training/Train_DQN.ipynb:146`), và `dqn_predict_660(X) = reduce_mean(q_network(X))` `topk_shap_analysis.ipynb:541-550` (Q-values).
    *   A2C_mod: `a2c_mod_predict(x) = π = softmax(layer4)` `XAI/SHAP-temp.ipynb:297-301` (`layer4 32→14` `Training/A2C-mod.ipynb:128` + softmax `145`), và `a2c_predict_660(X) = reduce_mean(actor(per-product))` `topk:570-584`.
    *   Đã có nỗ lực normalize DQN qua softmax để mimic π, nhưng vẫn không tương đương về mặt toán học (Q unbounded vs π constrained).
*   **Logits có sẵn nhưng chưa dùng cho SHAP:** `A2C-mod.ipynb:381` `log π` trong `categorical`, `451` `log p_batch` để tính `p_new`, DQN `q_values` chính là logits. Chưa có SHAP trên logits.
*   **Advantage cần retrain:** DQN không có `V(s)` riêng, muốn `A(s,a)=Q(s,a)-V(s)` phải thêm head hoặc dùng `max Q` làm baseline - cần retrain hoặc ước lượng post-hoc, effort cao.

### 3. Hướng giải quyết chi tiết (Chọn 13A không retrain)

**Phương án 13A (Không retrain, khuyến nghị):** Dùng **pre-softmax logits** làm common target - cả hai model đều có logits linear 14 trước softmax, cùng scale unbounded, cùng là score trước khi normalize.

**Bước 13.1 - Tạo wrapper logits mới (không retrain):**
```python
# DQN logits: q_values linear (đã là logits)
def dqn_logits_660(X):
    q_values = q_network(X, training=False)  # [B,220,14] linear
    return tf.reduce_mean(q_values, axis=1).numpy()  # [B,14] logits

# A2C_mod logits: layer4 output trước softmax
# Cần lấy layer4.weight: actor.layer4(X) trước softmax
# Tạo actor_logits model: input [B*P,3] -> layer1-3 -> layer4 (linear) -> [B,P,14] -> mean -> [B,14]
def a2c_logits_660(X):
    B = X.shape[0]
    s3d = tf.transpose(tf.reshape(X, [B,3,220]), [0,2,1])  # [B,220,3]
    s_pp = tf.reshape(s3d, [B*220,3])
    # forward qua layer1-3 + layer4 linear (không softmax)
    logits_pp = actor_logits_forward(s_pp)  # [B*220,14]
    logits_3d = tf.reshape(logits_pp, [B,220,14])
    return tf.reduce_mean(logits_3d, axis=1).numpy()  # [B,14]
```

**Bước 13.2 - Chạy SHAP trên logits với sensitivity n_states=10/20/50 (Phương án B, ~2 tiếng cho 660 chiều, để tránh reviewer hỏi "lỡ 50 mẫu khác thì sao"):**
*   Background: `background_660` 100 mẫu `topk_shap_analysis.ipynb:450-457` (`inv/sales ~U(0,1), waste=0.025*inv+N(0,0.005)`).
*   Test states: **3 mức sensitivity 10, 20, 50 mẫu** từ `data/test.tfrecords` (thay vì chỉ 10 để chứng minh 10-state result đại diện). Mỗi mức chạy riêng, tính Jaccard Top-20 giữa các mức.
*   Explainer: `shap.KernelExplainer(logits_predict, sampled_background)` như `XAI/SHAP-temp.ipynb:572`, `nsamples=2000` cho 660 chiều.
*   Tính `Mean|SHAP|` và Top-20 cho cả DQN logits và A2C_mod logits ở mỗi mức n_states, so sánh với baseline `softmax(Q)` vs `π` và giữa các n_states với nhau.

**Bước 13.3 - Đánh giá feasibility & sensitivity:**
*   Nếu Top-20 logits vẫn là Sales SKU64,163... (giống baseline) → kết luận robust, common target khả thi và không đổi kết luận chính.
*   Nếu Top-20 đổi nhiều (ví dụ Waste lên Top) → báo cáo trong paper: common target cho kết quả khác, nên giữ disclaimer Task 14.

**Tại sao chọn hướng này?**
*   **Không retrain:** Chỉ cần đọc checkpoint đã có `output Training/checkpointDQN` (ckpt-60) và `output Training/outputA2Cmod/checkpoints_a2cmod` (ckpt-64), lấy weight `layer4`/`q_values` đã lưu, không tốn ngày train lại như Task 13 advantage.
*   **Cùng scale:** Logits cả hai đều unbounded linear score, so magnitude/stability công bằng hơn Q vs π.
*   **Sensitivity 10/20/50:** Tránh reviewer hỏi "lỡ 50 mẫu khác thì sao?" như bạn lo lắng. Nếu Jaccard giữa n=10 vs n=50 >0.83 như Task 11 đã thấy với k=10/20/50, chứng minh 10-state result đại diện. Thời gian tăng từ 15 phút (10 mẫu) lên ~2 tiếng (10+20+50) với Partition, ~4-5 tiếng với KernelSHAP, vẫn chấp nhận được cho Strong scope.

**Phương án 13B (Có retrain, không khuyến nghị):** Nếu checkpoint không lưu được logits (ví dụ A2C_mod chỉ lưu π), phải retrain thêm head Advantage - tốn 600ep x 900 steps, không cần thiết cho Strong scope. Sẽ ghi trong plan là "if logits not extractable, fallback to Task 14".

**Deliverable Task 13:**
*   Đoạn văn tiếng Việt + bảng so sánh Top-5 logits vs baseline + bảng sensitivity Jaccard giữa n_states=10/20/50 (trong `task11-9/outputTask11-9.md` phần Task 13)
*   File `task11-9/outputTask11_common_target.csv` (Top-20 logits cho DQN và A2C_mod, 10/20/50 states)
*   File `task11-9/outputTask11_n_states_sensitivity.csv` (Jaccard/Spearman giữa n=10 vs 20 vs 50)
*   Script `task11-9/scripts/common_target_test.ipynb` (wrapper logits, chạy 10/20/50 states, ~2-5 tiếng)

---

## Task 14: Nếu không đổi common target, tuyên bố rõ SHAP comparison chỉ qualitative, không so trực tiếp magnitude/stability

### 1. Yêu cầu trong `Task 11-9.md:11`

> **Task:** Nếu không đổi common target, tuyên bố rõ SHAP comparison chỉ qualitative, không so trực tiếp magnitude/stability.  
> **Type:** Writing  
> **Requires Retrain:** No  
> **Recommend Scope:** Must  
> **Deliverable:** Moderated interpretation

### 2. Hiện trạng

*   Bài báo hiện tại `Xai_Inventory_Submit_17Mar.md:1185` (Section 3.3.3) và `1241` (Table 7b) so sánh trực tiếp magnitude `Mean|SHAP|` giữa DQN (0.00281) và A2C_mod (0.00044) và stability (DQN 1.00 vs A2C_mod 0.29) mà chưa tuyên bố đây là hai target khác scale.
*   R1 #5 yêu cầu: hoặc dùng common target (Task 13), hoặc tuyên bố rõ đây là qualitative.

### 3. Hướng giải quyết chi tiết (Writing, không cần experiment)

**Bước 14.1 - Chèn disclaimer vào 2 vị trí trong bài báo (tiếng Việt trước, sau dịch Anh):**
*   **Vị trí 1 - Section 3.3.3 SHAP Implementation Details (`Xai_Inventory_Submit_17Mar.md:1185`):** Thêm đoạn sau Table 7a Scope Matrix:
    > "Lưu ý: DQN được giải thích trên softmax(Q) và A2C_mod trên π(a*|s) - hai đại lượng khác scale (Q unbounded vs π constrained [0,1] và phụ thuộc tất cả actions qua softmax). Do đó, so sánh trực tiếp magnitude và stability giữa hai agents trong Table 7b/7c là mô tả định tính, không phải định lượng. Phân tích sensitivity trên common target logits được báo cáo trong Supplementary (Task 13) để kiểm tra robustness."
*   **Vị trí 2 - Section 4.5.2 và 4.5.4 (`1241`, `1256`):** Thêm footnote dưới Table 7b và 7c: `* Comparison between DQN and A2C_mod is qualitative due to different explanation targets (softmax Q vs π); see Task 13 common-target analysis in Supplementary.`

**Bước 14.2 - Moderated interpretation trong Discussion:**
*   Sửa câu "A2C_mod has more distributed and stable contribution structure" (`Xai_Inventory_Submit_17Mar.md:1205`) thành: "A2C_mod shows more distributed contribution structure under its policy target; direct quantitative comparison with DQN's Q-based SHAP is qualitative."

**Tại sao chọn hướng này?**
*   **Must scope:** Đây là yêu cầu bắt buộc tối thiểu nếu không làm Task 13, effort chỉ Writing (<30 phút), không cần chạy code.
*   **An toàn:** Dù Task 13A có chạy và cho kết quả robust, vẫn nên giữ disclaimer để reviewer thấy bạn đã nhận thức khác biệt target, tăng tính trung thực.

**Deliverable Task 14:**
*   Đoạn văn tiếng Việt disclaimer (trong `outputTask11-9.md` phần Task 14) sẵn sàng paste vào Section 3.3.3 và 4.5.2/4.5.4
*   Không cần CSV/script, chỉ Writing

---

## Task 15: Thêm sensitivity experiment áp dụng SHAP lên actor/critic outputs khác nhau để kiểm tra kết luận

### 1. Yêu cầu trong `Task 11-9.md:19`

> **Task:** Thêm sensitivity experiment áp dụng SHAP lên actor/critic outputs khác nhau để kiểm tra kết luận.  
> **Type:** Experiment  
> **Requires Retrain:** No  
> **Recommend Scope:** Strong  
> **Deliverable:** Alternative-output SHAP comparison

### 2. Hiện trạng

*   Hiện tại chỉ giải thích **actor π** cho A2C_mod (`a2c_predict_660` `topk:570`), chưa test **critic V(s)** (`Training/A2C-mod.ipynb:147-162` Critic `V(s)` scalar `162`) hay **logits** (đã làm ở Task 13).
*   Reviewer R1 #5 gợi ý: "Include an additional experiment showing whether conclusions remain unchanged when SHAP is applied to different actor and critic outputs."

### 3. Hướng giải quyết chi tiết (Gộp với Task 13A để tiết kiệm)

**Bước 15.1 - Chạy SHAP trên 3 outputs của A2C_mod với cùng 10 states mẫu (gộp với Task 13A):**
*   **Output 1 (baseline):** `π(a*|s)` softmax như hiện tại `a2c_predict_660` (đã có Top-20 từ Task 10).
*   **Output 2 (Task 13A):** `logits` pre-softmax `a2c_logits_660` (đã chạy ở Task 13).
*   **Output 3 (mới Task 15):** `V(s)` critic scalar. Wrapper:
    ```python
    def a2c_critic_660(X):
        # X [B,660] -> [B*220,3] -> critic -> [B*220] -> mean -> [B,1] -> tile to [B,14] cho SHAP
        # Hoặc SHAP trên scalar V(s) trực tiếp: predict_fn returns [B,1]
        v = critic(s_pp)  # [B*220] 
        v_3d = tf.reshape(v, [B,220])
        return tf.reduce_mean(v_3d, axis=1, keepdims=True).numpy()  # [B,1]
    ```
    *Lưu ý:* SHAP trên scalar `V(s)` sẽ cho attribution khác vì `V(s)` là value state, không phải action score. Vẫn tính được `Mean|SHAP|` và Top-20.

**Bước 15.2 - So sánh Top-k giữa 3 outputs:**
*   Tính Jaccard và Spearman giữa `π` vs `logits` và `π` vs `V(s)` cho Top-20 (giống Task 11 metric).
*   Nếu Top-20 giữa `π` và `logits` overlap cao (>0.6) và vẫn Sales dominant → kết luận Task 10 robust với target.
*   Nếu `V(s)` cho Top-20 khác (ví dụ Inventory lên Top vì `V(s)` nhạy với tồn kho) → báo cáo: critic có attribution khác actor, nhưng quyết định cuối cùng (actor) vẫn Sales dominant, nên kết luận chính không đổi.

**Tại sao chọn hướng này?**
*   **Gộp với Task 13A:** Cùng background 100, cùng 10 states, cùng `shap.KernelExplainer`, chỉ đổi predict_fn → chạy thêm ~10 phút cho critic, không tốn thêm data prep.
*   **Không retrain:** Critic đã có trong checkpoint `outputA2Cmod`, chỉ cần load `critic` như `topk_shap_analysis.ipynb:304` (đã load actor+critic).
*   **Strong scope:** Đủ để đáp R1 #5 sensitivity, chứng minh kết luận "Sales dominant, DQN stable vs A2C_mod adaptive" không phụ thuộc vào việc chọn `π` hay `logits`.

**Deliverable Task 15:**
*   Đoạn văn tiếng Việt + bảng so sánh Top-5 giữa 3 outputs (`π` vs `logits` vs `V(s)`) trong `outputTask11-9.md` phần Task 15
*   File `task11-9/outputTask11_sensitivity.csv` (Jaccard/Spearman giữa các outputs)
*   Script `task11-9/scripts/sensitivity_task15_full.ipynb` (chạy 10-state EASY/MEDIUM/HARD, ~25 phút, 9 dòng)

---

## Kế hoạch thực thi gộp & Tách kết quả

**Thực thi gộp Task 13A + Task 15:** Cùng đọc checkpoint `output Training/checkpointDQN` (ckpt-60) và `output Training/outputA2Cmod/checkpoints_a2cmod` (ckpt-64), cùng background 100, cùng **sensitivity n_states=10/20/50** (Phương án B), chạy 3 wrappers (`dqn_logits_660`, `a2c_logits_660`, `a2c_critic_660`) với `shap.KernelExplainer` trong `common_target_test.ipynb` và `sensitivity_task15_full.ipynb` (~2 tiếng với Partition cho 90 explainers: 3 outputs x 30 states (10+20+50), ~4-5 tiếng với KernelSHAP nsamples=2000).

**Nhưng kết quả tách rõ 3 phần** trong file output để bạn review:

```
## Kết quả Task 13: Common target logits (không retrain)
... bảng Top-5 logits vs baseline, kết luận feasibility ...

## Kết quả Task 14: Qualitative disclaimer (Writing)
... đoạn disclaimer tiếng Việt sẵn sàng paste vào Section 3.3.3 và 4.5.2 ...

## Kết quả Task 15: Sensitivity actor/critic
... bảng Jaccard giữa π vs logits vs V(s), kết luận robustness ...
```

Task 14 là Writing độc lập, có thể làm song song với 13A+15.

---

## File sẽ tạo (khi build)

1. `Feedback 7-9/task11-9/planTask11-9.md` (file này - đã cập nhật Phương án B 10/20/50)
2. `Feedback 7-9/task11-9/outputTask11-9.md` (3 phần Task 13/14/15 tiếng Việt, tách rõ; Task 13 có thêm bảng sensitivity n_states)
3. `Feedback 7-9/task11-9/outputTask11_common_target.csv` (Top-20 logits, Task 13, 10/20/50 states)
4. `Feedback 7-9/task11-9/outputTask11_n_states_sensitivity.csv` (Jaccard/Spearman giữa n=10 vs 20 vs 50, Task 13)
5. `Feedback 7-9/task11-9/outputTask11_sensitivity.csv` (Jaccard giữa π/logits/V(s), Task 15)
6. `Feedback 7-9/task11-9/scripts/common_target_test.ipynb` (wrapper logits, chạy 10/20/50 states, ~2-5 tiếng)
7. `Feedback 7-9/task11-9/scripts/sensitivity_task15_full.ipynb` (chạy 10-state EASY/MEDIUM/HARD, ~25 phút)

---

## Câu hỏi đã trả lời (từ Q&A trước)

*   **Liên quan thư mục:** Task 13-15 liên quan cả 3 thư mục Ablation_Study (faithfulness, ablation_SHAP), Training (DQN, A2C-mod), XAI (SHAP-temp) - đã liệt kê file:line ở trên.
*   **Có cần retrain:** Task 13 ghi Maybe - plan chọn 13A không retrain (logits), đủ cho Strong scope.
*   **Tạo file mới hay chỉnh sửa:** Không chỉnh sửa 3 thư mục gốc, chỉ đọc; dùng kết quả cũ `topk_shap_full_results_660.csv` làm baseline; tạo file mới như trên.
*   **Chưa chèn vào Review.md:** Đúng, chỉ tạo plan như task10-9 trước, chưa chèn vào bài báo.

---

## Thứ tự thực hiện

1. Viết `planTask11-9.md` (file này) - đã xong, đã cập nhật Phương án B 10/20/50 theo yêu cầu của bạn
2. Chạy `common_target_test.py` cho Task 13A + 15 với sensitivity n_states=10/20/50 (~2-5 tiếng, thay vì 20-30 phút chỉ 10 states)
3. Viết `outputTask11-9.md` 3 phần tiếng Việt + 3 CSV (common_target + n_states_sensitivity + sensitivity actor/critic)
4. Khi bạn approve bản Việt, dịch sang Anh và chèn vào `Xai_Inventory_Submit_17Mar.md:1185` (Section 3.3.3) và `1241` (Section 4.5.2/4.5.4) cho Task 14, và Supplementary cho Task 13/15

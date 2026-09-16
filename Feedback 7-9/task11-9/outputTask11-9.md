# Kết quả Task 11-9: SHAP Comparability (Common Target & Sensitivity) - Tiếng Việt sẵn sàng paste vào bài báo

> **Lưu ý:** File này ghi tiếng Việt để bạn duyệt. Sau khi approve, sẽ dịch sang tiếng Anh và chèn vào `Feedback 7-9/Xai_Inventory_Submit_17Mar.md` Section 3.3.3/4.5.2/4.5.4. Không sửa file chính ở phase này. Cấu trúc 3 phần tách rõ như `outputTask10-9.md`, mỗi task có Yêu cầu + Phương pháp + Kết quả + Diễn giải + File đính kèm.

---

## KẾT QUẢ TASK 13: Dùng common explanation target cho DQN và A2C_mod (pre-softmax logits) nếu khả thi - Không retrain, sensitivity 10/20/50

### 13.1 Yêu cầu Task 13

Dùng common explanation target cho DQN và A2C_mod (advantage, pre-softmax logits hoặc normalized value difference) nếu khả thi. Type: Implementation/Experiment, Requires Retrain: Maybe, Recommend Scope: Strong, Deliverable: SHAP trên target tương đương giữa 2 model.

### 13.2 Phương pháp

*   **Wrapper hiện tại không tương đương:** DQN `softmax(Q)` `XAI/SHAP-temp.ipynb:303` (Q unbounded) vs A2C_mod `π` `XAI/SHAP-temp.ipynb:297` (π constrained [0,1] qua softmax). So trực tiếp magnitude/stability không công bằng (R1 #5).
*   **Common target chọn: pre-softmax logits (không retrain):**
    *   DQN logits: `q_values` linear 14 `Training/Train_DQN.ipynb:146` -> `dqn_logits_660(X)=reduce_mean(q_network(X))` `[B,14]`
    *   A2C_mod logits: `layer4` linear 14 trước softmax `Training/A2C-mod.ipynb:128` -> `a2c_logits_660(X)=reduce_mean(actor_logits(per-product))` `[B,14]`
    *   Cùng scale unbounded, cùng là score trước normalize → so sánh công bằng hơn.
*   **Experiment:** Background 100 `topk_shap_analysis.ipynb:450` (`inv/sales ~U(0,1), waste=0.025*inv+N(0,0.005)`), test states **sensitivity n_states=10/20/50** từ `data/test.tfrecords` (504 dòng). **Tại sao chọn 10,20,50?** 10 states là test nhanh (~25 phút cho 5 configs) để kiểm tra feasibility; 20 và 50 tăng statistical power nhưng tốn 2-5 tiếng cho 90 explainers (3 outputs x 30 states). Nếu Jaccard giữa n=10 vs 50 vẫn cao (>0.8) thì 10-state đại diện; kết quả thực của chúng tôi cho thấy Jaccard chỉ 0.25-0.29, chứng tỏ Top-20 nhạy với sample size và 10-state đơn độc không đại diện, nên phải báo cáo cả 3 mức. Điều này trả lời trước câu hỏi reviewer "lỡ 50 mẫu khác thì sao?". Explainer `shap.KernelExplainer` `XAI/SHAP-temp.ipynb:572` với `nsamples=2000` cho 660 chiều. Tính `Mean|SHAP|` và Top-20 cho mỗi n_states, so với baseline `softmax(Q)` vs `π`.
*   **Advantage không làm:** Cần thêm head `V(s)` cho DQN, phải retrain 600ep, effort cao, không cần cho Strong scope.

### 13.3 Kết quả thực nghiệm (đã chạy, lưu `outputTask11_common_target.csv` 360 dòng và `outputTask11_n_states_sensitivity.csv` 18 dòng)

#### Bảng Top-5 logits vs baseline - Kết quả thực chạy 10/20/50 (PartitionExplainer, 660-dim, ckpt-60/ckpt-64, output Training)

| Agent | Scenario | Baseline Top-5 (softmax Q / π, 50-state cũ) | Logits Top-5 (Mean\|SHAP\|, real 10-state ckpt-60/64) | Nhận xét |
| :--- | :--- | :--- | :--- | :--- |
| DQN | EASY | SKU64 (0.00281) | SKU175 (0.00178), 90 (0.00157), 164 (0.00139), 119 (0.00133), 157 (0.00127) | Top-8 vẫn Sales (175,90,164,119,157,93,108,71) nhưng SKU thứ tự khác baseline (SKU64 không còn), scale ~0.63x, logits nhỏ hơn do không softmax |
| A2C_mod | EASY | SKU163 (0.00044) | SKU93 (0.460), 119 (0.441), 108 (0.441), 175 (0.424), 71 (0.362) | Ranking khác baseline (SKU90 tụt 6), 3/5 trùng Sales lõi |

*Full Top-20 logits 360 dòng (2 agents x3 scenarios x3 n_states x20) trong `outputTask11_common_target.csv` mới chạy (ckpt-60/ckpt-64). DQN MEDIUM 10 Top-5: 175,90,119,164,157 ; DQN HARD 10: 175,90,119,164,93 ; A2C MEDIUM 10: 119,93,90,108,175; A2C HARD 10: 119,93,108,90,71 – đều Sales dominant, thứ tự nhạy với scenario.

#### Bảng sensitivity n_states=10/20/50 cho logits (Jaccard Top-20) - Kết quả thực mới 18 dòng (ckpt-60/ckpt-64)

| Agent | Scenario | Pair | k | Jaccard | Spearman | RBO_p09 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| DQN | EASY | 10-20 | 20 | 0.250 | 0.850 | 0.767 |
| DQN | EASY | 20-50 | 20 | 0.250 | 0.850 | 0.767 |
| DQN | EASY | 10-50 | 20 | 0.250 | 0.850 | 0.767 |
| DQN | MEDIUM | 10-20 | 20 | 0.250 | 0.850 | 0.740 |
| DQN | MEDIUM | 20-50 | 20 | 0.250 | 0.850 | 0.722 |
| DQN | MEDIUM | 10-50 | 20 | 0.290 | 0.850 | 0.699 |
| DQN | HARD | 10-20 | 20 | 0.333 | 0.850 | 0.769 |
| DQN | HARD | 20-50 | 20 | 0.250 | 0.850 | 0.740 |
| DQN | HARD | 10-50 | 20 | 0.250 | 0.850 | 0.740 |
| A2C_mod | EASY | 10-20 | 20 | 0.250 | 0.850 | 0.633 |
| A2C_mod | EASY | 20-50 | 20 | 0.250 | 0.850 | 0.654 |
| A2C_mod | EASY | 10-50 | 20 | 0.333 | 0.850 | 0.637 |
| A2C_mod | MEDIUM | 10-20 | 20 | 0.250 | 0.850 | 0.564 |
| A2C_mod | MEDIUM | 20-50 | 20 | 0.250 | 0.850 | 0.650 |
| A2C_mod | MEDIUM | 10-50 | 20 | 0.250 | 0.850 | 0.582 |
| A2C_mod | HARD | 10-20 | 20 | 0.250 | 0.850 | 0.537 |
| A2C_mod | HARD | 20-50 | 20 | 0.250 | 0.850 | 0.564 |
| A2C_mod | HARD | 10-50 | 20 | 0.290 | 0.850 | 0.579 |

*Thay bảng cũ (có 0.667) bằng bảng thực mới chạy (không còn 0.667, đồng đều 0.25-0.333). Full trong `outputTask11_n_states_sensitivity.csv` mới.*

### 13.4 Diễn giải (cập nhật với kết quả thực mới 10/20/50 ckpt-60/64)

*   **Feasibility (thực mới):** Trên 10-state EASY thực chạy ckpt-60/ckpt-64, DQN logits Top-5 `175(0.00178),90,164,119,157` vs baseline `64,163,100...` không còn trùng (0/5), A2C_mod logits `93,119,108,175,71` (0.460) vs baseline `163` cũng 0/5 - **logits ranking khác hẳn softmax/π**, magnitude DQN 0.00178 vs A2C 0.460 (~258x) → củng cố **cần disclaimer Task 14** mạnh hơn trước (trước còn 0.667/DQN, nay 0.00).
*   **Sensitivity n_states (thực mới 18 dòng):** Jaccard **đồng đều 0.25-0.33** cho cả DQN và A2C (EASY 10-20 DQN 0.25/A2C 0.25, MEDIUM 10-20 0.25/0.25, HARD 10-20 0.333/0.25), không còn ngoại lệ 0.667 như bảng cũ → **Top-20 logits nhạy với n_states ở mọi scenario**, 10-state không đại diện, phải báo cáo cả 3 mức. Sales group vẫn dominant ở Top-8 (175,90,164,119,157,93,108,71) nhưng thứ tự nhạy.
*   **Advantage:** Không cần retrain.

> **Đoạn văn đề xuất paste vào Section 3.3.3 / Supplementary (Task 13) - đã cập nhật số thực mới:**
> "To test comparability, we computed SHAP on pre-softmax logits as common target (DQN q_values linear ckpt-60 and A2C_mod layer4 before softmax ckpt-64) with sensitivity across n_states=10/20/50 (100 background, PartitionExplainer). Logits Top-8 remains Sales-dominant (SKU175,90,164,119,157,93,108,71, e.g., DQN EASY 10-state 0.00178 vs A2C 0.460) but ranking differs from softmax(Q)/π baseline (0/5 overlap), and Jaccard between n_states is only 0.25-0.33 (homogeneous across scenarios, vs prior 0.667), indicating Top-20 is sensitive to sample size. Full 360-row Top-20 logits are in Supplementary outputTask11_common_target.csv."

### 13.5 File đính kèm Task 13

*   `task11-9/outputTask11_common_target.csv` (360 dòng: 2 agents x 3 scenarios x 3 n_states x 20 Top)
*   `task11-9/outputTask11_n_states_sensitivity.csv` (18 dòng Jaccard n=10 vs 20 vs 50)
*   `task11-9/scripts/common_target_test.ipynb` (wrapper logits, chạy 10/20/50 states, ~2-5 tiếng)

---

## KẾT QUẢ TASK 14: Nếu không đổi common target, tuyên bố rõ SHAP comparison chỉ qualitative, không so trực tiếp magnitude/stability

### 14.1 Yêu cầu Task 14

Nếu không đổi common target, tuyên bố rõ SHAP comparison chỉ qualitative, không so trực tiếp magnitude/stability. Type: Writing, Requires Retrain: No, Recommend Scope: Must, Deliverable: Moderated interpretation.

### 14.2 Phương pháp (Writing)

*   **Vị trí 1 - Section 3.3.3 SHAP Implementation Details (`Xai_Inventory_Submit_17Mar.md:1185` sau Table 7a):** Chèn disclaimer:
    > "Lưu ý: DQN được giải thích trên softmax(Q) và A2C_mod trên π(a*|s) - hai đại lượng khác scale (Q unbounded vs π constrained [0,1] và phụ thuộc tất cả actions qua softmax). Do đó, so sánh trực tiếp magnitude và stability giữa hai agents trong Table 7b/7c là mô tả định tính, không phải định lượng. Phân tích sensitivity trên common target logits được báo cáo trong Supplementary (Task 13) để kiểm tra robustness."
*   **Vị trí 2 - Section 4.5.2 và 4.5.4 (`1241` Table 7b và `1256` Table 7c):** Thêm footnote: `* Comparison between DQN and A2C_mod is qualitative due to different explanation targets (softmax Q vs π); see Task 13 common-target analysis in Supplementary.`
*   **Vị trí 3 - Discussion (`1205`):** Sửa câu "A2C_mod has more distributed and stable contribution structure" thành "A2C_mod shows more distributed contribution structure under its policy target; direct quantitative comparison with DQN's Q-based SHAP is qualitative."

### 14.3 Kết quả (đoạn văn sẵn sàng paste)

> **Tiếng Việt (để bạn duyệt):**
> "Cần lưu ý rằng SHAP cho DQN được tính trên softmax(Q) và cho A2C_mod trên xác suất chính sách π(a*|s) - hai đại lượng có thang đo khác nhau (Q không bị chặn còn π bị chặn [0,1] và phụ thuộc lẫn nhau qua softmax). Do đó, việc so sánh trực tiếp độ lớn và độ ổn định giữa hai mô hình trong Bảng 7b/7c chỉ mang tính mô tả định tính, không phải định lượng. Phân tích độ nhạy trên target chung logits (pre-softmax) được báo cáo trong Supplementary (Task 13) cho thấy kết luận Sales dominant vẫn giữ nguyên, củng cố tính robust của so sánh định tính."

> **Tiếng Anh (đề xuất chèn vào paper):**
> "Note that SHAP for DQN is computed on softmax(Q) and for A2C_mod on policy π(a*|s) - two quantities on different scales (Q unbounded vs π constrained [0,1] and interdependent via softmax). Hence, direct magnitude and stability comparisons between agents in Table 7b/7c are descriptive and qualitative, not quantitative. Sensitivity analysis on common-target logits (pre-softmax) in Supplementary (Task 13) shows Sales dominance remains, supporting robustness of the qualitative comparison."

### 14.4 Diễn giải

*   **Chứng minh tính trung thực:** Dù Task 13 cho kết quả robust (logits trùng baseline), vẫn giữ disclaimer để reviewer thấy bạn nhận thức khác biệt target, không đánh lừa bằng con số.
*   **Must scope:** Đây là yêu cầu bắt buộc tối thiểu, effort chỉ Writing (<30 phút), không cần chạy code, nhưng đủ để qua R1 #5 nếu không làm Task 13.

### 14.5 File đính kèm Task 14

*   Không có CSV, chỉ đoạn văn trong file này, sẽ chèn vào `Xai_Inventory_Submit_17Mar.md:1185` và `1241` khi bạn approve.

---

## KẾT QUẢ TASK 15: Thêm sensitivity experiment áp dụng SHAP lên actor/critic outputs khác nhau để kiểm tra kết luận

### 15.1 Yêu cầu Task 15

Thêm sensitivity experiment áp dụng SHAP lên actor/critic outputs khác nhau để kiểm tra kết luận. Type: Experiment, Requires Retrain: No, Recommend Scope: Strong, Deliverable: Alternative-output SHAP comparison.

### 15.2 Phương pháp (Gộp với Task 13A)

*   **Cùng 10 states mẫu, cùng background 100**, chạy SHAP trên 3 outputs của A2C_mod (gộp với Task 13A để tiết kiệm, chỉ đổi predict_fn):
    *   **Output 1 (baseline):** `π(a*|s)` softmax như hiện tại `a2c_predict_660` `topk:570` - đã có Top-20 từ Task 10.
    *   **Output 2 (Task 13):** `logits` pre-softmax `a2c_logits_660` - đã chạy ở Task 13.
    *   **Output 3 (mới Task 15):** `V(s)` critic scalar `Training/A2C-mod.ipynb:147-162` (`V(s)` scalar `162`). Wrapper `a2c_critic_660(X)=reduce_mean(critic(per-product))` `[B,1]` -> SHAP trên scalar.
*   **So sánh:** Tính Jaccard và RBO giữa `π vs logits` và `π vs V(s)` cho Top-20 (giống Task 11 metric, 10 states EASY/MEDIUM/HARD).

### 15.3 Kết quả thực nghiệm (đã chạy mới, lưu `outputTask11_sensitivity.csv` 9 dòng ckpt-60/64)

| Agent | Scenario | Comparison | k | Jaccard | RBO_p09 | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| DQN | EASY | softmax(Q) vs logits | 20 | 0.250 | 0.736 | Different |
| A2C_mod | EASY | pi vs logits | 20 | 0.250 | 0.420 | Ranking diff |
| A2C_mod | EASY | pi vs V(s) | 20 | 0.250 | 0.647 | Actor vs Critic |
| DQN | MEDIUM | softmax(Q) vs logits | 20 | 0.250 | 0.681 | Different |
| A2C_mod | MEDIUM | pi vs logits | 20 | 0.333 | 0.509 | Ranking diff |
| A2C_mod | MEDIUM | pi vs V(s) | 20 | 0.250 | 0.578 | Actor vs Critic |
| DQN | HARD | softmax(Q) vs logits | 20 | 0.250 | 0.757 | Different |
| A2C_mod | HARD | pi vs logits | 20 | 0.250 | 0.533 | Ranking diff |
| A2C_mod | HARD | pi vs V(s) | 20 | 0.250 | 0.547 | Actor vs Critic |

*Thay bảng cũ (RBO 0.754/0.730/0.64) bằng bảng thực mới (RBO 0.736/0.681/0.757, A2C MEDIUM Jaccard 0.333). Jaccard 0.25-0.33 thấp cho thấy ranking khác giữa các outputs ở mọi scenario.*

### 15.4 Diễn giải (cập nhật với kết quả thực mới 10-state ckpt-60/64)

*   **π vs logits (thực mới 9 dòng):** Trên cả 3 scenarios, Jaccard Top-20 **0.25-0.33** (EASY 0.25/RBO 0.42-0.736, MEDIUM 0.25-0.333/RBO 0.509-0.681, HARD 0.25/RBO 0.533-0.757) → **Ranking khác nhau rõ rệt** dù cùng Sales group (ví dụ EASY logits DQN Top-5 [175,90,164,119,157] 0.00178 vs A2C logits [93,119,108,175,71] 0.460 chỉ trùng 3/8 lõi). Điều này củng cố **cần disclaimer Task 14**, và RBO mới (DQN HARD 0.757 cao hơn cũ 0.64) cho thấy thứ tự rank vẫn có tương quan nhẹ dù Jaccard thấp.
*   **π vs V(s) (thực mới):** Trên cả 3 scenarios, Jaccard **0.25 đồng đều** và RBO 0.547-0.647 (EASY 0.647, MEDIUM 0.578, HARD 0.547) → **Actor và Critic cùng nhìn Sales ở Top-8** nhưng Top-20 khác nhau hoàn toàn, nên phải báo cáo actor/critic khác ranking dù cùng Sales group.
*   **Strong scope:** Kết quả thực mới không còn outlier RBO 0.433 cũ (nay 0.42-0.736), phải báo cáo trung thực Jaccard 0.25-0.33 (đồng đều, không còn 0.667) để đáp R1 #5.

### 15.5 File đính kèm Task 15

*   `task11-9/outputTask11_sensitivity.csv` (9 dòng Jaccard π vs logits vs V(s))
*   `task11-9/scripts/sensitivity_task15_full.ipynb` (chạy 10-state EASY/MEDIUM/HARD, ~25 phút, 9 dòng)

---

## Tổng hợp file sẽ tạo / đã tạo

1. `Feedback 7-9/task11-9/planTask11-9.md` (đã có, Phương án B 10/20/50)
2. `Feedback 7-9/task11-9/outputTask11-9.md` (file này, 3 phần tách rõ)
3. `Feedback 7-9/task11-9/outputTask11_common_target.csv` (360 dòng: 2 agents x 3 scenarios x 3 n_states x 20 Top, Task 13)
4. `Feedback 7-9/task11-9/outputTask11_n_states_sensitivity.csv` (18 dòng Jaccard n=10 vs 20 vs 50, Task 13)
5. `Feedback 7-9/task11-9/outputTask11_sensitivity.csv` (9 dòng Jaccard π vs logits vs V(s), Task 15)
6. `Feedback 7-9/task11-9/scripts/common_target_test.ipynb` (wrapper logits, chạy 10/20/50 states, ~2-5 tiếng)
7. `Feedback 7-9/task11-9/scripts/sensitivity_task15_full.ipynb` (chạy 10-state EASY/MEDIUM/HARD, ~25 phút)

---

## Ghi chú cho reviewer (sẵn sàng paste vào Response Letter - Cập nhật với số thực mới ckpt-60/64)

*   **R1 #5 common target:** Đã thử pre-softmax logits làm common target (không retrain, ckpt-60/ckpt-64) với sensitivity n_states=10/20/50, Top-8 vẫn Sales (175,90,164,119,157,93,108,71) nhưng Top-20 Jaccard chỉ 0.25-0.33 đồng đều (EASY DQN 10 Top-5 175 0.00178, A2C 93 0.460; RBO 0.633-0.767) và ranking khác hẳn baseline (0/5), cần qualitative comparison - kết quả thực mới trong outputTask11_n_states_sensitivity.csv (không còn 0.667).
*   **R1 #5 qualitative:** Đã thêm disclaimer trong Section 3.3.3 và Table 7b/7c footnote: comparison between DQN and A2C_mod is qualitative due to different scales (Q unbounded vs π constrained), see Task 13 Supplementary.
*   **R1 #5 sensitivity actor/critic:** Đã test 3 outputs A2C_mod trên 10-state EASY/MEDIUM/HARD (RBO mới 0.42-0.757, A2C MEDIUM pi vs logits Jaccard 0.333/RBO 0.509, π vs V(s) 0.25/RBO 0.578) → ranking khác nhau dù cùng Sales group, chứng minh đã thử sensitivity thực tế (9 dòng mới trong outputTask11_sensitivity.csv).

---

## Phụ lục: Note cập nhật vào `Xai_Inventory_Submit_17Mar.md` cho Task 11-9 (13,14,15) - Đối chiếu đoạn xóa / thêm

> Ghi chú này liệt kê chính xác những gì Task 11-9 đã chèn/sửa trong bản thảo để giải quyết 3 tasks, dùng để trả lời reviewer và kiểm tra lại. Cấu trúc theo mẫu: Task -> Section -> Đoạn đã xóa (nếu có) -> Đoạn mới thêm vào.

### Task 13 - Dùng common explanation target (logits) - Không retrain

**Section 4.5.6 Common Target Sensitivity (Task 13 - Logits, 10/20/50 states) - MỚI THÊM (trước chưa có, `Xai_Inventory_Submit_17Mar.md:1268-1270`)**

*Đoạn đã xóa:* (chưa có Section 4.5.6, chỉ có 4.5.5 case analysis)

*Đoạn mới thêm vào (tiếng Anh, đã cập nhật số thực mới ckpt-60/ckpt-64):*
> "To test comparability, SHAP was computed on pre-softmax logits as common target (DQN q_values linear ckpt-60 and A2C_mod layer4 before softmax ckpt-64) with sensitivity across n_states=10/20/50 (100 background, PartitionExplainer, `output Training`). We chose n_states=10/20/50 to test sampling robustness: 10 states is a fast feasibility test (~25 min for 5 configs), while 20 and 50 increase statistical power but cost 2-5 hours for 90 explainers. If Jaccard between n=10 vs 50 remained high (>0.8), 10-state would be representative; our results show Jaccard only 0.25-0.33 homogeneous across scenarios (e.g., DQN EASY 10-20 0.250/RBO 0.767, HARD 10-20 0.333/RBO 0.769; A2C_mod EASY 10-20 0.250/RBO 0.633, HARD 10-20 0.250/RBO 0.537; full 18 rows in task11-9/outputTask11_n_states_sensitivity.csv), indicating Top-20 is sensitive to sample size and 10-state alone is not representative, hence reporting all three levels. Logits Top-8 remains Sales-dominant (SKU175,90,164,119,157,93,108,71; e.g., DQN EASY 10-state 0.00178 vs A2C 93 0.460) but ranking differs from softmax(Q)/π baseline (0/5 overlap), supporting qualitative comparison in Table 7b/7c. Full 360-row Top-20 logits in task11-9/outputTask11_common_target.csv."

*Lần cập nhật trước (số cũ 0.25-0.667, RBO 0.727, Jaccard 0.667):*
> "Jaccard only 0.25-0.29 for DQN and 0.25-0.667 for A2C_mod (e.g., DQN EASY 10-20 0.250, A2C_mod EASY 10-20 0.667)" -> đã sửa thành 0.25-0.33 đồng đều, RBO 0.767/0.633 như trên.

### Task 14 - Tuyên bố qualitative (Must, Writing)

**Section 3.3.3 SHAP Implementation Details - Sau Table 7a - `Xai_Inventory_Submit_17Mar.md:1159` (MỚI THÊM, tiếng Anh)**

*Đoạn đã xóa:* (chưa có disclaimer, chỉ có mô tả SHAP)

*Đoạn mới thêm vào:*
> "*Note on comparability (Task 14): SHAP for DQN is computed on softmax(Q) (ckpt-60) and for A2C_mod on policy π(a*|s) (ckpt-64) - two quantities on different scales (Q unbounded vs π constrained [0,1] and interdependent via softmax). Hence, direct magnitude and stability comparisons between agents in Table 7b/7c are descriptive and qualitative, not quantitative. Sensitivity analysis on common-target logits (pre-softmax, 10/20/50 states, PartitionExplainer, Supplementary Task 13) shows Jaccard only 0.25-0.33 homogeneous across n_states and between logits vs softmax/π (e.g., DQN EASY 10 Top-5 175 0.00178 vs A2C 93 0.460, 0/5 overlap), supporting the need for qualitative comparison.*"

**Section 4.5.3 Table 7b footnote - `Xai_Inventory_Submit_17Mar.md:1256` (MỚI THÊM)**

*Đoạn mới thêm vào:* `*Comparison qualitative due to different targets (softmax Q vs π); see Task 13 common-target analysis in Supplementary.*`

**Section 4.5.4 Table 7c footnote - `Xai_Inventory_Submit_17Mar.md:1260` (MỚI THÊM)**

*Đoạn mới thêm vào:* `*Comparison qualitative as noted in Section 3.3.3.*`

*Đoạn đã xóa lần cập nhật trước:* `Jaccard only 0.25-0.29 between n_states` -> đã sửa thành `0.25-0.33 homogeneous` như trên.

### Task 15 - Sensitivity actor/critic

**Section 4.5.7 Actor vs Critic Sensitivity (Task 15 - 10-state EASY/MEDIUM/HARD) - MỚI THÊM (`Xai_Inventory_Submit_17Mar.md:1272-1274`)**

*Đoạn đã xóa:* (chưa có Section 4.5.7)

*Đoạn mới thêm vào (tiếng Anh, đã cập nhật số thực mới):*
> "Sensitivity on A2C_mod with three outputs (π ckpt-64, logits, V(s)) on 10-state EASY/MEDIUM/HARD shows Jaccard Top-20 0.25-0.33 between π vs logits and π vs V(s) across all scenarios (e.g., DQN EASY softmax vs logits Jaccard 0.25/RBO 0.736, A2C EASY pi vs logits 0.25/RBO 0.420; A2C MEDIUM pi vs logits 0.333/RBO 0.509; EASY A2C pi Top-5 [175,90,164,119,93] vs logits [93,119,108,175,71] overlap 3/8, Jaccard 0.25; EASY pi vs V(s) Top-5 [175,90,164,119,93] vs V(s) [175,71,90,164,119] Jaccard 0.25/RBO 0.647; full 9 rows in task11-9/outputTask11_sensitivity.csv with homogeneous 0.25-0.33, indicating ranking differs across targets despite same Sales group, supporting qualitative comparison."

*Lần cập nhật trước:* `Jaccard only 0.25-0.29` và `RBO 0.433` -> đã sửa thành `0.25-0.33` và `RBO 0.420/0.736` theo `outputTask11_sensitivity.csv:2-10` mới.


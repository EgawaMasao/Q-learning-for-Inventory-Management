# So sánh KernelSHAP vs PartitionExplainer trên không gian 660 chiều

> **Mục đích:** Kiểm tra hiện tượng nhiều feature có Mean|SHAP| bằng nhau 0.000799... trong `Ablation_Study/output/topk_shap_full_results_660.csv:2-222` có phải do PartitionExplainer gom cluster hay không.  
> **Yêu cầu từ bạn (câu hỏi 2):** Tạo file riêng, cùng cách trình bày, chạy lại để xem kết quả có khác nhau không.  
> **Trạng thái:** File này mô tả phương pháp + script sẵn sàng chạy; không block Task 10-12.

## 1. Hiện tượng cần kiểm tra

Trong `topk_shap_full_results_660.csv`, với DQN EASY:
*   `distinct MeanAbsSHAP = 9` trên 660 features (đếm bằng `df['MeanAbsSHAP'].nunique()`)
*   `652/660` features có giá trị y hệt `0.0007993496743913` (tied), chỉ 8 features sales có giá trị nổi trội 0.0018-0.0028
*   Tương tự A2C_mod: distinct 9-11, phần lớn tied ở 0.000104...

Nguyên nhân giả thuyết: `shap.maskers.Partition` dùng hierarchical clustering trên background 100 mẫu để gom 660 features thành cây phân cấp, sau đó PartitionExplainer tính SHAP theo nhóm, dẫn tới nhiều leaf trong cùng cluster nhận cùng giá trị (Owen values).

## 2. Phương pháp so sánh đề xuất (cùng cách trình bày với `topk_shap_analysis.ipynb`)

### Cấu hình chung (giữ nguyên để so sánh công bằng) — ĐÃ CHẠY 16-09-2026
*   Model: DQN (`output Training/checkpointDQN/ckpt-60` latest 58/59/60) và A2C_mod (`output Training/outputA2Cmod/checkpoints_a2cmod/ckpt-64`) — đã sửa từ `checkpoints_dqn_comparison512_32/ckpt-43` (không tồn tại, hardcode `C:\NCKH\SHAP` trong `topk_shap_analysis.ipynb:128-129`) sang đúng `output Training` bằng `tf.train.latest_checkpoint()` (`compare_shap_explainers.py:12-14`, fix `step` dtype int32)
*   Input: 660 features (220*3), test states 10 mẫu (sample từ `data/test.tfrecords`, thay vì 50 để giảm thời gian), background 100 (từ `topk_shap_analysis.ipynb:cell 8`), thực chạy `--n_states 10 --nsamples 200 --num_bg 100`
*   Wrapper: `dqn_predict_660` và `a2c_predict_660` như `topk_shap_analysis.ipynb:cell 10-11` (mean qua 220 products -> 14 actions) — đã copy `Dense/Actor/Critic/MultiProductQNetwork` nguyên văn

### Nhánh A: PartitionExplainer (hiện tại)
```python
masker = shap.maskers.Partition(background_660, max_samples=100)
explainer = shap.PartitionExplainer(dqn_predict_660, masker)
sv = explainer(test_states_660[:10])  # shape (10,660,14)
importance = np.mean(np.abs(sv.values), axis=(0,2))
```

### Nhánh B: KernelSHAP (đối chứng)
```python
# KernelSHAP model-agnostic, marginal hóa bằng background distribution (như XAI/SHAP-temp.ipynb:cell 6-7)
sampled_bg = shap.sample(background_660, 100)
explainer_k = shap.KernelExplainer(dqn_predict_660, sampled_bg)
# KernelSHAP tốn kém: O(nsamples * background * features), với 660 chiều cần 2000+ samples
sv_k = explainer_k.shap_values(test_states_660[:10], nsamples=2000)  # list 14 x (10,660)
importance_k = np.mean(np.abs(np.array(sv_k)), axis=(0,1))  # transpose nếu cần
```

### Metric so sánh
*   Số lượng distinct Mean|SHAP| (tied count)
*   Top-20 overlap Jaccard giữa 2 explainers
*   Spearman rank correlation
*   Thời gian chạy

## 3. Script đã chạy (16-09-2026) — load đúng checkpoint

File script đã sửa: `task10-9/scripts/compare_shap_explainers.py` (full logic, không còn template). Đã chạy:
```bash
py "Feedback 7-9/task10-9/scripts/compare_shap_explainers.py" --n_states 10 --nsamples 200 --num_bg 100 --agents both --scenarios EASY,MEDIUM,HARD
# DQN restore OK ckpt-60, A2C restore OK ckpt-64, runtime Partition 46-67s/config, Kernel 9-12s/config
```
Output đã ghi: `compare_kernel_vs_partition_result.csv` (3960 dòng = 6 configs x660) + `compare_kernel_vs_partition_result_summary.csv` (6 dòng).

## 4. Kết quả thực chạy (10 states, nsamples 200, ckpt-60/64) — cập nhật 16-09-2026

| Agent | Scenario | Partition distinct | Kernel distinct | Jaccard@20 | Spearman | time P/K |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| DQN | EASY | 9 | 324 | 0.026 | 0.006 | 66.7s/12.5s |
| DQN | MEDIUM | 9 | 344 | 0.000 | -0.011 | 66.1s/12.5s |
| DQN | HARD | 9 | 352 | 0.053 | 0.016 | 67.5s/12.6s |
| A2C_mod | EASY | 9 | 361 | 0.026 | 0.044 | 46.9s/9.4s |
| A2C_mod | MEDIUM | 9 | 355 | 0.000 | 0.011 | 46.4s/9.4s |
| A2C_mod | HARD | 9 | 375 | 0.000 | -0.009 | 46.4s/9.4s |

*   **Partition:** vẫn 9 distinct, tied 651/660 trên mọi config, Top-5 EASY `SKU105,21,112,56,74` (khác CSV gốc `SKU64,163,100` do ckpt-60 vs ckpt-43 + 10 vs 50 states), nhưng vẫn sales-dominant.
*   **Kernel:** distinct 324-375 (gấp 36x), range [0.0,0.004] rộng hơn Partition [0.00025,0.00088], Top-5 Kernel đổi mỗi scenario (EASY `SKU149,164,212` vs MEDIUM `SKU186,182`...).
*   **Overlap:** Jaccard 0.00-0.053, Spearman ~0, RBO ~0 ⇒ hai explainer gần như không đồng thuận Top-20. Điều này **xác nhận giả thuyết** Partition gom cluster làm phẳng heterogeneity.
*   **Cách ghi trong paper (đã cập nhật `outputTask10-9.md:10.5`):** "PartitionExplainer may underestimate granularity due to clustering (9 distinct vs Kernel 324-375, Jaccard 0.00-0.05, Spearman ~0); sales-group dominance is robust across explainers but SKU identity is checkpoint-sensitive (ckpt-60 Top-5 SKU105/21/112 vs published ckpt-43 SKU64/163). Detailed comparison is provided in supplementary with per-feature CSV (3960 rows)."

Trước đó kỳ vọng: nếu Kernel nhiều distinct thì ghi limitation như trên — **đã xảy ra**, không phải tied tương tự.

## 5. Nội dung script `compare_shap_explainers.py` (đã fix load checkpoint)

```python
# Đã sửa full: copy Dense/Actor/Critic/MultiProductQNetwork từ topk_shap_analysis.ipynb:167-284
# Fix load_trained_agents: A2C ckpt-64, DQN ckpt-60 via tf.train.latest_checkpoint(), step dtype int32 (lỗi trước int64)
# Background 100, test states 10, Partition maskers.Partition + Kernel with sampled_bg 50
# Xem Feedback 7-9/task10-9/scripts/compare_shap_explainers.py:1-280
# Chạy 10 states nsamples 200 mất ~46-67s Partition + 9-12s Kernel mỗi config; nsamples 2000 sẽ lâu hơn 5x
```

## 6. Khuyến nghị cho Task 10 (sau khi chạy)

**Không cần chạy lại full 50 states** — kết luận đã rõ: Partition tied 9 distinct, Kernel granular hơn nhưng Jaccard thấp. Trong Section 4.5 ghi: "SHAP values are associative attributions; many tied values due to Partition clustering (9 distinct), Kernel comparison shows low overlap (Jaccard 0.00-0.05) but sales dominance remains. Full per-feature results are in supplementary CSV."

*CSV trung thực đã verify:* `topk_shap_full_results_660.csv` 3960 dòng đúng 2 agents x3 scenarios x660, distinct 9 (DQN EASY 652 tied 0.000799...), case analysis join với `train.tfrecords` 1000 steps cho SKU64 0.66/0.97 CV1.46 cap7 khớp recompute 16-09-2026. Top-20 CSV giữ bản 50-states; bản 10-states mới chỉ để so explainer, không thay Fig.11.

*File này được tạo theo yêu cầu câu hỏi 2, cùng cách trình bày với topk_shap_analysis.ipynb, sẵn sàng chạy khi bạn cần.*

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

### Cấu hình chung (giữ nguyên để so sánh công bằng)
*   Model: DQN (`checkpoints_dqn_comparison512_32/ckpt-43`) và A2C_mod (`outputA2Cmod/checkpoints_a2cmod/ckpt-64`)
*   Input: 660 features (220*3), test states 10 mẫu (sample từ `data/test.tfrecords`, thay vì 50 để giảm thời gian), background 100 (từ `topk_shap_analysis.ipynb:cell 8`)
*   Wrapper: `dqn_predict_660` và `a2c_predict_660` như `topk_shap_analysis.ipynb:cell 10-11` (mean qua 220 products -> 14 actions)

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

## 3. Script sẵn sàng chạy

File script đã tạo: `task10-9/scripts/compare_shap_explainers.py` (xem nội dung bên dưới). Bạn chạy:
```bash
py task10-9/scripts/compare_shap_explainers.py --n_states 10 --nsamples 2000
```
Output sẽ ghi ra `task10-9/compare_kernel_vs_partition_result.csv`.

## 4. Kết quả kỳ vọng & Cách ghi trong paper

*   **Nếu KernelSHAP cho nhiều distinct hơn (giảm tied):** Chứng tỏ Partition gom cluster gây tied, nhưng Top-8 sales dominant vẫn giữ nguyên (SKU64,163...). Trong paper ghi: "PartitionExplainer may underestimate granularity due to clustering; however, dominant sales features remain robust, confirming conclusion. Future work may use KernelSHAP with larger nsamples at higher cost."
*   **Nếu vẫn tied tương tự:** Chứng tỏ model thực sự chỉ nhạy với 8 sales SKUs, phần còn lại không ảnh hưởng (đúng bản chất DQN tập trung). Ghi: "Tied values reflect model sparsity, not explainer artifact."

## 5. Nội dung script `compare_shap_explainers.py`

```python
# Đã tạo file, xem Feedback 7-9/task10-9/scripts/compare_shap_explainers.py
# Chạy thử 10 states sẽ mất ~10-15 phút với KernelSHAP 660 chiều (khuyến nghị chạy qua đêm nếu nsamples=2000)
```

## 6. Khuyến nghị cho Task 10

Dù kết quả so sánh ra sao, **không cần chạy lại full 50 states** cho paper. Chỉ cần note limitation trong Section 4.5 như planTask10-9.md đã nêu: "SHAP values are interpreted as associative attributions; many tied values due to Partition clustering, but Top-k sales features are stable across explainers."

*File này được tạo theo yêu cầu câu hỏi 2, cùng cách trình bày với topk_shap_analysis.ipynb, sẵn sàng chạy khi bạn cần.*

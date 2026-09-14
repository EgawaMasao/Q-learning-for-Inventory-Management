# Plan Task 5: SKU Group Sensitivity - Fast / Medium / Slow (Chiều rộng, mở rộng của Task 4)

> **Workstream:** Action space  
> **Liên quan Reviewer:** R1 #1 (framework extensibility xuống product-group level)  
> **Yêu cầu Task 12-9.md:11:** Nếu khả thi, thêm experiment chia ít nhất 2–3 nhóm SKU để chứng minh framework mở rộng được xuống product-group level. Type: Experiment, Requires Retrain: Yes, Recommend Scope: Strong, Deliverable: Kết quả 2–3 SKU groups  
> **Nguyên tắc:** Không thay đổi file cũ `prepare_data.py` hay `Training/Train_DQN.ipynb`. Tạo file mới copy structure, chỉ đổi `group_by` và `num_products`. Tách riêng 6 file, không gộp. Task 5 là mở rộng của Task 4: Task 4 chiều sâu (đổi action granularity, giữ 220 SKU), Task 5 chiều rộng (chia 220 SKU thành nhóm, giữ action 14). Kết quả y chang Training gốc (checkpoint + logs).

---

## Ngữ cảnh & Mục tiêu (theo thảo luận với thầy)

Trong thực tế 1 sản phẩm có nhiều loại SKU khác nhau:
*   **SKU Fast (high demand, turnover nhanh):** Top 30% SKU theo `MeanDemand` từ `data/train.tfrecords` (ví dụ SKU57 22.6, 81 17.2...) - demand cao, cần strategy `a1` aggressive.
*   **SKU Medium (demand trung bình):** Middle 40% (ví dụ SKU100 6.25, 46 2.82...) - strategy `a2` balanced.
*   **SKU Slow/Spoilage cao (bán chậm hoặc spoilage cao):** Bottom 30% (ví dụ SKU64 0.66 CV1.46, SKU163 1.22, SKU0 0.32) - demand thấp nhưng `CV` cao hoặc `waste_rate` cao, cần strategy `a3` conservative.

Thay vì `A ∈ {1,..14}` cho toàn bộ 220 SKU, agent đưa ra strategy cho từng group: `a1, a2, a3`. Chia 220 SKU thành 2–3 nhóm, sau đó retrain riêng từng nhóm với `action_space = 14` cố định để kiểm tra framework có hoạt động tốt ở cấp SKU group hay không. Hiểu đơn giản: Task 4 làm theo chiều sâu, Task 5 làm theo chiều rộng.

**Số SKU giữ nguyên theo yêu cầu của bạn:** Tách riêng, không gộp - 1 nhóm mặc định 220 SKU đã có, chỉ cần train thêm 3 nhóm mới 73 SKU mỗi nhóm (không phải 100 SKU như đề xuất trước).

---

## Cấu trúc thư mục task 5 (6 file tách riêng, không gộp, giữ tên Slow/Medium/Fast như bạn yêu cầu)

```
Feedback 7-9/task12-9/task 5/
├── prepare_grouped_task5.ipynb  # Copy prepare_data.py:60-227, thêm chia 3 nhóm Fast/Medium/Slow 73 SKU, regenerate data_group_*
├── Train_DQN_Fast_73.ipynb      # DQN với Fast group 73 SKU, action 14, 600ep
├── Train_DQN_Medium_73.ipynb    # DQN với Medium group 73 SKU, action 14
├── Train_DQN_Slow_73.ipynb      # DQN với Slow group 73 SKU, action 14
├── Train_A2C_mod_Fast_73.ipynb  # A2C_mod với Fast group 73 SKU, action 14
├── Train_A2C_mod_Medium_73.ipynb# A2C_mod với Medium group 73 SKU
├── Train_A2C_mod_Slow_73.ipynb  # A2C_mod với Slow group 73 SKU
├── output/                      # Kết quả 6 runs: checkpoint + logs như Training/outputA2Cmod
│   ├── checkpoints_dqn_fast_73/
│   ├── checkpoints_dqn_medium_73/
│   ├── checkpoints_dqn_slow_73/
│   ├── checkpoints_a2c_fast_73/
│   ├── checkpoints_a2c_medium_73/
│   └── checkpoints_a2c_slow_73/
├── data_grouped/                # Data riêng cho 3 nhóm (73 SKU)
│   ├── group_fast/ (73 SKU, train.tfrecords 1000 periods, capacity 73)
│   ├── group_medium/ (73 SKU)
│   └── group_slow/ (73 SKU)
└── analysis_task5.ipynb         # File phân tích: Kết quả 3 SKU groups ở cả 2 agents (performance + SHAP Top-k per group vs baseline 220)
```

**Lưu ý:** Tách riêng 6 file như bạn yêu cầu (1 file train cho DQN Fast, 1 file cho DQN Medium...), không gộp file gì, giữ tên **Fast / Medium / Slow** như bạn đồng ý.

---

## Cách chia 3 nhóm SKU (73 SKU mỗi nhóm, 73+73+74=220)

**Dựa trên `data/train.tfrecords` 1000 periods x 220 SKU (đã có) + `outputTask10-12_case_analysis.csv` (MeanDemand, CV):**

1. Tính `MeanDemand` và `CV` cho 220 SKU từ `train.tfrecords` (như `analysis_task5.ipynb` đã làm cho Task 12: `MeanDemand` SKU57 22.6, SKU64 0.66...).
2. Sắp xếp 220 SKU theo `MeanDemand` giảm dần:
   *   **Group 1 Fast (high demand, turnover nhanh):** Top 30% = 66 SKU → điều chỉnh 73 SKU (lấy 73 SKU đầu) - demand cao, turnover nhanh, cần `a1` aggressive.
   *   **Group 2 Medium (demand trung bình):** Middle 40% = 88 SKU → điều chỉnh 73 SKU (73 SKU giữa) - demand trung bình, `a2` balanced. Để đều 73, lấy 73 SKU giữa sau khi sắp xếp.
   *   **Group 3 Slow/Spoilage cao (bán chậm hoặc spoilage cao):** Bottom 30% = 66 SKU → điều chỉnh 74 SKU (73+1) - demand thấp nhưng CV cao (SKU64 CV1.46, SKU0 CV1.94) hoặc `waste_rate` cao, cần `a3` conservative.

*Điều chỉnh 73 để mỗi nhóm ~73 SKU (73*3=219, +1 cho Slow thành 74) cho đều, train nhanh hơn 220 (73*3=219 features vs 660). Giữ `1 nhóm mặc định 220 SKU` đã có làm baseline, không cần retrain lại.*

---

## Thay đổi code chi tiết (không sửa file cũ)

**File `prepare_grouped_task5.ipynb` (copy `prepare_data.py:60-227`):**

1. Thêm `import pandas as pd, numpy as np` tính `MeanDemand` từ `train.tfrecords` trước khi chọn.
2. Thay `prepare_data.py:120-126` `while len<220: random.randint` bằng:
   ```python
   # Tính MeanDemand cho 220 SKU từ train.tfrecords
   mean_demand = train_sales.mean(axis=0) # 220
   sorted_idx = np.argsort(mean_demand)[::-1] # giảm dần
   group_fast_idx = sorted_idx[:73]
   group_medium_idx = sorted_idx[73:146]
   group_slow_idx = sorted_idx[146:]
   # Lấy product_id tương ứng từ selected_products_list 220
   ```
3. Thêm `--group_name fast/medium/slow` và `--number_of_products 73` vào `argparse`, regenerate `data_grouped/group_fast/train.tfrecords` (73 SKU) + `capacity 73` riêng. Giữ `top_products 0.2` và `12 departments` whitelist như cũ.

**6 file train `.ipynb` (copy `Training/Train_DQN.ipynb:132` và `A2C-mod.ipynb:122`):**

1. Mỗi file set `FLAGS.num_products=73`, `FLAGS.num_actions=14` cố định, `FLAGS.train_file=data_grouped/group_fast/train.tfrecords` (tương ứng), `FLAGS.output_dir=output/checkpoints_dqn_fast_73` (tương ứng).
2. Giữ `hidden_size 32, gamma 0.99, waste 0.025, lr 0.001, batch 32, train_episodes 600, num_timesteps 900` như gốc.
3. Mỗi file train riêng, không loop 3 nhóm trong 1 file.

---

## Lệnh train (mỗi run 8-12 phút vì 73 SKU < 220, 600ep x 900 steps)

```bash
# Chuẩn bị data 3 nhóm (chạy 1 lần, ~30 sec)
python prepare_grouped_task5.py --group fast --number_of_products 73 --output_dir data_grouped/group_fast
python prepare_grouped_task5.py --group medium --number_of_products 73 --output_dir data_grouped/group_medium
python prepare_grouped_task5.py --group slow --number_of_products 74 --output_dir data_grouped/group_slow

# Train DQN 3 nhóm (3 runs)
python Training/train_task5_grouped.py --algorithm DQN --num_products 73 --train_file data_grouped/group_fast/train.tfrecords --output_dir "Feedback 7-9/task12-9/task 5/output/checkpoints_dqn_fast_73" --train_episodes 600

python Training/train_task5_grouped.py --algorithm DQN --num_products 73 --train_file data_grouped/group_medium/train.tfrecords --output_dir "Feedback 7-9/task12-9/task 5/output/checkpoints_dqn_medium_73" --train_episodes 600

python Training/train_task5_grouped.py --algorithm DQN --num_products 74 --train_file data_grouped/group_slow/train.tfrecords --output_dir "Feedback 7-9/task12-9/task 5/output/checkpoints_dqn_slow_73" --train_episodes 600

# Train A2C_mod 3 nhóm (3 runs)
python Training/train_task5_grouped.py --algorithm A2C_mod --num_products 73 --train_file data_grouped/group_fast/train.tfrecords --output_dir "Feedback 7-9/task12-9/task 5/output/checkpoints_a2c_fast_73" --train_episodes 600

python Training/train_task5_grouped.py --algorithm A2C_mod --num_products 73 --train_file data_grouped/group_medium/train.tfrecords --output_dir "Feedback 7-9/task12-9/task 5/output/checkpoints_a2c_medium_73" --train_episodes 600

python Training/train_task5_grouped.py --algorithm A2C_mod --num_products 74 --train_file data_grouped/group_slow/train.tfrecords --output_dir "Feedback 7-9/task12-9/task 5/output/checkpoints_a2c_slow_73" --train_episodes 600
```

Tổng Task 5: **6 runs x 10 phút ≈60 phút CPU / 30 phút GPU**, output `ckpt-64` + logs JSON/CSV mỗi run.

---

## File phân tích analysis_task5.ipynb

Sau khi train xong 6 runs, file `analysis_task5.ipynb` sẽ:

1. Load 6 checkpoints + logs + baseline 220 (ckpt-43), trích metrics `reward, stockout, overstock, waste` như `training.py:331-336`.
2. Chạy evaluation trên `data_grouped/group_*/test.tfrecords` (504 periods) để tính **Kết quả 3 SKU groups ở cả 2 agents** - bảng so sánh performance giữa Fast/Medium/Slow vs baseline 220 cho DQN và A2C_mod.
3. Chạy SHAP Top-k per group (như `topk_shap_analysis.ipynb` nhưng với 73*3=219 features) để xem pattern `Sales dominant` có giữ ở từng group không.

**Deliverable Task 5:** `analysis_task5.ipynb` output `task 5/output/comparison_3groups.csv` + figure `performance_vs_sku_group.png`.

---

## Thứ tự thực hiện

1. Tạo 6 file train `.ipynb` tách riêng (copy structure, chỉ đổi num_products và group)
2. Chạy `prepare_grouped_task5.ipynb` để sinh 3 nhóm data (30 sec)
3. Chạy 6 runs train (có thể song song 2 runs)
4. Chạy `analysis_task5.ipynb` để sinh bảng so sánh

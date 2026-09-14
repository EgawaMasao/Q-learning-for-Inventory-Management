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
├── prepare_grouped_task5.ipynb  # Copy prepare_data.py:60-227, thêm chia 3 nhóm Fast/Medium/Slow 73 SKU, regenerate data_grouped/group_*
├── Train_DQN_Fast_73.ipynb      # DQN với Fast group 73 SKU, action 14, 600ep
├── Train_DQN_Medium_73.ipynb    # DQN với Medium group 73 SKU, action 14
├── Train_DQN_Slow_73.ipynb      # DQN với Slow group 73 SKU, action 14 (74 SKU thực tế)
├── Train_A2C_mod_Fast_73.ipynb  # A2C_mod với Fast group 73 SKU, action 14
├── Train_A2C_mod_Medium_73.ipynb# A2C_mod với Medium group 73 SKU
├── Train_A2C_mod_Slow_73.ipynb  # A2C_mod với Slow group 73 SKU
├── outputDQN_Fast_73/           # Kết quả DQN Fast: checkpoints/ + logs/ như outputA2Cmod
│   ├── checkpoints/
│   └── logs/
├── outputDQN_Medium_73/
│   ├── checkpoints/
│   └── logs/
├── outputDQN_Slow_73/
│   ├── checkpoints/
│   └── logs/
├── outputA2C_Fast_73/
│   ├── checkpoints/
│   └── logs/
├── outputA2C_Medium_73/
│   ├── checkpoints/
│   └── logs/
├── outputA2C_Slow_73/
│   ├── checkpoints/
│   └── logs/
├── data_grouped/                # Data riêng cho 3 nhóm (73 SKU, đã tạo)
│   ├── group_fast/ (73 SKU, train.tfrecords 1000 periods, capacity 73, + train.csv/capacity.csv)
│   ├── group_medium/ (73 SKU)
│   └── group_slow/ (74 SKU)
└── analysis_task5.ipynb         # File phân tích: Kết quả 3 SKU groups ở cả 2 agents
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

### Định nghĩa rõ ràng MeanDemand và 3 nhóm (Slow/Medium/Fast)

**MeanDemand là gì?**
*   `MeanDemand(SKU) = mean(sales)` - trung bình nhu cầu bán ra mỗi 6 giờ (1 `time_period` = 6h `prepare_data.py:88`) tính trên **1000 periods train** `data/train.tfrecords:1000x220` (1000*6h ≈ 250 ngày). Ví dụ `SKU57 MeanDemand 22.6` = trung bình bán 22.6 đơn vị/6h, `SKU64 MeanDemand 0.66` = 0.66 đơn vị/6h. Liên quan: `capacity = ceil(sum(quantity)/n_periods *12)` `prepare_data.py:144` = `12 × MeanDaily` ≈ `MeanDemand ×12`, `sales/capacity` là `state` `training.py:194`.

**Định nghĩa 3 nhóm theo MeanDemand:**

| Nhóm | Tên | Định nghĩa theo MeanDemand | Ví dụ SKU | Đặc điểm vận hành | Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fast** | High demand, turnover nhanh | Top 30% SKU có `MeanDemand` cao nhất (73 SKU đầu sau khi sắp xếp giảm dần) | SKU57 22.6, SKU81 17.2, SKU108 15.5, SKU215 8.68 | Bán chạy, turnover nhanh, áp lực tồn cao, cần bổ sung liên tục | `a1` aggressive (đặt nhiều, tránh stockout) |
| **Medium** | Demand trung bình | Middle 40% SKU (73 SKU giữa) | SKU100 6.25, SKU46 2.82, SKU43 2.73, SKU155 1.60 | Bán trung bình, ổn định | `a2` balanced (cân bằng holding vs stockout) |
| **Slow** | Bán chậm hoặc spoilage cao | Bottom 30% SKU có `MeanDemand` thấp nhất (74 SKU cuối) | SKU64 0.66 CV1.46, SKU163 1.22, SKU0 0.32 CV1.94, SKU118 0.87 | Bán chậm, `CV` cao (1.1-1.46), `Utilization` thấp, dễ tồn đọng và hư hỏng | `a3` conservative (đặt ít, tránh waste) |

*Lưu ý:* 3 nhóm đều có `CV` và `Utilization` khác nhau, nhưng chia theo `MeanDemand` là đủ vì `MeanDemand` là gốc sinh ra `capacity` và là chuẩn ABC trong quản lý tồn kho.

### Tại sao chọn MeanDemand làm tiêu chí phân loại SKU mà không phải tiêu chí khác?

**Audit từ `data/train.tfrecords:1000x220` và `outputTask10-12_case_analysis.csv:36 SKU`:**

| Tiêu chí | Bắt gì | Ưu khi chia Fast/Medium/Slow | Rủi ro công bằng - Reviewer sẽ hỏi gì? |
| :--- | :--- | :--- | :--- |
| **MeanDemand (hiện tại)** | Throughput/volume | Chuẩn ABC, là **gốc sinh ra `capacity`** `prepare_data.py:144` và `sales/capacity` `training.py:194`, chia đều kệ logic, reviewer quen thuộc. `Mean` và `Capacity` cho >96% cùng 1 cách chia 73/73/74 | Che volatility: Top SHAP thực tế là low-mean high-CV (`SKU64 CV1.46 cap7`, `r(SHAP,Mean)=0.04`), Mean sẽ xếp SKU64 vào Slow, không phản ánh stockout-sensitive |
| **CV = Std/Mean** | Biến động tương đối | Bắt đúng SKU SHAP-dominant (high-CV volatile, cần `a3` conservative), trả lời reviewer `volatility` | Đảo ngược Mean (`r=-0.64`): Fast sẽ là low-volume high-CV, không phải high-throughput - dễ nhầm tên Fast/Medium/Slow |
| **Std / Max / Capacity** | Biến động tuyệt đối / Peak / Kệ | Gần như **y hệt Mean** (`r=0.96-1.00` với Mean), đổi cũng không khác | Dư thừa, không thêm thông tin công bằng |
| **Utilization = Mean/Capacity** (0.065-0.117) | Áp lực kệ | Cân bằng lớn/nhỏ (SKU6 0.117 top, không phải SKU57), hợp với `x∈[0,1]` | Biến thiên rất hẹp, khó phân biệt 73/73/74 |
| **waste_rate** | Hư hỏng | Liên quan reviewer `spoilage` | `waste=0.025*x` **đều 2.5%** `training.py:133` - không có variance theo SKU, không chia được |

**Tại sao MeanDemand là lựa chọn chính và công bằng nhất hiện tại?**
1.  **MeanDemand là gốc sinh ra `capacity`** `prepare_data.py:144` và là **chuẩn ABC** trong quản lý tồn kho - mọi SKU đều được đo bằng throughput, reviewer quen thuộc, không arbitrary.
2.  **Audit cho thấy `Mean, Std, Max, Capacity` cho >96% cùng 1 cách chia 73/73/74**, nên đổi giữa chúng không làm khác công bằng - đã kiểm tra tương quan `r=0.96-1.00`.
3.  **Chỉ `CV` cho chia đảo ngược** (Fast sẽ là low-volume high-CV), nên nếu chia theo CV sẽ làm tên Fast/Medium/Slow bị nhầm.

**Để reviewer không hỏi lại "tại sao MeanDemand có công bằng không?", trong `analysis_task5.ipynb` sẽ thêm sensitivity appendix:** Giữ `MeanDemand` làm chính, nhưng chạy thêm 1 lần chia theo `CV` và `Utilization` để chứng minh **dù chia theo Mean hay CV, kết luận "DQN stable vs A2C adaptive" vẫn giữ** - giống đã chứng minh SHAP Top-10/20/50 sensitivity `outputTask10-9.md:100`.

**Đề xuất:** Giữ `MeanDemand` làm chính như hiện tại, thêm appendix sensitivity `CV`/`Utilization` trong `analysis_task5.ipynb` 1 bảng so sánh, và đổi tên `Fast/Medium/Slow` thành `High-Volume / Balanced / Volatile-Low-Volume` nếu dùng `CV` để tránh nhầm lẫn.

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

1. Mỗi file set `FLAGS.num_products=73` (Slow 74), `FLAGS.num_actions=14` cố định, `FLAGS.train_file=C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\data_grouped\group_fast\train.tfrecords` (tương ứng), `FLAGS.output_dir=C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\outputDQN_Fast_73\checkpoints` (tương ứng, tuyệt đối như task 4).
2. Giữ `hidden_size 32, gamma 0.99, waste 0.025, lr 0.001, batch 32, train_episodes 600, num_timesteps 900` như gốc.
3. Mỗi file train riêng, không loop 3 nhóm trong 1 file.

---

## Lệnh train (mỗi run 8-12 phút vì 73 SKU < 220, 600ep x 900 steps)

```bash
# Chuẩn bị data 3 nhóm (đã chạy xong, 30 sec, data_grouped/group_*/ đã có 4 TFRecords + 4 CSV)
# Đã sinh: group_fast 73 SKU (train 1000x73), group_medium 73, group_slow 74

# Train DQN 3 nhóm (3 runs) - dùng đường dẫn tuyệt đối như task 4
python Training/train_task5_grouped.py --algorithm DQN --num_products 73 --train_file "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\data_grouped\group_fast\train.tfrecords" --output_dir "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\outputDQN_Fast_73\checkpoints" --train_episodes 600

python Training/train_task5_grouped.py --algorithm DQN --num_products 73 --train_file "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\data_grouped\group_medium\train.tfrecords" --output_dir "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\outputDQN_Medium_73\checkpoints" --train_episodes 600

python Training/train_task5_grouped.py --algorithm DQN --num_products 74 --train_file "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\data_grouped\group_slow\train.tfrecords" --output_dir "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\outputDQN_Slow_73\checkpoints" --train_episodes 600

# Train A2C_mod 3 nhóm (3 runs)
python Training/train_task5_grouped.py --algorithm A2C_mod --num_products 73 --train_file "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\data_grouped\group_fast\train.tfrecords" --output_dir "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\outputA2C_Fast_73\checkpoints" --train_episodes 600

python Training/train_task5_grouped.py --algorithm A2C_mod --num_products 73 --train_file "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\data_grouped\group_medium\train.tfrecords" --output_dir "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\outputA2C_Medium_73\checkpoints" --train_episodes 600

python Training/train_task5_grouped.py --algorithm A2C_mod --num_products 74 --train_file "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\data_grouped\group_slow\train.tfrecords" --output_dir "C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task12-9\task 5\outputA2C_Slow_73\checkpoints" --train_episodes 600
```

Tổng Task 5: **6 runs x 10 phút ≈60 phút CPU / 30 phút GPU**, output `outputDQN_Fast_73/checkpoints/ckpt-64` + `logs/` + CSV mỗi run (đã tạo data_grouped sẵn, chỉ cần train).

---

## File phân tích analysis_task5.ipynb

Sau khi train xong 6 runs, file `analysis_task5.ipynb` sẽ:

1. Load 6 checkpoints + logs (`outputDQN_Fast_73/checkpoints/ckpt-64`...) + baseline 220 (`checkpoints_dqn_comparison512_32/ckpt-43`), trích metrics `reward, stockout, overstock, waste` như `training.py:331-336`.
2. Chạy evaluation trên `data_grouped/group_*/test.tfrecords` (504 periods, đã có) để tính **Kết quả 3 SKU groups ở cả 2 agents** - bảng so sánh performance giữa Fast/Medium/Slow vs baseline 220 cho DQN và A2C_mod.
3. Chạy SHAP Top-k per group (như `topk_shap_analysis.ipynb` nhưng với 73*3=219 features) để xem pattern `Sales dominant` có giữ ở từng group không.

**Deliverable Task 5:** `analysis_task5.ipynb` output `outputDQN_Fast_73/comparison_3groups.csv` (hoặc `task 5/output/comparison_3groups.csv` chung) + figure `performance_vs_sku_group.png`.

---

## Thứ tự thực hiện

1. Tạo 6 file train `.ipynb` tách riêng (copy structure, chỉ đổi num_products và group)
2. Chạy `prepare_grouped_task5.ipynb` để sinh 3 nhóm data (30 sec)
3. Chạy 6 runs train (có thể song song 2 runs)
4. Chạy `analysis_task5.ipynb` để sinh bảng so sánh

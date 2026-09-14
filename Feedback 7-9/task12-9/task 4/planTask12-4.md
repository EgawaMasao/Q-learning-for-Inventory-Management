# Plan Task 4: Action Space Sensitivity 7 / 14 / 28 (Chiều sâu)

> **Workstream:** Action space  
> **Liên quan Reviewer:** R1 #1 (14 portfolio-level strategies oversimplify)  
> **Yêu cầu Task 12-9.md:3:** Chạy sensitivity/ablation với độ phân giải action space khác nhau, ví dụ 7 / 14 / 28 strategies. Type: Experiment, Requires Retrain: Yes, Recommend Scope: Strong, Deliverable: Bảng so sánh performance + explanation robustness  
> **Nguyên tắc:** Không thay đổi file cũ `Training/Train_DQN.ipynb` và `Training/A2C-mod.ipynb` hay `training.py`. Tạo file mới copy structure, chỉ đổi `num_actions` và `action_space`. Tách riêng 6 file, không gộp. Kết quả y chang Training gốc (checkpoint + logs JSON/CSV).

---

## Ngữ cảnh

Hiện tại `Xai_Inventory_Submit_17Mar.md:298` và `training.py:201` dùng 14 actions portfolio-level `[0,0.005,0.01,0.0125,0.015,0.0175,0.02,0.03,0.04,0.08,0.12,0.2,0.5,1.0]` cho cả 220 SKU cùng 1 action. Reviewer cho rằng oversimplify - cần test 7 (coarse) và 28 (fine) để chứng minh reported performance và explanation patterns có robust với action granularity không. Task 4 làm theo chiều sâu (đổi độ phân giải action, giữ nguyên 220 SKU, data cũ).

14 đã có checkpoint `checkpoints_dqn_comparison512_32/ckpt-43` và `outputA2Cmod/ckpt-64`, theo yêu cầu mới **không train lại 14**, chỉ train 7 và 28 (4 runs) và dùng checkpoint 14 sẵn làm baseline để so sánh 7/14/28.

---

## Cấu trúc thư mục task 4 (6 file tách riêng, không gộp)

```
Feedback 7-9/task12-9/task 4/
├── Train_DQN_7.ipynb          # DQN với 7 actions, 220 SKU, 600ep - train lại từ đầu
├── Train_A2C_mod_7.ipynb      # A2C_mod với 7 actions, 220 SKU - train lại từ đầu
├── Train_DQN_28.ipynb         # DQN với 28 actions, 220 SKU - train lại từ đầu
├── Train_A2C_mod_28.ipynb     # A2C_mod với 28 actions, 220 SKU - train lại từ đầu
├── outputDQN_7/               # Kết quả DQN 7: checkpoints/ + logs/ như outputA2Cmod (ckpt-64 + logs JSON/CSV)
│   ├── checkpoints/
│   └── logs/
├── outputA2C_7/               # Kết quả A2C_mod 7: checkpoints/ + logs/
│   ├── checkpoints/
│   └── logs/
├── outputDQN_28/              # Kết quả DQN 28: checkpoints/ + logs/
│   ├── checkpoints/
│   └── logs/
├── outputA2C_28/              # Kết quả A2C_mod 28: checkpoints/ + logs/
│   ├── checkpoints/
│   └── logs/
└── analysis_task4.ipynb       # File phân tích sau train: Bảng so sánh performance (reward, service, holding, waste, ordering, stockout) + explanation robustness (RDX/MSX composition, SHAP FCS/CAS) giữa 3 action spaces (7, 14 baseline đã có từ checkpoints_dqn_comparison512_32/ckpt-43 và outputA2Cmod/ckpt-64, + 28 mới) của cả 2 agents
```

**Lưu ý:** Tách riêng 4 file như bạn yêu cầu (1 file train 7 action cho DQN, 1 file train 7 action cho A2C_mod, tương tự cho 28), **mỗi file train lại từ đầu**, không train lại 14 vì đã có checkpoint 14 baseline sẵn. Mỗi output tách thành thư mục riêng `outputDQN_7`, `outputA2C_7`, `outputDQN_28`, `outputA2C_28` gồm `checkpoints/` + `logs/` y chang `outputA2Cmod`.

---

## Thiết kế action space 7 / 14 / 28

| Action space | Số bins | Array (normalized to capacity [0,1]) | Ý nghĩa |
|---|---|---|---|
| **7 (coarse)** | 7 | `[0, 0.01, 0.02, 0.04, 0.12, 0.5, 1.0]` | Thô, bỏ granularity mịn 0-0.02, chỉ giữ extremes + median |
| **14 (baseline)** | 14 | `[0, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.5, 1.0]` | Hiện có, non-uniform 6 bins mịn 0-0.02 rồi nhảy |
| **28 (fine)** | 28 | `[0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0]` | Mịn, tăng granularity ở 0-0.2 |

*Giữ non-uniform như baseline để phản ánh thực tế replenishment: nhiều mức nhỏ để fine-tune, ít mức lớn để tránh overstock.*

---

## Thay đổi code chi tiết (không sửa file cũ)

**Copy `training.py:201-305` và `Training/A2C-mod.ipynb:ec4f3b3e` vào 6 file mới, chỉ đổi:**

1. `training.py:37 Actor layer4 (32→num_actions)`, `267 TensorArray [P, num_actions]`, `367 p_batch reshape`, `426 p_new=softmax(log p + delta/|i-a|+1)` - tự adapt qua `FLAGS.num_actions`.
2. `training.py:202 predict() action_space=tf.tile([...14 vals...])` và `305 train() action_space=tf.tile([...])` - thay bằng `if FLAGS.num_actions==7: ACTION_SPACE=np.array([0,0.01,...]) elif 28: [...]` .
3. `Training/Train_DQN.ipynb:138 DQNAgentRDX(num_actions=7/14/28)` và `A2CStyleInventoryEnv.action_space=np.array([...])`.
4. `Training/A2C-mod.ipynb:ec4f3b3e FLAGS.num_actions=7/14/28` và `f3022a91 action_space=tf.tile([...])`.

**Giữ nguyên:** `num_products 220`, `num_features 3`, `hidden_size 32`, `train_episodes 600`, `num_timesteps 900`, `batch_size 32`, `gamma 0.99`, `waste 0.025`, `actor_lr 0.001`, `critic_lr 0.001`, `entropy 0.001`, `dropout 0.1`.

---

## Lệnh train (mỗi run 10-15 phút CPU, 600ep x 900 steps = 540k steps)

```bash
# DQN 7 - train lại từ đầu
python Training/train_task4_action_space.py --algorithm DQN --num_actions 7 --output_dir "Feedback 7-9/task12-9/task 4/outputDQN_7/checkpoints" --train_episodes 600 --num_products 220 --batch_size 32 --hidden_size 32 --train_file data/train.tfrecords
# log -> outputDQN_7/logs/

# A2C_mod 7 - train lại từ đầu
python Training/train_task4_action_space.py --algorithm A2C_mod --num_actions 7 --output_dir "Feedback 7-9/task12-9/task 4/outputA2C_7/checkpoints" --train_episodes 600 --num_products 220
# log -> outputA2C_7/logs/

# DQN 28 - train lại từ đầu
python Training/train_task4_action_space.py --algorithm DQN --num_actions 28 --output_dir "Feedback 7-9/task12-9/task 4/outputDQN_28/checkpoints" --train_episodes 600 --num_products 220

# A2C_mod 28 - train lại từ đầu
python Training/train_task4_action_space.py --algorithm A2C_mod --num_actions 28 --output_dir "Feedback 7-9/task12-9/task 4/outputA2C_28/checkpoints" --train_episodes 600

# DQN 14 và A2C_mod 14 không train lại, dùng checkpoint sẵn có:
# DQN 14: checkpoints_dqn_comparison512_32/ckpt-43
# A2C_mod 14: outputA2Cmod/checkpoints_a2cmod/ckpt-64
```

Tổng Task 4: **4 runs (7 và 28 cho cả 2 agents) x 15 phút ≈60 phút CPU / 30 phút GPU** (không train lại 14, dùng checkpoint sẵn), output `ckpt-64` (40KB) + logs JSON/CSV (7KB) mỗi run như `outputA2Cmod/logsA2Cmod`.

---

## File phân tích analysis_task4.ipynb

Sau khi train xong 6 runs, file `analysis_task4.ipynb` sẽ:

1. Load 6 checkpoints + logs, trích metrics `reward, stockout, overstock, waste, quantile` như `training.py:331-336` `r = 1 - z - overstock - q - quan`.
2. Chạy lại evaluation trên `data/test.tfrecords:504 periods` với `496 test cycles` như `Xai_Inventory_Submit_17Mar.md:721` để tính **Bảng so sánh performance** (total reward, service, holding, waste, ordering, stockout) giữa 7/14/28 cho DQN và A2C_mod.
3. Chạy lại RDX/MSX và SHAP (như `Ablation_Study/ablation_RDX.ipynb` và `XAI/SHAP-temp.ipynb`) trên 3 action spaces để tính **explanation robustness** (RDX composition, MSX size, SHAP FCS) - kiểm tra pattern có đổi không khi bins mịn/thô.

**Deliverable Task 4:** `analysis_task4.ipynb` output `task 4/output/comparison_table_7_14_28.csv` + figure `performance_vs_action_granularity.png`.

---

## Thứ tự thực hiện

1. Tạo 6 file train `.ipynb` tách riêng như trên (copy structure, chỉ đổi num_actions)
2. Chạy 6 runs train (có thể song song 2 runs cùng lúc)
3. Chạy `analysis_task4.ipynb` để sinh bảng so sánh

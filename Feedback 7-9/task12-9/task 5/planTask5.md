# Plan Task 5: SKU Group Scalability (Product-Group Level) - Chi tiết

> **Workstream:** Action space  
> **Task ID:** 5 — `Feedback 7-9/task12-9/Task 12-9.md:9-15`  
> **Nguồn dữ liệu đã có:** `Training/A2C-mod.ipynb:70-82` (FLAGS `num_products=220`, `num_features=3`, `hidden_size=32`), `Training/DQN.ipynb` (DQN `num_products=220` mirror), `prepare_data.py:30-56` (`sales_example`, `capacity_example`, `stock_example`), `Feedback 7-9/task12-9/task 5/prepare_grouped_task5.ipynb:1-55` (chia 220 SKU theo MeanDemand → Fast 73/Medium 73/Slow 74), `Feedback 7-9/task12-9/task 5/data_grouped/group_*/train|test|capacity|stock.tfrecords` (1000×73, đã sinh), `Feedback 7-9/task12-9/task 5/outputA2C_*_73/checkpoints` (A2C ckpt-60..63) + `outputDQN_*_73/checkpoints` (ckpt-61), `Ablation_Study/faithfulness/` và `XAI/SHAP-temp.ipynb` (không liên quan trực tiếp, chỉ tham khảo)  
> **Nguyên tắc chung:** Ưu tiên **không mock/dummy** — chỉ dùng checkpoint/logs thật đã train xong. So sánh **chỉ nội bộ 3 groups** (không so baseline 220) theo yêu cầu mới. Kết quả ghi tiếng Việt, sẵn sàng copy vào Supplementary sau khi approve.

---

## Ngữ cảnh chung & Mục tiêu

Bài báo hiện tại train trên toàn bộ 220 SKU (`data/train.tfrecords` 1000 periods). Reviewer Workstream *Action space* yêu cầu chứng minh framework **không hardcode 220** và mở rộng được xuống *product-group level* (ví dụ theo velocity/nhóm ngành hàng). Nếu 220 SKU chia thành 2–3 nhóm (Fast/Medium/Slow) mà cả A2C-mod và DQN đều train được và cho learning curves hội tụ khác biệt nhưng hợp lý, thì tính *scalable* được chứng minh — quan trọng cho triển khai thực tế (mỗi cửa hàng chỉ quản 70–80 SKU chủ lực).

`Task 12-9.md:10` ghi: *“Nếu khả thi, thêm experiment chia ít nhất 2–3 nhóm SKU để chứng minh framework mở rộng được xuống product-group level.”* — **Strong scope**, `Requires Retrain: Yes` đã được thực hiện trước (6 notebooks train riêng).

**Cách thực thi Task 5:** Không cần retrain thêm — chỉ cần **evaluation & so sánh**: (1) A2C trên 3 groups, (2) DQN trên 3 groups, (3) A2C vs DQN trong từng group. Kết quả tách rõ 3 bảng để đáp ứng yêu cầu "so sánh gồm cả A2C trong 3 trường hợp SKU, DQN trong 3 trường hợp SKU và A2C vs DQN trong 3 trường hợp SKU".

---

## Task 5: Chia 2–3 nhóm SKU — So sánh nội bộ 3 groups

### 1. Yêu cầu trong `Task 12-9.md:9-15`

> **Task:** Nếu khả thi, thêm experiment chia ít nhất 2–3 nhóm SKU để chứng minh framework mở rộng được xuống product-group level.  
> **Type:** Experiment  
> **Requires Retrain:** Yes  
> **Recommend Scope:** Strong  
> **Deliverable:** Kết quả 2–3 SKU groups

### 2. Hiện trạng

*   **Data đã chia (không mock):**
    *   `prepare_grouped_task5.ipynb:44-54` đọc `data/train.tfrecords` 220 SKU, tính `mean_demand = train_sales_220.mean(axis=0)` (`:46`), `sorted_idx = argsort()[::-1]` → `group_fast_idx = sorted_idx[:73]` (top 73, SKU57 mean 22.6), `group_medium_idx = sorted_idx[73:146]` (mid 73, SKU100 mean 6.25), `group_slow_idx = sorted_idx[146:]` (bottom 74, SKU64 mean 0.66 CV 1.46). 
    *   `prepare_grouped_task5.ipynb:97-114` sinh `data_grouped/group_{fast,medium,slow}/train|test|capacity|stock.tfrecords` qua `prepare_data.capacity_example`/`stock_example`/`sales_example`. Đã verify: `train.tfrecords` 330000/334000 bytes, `test.tfrecords` 166320/168336 bytes (`task 5/data_grouped/group_*/`).
    *   Lưu ý: Slow = 74 (220 không chia hết cho 3) — giữ nguyên để không mất SKU, cần ghi chú trong báo cáo.

*   **Training đã xong (6 models, 600 episodes × 900 timesteps, 14 actions):**
    *   A2C: `Train_A2C_mod_Fast_73.ipynb:3` `FLAGS.num_products=73/74`, `num_features=219/222`, `output_dir=.../outputA2C_*_73/checkpoints` — checkpoint `ckpt-63` (Fast, 63 files) + `ckpt-60` (Medium/Slow, 60 files). Logs: Medium/Slow có `training_summary_*.json` 600 eps, Fast reconstruct từ `training_log_*_episode_*.json` 639 files → dedup 460 unique (missing 461-600) — cảnh báo trong notebook.
    *   DQN: `Train_DQN_*_73.ipynb:3` `num_products=73/74`, `hidden_size=128`, `GroupNormalization` — checkpoint `ckpt-61` (5 files cuối, save_every=10) nhưng `training_summary_*.json` đủ 600 eps, `reward/stockout/waste/overstock/quantile/loss` đầy đủ.

*   **Thư mục liên quan:**
    *   `Training/A2C-mod.ipynb:122-162` và `Training/DQN.ipynb:132-153` là template gốc — liên quan **cao**, dùng để adapt FLAGS và parsers `FixedLenFeature([N])`.
    *   `Ablation_Study/faithfulness/topk_shap_analysis.ipynb` + `XAI/SHAP-temp.ipynb` — liên quan **thấp**, phục vụ XAI robustness (Task 11-9), không cần cho scalability SKU. Chỉ tham khảo nếu muốn chạy SHAP per-group sau này.

*   **Gap cho deliverable:**
    *   Chưa có bảng tổng hợp so sánh cross-group & cross-algo (3 bảng yêu cầu).
    *   Chưa có figure convergence và thống kê improvement.

### 3. Hướng giải quyết chi tiết (Không retrain, chỉ evaluation)

**Phương án đã chọn (đã implement trong `evaluate_task5_SKU_groups.ipynb`):** Dùng logs thật + checkpoints thật, không mock.

**Bước 5.1 - Load training summaries (không mock):**
```python
# A2C Medium/Slow: training_summary_*.json (600 eps)
# A2C Fast: reconstruct từ per-episode json (dedup theo episode, lay file timestamp lon nhat)
# DQN 3 groups: training_summary_*.json (reward/stockout/waste/overstock/quantile)
df_a2c_fast, _ = reconstruct_a2c_fast_from_logs("outputA2C_Fast_73/logs")  # 460 eps
df_a2c_medium = load_a2c_summary_from_file("outputA2C_Medium_73/logs/training_summary_*.json")  # 600 eps
df_a2c_slow = load_a2c_summary_from_file("outputA2C_Slow_73/logs/training_summary_*.json")  # 600 eps
df_dqn_fast/med/slow = load_dqn_summary("outputDQN_*_73/logs/training_summary_*.json")  # 600 eps each
```
*   Chuẩn hóa cột: `reward = rewards_mean`, `stockout = stockouts_mean`, `waste = waste_mean` để khớp DQN.
*   Gán `group = Fast/Medium/Slow`, `algo = A2C_mod/DQN`.

**Bước 5.2 - Tạo 3 bảng so sánh (chỉ nội bộ 3 groups):**
*   **Bảng 1 — A2C trên 3 groups:** `summarize_a2c(df)` tính `Reward_all_mean`, `Reward_last100_mean/std`, `Reward_best`, `Stockout_last100`, `Waste_last100`, `CriticLoss_last100`, `ActorLoss_last100` cho Fast (460 eps, last100=361-460), Medium (600 eps), Slow (600 eps).
*   **Bảng 2 — DQN trên 3 groups:** `summarize_dqn(df)` tính thêm `Overstock_last100`, `Quantile_last100`, `Loss_last100`.
*   **Bảng 3 — A2C vs DQN per group:** Merge theo Group, tính `Reward_delta = A2C−DQN`, `Stockout_delta`, `Waste_delta`, `Winner_reward` (DQN thắng cả 3 groups trong logs hiện tại: Δ -0.59 Fast, -0.60 Medium, -0.42 Slow).

**Bước 5.3 - Figures (6 PNG, `matplotlib Agg`):**
*   `fig_A2C_reward_3groups.png` — 3 curves A2C Fast/Medium/Slow (episode vs reward)
*   `fig_DQN_reward_3groups.png` — 3 curves DQN
*   `fig_A2C_vs_DQN_per_group.png` — 6 curves solid=A2C dashed=DQN
*   `fig_bar_reward_last100.png` + `fig_bar_stockout_last100.png` + `fig_bar_waste_last100.png` — bar chart per group

**Bước 5.4 - Thống kê bổ sung:**
*   `improvement(df)` = `last100 - first50` và `pct` để chứng minh học được: DQN Fast +722%, Medium +2758%, Slow +5448%; A2C Fast +1.48%, Slow -0.5%, Medium -79% (suy giảm cần ghi nhận).

**Tại sao chọn hướng này?**
*   **Không retrain:** Cả 6 models đã train xong (checkpoint + 600 eps logs), chỉ cần evaluation → tiết kiệm ngày train lại.
*   **Không mock:** Đọc file thật `data_grouped/*.tfrecords` và `training_summary_*.json`, code `np.mean(rewards)` trên mảng thật.
*   **Đáp ứng yêu cầu mới:** "chỉ so sánh nội bộ 3 groups" → không so baseline 220, tách rõ 3 bảng A2C-3groups / DQN-3groups / A2CvsDQN-per-group.
*   **Xử lý thiếu dữ liệu:** Báo cáo rõ Fast chỉ 460/600 eps dù ckpt-63 tồn tại; Slow 74 SKU — trung thực, tăng độ tin cậy với reviewer.

**Deliverable Task 5:**
*   Notebook `task 5/evaluate_task5_SKU_groups.ipynb` (9 code cells, đã chạy verified `run_eval2.py` → `output_grouped/`)
*   `task 5/output_grouped/table1_A2C_3groups.csv` (700 bytes, 3 rows: Fast 460 eps, Medium/Slow 600 eps)
*   `task 5/output_grouped/table2_DQN_3groups.csv` (762 bytes)
*   `task 5/output_grouped/table3_A2C_vs_DQN_per_group.csv` (774 bytes, DQN thắng cả 3 groups)
*   `task 5/output_grouped/table_stats_improvement.csv` (569 bytes)
*   `task 5/output_grouped/figures/` (6 PNG, 41-309KB)

---

## Kế hoạch thực thi gộp & Tách kết quả

**Thực thi gộp:** Một notebook `evaluate_task5_SKU_groups.ipynb` đọc cả 6 logs + 3 data_grouped, cùng background `train/test.tfrecords` đã chia, chạy 3 wrappers summary (`load_a2c_summary`, `reconstruct_a2c_fast`, `load_dqn_summary`) với `pandas` + `matplotlib` trong ~1 phút (không cần GPU). Không phụ thuộc `Ablation_Study`/`XAI`.

**Nhưng kết quả tách rõ 3 phần** trong file output để review:
```
## Bảng 1: A2C trên 3 SKU groups (Fast/Medium/Slow)
... Reward_all/last100, Stockout, Waste, CriticLoss ...

## Bảng 2: DQN trên 3 SKU groups
... Reward, Stockout, Waste, Overstock, Quantile, Loss ...

## Bảng 3: A2C vs DQN per group
... delta reward/stockout/waste, Winner per group ...

## Thống kê improvement first50 vs last100
... DQN +722%..+5448%, A2C Fast +1.48% ...
```

---

## File sẽ tạo (khi build)

1. `Feedback 7-9/task12-9/task 5/planTask5.md` (file này)
2. `Feedback 7-9/task12-9/task 5/evaluate_task5_SKU_groups.ipynb` (đã tạo, 9 code cells, verified)
3. `Feedback 7-9/task12-9/task 5/output_grouped/table1_A2C_3groups.csv`
4. `Feedback 7-9/task12-9/task 5/output_grouped/table2_DQN_3groups.csv`
5. `Feedback 7-9/task12-9/task 5/output_grouped/table3_A2C_vs_DQN_per_group.csv`
6. `Feedback 7-9/task12-9/task 5/output_grouped/table_stats_improvement.csv`
7. `Feedback 7-9/task12-9/task 5/output_grouped/figures/fig_A2C_reward_3groups.png`
8. `Feedback 7-9/task12-9/task 5/output_grouped/figures/fig_DQN_reward_3groups.png`
9. `Feedback 7-9/task12-9/task 5/output_grouped/figures/fig_A2C_vs_DQN_per_group.png`
10. `Feedback 7-9/task12-9/task 5/output_grouped/figures/fig_bar_reward_last100.png` (+ stockout/waste)

---

## Phân tích thư mục liên quan (tóm tắt)

| Thư mục | Liên quan | Vì sao |
|---------|-----------|--------|
| `Training/A2C-mod.ipynb:70-82`, `Training/DQN.ipynb` | **Cao** | Template gốc FLAGS, parsers `FixedLenFeature([220])` → adapt sang 73/74 là core của scalability |
| `Feedback 7-9/task12-9/task 5/prepare_grouped_task5.ipynb` | **Cao** | Đã chia data, sinh `data_grouped` — artifact đầu vào |
| `Feedback 7-9/task12-9/task 5/data_grouped` | **Cao** | Input thực cho 6 models |
| `Feedback 7-9/task12-9/task 5/outputA2C_*_73` + `outputDQN_*_73` | **Cao** | Checkpoint/logs thật, nguồn bảng |
| `Ablation_Study/faithfulness/` | **Thấp** | XAI faithfulness, không cần cho Task 5 |
| `XAI/SHAP-temp.ipynb` | **Thấp** | SHAP comparability Task 11-9, optional nếu muốn SHAP per-group sau |

---

## Câu hỏi đã trả lời

*   **Task yêu cầu làm gì?** Chia 220 SKU thành 2–3 nhóm (Fast/Medium/Slow) và chứng minh framework mở rộng được xuống product-group level — Strong scope, đã retrain 6 models.
*   **Hướng xử lý đề xuất?** Không retrain thêm, chỉ evaluation trên logs/checkpoints thật → 3 bảng A2C-3groups / DQN-3groups / A2CvsDQN-per-group + 6 figures, code `reconstruct_a2c_fast_from_logs` dedup episode.
*   **Kết quả chứng minh gì?** Cả A2C và DQN đều train được với `num_products` linh hoạt 73/74, cùng pipeline TFRecords — framework scalable. DQN thắng A2C cả 3 groups về reward hội tụ (Δ -0.42..-0.60), nhưng A2C vẫn hội tụ (Fast/Slow ổn định, Medium suy giảm cần lưu ý).
*   **Có giải quyết được task không?** Có — đã có 3 groups × 2 algos, đủ deliverable `Kết quả 2–3 SKU groups`, chỉ cần báo cáo trung thực thiếu 140 eps Fast và 74 SKU Slow.

---

## Thứ tự thực hiện

1. Viết `planTask5.md` (file này) — đã xong
2. Chạy `evaluate_task5_SKU_groups.ipynb` (đã verified qua `run_eval2.py`, ~1 phút, sinh 4 CSV + 6 PNG trong `output_grouped/`)
3. Review 3 bảng tiếng Việt trong `output_grouped/` với mentor
4. Khi approve, copy bảng/figures vào Supplementary `Xai_Inventory_Submit_*.md` Section *Scalability* và ghi chú Fast 460/600 + Slow 74 SKU


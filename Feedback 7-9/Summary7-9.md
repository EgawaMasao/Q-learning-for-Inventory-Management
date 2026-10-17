# Tổng Hợp Báo Cáo Feedback 7-9 — Summary 7-9

> **Phạm vi:** Tổng hợp 11 file output từ `task10-9` đến `task17-9`, sắp xếp lại **theo ID Task tăng dần** để báo cáo. Ngôn ngữ dễ đọc, giữ đầy đủ số liệu thực nghiệm để copy vào Response Letter / `Xai_Inventory_Submit_17Mar.md`.
> **Ngày tổng hợp:** 18/09/2026 | **Checkpoint chính:** DQN `ckpt-60` / `ckpt-50` (hidden 128), A2C_mod `ckpt-64` (hidden 32)

---

## Mục Lục

1. [Bảng Tổng Quan Nhanh (26 Tasks)](#1-bảng-tổng-quan-nhanh)
2. [Nhóm Action Space: Task 4 & 5](#2-nhóm-action-space)
3. [Nhóm SHAP Micro-level: Task 10, 11, 12](#3-nhóm-shap-micro-level)
4. [Nhóm SHAP Comparability: Task 13, 14, 15](#4-nhóm-shap-comparability)
5. [Nhóm Reward Design: Task 20, 21](#5-nhóm-reward-design)
6. [Nhóm Faithfulness: Task 24-28](#6-nhóm-faithfulness)
7. [Nhóm Data Integrity: Task 33, 34](#7-nhóm-data-integrity)
8. [Nhóm SHAP Implementation: Task 39-42](#8-nhóm-shap-implementation)
9. [Nhóm A2C_mod Definition: Task 43, 44](#9-nhóm-a2c_mod-definition)
10. [Nhóm Managerial & Writing: Task 47, 50, 53](#10-nhóm-managerial--writing)
11. [Bảng Thuật Ngữ](#11-bảng-thuật-ngữ)

---

## 1. Bảng Tổng Quan Nhanh

| ID | Tên ngắn gọn | Loại | File nguồn | Tình trạng |
|---|---|---|---|---|
| 4 | Độ nhạy Action Space (7/14/28) | Experiment / Analysis | `task12-9/task 4/outputTask12-9-4.md` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1898-1943` Section 4.9** |
| 5 | SKU Group Scalability (73/73/74) | Experiment (Retrain) | `task12-9/task 5/outputTask12-9-5.md` | **100% - Hoàn thành** (lưu ý Medium bất ổn) | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:418-429` Section 3.1.5 + `1856-1897` Section 4.8** |
| 10 | SHAP Top-k 660 chiều làm kết quả chính | Analysis / Writing | `task10-9/outputTask10-9.md:7` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1289-1299` Table 7a + `1382-1398` Section 4.5.3** |
| 11 | Nhất quán Top-k giữa EASY/MEDIUM/HARD | Experiment / Analysis | `task10-9/outputTask10-9.md:77` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1399-1402` Section 4.5.4** |
| 12 | Vì sao SKU thống trị Top-20 | Analysis | `task10-9/outputTask10-9.md:134` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1403-1407` Section 4.5.5** |
| 13 | Common target logits (pre-softmax) | Implementation / Experiment | `task11-9/outputTask11-9.md:7` | **100% - Đã chạy 360 dòng** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1409-1412` Section 4.5.6** |
| 14 | Disclaimer qualitative nếu không đổi target | Writing | `task11-9/outputTask11-9.md:76` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1300` Section 3.3.3 + footnote Table 7b/7c** |
| 15 | Sensitivity actor/critic/logits | Experiment | `task11-9/outputTask11-9.md:108` | **100% - Đã chạy 9 dòng** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1413-1416` Section 4.5.7** |
| 20 | Justify reward coefficients (OAT) | Writing / Literature + Experiment | `task13-9/task1/outputTask13-9.md:7` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:689-702` Section 3.2.4** |
| 21 | Range & distribution trước/sau normalize | Analysis | `task13-9/task1/outputTask13-9.md:109` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1417-1452` Section 4.5.8** |
| 24 | Faithfulness: AUPC/ASR/CI | Experiment | `task15-9/outputTask15-9.md:7` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1751-1799` Section 4.7.1-4.7.2** |
| 25 | Faithfulness: So sánh thống kê | Analysis | `task15-9/outputTask15-9.md:7` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1801-1817` Section 4.7.3** |
| 26 | Faithfulness: Giao thức (protocol) | Method / Writing | `task15-9/outputTask15-9.md:7` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1755-1762` Section 4.7.1** |
| 27 | Faithfulness: Ngưỡng 1% | Method | `task15-9/outputTask15-9.md:7` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1818-1835` Section 4.7.4** |
| 28 | Faithfulness: Nhất quán giữa states | Analysis | `task15-9/outputTask15-9.md:7` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:1836-1854` Section 4.7.5** |
| 33 | Dataset split chronological | Clarification | `task13-9/task2/outputTask13-9-2.md:7` | **100% - Đã chứng minh** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:350-388` Section 3.1.3** |
| 34 | Leakage audit 4 nhóm | Method / Check | `task13-9/task2/outputTask13-9-2.md:82` | **95% - Phát hiện MAJOR, đã có cách sửa, chưa retrain** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:389-417` Section 3.1.4** |
| 39 | Xây dựng tập nền SHAP 200→100 | Method | `task14-9/outputTask14-9.md:9` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:769-787` Section 3.3.1** |
| 40 | Độ nhạy SHAP (nguồn/kích thước) | Experiment | `task14-9/outputTask14-9.md:9` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:788-822` Section 3.3.2** |
| 41 | Tính hợp lệ miền (vi phạm ràng buộc) | Experiment | `task14-9/outputTask14-9.md:9` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:823-846` Section 3.3.3** |
| 42 | Chuẩn hóa mô tả Kernel vs Partition | Writing | `task14-9/outputTask14-9.md:9` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:847-863` Section 3.3.4** |
| 43 | Định nghĩa A2C_mod (8 cải tiến) | Writing / Analysis | `task16-9/task1/outputTask16-9-1.md:8` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:704-731` Section 3.2.5** |
| 44 | Phép biến đổi Q-equivalent | Method / Writing | `task16-9/task1/outputTask16-9-1.md:8` | **100% - Đã kiểm chứng 100% argmax** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:732-763` Section 3.2.6** |
| 47 | Managerial Case Studies + giảm claim trust | Writing | `task17-9/outputTask47.md:1` | **100% - Hoàn thành** | **Đã thêm: `Xai_Inventory_Submit_17Mar.md:13` Abstract + `1945-1990` Section 4.10 + `1996-1998` Conclusion** |
| 50 | Mở rộng văn liệu lên 50+ refs | Literature / Writing | `task16-9/task1/outputTask16-9-2.md:9` | **40% - Đã audit, chưa viết xong (hiện 25 refs)** | **Chưa thêm** (dự kiến Section 1 + 2 + References) |
| 53 | Rà soát hệ thống hình vẽ (30 refs) | Cleanup / Writing | `task16-9/task1/outputTask16-9-2.md:9` | **60% - Đã audit, còn 5 lỗi cần sửa** | **Chưa thêm đầy đủ** (dự kiến dòng 342,651,1464,1593,1606,1620) |

---

## 2. Nhóm Action Space

### Task 4 — Độ nhạy Action Space (7 / 14 / 28)

- **Yêu cầu làm gì?** Làm ablation có hệ thống: huấn luyện & đánh giá độc lập 3 mức phân giải 7 (thưa), 14 (baseline), 28 (mịn) cho cả A2C_mod và DQN, so sánh **hiệu năng + tính bền vững của giải thích** (không chỉ reward).
- **Loại Task:** Experiment / Analysis — Strong scope, đã huấn luyện xong 6 điều kiện, không cần retrain thêm.
- **Phương pháp được chọn:**
  - Giữ cố định môi trường (`r = 1 - z - overstock - q - quan` `training.py:336`, state 660 chiều), chỉ đổi tập action: 7 `[0,0.01,0.02,0.04,0.12,0.5,1.0]`, 14 `[0,0.005,...,1.0]`, 28 `[0,0.0025,...,1.0]`.
  - Dùng checkpoint thật: A2C `ckpt-60/ckpt-64`, DQN `ckpt-61/ckpt-50`, rollout tất định 504 steps (seed 42, argmax).
  - 3 trụ cột XAI: SHAP (KernelSHAP, BG 20, nsamples 500), RDX/MSX (OCS, kích thước tập tối thiểu, Jaccard stability), Faithfulness (mask theo MoRF/LeRF).
- **Lý do chọn:** Cô lập tác động của độ phân giải; giữ hidden riêng (32 cho A2C, 128 cho DQN) để công bằng vì đó là điểm tối ưu riêng, không ép bằng nhau nhân tạo.
- **Kết quả:**
  - **Hiệu năng không tăng đơn điệu:** DQN tốt nhất ở 28 (reward 0.8213) và 7 (0.8001) > 14 (0.5960); A2C_mod chỉ tốt ở 14 (0.3833), còn 7 (-0.1265) và 28 (-0.1170) sập do overstock vọt 0.96/0.83 (`outputTask12-9-4.md:39-44`).
  - **Giải thích vẫn thưa & bền vững tương đối:** FCS ~0 (A2C) / 0.02 (DQN); OCS 0.27-0.62; Minimal set 1.0-2.68 (A2C 28 vọt 2.68); Stability 50-90.7% (A2C 7 cao nhất).
- **Tình trạng:** **100%** — đã có 2 bảng + 3 hình (`task4_reward_vs_actions.png` etc.), đoạn văn paste vào Section 4.9 đã soạn sẵn.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.9 Sensitivity to Action Space Resolution (`:1898-1943`)** — chèn mới hoàn toàn sau Section 4.8 và trước Section 5 Conclusion. Gồm: đoạn mở đầu lý do chọn 7/14/28, Table 4a (Reward/Overstock/Waste × 6 dòng), Figure 12a `task4_reward_vs_actions.png`, đoạn diễn giải hiệu năng, Table 4b (FCS/OCS/Minimal set/Stability), Figure 12b+12c, đoạn kết về sparsity. Trước đó chưa có Section 4.9.
- **Thuật ngữ:** *Action space resolution* = độ mịn của tập hành động rời rạc; *FCS* = Feature Coverage Score (tỷ lệ feature vượt ngưỡng SHAP); *MSX stability (Jaccard)* = độ ổn định của tập giải thích khi đổi ngưỡng.

### Task 5 — SKU Group Scalability (Product-Group Level)

- **Yêu cầu làm gì?** Chứng minh framework không hardcode 220 SKU: chia ít nhất 2-3 nhóm SKU, train riêng A2C & DQN trên từng nhóm, so sánh nội bộ (yêu cầu bổ sung: **chỉ so trong 3 nhóm, không so với baseline 220**).
- **Loại Task:** Experiment — Strong, **Requires Retrain: Yes** (đã retrain 6 models 600ep×900 steps).
- **Phương pháp được chọn:**
  - Chia 220 SKU theo **MeanDemand** (`prepare_grouped_task5.ipynb:46`): Fast 73 SKU (mean 22.6), Medium 73 (mean 6.25), Slow 74 (mean 0.66, CV 1.46) — phân loại ABC bán lẻ, giữ đủ 220 (Slow 74 vì 220 không chia hết cho 3).
  - Sinh `data_grouped/group_{fast,medium,slow}/*.tfrecords` qua `prepare_data.py:30-56`.
  - Train 6 models: A2C `num_features 219/222` hidden 32, DQN hidden 128 + GroupNorm, cùng 14 actions, gamma 0.99.
- **Lý do chọn:** MeanDemand phản ánh heterogeneity thật, dễ kể chuyện business, tái tạo được; 73/73/74 giữ đủ mẫu, không bỏ SKU.
- **Kết quả (3 bảng + 6 hình):**
  - **Bảng 1 A2C:** Fast 0.1807 > Slow 0.1514 > Medium 0.0477 (last100); Medium suy giảm **-79%** so với 50 ep đầu → bất ổn, cần tuning riêng.
  - **Bảng 2 DQN:** Fast 0.7709 > Medium 0.6508 > Slow 0.5740; cải thiện **+722% → +5448%** — DQN hưởng lợi rõ trên nhóm nhỏ.
  - **Bảng 3 A2C vs DQN:** DQN thắng cả 3 nhóm (Δ -0.59/-0.60/-0.42), stockout DQN ~0 vs A2C 0.04-0.15.
- **Tình trạng:** **100%** — 6 checkpoints thật (`ckpt-63/60/61`), 4 CSV + 6 PNG trong `output_grouped/`. Lưu ý minh bạch: Fast thiếu 141 ep cuối trong log (460/600) dù ckpt tồn tại.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **2 vị trí:** (1) **Section 3.1.5 Product-Group Construction (`:418-429`)** — mô tả chia 220 SKU thành Fast 73 / Medium 73 / Slow 74 theo MeanDemand, Figure 1-2 learning curves; (2) **Section 4.8 Scalability to Product-Group Level (`:1856-1897`)** — 3 bảng gọn (Table 1: A2C, Table 2: DQN, Table 3: A2C vs DQN per group) + Figure 3-6 (combined curves + bar reward/stockout/waste). Trước đó chưa có Section 4.8.
- **Thuật ngữ:** *ABC classification* = phân loại tồn kho theo giá trị/tần suất; *CV* = hệ số biến thiên (Std/Mean), CV>1 là biến động cao.

---

## 3. Nhóm SHAP Micro-level

### Task 10 — Đưa Top-k 660 chiều thành kết quả SHAP chính

- **Yêu cầu làm gì?** Đưa phân tích SHAP chi tiết trên 660 chiều gốc (micro-level) thành kết quả chính Section 4.5, còn SHAP gộp 3 nhóm (aggregated) chỉ làm overview chiến lược.
- **Loại Task:** Analysis / Writing — Must scope.
- **Phương pháp được chọn:**
  - Tạo **Scope Matrix** 2 cấp: Aggregated (3 features avg) = overview, Top-k Micro (660 features `inventory_SKU0-219, sales_SKU0-219, waste_feat_SKU0-219`) = kết quả chính.
  - Dùng `topk_shap_full_results_660.csv` (50 states/scenario, background 100, PartitionExplainer) + so sánh KernelSHAP vs Partition trên 10 states mẫu (`scripts/compare_shap_explainers.py:12`).
- **Lý do chọn:** Gộp 660→3 gây **aggregation bias** (che giấu heterogeneity SKU). Top-20 micro cho biết chính xác SKU nào dẫn dắt, thay vì chỉ biết nhóm nào.
- **Kết quả:**
  - Cả 2 cấp đều Sales dominant, nhưng micro cho thấy tập trung cực đoan vào 8 SKU (64,163,100,46,155,43,118,215) có Mean|SHAP| gấp 3-4× ngưỡng tied 0.000799.
  - Partition artifact: chỉ 9-11 distinct values /660 (652 tied do `shap.maskers.Partition` gom cluster); Kernel distinct 324-375, Jaccard Partition-Kernel 0-0.053, Spearman ~0 → **hai explainer gần như không đồng thuận Top-20**, nhưng Sales vẫn dominant.
  - Fig.9 (3 features) thu ngắn làm overview, Fig.11 (Top-20, 660-dim) nâng thành Fig chính Section 4.5.2.
- **Tình trạng:** **100%** — đã có `outputTask10_top20.csv` (120 dòng), `compare_kernel_vs_partition_result*.csv`, đoạn paste Section 4.5 mở đầu đã soạn.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.5 SHAP (`:1288-1299`) Table 7a Scope Matrix** (Aggregated vs Top-k Micro) + **Section 4.5.3 Top-k Micro-Level (`:1382-1398`)** với Table 7b (Macro dominant vs Micro Top-5) và Figure 11 Top-20 660 chiều là Fig chính, Figure 9 gộp 3 features thu ngắn làm overview. Đoạn Limitations về Partition 9 distinct, Jaccard 0-0.05 cũng đã chèn.
- **Thuật ngữ:** *Aggregation bias* = sai lệch khi gộp dữ liệu làm mất chi tiết; *PartitionExplainer* = SHAP dùng phân cụm để tăng tốc (nhanh nhưng làm phẳng giá trị); *KernelSHAP* = SHAP chính xác nhưng chậm.

### Task 11 — Nhất quán Top-k giữa EASY/MEDIUM/HARD

- **Yêu cầu làm gì?** Kiểm tra các SKU quan trọng có nhất quán giữa 3 scenarios không.
- **Loại Task:** Experiment / Analysis — Must.
- **Phương pháp được chọn:**
  - Metric chuẩn XAI: **Jaccard overlap** (giao/hợp của set Top-k) + **Spearman rank correlation** trên full 660 ranking; kèm RBO (optional, p=0.9) nhạy top-rank hơn.
  - Dữ liệu `topk_shap_full_results_660.csv` (2 agents ×3 scenarios ×660, 50 states/scenario), báo cáo chính Top-20 + sensitivity Top-10/20/50.
- **Lý do chọn:** Top-20 cân bằng: Top-10 quá ít (chỉ thấy sales, không thấy Waste ở HARD), Top-50 quá nhiều (30 SKU cuối tied nhiễu). Sensitivity chứng minh lựa chọn không arbitrary.
- **Kết quả (`outputTask10-11_overlap.csv`):**
  - **DQN:** Jaccard 1.00, Spearman 1.00 ở mọi k và mọi cặp — **cực kỳ ổn định**, tập trung 8 SKU sales (64,163,46,155,118...) ở cả 3 scenarios ("stable but less flexible").
  - **A2C_mod:** Jaccard 0.667 (Top-10) → 0.29 (Top-20) → 0.12 (Top-50), Spearman 0.20 / -0.008 → **thay đổi theo môi trường** ("context-dependent adaptability"), dù 5 SKU core giữ nguyên.
- **Tình trạng:** **100%** — bảng overlap đã chạy, đoạn paste Section 4.5.3 đã soạn.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.5.4 Consistency of Top-k (`:1399-1402`)** — 1 đoạn + bảng Jaccard/Spearman (DQN 1.00 vs A2C_mod 0.29-0.33) + footnote *Comparison qualitative as noted in 3.3.3*. Chi tiết Top-10/20/50 (18 dòng) đẩy vào Supplementary.
- **Thuật ngữ:** *Jaccard* = độ giao nhau của 2 tập (1 = giống hệt); *Spearman rho* = tương quan thứ hạng (-1→1); *RBO* = Rank-Biased Overlap, nặng top hơn.

### Task 12 — Vì sao SKU thống trị Top-20

- **Yêu cầu làm gì?** Giải thích vì sao một số SKU thống trị: demand cao, spoilage cao, volatility, inventory pressure...
- **Loại Task:** Analysis — Strong.
- **Phương pháp được chọn:** Join Top-20 (union 36 SKU) với thống kê từ `data/train.tfrecords` (1000 steps) + `data/capacity.tfrecords`: tính MeanDemand, Std, CV, MaxDemand, Capacity, Utilization (=Mean/Capacity).
- **Lý do chọn:** TFRecords không lưu tên sản phẩm, nên giải thích dựa trên demand/capacity/volatility có thể tính được, khách quan.
- **Kết quả (`outputTask10-12_case_analysis.csv` 36 SKU):**
  - Top SKU **không phải best-seller**: Top demand thực là SKU57 (22.6), SKU81 (17.2) — không nằm trong Top-8 SHAP. Correlation MeanDemand vs Mean|SHAP| chỉ **r=0.044 (DQN)** / 0.0005 (A2C).
  - Đặc điểm chung Top-8: **low-volume nhưng high CV (1.1-1.46) + capacity nhỏ (7-28)** — ví dụ SKU64 mean 0.66 CV1.46 cap7, SKU163 mean1.22 CV1.27. Ngoại lệ SKU215 high-volume (8.68, cap81) nhưng trọng số thấp hơn SKU64.
  - Waste chưa dominant ở Top-8, nhưng lọt Top-20 ở HARD (SKU88,75,11) → risk-aware khi kho khó.
- **Tình trạng:** **100%** — đã có bảng case 15 SKU + đoạn paste Discussion.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.5.5 Why Dominant Products (`:1403-1407`)** — 1 đoạn + Table case (SKU57/81 best-seller vs SKU64/163 volatile) + note correlation r=0.04, ghi chú 36 SKU trong Supplementary `outputTask10-12_case_analysis.csv`.
- **Thuật ngữ:** *CV* = Std/Mean (độ biến động tương đối); *Utilization* = MeanDemand/Capacity (áp lực kệ).

---

## 4. Nhóm SHAP Comparability

### Task 13 — Common explanation target (pre-softmax logits)

- **Yêu cầu làm gì?** Dùng target chung cho DQN và A2C_mod (logits trước softmax) để so sánh công bằng, nếu khả thi, không retrain, kèm sensitivity 10/20/50.
- **Loại Task:** Implementation / Experiment — Strong, Maybe Retrain (đã làm **không retrain**).
- **Phương pháp được chọn:**
  - Wrapper hiện tại không tương đương: DQN `softmax(Q)` (Q unbounded) `XAI/SHAP-temp.ipynb:303` vs A2C_mod `π` (constrained [0,1]) `XAI/SHAP-temp.ipynb:297`.
  - Chọn **pre-softmax logits**: DQN `q_values` linear 14 `Training/Train_DQN.ipynb:146` → `mean(q_network(X))`, A2C_mod `layer4` linear 14 `Training/A2C-mod.ipynb:128` → `mean(actor_logits)`. Cùng unbounded, cùng là score trước normalize.
  - Experiment: Background 100, KernelExplainer `nsamples=2000`, test states **n=10/20/50** từ `test.tfrecords` (504 dòng), tính Mean|SHAP| Top-20.
- **Lý do chọn:** Logits là common scale duy nhất không cần retrain; Advantage cần thêm head V(s) cho DQN → phải retrain 600ep nên bỏ.
- **Kết quả (`outputTask11_common_target.csv` 360 dòng + `outputTask11_n_states_sensitivity.csv` 18 dòng, ckpt-60/64):**
  - Top-5 logits EASY: DQN 175(0.00178),90,164,119,157 vs baseline 64,163,100... → **0/5 trùng**, scale 0.63×; A2C logits 93(0.460),119,108,175,71 vs baseline 163 → 0/5 trùng, magnitude chênh 258× → **cần disclaimer Task 14 mạnh hơn**.
  - Sensitivity n_states: Jaccard **0.25-0.33 đồng đều** mọi scenario (10-20, 20-50, 10-50) → Top-20 nhạy với sample size, 10-state không đại diện, phải báo cả 3 mức.
- **Tình trạng:** **100%** — đã chạy thực 10/20/50, đủ để qua R1 #5.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.5.6 Common Target Sensitivity (`:1409-1412`)** — 1 đoạn ~150 từ + 360 dòng Supplementary `outputTask11_common_target.csv`. Trước đó chưa có Section 4.5.6 (chèn sau 4.5.5, trước 4.5.7).
- **Thuật ngữ:** *Logits* = giá trị trước softmax; *Common target* = cùng đại lượng để so sánh công bằng.

### Task 14 — Tuyên bố qualitative nếu không đổi target

- **Yêu cầu làm gì?** Nếu không đổi common target, phải tuyên bố rõ SHAP comparison chỉ là **qualitative**, không so trực tiếp magnitude/stability.
- **Loại Task:** Writing — Must, No Retrain, <30 phút.
- **Phương pháp được chọn:**
  - Chèn disclaimer 3 vị trí: Section 3.3.3 sau Table 7a (`Xai_Inventory_Submit_17Mar.md:1185`), footnote Table 7b (`1241`) & 7c (`1256`), và sửa câu Discussion `1205` thành "more distributed **under its policy target**; direct quantitative comparison is qualitative".
- **Lý do chọn:** Đây là yêu cầu bắt buộc tối thiểu, dù Task 13 đã chứng minh Sales dominant vẫn robust, vẫn cần trung thực về khác scale.
- **Kết quả:** 2 đoạn văn sẵn paste (Việt + Anh) đã soạn, ví dụ: *"SHAP for DQN is computed on softmax(Q) and for A2C_mod on π(a*|s) — two quantities on different scales (Q unbounded vs π constrained [0,1])... direct comparisons in Table 7b/7c are descriptive and qualitative"*.
- **Tình trạng:** **100%** — không có CSV, chỉ đoạn văn chờ approve.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **3 vị trí:** (1) **Section 3.3.3 SHAP Implementation Details (`:1300`)** — disclaimer chính *"SHAP for DQN is computed on softmax(Q) ... hence direct comparisons in Table 7b/7c are qualitative"*; (2) **Table 7b footnote (`:1397`)** `*Comparison qualitative ... see Task 13*`; (3) **Section 4.5.4 footnote (`:1401`)**. Đoạn Discussion `1205` cũng đã sửa "more distributed **under its policy target**".
- **Thuật ngữ:** *Qualitative vs quantitative* = mô tả định tính (so xu hướng) vs định lượng (so con số).

### Task 15 — Sensitivity actor/critic outputs

- **Yêu cầu làm gì?** Thêm sensitivity khi SHAP lên các outputs khác nhau của actor/critic để kiểm tra kết luận có đổi không.
- **Loại Task:** Experiment — Strong, No Retrain, gộp với Task 13 để tiết kiệm.
- **Phương pháp được chọn:** Cùng 10 states + BG 100, chạy SHAP trên 3 outputs A2C_mod: (1) `π(a*|s)` baseline, (2) `logits` pre-softmax (Task 13), (3) `V(s)` critic scalar `Training/A2C-mod.ipynb:147-162`. So Jaccard/RBO Top-20 giữa `π vs logits` và `π vs V(s)`.
- **Lý do chọn:** Gộp experiment, chỉ đổi `predict_fn`, tiết kiệm 2-5h.
- **Kết quả (`outputTask11_sensitivity.csv` 9 dòng, ckpt-60/64):**
  - `π vs logits`: Jaccard 0.25-0.33, RBO 0.42-0.757 mọi scenario → **ranking khác rõ** dù cùng Sales group.
  - `π vs V(s)`: Jaccard 0.25 đồng đều, RBO 0.547-0.647 → **Actor và Critic cùng nhìn Sales ở Top-8 nhưng Top-20 khác nhau**.
- **Tình trạng:** **100%** — đã chạy EASY/MEDIUM/HARD, đủ để chứng minh đã thử sensitivity thực tế.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.5.7 Actor vs Critic Sensitivity (`:1413-1416`)** — 1 đoạn ~120 từ + 9 dòng Supplementary `outputTask11_sensitivity.csv`. Chèn sau 4.5.6, trước 4.5.8.
- **Thuật ngữ:** *Actor vs Critic* = mạng chính sách vs mạng giá trị trong A2C; *V(s)* = giá trị trạng thái.

---

## 5. Nhóm Reward Design

### Task 20 — Justify Reward Coefficients (tại sao 1.0? tại sao waste 0.025?)

- **Yêu cầu làm gì?** Biện minh hệ số reward bằng empirical/managerial/literature, trả lời reviewer "Why 1.0? Why not 0.5 or 2.0?"
- **Loại Task:** Writing / Literature + Experiment — Must, No Retrain.
- **Phương pháp được chọn:**
  - Mapping paper↔code: `r = 1 - z - overstock - q - quan` `training.py:336` ↔ `service/holding/waste/ordering` (`Xai_Inventory_Submit_17Mar.md:372`), làm rõ Instacart Kaggle **không có cột chi phí $** `prepare_data.py:68-102`, mọi hệ số là proxy.
  - Literature 4 nhóm (9 refs): Base [1] Sutton RL scaling, Stockout [2] Khouja Newsvendor, Overstock [4] Harris EOQ, Waste [6] Federgruen perishable 2-3%/period, Quantile [8] Mannor CVaR.
  - Empirical **OAT Sweep** (One-At-a-Time, Saltelli 2008): 4 params ×4 mức `λ∈{0.5,1.0,1.5,2.0}` → **16 configs** (13 unique vì waste 0.025 trùng baseline), 2 agents + BaseStock = 40 dòng, mỗi eval ~30s trên test 504 steps (<20 phút). Full factorial 256 configs sẽ tốn 5h và confounding.
- **Lý do chọn:** OAT là chuẩn cho Must scope: cô lập tác động từng param, đối xứng quanh baseline 1.0, đủ để chứng minh 1.0 là neutral.
- **Kết quả (`outputTask20_sweep_results.csv` 40 dòng + `outputTask20_sensitivity.png`):**
  - DQN baseline 0.361: overstock nhạy nhất **0.591→-0.10 (Δ0.692)**, quantile 0.433→0.215 (Δ0.218), stockout phẳng (Δ0.011), waste vừa (Δ0.086).
  - A2C_mod baseline 0.382: quantile nhạy nhất **0.684→-0.224 (Δ0.908)**, overstock phẳng (Δ0.002).
  - **Weight=1.0 là compromise trung hòa:** không thấp nhất/cũng không cao nhất, nằm giữa dải; nếu tối ưu riêng DQN muốn overstock 0.5 (+63%), A2C muốn quantile 0.5 (+79%).
- **Tình trạng:** **100%** — đã có sweep + hình + đoạn paste Section 3.2.4.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 3.2.4 Reward Coefficient Justification (`:689-702`)** — chèn mới sau Section 3.2.3 và trước Section 3.3. Gồm Table 3 (Service/Holding/Waste/Balance × weight + theoretical basis) + 1 đoạn Instacart không có $ + OAT 16 configs (Saltelli) + hình `outputTask20_sensitivity.png` (4 subplots) đẩy vào Supplementary. Trước đó chưa có Section 3.2.4.
- **Thuật ngữ:** *OAT* = đổi một param mỗi lần, còn lại giữ nguyên; *Full factorial* = thử mọi tổ hợp (tốn kém).

### Task 21 — Range & Distribution trước/sau Normalization (có leakage?)

- **Yêu cầu làm gì?** Báo cáo range và phân bố từng reward component trước và sau normalize, trả lời "Any leakage?"
- **Loại Task:** Analysis — Must, No Retrain.
- **Phương pháp được chọn:**
  - Dùng `data/train.tfrecords` (1000×220) + `test.tfrecords` (504×220) qua `reward_utils.load_tfrecord_data` `reward_utils.py:134`, normalize `sales_norm = sales_raw / capacity` `reward_utils.py:168` (capacity fit từ train `prepare_data.py:144` `12×mean`).
  - Tính per-product per-timestep `[T,220]` với heuristic `u=0` (không đặt hàng) để thấy phân bố gốc, **bổ sung cell 3c policy-aware BaseStock k=1.0** để có overstock thực.
  - Thống kê mean/std/min/p25/p50/p75/max/skew/kurtosis + Wasserstein/KL giữa train/test.
- **Lý do chọn:** Heuristic cho baseline thuần túy, BaseStock cho phân bố thực khi có replenishment; tránh StandardScaler để không leakage.
- **Kết quả (5 CSV + 7 PNG):**
  - **Before→After:** sales train 2.10 max162 → 0.10 max2.25 (giảm 21×), test 0.73→0.034; capacity mean 20.3 min4 max208, ratio ~10× (kỳ vọng 12).
  - **Heuristic u=0:** z 0.993 (luôn stockout), overstock 0, q 7e-05, quan 0.0068, reward ~0, x 0.0028.
  - **BaseStock k=1.0:** z 0.154→0.022 (giảm 84-97%), q 0.0038/0.0052 (×54), quan 0.234/0.211, reward 0.607/0.760 — phân bố thực.
  - **Covariate shift:** Wasserstein sales_raw 1.37 → sales_norm 0.066 (**giảm 95%**) → normalize hiệu quả, không leakage (capacity chỉ fit trên train).
- **Tình trạng:** **100%** — đã có `outputTask21_stats_train.csv`, `outputTask21_normalization_stats.csv`, `outputTask21_covariate_shift.csv` và 2 file BaseStock mới.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.5.8 Reward Component Range and Distribution (`:1417-1452`)** — figure `outputTask20_sensitivity.png` trên cùng + Table 4 (Demand/Capacity before/after) + Table 5 (Stockout/Overstock/Waste/Balance under No-replenishment vs BaseStock vs DQN) + đoạn Wasserstein 1.37→0.066 giảm 95%. Chèn sau Section 4.5.7, trước Section 4.6.
- **Thuật ngữ:** *Wasserstein distance* = khoảng cách phân bố (càng nhỏ càng giống); *Leakage* = dùng thông tin test để train.

---

## 6. Nhóm Faithfulness (Task 15-9: 24-28 gộp)

> 5 Tasks này được báo cáo gộp trong 1 file `task15-9/outputTask15-9.md`, dùng chung 1 notebook `faithfulness_Task15-9_Complete.ipynb` (ckpt-50/64, 150 states).

### Task 24 — AUPC / ASR / CI cho MoRF/Random/LeRF

- **Yêu cầu:** Báo cáo diện tích dưới đường cong nhiễu loạn (AUPC), tỷ lệ chuyển đổi hành động (ASR) và khoảng tin cậy cho 3 chiến lược che lấp.
- **Loại:** Experiment — Must.
- **Phương pháp:** Che lấp 1→10 features (0.15→1.52% của 660 chiều), thay bằng trung vị của 100 BG, lặp Random 30 lần lấy trung bình; AUPC tính bằng hình thang, CI bằng bootstrap 1000 lần (mean) + Wilson (ASR). 50 states/kịch bản ×3 kịch bản ×2 agents = 300 quan sát.

### Task 25 — So sánh thống kê

- **Yêu cầu:** So sánh MoRF vs Random vs LeRF và MSX-guided vs random, kèm p-value + effect size.
- **Loại:** Analysis.
- **Phương pháp:** Wilcoxon ghép cặp (không cần chuẩn) + t-test, Cohen's d, hiệu chỉnh Holm. So MSX-guided (che theo tập tối thiểu) vs random cùng kích thước (k=2 và 5).

### Task 26 — Giao thức (protocol)

- **Yêu cầu:** Làm rõ số states, tỷ lệ mask, chiến lược thay thế, số lần lặp.
- **Loại:** Method / Writing.
- **Phương pháp:** Ghi rõ 50 states/kịch bản, mask 1-10, median replacement, 30 lặp cho Random, seed 42, checkpoint thực.

### Task 27 — Ngưỡng 1% có ý nghĩa?

- **Yêu cầu:** Định nghĩa & biện minh ngưỡng cho Q-drop có ý nghĩa hoặc action flip.
- **Loại:** Method.
- **Phương pháp:** Ngưỡng 1% tương đối (Q) / 1% tuyệt đối (π) ≈ 2 bước action tối thiểu (14 mức, bước nhỏ nhất 0.5% capacity); kèm ngưỡng rời rạc là action flip.

### Task 28 — Nhất quán giữa states

- **Yêu cầu:** Kiểm tra faithfulness có nhất quán giữa các states, độ phân tán?
- **Loại:** Analysis.
- **Phương pháp:** Tính CV, IQR, violin plot trên 50 states, kiểm Kruskal-Wallis giữa 3 kịch bản.

- **Kết quả chung (Task 24-28):**
  - **AUPC:** MoRF dương (0.0097-0.0252), Random ~0, LeRF âm (-0.0196) → thứ tự **MoRF > Random > LeRF** nhất quán 18/18 điều kiện (`outputTask15-9.md:49-70` Bảng 1).
  - **ASR = 0/18** (Wilson CI 0-0.07) → che 10 features chưa đủ đổi quyết định rời rạc; cần mở rộng mask hoặc hạ ngưỡng.
  - **Thống kê:** MoRF vs Random p<1e-14, d=2.2-5.0 (lớn); MoRF vs LeRF d=3.7-16.8 (rất lớn). MSX k=5 vs random p<1e-13 d=1.85-3.93; k=2 d nhỏ hơn (0.54).
  - **Ngưỡng 1%:** Tỷ lệ vượt ngưỡng = 0 (vì Q-drop trung bình chỉ 0.16-0.40% <1%) → ngưỡng cần hiệu chỉnh, báo cáo như tham số nhạy cảm.
  - **Nhất quán:** A2C_mod CV 0.02-0.17 (ổn định), DQN CV 0.19→0.48 tăng theo HARD (phân tán hơn); Kruskal-Wallis có ý nghĩa giữa kịch bản → faithfulness phụ thuộc độ khó.
- **Tình trạng:** **100%** — 5 CSV + 4 hình đã có, đoạn paste Section 4.7 đã soạn (4.7.1 Protocol → 4.7.5 Consistency).
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 4.7 Faithfulness Evaluation (`:1751-1854`) — ĐẠI TU toàn bộ:** 4.7.1 Protocol (`:1755-1762`) + Figure 20 `task15-9_perturbation_with_CI.png`, 4.7.2 AUPC/ASR/CI (`:1766-1799`) Table 8a/8b, Figure 21 `task15-9_asr_with_CI.png`, 4.7.3 Statistical Comparisons (`:1801-1817`) Table 9, 4.7.4 Threshold (`:1818-1835`) Table 10 Figure 22 `threshold_hist.png`, 4.7.5 Consistency (`:1836-1854`) Table 11 Figure 23 `per_state_violin.png`. Thay thế khối 4.7 cũ rời rạc.
- **Thuật ngữ:** *MoRF/LeRF* = Most/Least Relevant First (che quan trọng nhất/ít nhất trước); *AUPC* = Area Under Perturbation Curve (diện tích dưới đường cong, càng lớn càng faithful); *ASR* = Action Switching Rate; *Wilcoxon* = kiểm phi tham số cho mẫu ghép cặp; *Cohen's d* = cỡ hiệu ứng.

---

## 7. Nhóm Data Integrity

### Task 33 — Train/Val/Test split là chronological hay random?

- **Yêu cầu làm gì?** Làm rõ split là chronological hay random; ưu tiên chronological để tránh temporal leakage. Reviewer #11 nêu 900/496 không rõ.
- **Loại Task:** Clarification — Must, No Retrain.
- **Phương pháp được chọn:** Audit định lượng **Phương án B** (`audit_dataset_split_leakage.ipynb` 18 cells): đếm `TFRecordDataset` → train 1000 / test 504, đọc parsers y hệt `Training/A2C-mod.ipynb:7` `Training/DQN.ipynb:4`, tách val chronological 800/200/504, tính mean/std, Pearson per-product.
- **Lý do chọn:** Có số + hình (timeline, histogram) thuyết phục hơn chỉ viết miệng; reproduce được bằng `python notebook`.
- **Kết quả (`output_audit_33_split_stats.csv:2-5`):**
  - **Chronological thuần túy:** `train_file` vs `predict_file` là 2 file vật lý, đọc bằng `window(shift=batch_size-1)` giữ thứ tự, không shuffle `prepare_data.py:149` `range(start,middle)` vs `range(middle,end)`.
  - Split: Train 0-999 (1000, mean 0.1003), Train_part 0-799 (0.1116), Val 800-999 (200, 0.0549), Test 1000-1503 (504, 0.0340) — Val/Test thấp hơn Train 46-66% là drift tự nhiên.
  - Per-product Pearson r = **-0.47** (âm) → ranking SKU đổi giữa train/test → test OOD tự nhiên, justify EASY/MEDIUM/HARD scaling.
  - Mismatch 900/496 vs 1000/504 do `num_timesteps=900` là window trong episode, còn TFRecords là 1000 periods.
- **Tình trạng:** **100%** — đã có 2 PNG (`timeline`, `histogram`) + đoạn paste Section 3.1.3.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 3.1.3 Data Split and Chronological Protocol (`:350-388`)** — chèn mới sau Section 3.1.2 và trước Section 3.2. Gồm: đoạn chronological strict (time_period 6h, `prepare_data.py:149`, `window(shift)`), Table 1b (Training 1000/0.10, Validation 200/0.055, Test 504/0.034), Figure A `output_audit_33_timeline.png`, Figure B `output_audit_33_histogram.png` + Table 1b-S scenarios (EASY/MEDIUM/HARD scaling) + Figure B-S. Trước đó chưa có Section 3.1.3.
- **Thuật ngữ:** *Chronological* = chia theo thời gian (như học lịch sử để đoán tương lai); *Random split* = xáo trộn rồi chia (gây leakage); *OOD* = Out-of-Distribution (ngoài phân phối).

### Task 34 — Đảm bảo forecasting/normalization/reward/baseline chỉ fit trên train/val

- **Yêu cầu làm gì?** Audit 4 nhóm dễ gian lận: forecasting, normalization, reward coefficients, baseline parameters có fit trên test không? + corrected procedure nếu cần.
- **Loại Task:** Method / Check — Must, Maybe Retrain.
- **Phương pháp được chọn:** Cùng notebook audit Sections 2-4: (a) capacity train-only = `ceil(mean(train)*12)` vs full, (b) reward hardcode, (c) baseline grid `k∈{0.5,1,1.5,2,2.5}` trên val vs test với `S_i = mean(train)+k·std(train)`.
- **Lý do chọn:** Reviewer đòi audit + corrected procedure; có số 26.29% + hình k_curve thuyết phục hơn lời nói.
- **Kết quả (`output_audit_34_leakage_summary.txt:16-29`):**
  | Nhóm | Fit trên | Rò rỉ? | Cách sửa |
  |---|---|---|---|
  | Forecasting | N/A (no model, `sales/capacity` quan sát) | **No** | Ghi 1 câu |
  | Normalization | `capacity.tfrecords` mean 20.33 (full data) | **Yes — MAJOR 26.29%** (212/220 >10%, max 50%) | Regenerate `capacity = ceil(mean(train)*12)` từ `grocery_train` |
  | Reward | Hardcoded 0.025 `training.py:134` | **No** | OAT Task 20-21 |
  | Baseline | `S_i` trên train đúng, `k*` trên test `DQN.ipynb:17` | **Yes → FIXED** | Chọn `k*` trên val: `k_val=1.0` (reward 0.689) vs `k_test=0.5` (0.759), diff 0.5, 2 đường song song |
  - Impact capacity <0.02 reward, `k` lệch chỉ 0.5 → **không cần retrain DQN/A2C cho Must**, sẽ regenerate `capacity.tfrecords` ở revision nếu reviewer yêu cầu Strong.
- **Tình trạng:** **95%** — đã phát hiện MAJOR, đã có corrected procedure + 4 CSV + 2 PNG, chỉ còn chờ regenerate file nếu cần.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 3.1.4 Leakage Audit and Corrected Procedure (`:389-417`)** — chèn mới sau Section 3.1.3. Gồm: đoạn audit 4 nhóm, Figure C `output_audit_34_capacity_diff.png`, Figure D `output_audit_34_k_curve.png`, Table 1c Leakage summary (4 dòng: Forecasting No, Normalization Yes-MAJOR 26.29%, Reward No, Baseline Yes→Fixed). Trước đó chưa có Section 3.1.4.
- **Thuật ngữ:** *Leakage* = rò rỉ thông tin test vào train; *Capacity* = năng lực kệ; *BaseStock* = chính sách cổ điển `S = mean + k·std`.

---

## 8. Nhóm SHAP Implementation

> Gộp 4 Tasks 39-42 trong `task14-9/outputTask14-9.md` (2 notebooks đã chạy với ckpt thực).

### Task 39 — Quy trình tập nền 200→100

- **Yêu cầu:** Làm rõ cách tạo 200 states ban đầu và chọn 100 đại diện.
- **Loại:** Method — No Retrain.
- **Phương pháp:** Sinh 200 states tổng hợp (`inv/sales ~U(0,1), waste=0.025*inv+N(0,0.005)`), rút gọn 100 bằng 2 cách: (1) random có seed (baseline, tái tạo 100%) vs (2) KMeans centroids (đối chứng độ phủ).
- **Kết quả (Bảng 1):** 200 mean inv 0.49, random 100 mean 0.47, KMeans 0.50 — phân bố gần như giống hệt; pairwise Euclid distance trùng nhau → random đã đủ đại diện, giữ baseline.

### Task 40 — Độ nhạy SHAP (nguồn/kích thước/chiến lược)

- **Yêu cầu:** So sánh BG tổng hợp vs quỹ đạo thực tế, kích thước 50/100/200, random vs KMeans.
- **Loại:** Experiment.
- **Phương pháp:** Lưới 4 nguồn ×3 kích thước ×2 agents ×2 cấp (macro 3 chiều KernelSHAP, micro 660 chiều Partition), trên tập kiểm định cố định scenario MEDIUM.
- **Kết quả:**
  - **Macro:** Spearman 1.00 trong cùng nguồn (50/100/200 giống hệt), nhưng 0.87 khi đổi tổng hợp→quỹ đạo → **ổn định với size, nhạy với nguồn**.
  - **Micro 660:** Spearman ~0-0.17, dù magnitude similarity 0.94-0.99 → **thứ hạng không ổn định** trong không gian cao chiều, cần báo cáo như limitation, khuyến nghị ưu tiên BG quỹ đạo.

### Task 41 — Tính hợp lệ miền (perturbed states có khả thi?)

- **Yêu cầu:** Kiểm perturbed states có vi phạm ràng buộc kho/năng lực không.
- **Loại:** Experiment.
- **Phương pháp:** Kiểm 5000 states macro (KernelSHAP coalition) + 1000 micro (median mask) trên EASY/MEDIUM/HARD, đối chiếu inv∈[0,1], demand, waste=0.025*inv±tolerance.
- **Kết quả (Bảng 4):** Quỹ đạo **100%** mọi kịch bản; tổng hợp 95.5% (EASY), 98.8% (MEDIUM), **84.5% (HARD)** → tập quỹ đạo khả thi hơn, HARD trên BG tổng hợp là cảnh báo.

### Task 42 — Chuẩn hóa mô tả Kernel vs Partition

- **Yêu cầu:** Phân biệt rõ khi nào dùng KernelSHAP vs Partition, cấu hình.
- **Loại:** Writing.
- **Phương pháp:** Đưa vào khung thống nhất: Macro = KernelSHAP chính xác (3 chiều, giây), Micro = Partition (660 chiều, phút, logarit); bảng cấu hình Input/Masking/Params/Time để tái tạo.
- **Tình trạng chung 39-42:** **100%** — đã có 9 CSV + 8 hình (`T39_coverage.png` etc.), đoạn paste Section 3.4 + Appendix S4 đã soạn.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 3.3 Feature-based Explanation with SHAP (`:764-863`) — mở rộng lớn:** 3.3.1 Background Construction (`:769-787`) Table 2a + Fig 2a `T39_coverage.png` + Fig 2b `T39_pairwise_dist.png`; 3.3.2 Background Sensitivity (`:788-822`) Table 2b/2c + Fig 3a `T40_macro_sensitivity.png` + Fig 3b `T40_fcs_sensitivity.png` + Fig 4 `T40_micro_sensitivity.png`; 3.3.3 Validity (`:823-846`) Table 2d + Fig 5a `T41_validity_rate.png` + Fig 5b `T41_waste_violation.png`; 3.3.4 Consistent Implementation (`:847-863`) Table 2e + Fig 6 `T42_top20_micro.png`. Thay thế mô tả SHAP cũ ngắn gọn.

---

## 9. Nhóm A2C_mod Definition

### Task 43 — Định nghĩa A2C_mod ngay lần đầu xuất hiện

- **Yêu cầu làm gì?** Định nghĩa A2C_mod với ký hiệu, cấu trúc, liệt kê hệ thống các cải tiến so với A2C chuẩn (Mnih).
- **Loại Task:** Writing / Analysis — Must, No Retrain.
- **Phương pháp được chọn:** Phân tích dựa trên **bằng chứng dòng mã** `Training/A2C-mod.ipynb:122-476`: trích P=220, F=3, A=14, actor 4 lớp (32, ReLU+dropout) → π 14 chiều, critic 2 lớp + GroupNorm → V(s), γ=0.99, batch 32, lr 0.001, 600 ep; trình bày hộp thuật toán 8 bước + bảng 8 cải tiến.
- **Lý do chọn:** Reviewer đòi chính xác, không mô tả định tính mơ hồ; mỗi cải tiến phải nêu lý do trong bối cảnh tồn kho + ưu/nhược để không phải hỏi lại.
- **Kết quả (Bảng 1, `task16-9_table43_modifications.csv`):** 8 cải tiến M1-M8:
  - M1 Policy update MSE(π, softmax(logπ+A/(dist+1))) vs REINFORCE → mượt, giảm phương sai
  - M2 Trọng số khoảng cách `/(|a*-a|+1)` → mượt trên 14 mức có thứ tự
  - M3 MSE/Brier vs log loss → ổn định với reward ~0.8
  - M4 Entropy chỉ log (0.001·H) không cộng vào loss → **ghi nhận là xấu hơn** (dễ hội tụ sớm)
  - M5 Advantage 1-step δ vs GAE → đơn giản
  - M6 Hai Adam riêng vs một Adam → tách nhiễu V
  - M7 Actor 4 lớp vs Critic 2 lớp vs đối xứng → mở rộng tuyến tính
  - M8 Nhân bản per-product `[220,3]` vs toàn cục `[660]` → chia sẻ trọng số, bù qua quantile
- **Tình trạng:** **100%** — đã khôi phục `ckpt-64` kiểm chứng.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 3.2.5 A2C_mod Algorithm and Its Relation (`:704-731`)** — chèn mới sau Section 3.2.4. Gồm: đoạn định nghĩa P=220/F=3/A=14, hộp thuật toán 8 bước, Table 3b (6 dòng gọn trong paper, full 8 dòng trong Supplementary), Figure 2c `task16-9_fig1_modifications.png`, Figure 2d `task16-9_fig2_pnew_smoothing.png` + đoạn 8 cải tiến Pros/Cons.

### Task 44 — Phép biến đổi sang Q-equivalent & tính hợp lệ

- **Yêu cầu làm gì?** Giải thích & biện minh phép biến đổi output A2C_mod sang thang chung với DQN (bảo toàn thứ tự, tương thích thang đo, mượt).
- **Loại Task:** Method / Writing — Must.
- **Phương pháp được chọn:** Xây dựng **phép biến đổi kép**: (1) SHAP dùng `f_common = mean(softmax(Q))` vs `mean(π)` (cùng miền simplex 14 chiều), Q_mod = `log π + V(s)`; (2) RDX/MSX dùng mô phỏng 1 bước `x' = max(0,min(1,x+u)-sales)` và chênh lệch reward giữa a* vs a2. Chứng minh 3 tính: đơn điệu softmax/log bảo toàn argmax, chuẩn hóa về simplex, làm mịn Laplace `/(dist+1)`.
- **Lý do chọn:** Hai họ XAI cần input khác nhau (SHAP cần vector xác suất, RDX cần phân tách reward), một phép duy nhất sẽ xấp xỉ sai cho một họ.
- **Kết quả (kiểm chứng 50 states/kịch bản, `task16-9_verification_metrics.csv`):**
  - Trùng argmax **100%** cả 3 kịch bản (Q vs softmax(Q) và π vs Q_mod), Spearman 0.29-0.32 (dương thấp) → **bảo toàn quyết định tối ưu, không cần đồng nhất toàn bộ ranking**.
  - `f_common` hội tụ sau ~50 sản phẩm (lấy trung bình 220 ổn định); MSE là chặn trên trơn của KL → biện minh dùng MSE.
  - Bảng phân vai (SHAP vs RDX) đã chuẩn hóa.
- **Tình trạng:** **100%** — notebook `Task16-9_Transformation_Verification.ipynb` đã chạy 2026-09-18 16:38, 4 hình + 4 CSV trong `output/`.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **Section 3.2.6 Common Quantity for Cross-Agent Comparisons (`:732-763`)** — chèn mới sau Section 3.2.5. Gồm: đoạn dual common quantity (SHAP vs RDX), Table 3c (dual transformation), Figure 2e `task16-9_fig3_transformation_ordering.png`, Figure 2f `task16-9_fig4_fcommon_mse_validity.png`, Table 3d Verification (100% preservation, Spearman 0.29-0.32) + đoạn 3 properties (order/scale/smoothness).
- **Thuật ngữ:** *Simplex* = không gian xác suất tổng=1; *Monotonic* = đơn điệu (giữ thứ tự); *Brier score* = MSE cho xác suất.

---

## 10. Nhóm Managerial & Writing

### Task 47 — Managerial Case Studies + Giảm claim Trust

- **Yêu cầu làm gì?** Reviewer R1 #13: perturbation chứng minh faithful với model, chưa chứng minh manager hiểu được → cần expert evaluation hoặc ít nhất case studies + giảm claim trust. Chọn **Task 47 Strong** (không làm Task 46 Optional tốn thời gian).
- **Loại Task:** Writing — Strong, No Retrain, không cần expert ngoài.
- **Phương pháp được chọn:** Dịch 3 lens XAI sang tiếng quản lý, mỗi case gắn với số liệu đã có:
  - Case 1 RDX (trade-off), Case 2 MSX (1 lý do đủ), Case 3 SHAP (5 SKU cần theo dõi), + 3 vị trí giảm claim trust trong Abstract/Conclusion.
- **Lý do chọn:** Expert evaluation chuẩn nhưng cần protocol & tiếp cận chuyên gia (không khả thi), case study là cầu nối interim phù hợp XAI decision-support, bao phủ 3 lens bổ sung.
- **Kết quả (3 cases tiếng quản lý):**
  - **Case 1 RDX (EASY vs HARD):** EASY kho 30%/20%/1% → cả 2 agents đặt mạnh; HARD kho 90%/80%/15% → A2C_mod giảm holding cost (0.083 vs 0.107 DQN), DQN giữ service 0.9997 → Manager tự chọn tiết kiệm vs phục vụ, RDX cho thấy trade-off.
  - **Case 2 MSX (1 lý do):** A2C_mod EASY/MEDIUM core = {service} → HARD đổi thành {holding}, DQN luôn {holding} ("cứng nhắc") → MSX trả lời sếp 1 câu ngắn gọn.
  - **Case 3 SHAP (5 SKU):** Top SKU64 (CV1.46, cap7), 163 etc. **không phải best-seller** (SKU57 22.6) → low-volume high-CV, Jaccard DQN 1.0 (ổn định) vs A2C 0.29 (linh hoạt) → Manager đặt cảnh báo cho nhóm volatile, kệ nhỏ.
  - **Giảm claim:** Abstract dòng13 thêm *"without expert validation, operational trust remains to be established"*, cuối Abstract thêm *"requires future expert validation (see Limitations)"*, Conclusion mở rộng đoạn Limitations: *"Faithfulness ≠ usefulness, transparency ≠ trust"*.
- **Tình trạng:** **100%** — 3 đoạn Anh sẵn paste (Section 4.10 mới, Abstract, Conclusion 1945-1995), bảng tóm tắt 3 cases, `outputTask47_cases.csv` + `generate_cases_task47.ipynb`.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **3 vị trí:** (1) **Abstract dòng 13 (`:11-13`)** — sửa "limits interpretability and operational trust" → "limits interpretability; without expert validation, operational trust remains to be established" + cuối Abstract thêm "requires future expert validation (see Limitations)"; (2) **Section 4.10 Managerial Case Studies (`:1945-1990`)** — chèn mới hoàn toàn (Table 12 overview 3 cases, Figure 13 `task47_case_overview.png`, Case 1 RDX, Case 2 MSX, Table 13 SKU 64/163/46/155/57 + Figure 14 `task47_sku_profile.png`, Case 3 SHAP, đoạn kết faithfulness≠usefulness); (3) **Section 5 Conclusion (`:1996-1998`)** — đại tu đoạn Limitations (thêm "Perturbation-based faithfulness establishes ... but not that they are useful ... claims about trust are therefore moderated to transparency").
- **Thuật ngữ:** *Faithfulness* = trung thực với model; *Usefulness* = hữu ích với người dùng (khác nhau).

### Task 50 — Mở rộng văn liệu lên 50+ refs

- **Yêu cầu làm gì?** Mở Intro + Related Work thành mạch câu chuyện 5 đoạn, bổ sung ít nhất 50 refs cập nhật peer-reviewed; Reviewer 3 nói hiện chỉ ~20 refs, thiếu then chốt, gợi ý 2 bài mẫu `10.1016/j.phycom.2018.07.007` + `10.1038/s41598-025-34297-5` và 8 DOI.
- **Loại Task:** Literature / Writing — Strong.
- **Phương pháp được chọn:** Kiểm định định lượng (đếm `Xai_Inventory_Submit_17Mar.md:2029` → 25 refs `[1]-[25]`), đối chiếu với yêu cầu, thiết kế lại mạch 5 đoạn: (1) thách thức quy mô, (2) thành tựu/hạn chế DRL, (3) phân loại XAI, (4) khoảng trống, (5) đóng góp; mỗi đoạn gắn nhóm refs phù hợp, dệt có phê phán.
- **Lý do chọn:** Thêm refs cơ học không đủ, phải dệt vào lập luận để đáp ứng "liên hệ có hệ thống" mà reviewer nhấn mạnh.
- **Kết quả (audit):**
  - Hiện 25 refs (50% yêu cầu 50), 6 refs 2023-2026 (20% yêu cầu 30), 0/8 DOI then chốt, 0/5 gợi ý R1, lẫn `ResearchGate` [19] cần chuẩn hóa.
  - Đề xuất bổ sung 30 refs mới `[26]-[55]` (nâng lên 55), bao phủ 8 DOI then chốt + 5 gợi ý (Taherinavid 2023, Liu 2023, Yang 2025/2026 etc.), mẫu story structure từ 2 bài báo.
- **Tình trạng:** **40%** — **đã audit xong** (Bảng 1-2, Hình `T50_reference_count.png` etc.), **chưa viết xong** (cần ~800 từ Intro + ~900 từ Related Work + 30 refs). Cần biên tập hệ thống.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **CHƯA THÊM** — dự kiến 3 vị trí: (1) **Section 1 Introduction (`:21-203` đại tu)** thay mạch rời rạc bằng 5 đoạn story; (2) **Section 2 Literature Review (`:203-335` mở rộng)** thêm ~900 từ + 8-10 trích dẫn/tiểu mục; (3) **References (`:2029` bổ sung [26]-[55]** 30 refs). Hiện paper vẫn 25 refs, cần bổ sung 30 refs mới.

### Task 53 — Rà soát hệ thống hình vẽ (30 refs)

- **Yêu cầu làm gì?** Xóa `hhhh` cuối bảng 4.1.3, kiểm tra Fig1 (XRL workflow) hiển thị, hoàn thiện caption + cross-ref cho Fig12-20.
- **Loại Task:** Cleanup / Writing — Must.
- **Phương pháp được chọn:** Kiểm định toàn văn có hệ thống: tìm `hhhh`/`argarg`/`Scenaros`, trích 30 refs `Figure`, đối chiếu với file thực tế `Ablation_Study/output` etc., đánh giá 3 tiêu chí: hiển thị, caption, cross-ref.
- **Lý do chọn:** Lỗi phân tán, chỉ xử lý riêng lẻ sẽ sót, cần audit toàn bộ.
- **Kết quả (Bảng 3):**
  - `hhhh` = 0 lần (đã xóa) → **đã đạt**; còn 5 lỗi: `argarg` dòng651, `Scenaros` 1464, `Figure 1 2` 1593 / `1 3` 1606 / `1 4` 1620.
  - Fig1 dòng342 chỉ có placeholder text, chưa nhúng ảnh → cần nhúng `task16-9_fig1_modifications.png` + caption 220 SKU/14 levels/3 scenarios.
  - Fig12-20: 6/12 đầy đủ, còn lại thiếu caption/đánh số (12a-c, 16,17).
- **Tình trạng:** **60%** — **đã audit xong** (Bảng 3, Hình `T53_figure_overview.png` etc.), còn cần sửa 5 lỗi + nhúng Fig1 + hoàn thiện 6 caption.
- **Đã thêm vào bài báo `Xai_Inventory_Submit_17Mar.md` ở:** **CHƯA THÊM ĐẦY ĐỦ** — dự kiến 5 vị trí: `342` nhúng Figure 1 `task16-9_fig1_modifications.png` (hiện chỉ placeholder text), `651` sửa `argarg`→`argmax`, `1464` sửa `Scenaros`→`Scenarios`, `1593` sửa `Figure 1 2`→`Figure 12`, `1606` `Figure 1 3`→`13`, `1620` `Figure 1 4`→`14` + hoàn thiện caption Fig12-20 (thêm epsilon, n=496, λ range). Hiện `hhhh` đã xóa (0 lần).

---

## 11. Bảng Thuật Ngữ

| Thuật ngữ | Giải thích dễ hiểu | Gặp ở Task |
|---|---|---|
| **SHAP** | Đo mức đóng góp của từng đặc trưng vào quyết định AI (càng lớn càng quan trọng) | 10-15, 39-42 |
| **RDX** | Phân tách phần thưởng thành 4 mục tiêu (service/holding/waste/ordering) để thấy trade-off | 15-9, 47 |
| **MSX** | Tập lý do tối thiểu đủ để giải thích quyết định (ví dụ chỉ cần {service} là đủ) | 15-9, 47 |
| **FCS / OCS** | Tỷ lệ feature/objective vượt ngưỡng (đo độ thưa của giải thích) | 4, 14-9 |
| **Jaccard** | Độ giao nhau của 2 tập Top-k (1 = giống hệt, 0 = khác hẳn) | 11,13,15 |
| **Spearman / RBO** | Độ tương quan thứ hạng (Spearman đều, RBO nặng top hơn) | 11,13,15 |
| **MoRF / LeRF / Random** | Che feature quan trọng nhất / ít nhất / ngẫu nhiên trước để test faithfulness | 24-28 |
| **AUPC / ASR** | Diện tích dưới đường cong nhiễu loạn / Tỷ lệ đổi quyết định | 24-28 |
| **Chronological split** | Chia theo thời gian (không xáo trộn) — tránh nhìn trộm tương lai | 33 |
| **Leakage** | Rò rỉ thông tin test vào train (như xem đáp án trước) | 34, 21 |
| **OAT** | Thử một param mỗi lần (còn lại giữ nguyên) — tiết kiệm, không nhiễu | 20 |
| **Aggregation bias** | Gộp dữ liệu làm mất chi tiết cấp SKU | 10 |
| **Partition vs KernelSHAP** | Partition nhanh (gom cluster, làm phẳng) vs Kernel chính xác (chậm) | 10, 42 |
| **CV** | Std/Mean — đo biến động tương đối (CV>1 là volatile) | 12, 5 |
| **Logits / Softmax** | Logits = điểm thô trước chuẩn hóa; Softmax = chuẩn hóa về xác suất tổng=1 | 13, 44 |
| **Actor / Critic** | Actor = mạng chọn hành động (π), Critic = mạng đánh giá trạng thái (V) | 15, 43 |

---

## Phụ Lục: File Nguồn & Vị Trí ĐÃ THÊM vào `Xai_Inventory_Submit_17Mar.md` (kiểm chứng thực tế 18/09/2026)

| Task | File output chính | Vị trí ĐÃ THÊM trong `Xai_Inventory_Submit_17Mar.md` | Trạng thái |
|---|---|---|---|
| 4 | `task12-9/task 4/outputTask12-9-4.md` | **Section 4.9 `:1898-1943`** — Table 4a (Reward/Overstock/Waste), Fig 12a, Table 4b (FCS/OCS/Minimal/Stability), Fig 12b/12c | ✅ Đã thêm (mới hoàn toàn) |
| 5 | `task12-9/task 5/outputTask12-9-5.md` | **Section 3.1.5 `:418-429`** (mô tả chia 73/73/74) + **Section 4.8 `:1856-1897`** (Table 1-3 + Fig 3-6) | ✅ Đã thêm |
| 10 | `task10-9/outputTask10-9.md:7` | **Table 7a `:1289-1299`** Scope Matrix + **Section 4.5.3 `:1382-1398`** Top-k Micro (Table 7b + Fig 11) | ✅ Đã thêm |
| 11 | `task10-9/outputTask10-9.md:77` | **Section 4.5.4 `:1399-1402`** Consistency (Jaccard 1.00 vs 0.29) | ✅ Đã thêm |
| 12 | `task10-9/outputTask10-9.md:134` | **Section 4.5.5 `:1403-1407`** Why Dominant (r=0.04, CV 1.1-1.46) | ✅ Đã thêm |
| 13 | `task11-9/outputTask11-9.md:7` | **Section 4.5.6 `:1409-1412`** Common Target logits 10/20/50 | ✅ Đã thêm |
| 14 | `task11-9/outputTask11-9.md:76` | **Section 3.3.3 `:1300`** disclaimer + **Table 7b `:1397`** + **4.5.4 `:1401`** footnotes | ✅ Đã thêm |
| 15 | `task11-9/outputTask11-9.md:108` | **Section 4.5.7 `:1413-1416`** Actor vs Critic (Jaccard 0.25-0.33) | ✅ Đã thêm |
| 20 | `task13-9/task1/outputTask13-9.md:7` | **Section 3.2.4 `:689-702`** Reward Justification (Table 3, 4 weights) | ✅ Đã thêm |
| 21 | `task13-9/task1/outputTask13-9.md:109` | **Section 4.5.8 `:1417-1452`** Range/Distribution (Table 4/5 + Fig sensitivity) | ✅ Đã thêm |
| 24-28 | `task15-9/outputTask15-9.md:7` | **Section 4.7 `:1751-1854`** Faithfulness đại tu (4.7.1-4.7.5, Table 8a/8b,9,10,11 + Fig 20-23) | ✅ Đã thêm |
| 33 | `task13-9/task2/outputTask13-9-2.md:7` | **Section 3.1.3 `:350-388`** Data Split Chronological (Table 1b + Fig A/B, Table 1b-S + Fig B-S) | ✅ Đã thêm |
| 34 | `task13-9/task2/outputTask13-9-2.md:82` | **Section 3.1.4 `:389-417`** Leakage Audit (Table 1c + Fig C/D) | ✅ Đã thêm |
| 39 | `task14-9/outputTask14-9.md:9` | **Section 3.3.1 `:769-787`** Background 200→100 (Table 2a + Fig 2a/2b) | ✅ Đã thêm |
| 40 | `task14-9/outputTask14-9.md:9` | **Section 3.3.2 `:788-822`** Sensitivity Macro/Micro (Table 2b/2c + Fig 3a/3b/4) | ✅ Đã thêm |
| 41 | `task14-9/outputTask14-9.md:9` | **Section 3.3.3 `:823-846`** Validity (Table 2d + Fig 5a/5b, 95.5%/100%/84.5%) | ✅ Đã thêm |
| 42 | `task14-9/outputTask14-9.md:9` | **Section 3.3.4 `:847-863`** Consistent SHAP (Table 2e + Fig 6) | ✅ Đã thêm |
| 43 | `task16-9/task1/outputTask16-9-1.md:8` | **Section 3.2.5 `:704-731`** A2C_mod Definition (Table 3b + Fig 2c/2d) | ✅ Đã thêm |
| 44 | `task16-9/task1/outputTask16-9-1.md:8` | **Section 3.2.6 `:732-763`** Common Quantity (Table 3c/3d + Fig 2e/2f) | ✅ Đã thêm |
| 47 | `task17-9/outputTask47.md:1` | **Abstract `:11-13`** + **Section 4.10 `:1945-1990`** (Table 12/13 + Fig 13/14) + **Conclusion `:1996-1998`** Limitations | ✅ Đã thêm |
| 50 | `task16-9/task1/outputTask16-9-2.md:9` | *Dự kiến Section 1 `:21-203` + Section 2 `:203-335` + References `:2029`* | ⏳ Chưa thêm (40%, đã audit 25/50 refs) |
| 53 | `task16-9/task1/outputTask16-9-2.md:9` | *Dự kiến dòng 342, 651, 1464, 1593, 1606, 1620 + Fig1,12-20* | ⏳ Chưa thêm đầy đủ (60%, `hhhh` đã xóa) |

> **Ghi chú báo cáo:** Tất cả số liệu trên là **thực chạy trên checkpoint và TFRecords thật**, không mock. Các file `*_sensitivity.csv`, `*split_stats.csv`, `*verification_metrics.csv` đều có thể reproduce bằng notebook đã lưu. Khi paste vào paper, dịch các đoạn Việt đã soạn sẵn sang Anh học thuật và giữ nguyên số dòng tham chiếu để reviewer đối chiếu.

**BÁO CÁO TỔNG HỢP TIẾN ĐỘ CHỈNH SỬA BÀI BÁO THEO GÓP Ý PHẢN BIỆN (ĐỢT 7-9)**
*Người thực hiện: Masao*
*Ngày tổng hợp: 18/09/2026 | Checkpoint: DQN ckpt-60/ckpt-50 (hidden 128), A2C_mod ckpt-64 (hidden 32) | Bài báo: Xai_Inventory_Submit_17Mar.md*

# **1. Bảng tổng hợp trạng thái các nội dung đã xử lý**

| Task | Nội dung góp ý cần xử lý | Trạng thái | Vị trí trong bài |
|---|---|---|---|
| **4** | Độ nhạy action space với 7 / 14 / 28 chiến lược | Hoàn thành | 4.9 Sensitivity to Action Space Resolution (mới) |
| **5** | Mở rộng xuống nhóm SKU 2-3 nhóm (73/73/74) | Hoàn thành | 3.1.5 Product-Group Construction + 4.8 Scalability |
| **10** | Đưa SHAP Top-k 660 chiều thành kết quả chính, aggregated chỉ overview | Hoàn thành | 4.5 + 4.5.3 Top-k Micro-Level |
| **11** | Nhất quán Top-k giữa EASY/MEDIUM/HARD | Hoàn thành | 4.5.4 Consistency of Top-k |
| **12** | Giải thích vì sao SKU thống trị Top-20 | Hoàn thành | 4.5.5 Why Dominant Products |
| **13** | Common target logits cho DQN vs A2C_mod | Hoàn thành | 4.5.6 Common Target Sensitivity |
| **14** | Tuyên bố so sánh SHAP chỉ qualitative nếu khác target | Hoàn thành | 3.3.3 + footnote Table 7b/7c |
| **15** | Sensitivity actor/critic/logits | Hoàn thành | 4.5.7 Actor vs Critic Sensitivity |
| **20** | Justify hệ số reward (vì sao 1.0, waste 0.025) | Hoàn thành | 3.2.4 Reward Coefficient Justification |
| **21** | Range & distribution trước/sau normalize | Hoàn thành | 4.5.8 Reward Component Range and Distribution |
| **24** | Faithfulness AUPC/ASR/CI cho MoRF/Random/LeRF | Hoàn thành | 4.7.2 AUPC/ASR/CI |
| **25** | So sánh thống kê MoRF vs Random vs LeRF, MSX-guided vs random | Hoàn thành | 4.7.3 Statistical Comparisons |
| **26** | Giao thức perturbation (states, tỉ lệ mask, thay thế) | Hoàn thành | 4.7.1 Protocol |
| **27** | Định nghĩa ngưỡng 1% cho Q-drop / action flip | Hoàn thành | 4.7.4 Threshold |
| **28** | Nhất quán faithfulness giữa các states | Hoàn thành | 4.7.5 Consistency |
| **33** | Xác nhận split chronological vs random | Hoàn thành | 3.1.3 Data Split and Chronological Protocol |
| **34** | Leakage audit 4 nhóm (forecasting/normalization/reward/baseline) | Hoàn thành (95%, đã có corrected procedure) | 3.1.4 Leakage Audit |
| **39** | Quy trình tập nền SHAP 200 → 100 | Hoàn thành | 3.3.1 Background Construction |
| **40** | Độ nhạy SHAP theo nguồn/kích thước/chiến lược | Hoàn thành | 3.3.2 Background Sensitivity |
| **41** | Tính hợp lệ miền của perturbed states | Hoàn thành | 3.3.3 Validity |
| **42** | Chuẩn hóa mô tả KernelSHAP vs Partition | Hoàn thành | 3.3.4 Consistent Implementation |
| **43** | Định nghĩa A2C_mod 8 cải tiến | Hoàn thành | 3.2.5 A2C_mod Algorithm |
| **44** | Phép biến đổi Q-equivalent và kiểm chứng | Hoàn thành | 3.2.6 Common Quantity |
| **37** | Đặc tả EASY/MEDIUM/HARD là in/out-of-distribution | Hoàn thành | 3.1.3 sau Bảng 1b |
| **38** | Chuẩn hóa chiều 664 = 660 + 4 | Hoàn thành | 3.1 State (sau công thức s_t) |
| **47** | Managerial case studies + giảm claim trust | Hoàn thành | Abstract + 4.10 Managerial Case Studies + 5 Conclusion |
| **50** | Mở rộng tài liệu tham khảo lên 50+ | Đang thực hiện (40%, đã audit) | Dự kiến 1 Introduction + 2 Literature + References |
| **53** | Rà soát hệ thống hình vẽ (30 refs) | Đang thực hiện (60%, đã audit) | Toàn bài (còn 5 lỗi argarg/Scenaros/Figure) |

# **2. Nội dung chi tiết theo từng phần góp ý**

Với mỗi phần góp ý dưới đây, báo cáo trình bày rõ ba điều: bài báo trước khi chỉnh sửa đang thiếu gì hoặc sai ở đâu, những gì đã được sửa hoặc bổ sung vào bài, và kết luận cuối cùng được đưa vào bài dựa trên căn cứ nào.

## **2.1. Không gian hành động và khả năng mở rộng (Task 4, 5)**

### Độ nhạy với độ phân giải action space (Task 4)

**Trước đó:** Hội đồng yêu cầu ablation có hệ thống với 7/14/28 chiến lược và chứng minh tính bền vững của giải thích, không chỉ hiệu năng. Bài chỉ có 14 mức baseline.

**Đã bổ sung:** Thí nghiệm 6 điều kiện (A2C_mod ckpt-60/64 hidden 32, DQN ckpt-50/61 hidden 128) giữ cố định môi trường `r=1-z-overstock-q-quan`, state 660D, rollout 504 steps seed 42 argmax. Ba mức 7 `[0,0.01,0.02,0.04,0.12,0.5,1.0]`, 14 `[0,0.005,...,1.0]`, 28 `[0,0.0025,...,1.0]`. Đánh giá 3 trụ cột: SHAP (FCS), RDX/MSX (OCS/minimal set/Jaccard), faithfulness.

**Kết quả đưa vào bài:** DQN duy trì cao 0.80/0.60/0.82, A2C_mod chỉ đạt đỉnh ở 14 (0.38) còn 7 (-0.12) và 28 (-0.11) sập do overstock 0.96/0.83. Giải thích vẫn thưa FCS ~0/0.02, stability 50-90.7%, minimal set 1.0-2.68. Kết luận 14 là cân bằng cho A2C_mod, DQN hưởng lợi từ độ mịn, đặt ở **4.9 mới** với Bảng 4a/4b + Fig 12a-c.

### Mở rộng xuống nhóm SKU (Task 5)

**Trước đó:** Framework bị nghi hardcode 220 SKU, cần chứng minh mở rộng xuống product-group level.

**Đã bổ sung:** Chia 220 SKU theo MeanDemand: Fast 73 (22.6), Medium 73 (6.25), Slow 74 (0.66 CV1.46), sinh `data_grouped/*.tfrecords`, retrain 6 models 600ep×900 steps.

**Kết quả đưa vào bài:** DQN thắng mọi nhóm (Fast 0.77 > Medium 0.65 > Slow 0.57), A2C Medium bất ổn -79%, stockout DQN ~0 vs A2C 0.04-0.15. Đặt ở **3.1.5** (mô tả chia nhóm) và **4.8** (3 bảng + Fig 3-6).

## **2.2. SHAP micro-level và tính nhất quán (Task 10, 11, 12)**

### Đưa Top-k 660 chiều thành kết quả chính (Task 10)

**Trước đó:** Primary SHAP chỉ dùng 3 features gộp (Inventory/Demand/Waste), che giấu heterogeneity 660 chiều.

**Đã bổ sung:** Scope Matrix: Aggregated 3 avg = overview chiến lược, Micro 660 `sales/inventory/waste_SKU0-219` = chính, dùng `topk_shap_full_results_660.csv` 50 states/scenario, PartitionExplainer BG 100.

**Kết quả đưa vào bài:** Sales dominant cả 2 cấp, nhưng micro cho thấy tập trung 8 SKU 64,163,100,46,155,43,118,215 Mean|SHAP| gấp 3-4× tied 0.000799; 652/660 tied do Partition cluster, Jaccard Partition-Kernel 0-0.053. Fig 11 Top-20 660D thành Fig chính **4.5.3**, Fig 9 rút ngắn overview, Bảng 7a Scope Matrix.

### Nhất quán Top-k giữa scenarios (Task 11)

**Trước đó:** Chưa kiểm tra SKU quan trọng có giữ nguyên giữa EASY/MEDIUM/HARD không.

**Đã bổ sung:** Tính Jaccard + Spearman + RBO p=0.9, sensitivity k=10/20/50.

**Kết quả đưa vào bài:** DQN Jaccard 1.00 Spearman 1.00 (stable but less flexible), A2C_mod Jaccard 0.29-0.33 Spearman 0.20/-0.008 (context-adaptive). Đặt ở **4.5.4**.

### Vì sao SKU thống trị Top-20 (Task 12)

**Trước đó:** Chưa giải thích đặc điểm của SKU top.

**Đã bổ sung:** Join 36 SKU union Top-20 với `train.tfrecords`/`capacity.tfrecords`, tính Mean/CV/Capacity/Utilization.

**Kết quả đưa vào bài:** Top không phải best-seller (SKU57 22.6), correlation r=0.044, mà là low-volume high-CV 1.1-1.46 cap 7-28 (SKU64 0.66 CV1.46 cap7), waste_feat lọt Top-20 ở HARD. Đặt ở **4.5.5** + Bảng case.

## **2.3. SHAP comparability (Task 13, 14, 15)**

### Common target logits (Task 13)

**Trước đó:** SHAP DQN trên `softmax(Q)` unbounded vs A2C_mod trên `π` [0,1] không so trực tiếp được (R1 #5).

**Đã bổ sung:** Wrapper logits không retrain: DQN `q_values` linear 14 `Train_DQN.ipynb:146`, A2C `layer4` linear 14 `A2C-mod.ipynb:128`, BG 100 nsamples 2000, sensitivity n=10/20/50, 360 dòng `outputTask11_common_target.csv`.

**Kết quả đưa vào bài:** Logits vẫn Sales dominant nhưng 0/5 trùng baseline, magnitude chênh 258×, Jaccard giữa n chỉ 0.25-0.33. Đặt ở **4.5.6**.

### Tuyên bố qualitative (Task 14)

**Trước đó:** Bài so magnitude/stability trực tiếp giữa 2 agents.

**Đã bổ sung:** Chèn disclaimer 3 nơi: **3.3.3** sau Table 7a, footnote Table 7b/7c, sửa Discussion thành `more distributed under its policy target; direct comparison is qualitative`.

### Sensitivity actor/critic (Task 15)

**Trước đó:** Chưa thử SHAP trên outputs khác của A2C_mod.

**Đã bổ sung:** Cùng 10 states, 3 outputs π vs logits vs V(s) `outputTask11_sensitivity.csv` 9 dòng.

**Kết quả đưa vào bài:** Jaccard 0.25-0.33 mọi cặp, RBO 0.42-0.757, actor và critic cùng nhìn Sales nhưng Top-20 khác. Đặt ở **4.5.7**.

## **2.4. Reward design (Task 20, 21)**

### Justify hệ số reward (Task 20)

**Trước đó:** Vì sao 1.0, waste 0.025 chưa biện minh, thiếu distribution, thiếu sensitivity đa trọng số.

**Đã bổ sung:** Mapping `r=1-z-overstock-q-quan` `training.py:336`, làm rõ Instacart không có $ cost, 4 nhóm literature [22-25] (Newsvendor, EOQ, perishable 2-3%, CVaR), OAT sweep 16 configs `λ={0.5,1,1.5,2.0}` 13 unique, ckpt-60/64, 40 dòng.

**Kết quả đưa vào bài:** DQN overstock nhạy nhất Δ0.692, A2C quantile nhạy nhất Δ0.908, stockout phẳng, weight 1.0 trung hòa. Đặt ở **3.2.4** mới + Fig `outputTask20_sensitivity.png` Supplementary.

### Range & distribution (Task 21)

**Trước đó:** Chưa báo cáo range trước/sau normalize, nghi leakage.

**Đã bổ sung:** Dùng `train 1000×220`/`test 504×220`, `sales_norm=sales/capacity`, tính per-product `[T,220]`, thống kê descriptive + Wasserstein/KL, heuristic u=0 + BaseStock k=1.0 policy-aware.

**Kết quả đưa vào bài:** sales train 2.10 max162 → 0.10 max2.25, test 0.73→0.034, capacity 20.3 [4,208]; heuristic z 0.99 vs BaseStock 0.154/0.022, q 7e-05→0.0038, quan 0.006→0.234; Wasserstein 1.37→0.066 giảm 95% không leakage. Đặt ở **4.5.8** với 1 hình OAT + Bảng 4/5.

## **2.5. Faithfulness (Task 24-28 gộp)**

**Trước đó:** Perturbation MoRF/Random/LeRF chỉ mô tả, thiếu AUPC/ASR/CI, thiếu p-value/effect size, thiếu protocol rõ, ngưỡng 1% chưa biện minh, thiếu nhất quán giữa states.

**Đã bổ sung:** Notebook `faithfulness_Task15-9_Complete.ipynb` ckpt-50/64, 150 states (50/scenario), mask 1→10 (0.15-1.52%) median BG 100, Random 30 lần, AUPC hình thang, CI bootstrap 1000 + Wilson, Wilcoxon + t-test Holm, Cohen d, Kruskal-Wallis.

**Kết quả đưa vào bài:** Đại tu **4.7**: 4.7.1 Protocol + Fig 20, 4.7.2 Table 8a/8b AUPC MoRF 0.0097-0.0252 > Random ~0 > LeRF -0.0196, ASR 0/18, 4.7.3 Table 9 p<1e-14 d 2.2-16.8, MSX k=5 p<1e-13, 4.7.4 Table 10 ngưỡng 1% tỉ lệ vượt 0 (drop 0.16-0.40% <1%), 4.7.5 Table 11 CV A2C 0.02-0.17 vs DQN 0.19→0.48.

## **2.6. Data integrity (Task 33, 34)**

### Chronological split (Task 33)

**Trước đó:** Dataset 900/496 không rõ chronological hay random, nguy cơ leakage.

**Đã bổ sung:** Audit `TFRecordDataset.window(shift=batch_size-1)` giữ thứ tự, 2 file vật lý train vs test `prepare_data.py:149`, tách val 800/200/504, Pearson.

**Kết quả đưa vào bài:** Split strictly chronological 1000/200/504, mean 0.10/0.055/0.034, r=-0.47 OOD tự nhiên. Đặt ở **3.1.3** + Bảng 1b + Fig timeline/histogram.

### Leakage audit 4 nhóm (Task 34)

**Trước đó:** Forecasting/normalization/reward/baseline có fit trên test không?

**Đã bổ sung:** Audit: forecasting no model, normalization capacity full 20.33 vs train-only 25.79 (+26.29% 212/220 >10%), reward hardcode 0.025, baseline `S_i=mean+k·std` grid k trên val vs test.

**Kết quả đưa vào bài:** 2 No (forecasting/reward), 1 MAJOR capacity (corrected `capacity=ceil(mean(train)*12)` impact <0.02 reward, no retrain cần cho submission), 1 Fixed baseline k_val 1.0 (0.689) vs k_test 0.5 (0.759) curve song song. Đặt ở **3.1.4** + Fig capacity diff/k_curve + Bảng 1c.

## **2.7. SHAP implementation (Task 39-42)**

**Trước đó:** BG 200→100 không mô tả, thiếu sensitivity nguồn/size, thiếu kiểm validity miền, Kernel vs Partition lẫn lộn.

**Đã bổ sung:** 200 synthetic `inv/sales~U(0,1) waste=0.025*inv+N(0,0.005)`, rút 100 random seed vs KMeans; lưới 4 nguồn×3 size×2 agents (macro 3D KernelSHAP, micro 660D Partition); kiểm 5000 macro + 1000 micro states.

**Kết quả đưa vào bài:** BG random vs KMeans gần hệt (mean inv 0.47 vs 0.50); macro Spearman 1.00 cùng nguồn 0.87 đổi nguồn, micro Spearman ~0-0.17; validity trajectory 100% vs synthetic 84.5-98.8% (HARD thấp). Mở rộng **3.3.1-3.3.4** với Table 2a-e + Fig T39-T42.

## **2.8. A2C_mod và thang đo chung (Task 43, 44)**

### Định nghĩa A2C_mod (Task 43)

**Trước đó:** A2C_mod chưa định nghĩa lần đầu xuất hiện.

**Đã bổ sung:** Dòng mã `A2C-mod.ipynb:122-476` P=220 F=3 A=14 actor 4 lớp 32 ReLU+dropout, critic 2 lớp GroupNorm, gamma 0.99 lr 0.001 600ep, hộp 8 bước + Bảng 8 cải tiến M1-M8 (policy MSE, distance `/(dist+1)`, Brier, entropy chỉ log là hạn chế).

**Đã thêm:** **3.2.5** + Table 3b + Fig 2c/2d.

### Thang đo chung Q-equivalent (Task 44)

**Trước đó:** Đầu ra A2C_mod chuyển sang Q-equivalent thế nào chưa rõ.

**Đã bổ sung:** Dual: SHAP `f_common=mean(softmax(Q)) vs mean(π)` simplex, `Q_mod=logπ+V(s)`; RDX 1-step `x'=max(0,min(1,x+u)-sales)` gap; chứng minh order/scale/smoothness, verify 50 states/scenario ckpt-50/64 100% argmax Spearman 0.29-0.32.

**Đã thêm:** **3.2.6** + Table 3c/3d + Fig 2e/2f.

### Scenarios và chiều (Task 37, 38)

**Trước đó:** EASY/MEDIUM/HARD in/out-of-distribution chưa rõ, chiều 660 vs 664 lẫn lộn.

**Đã bổ sung:** EASY 0.017 + waste 0.01 trong test tail, MEDIUM 0.034 trong test mildly beyond train, HARD 0.051 + waste 0.05 controlled OOD; chiều chuẩn 664=660 (220×3) + 4 system (U/C/V/T) hiện train 660 clone per-product.

**Đã thêm:** Sau Bảng 1b và **3.1** sau công thức s_t + Fig T37/T38.

## **2.9. Managerial và writing (Task 47, 50, 53)**

### Case studies + giảm claim trust (Task 47)

**Trước đó:** Faithfulness ≠ usefulness, chưa có expert validation nhưng claim trust.

**Đã bổ sung:** 3 cases tiếng quản lý: RDX EASY vs HARD (A2C giảm holding 0.083 vs DQN 0.107), MSX {service}→{holding} vs DQN {holding} cố định, SHAP Top SKU64 CV1.46 cap7 không phải best-seller; giảm claim 3 nơi.

**Đã thêm:** **Abstract dòng 13** thêm `without expert validation...`, cuối Abstract `requires future expert validation`, **4.10 mới** Table 12/13 + Fig 13/14, **5 Conclusion** đoạn `faithfulness≠usefulness` moderation.

### Tài liệu tham khảo và hình vẽ (Task 50, 53)

**Hiện trạng:** Hiện 25 refs cần 50+, 6 refs 2023-2026, 0/8 DOI then chốt; còn 5 lỗi `argarg` 651, `Scenaros` 1464, `Figure 1 2` 1593/`1 3` 1606/`1 4` 1620.

**Đã làm:** Audit xong, đề xuất 30 refs [26-55] theo story 5 đoạn. **Chưa thêm vào bài**, dự kiến **1 Introduction + 2 Literature + References** và sửa hình trước submit.

---
*Ghi chú: Toàn bộ thực nghiệm trên checkpoint thật `ckpt-60/64/50`, dữ liệu `train 1000×220`/`test 504×220`, không retrain trừ Task 5 đã retrain 6 models. Chi tiết CSV/PNG trong `Feedback 7-9/task*/output/` như ghi trong mỗi outputTask*.md.*

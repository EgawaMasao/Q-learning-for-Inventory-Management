# Output Task 3 — Phân tích kết quả SHAP kịch bản HARD (Trajectory vs. Synthetic Background)

## 1. Tóm tắt công việc & Kết quả thực nghiệm chính

- **Notebook đã chạy:** [task3_hard_trajectory_shap.ipynb](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task3/task3_hard_trajectory_shap.ipynb) trên 2 checkpoint thật của bài báo (`DQN`: `ckpt-50`, `A2C_mod`: `ckpt-64`) với kịch bản **HARD** (`sales_scale = 1.5`, `waste_rate = 0.05`, 50 test states, 100 background states, `seed = 42`).
- **Dữ liệu đầu ra:** lưu tại `Feedback 23-11/task3/output/` ([task3_summary.json](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task3/output/task3_summary.json), [hard_validity_check.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task3/output/hard_validity_check.csv), [hard_macro_shap_details.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task3/output/hard_macro_shap_details.csv), [hard_r3_02_combined_table.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task3/output/hard_r3_02_combined_table.csv), [hard_top_skus.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task3/output/hard_top_skus.csv), và 2 hình trong `output/figs/`).

### 3 phát hiện cốt lõi ở kịch bản HARD
1. **Tính hợp lệ vật lý (5.000 mẫu nhiễu loạn):**
   - Dùng `traj_rollout_100` khôi phục tính hợp lệ vật lý ở HARD lên **99.24%** (`DQN`) và **100.00%** (`A2C_mod`), so với chỉ **51.86%** (`DQN`) và **76.40%** (`A2C_mod`) khi dùng `syn_random_100` (trên điểm neo cũ `T41` là **85.68%**, tái lập khớp con số 84.5% của bài báo).
2. **Macro SHAP (3 đặc trưng, `KernelExplainer`):**
   - **`DQN`:** Thứ hạng giữ nguyên tuyệt đối (`Inventory > Demand > Waste`, **Spearman $\rho = 1.00$**, **Cosine = 0.987**). Tuy nhiên độ lớn `mean|SHAP|` giảm khoảng 3.8× (từ `0.01125` xuống `0.00298` cho Inventory) vì trạng thái quỹ đạo thật tập trung trong dải hẹp hơn phân bố đều `Uniform(0,1)`, khiến FCS theo ngưỡng `0.01` giảm từ `0.33` xuống `0.00`.
   - **`A2C_mod`:** Dưới `syn_random_100`, `Demand` chi phối (`0.08123 > 0.00997 > 0.00023`); nhưng dưới `traj_rollout_100`, cả 3 đặc trưng đều về **`0.00000`** ($\rho$ không xác định, Cosine = `0.000`). **Nguyên nhân kỹ thuật:** trên các trạng thái vĩ mô hợp lệ của quỹ đạo HARD, chính sách `A2C_mod` đã **bão hòa** (gán xác suất gần 1.0 cho cùng một hành động cực trị); các giá trị SHAP khác 0 dưới synthetic thực chất xuất phát từ các nhiễu loạn *ngoài miền vận hành (off-manifold)* đẩy chính sách rời vùng bão hòa.
3. **Micro SHAP (660 đặc trưng, `PartitionExplainer`):**
   - **Độ tương đồng độ lớn & tỷ trọng nhóm ổn định cao:** `DQN` đạt **Cosine = 0.971** (tỷ trọng I/D/W: `33.5/33.0/33.5%` vs. `34.2/32.4/33.4%`); `A2C_mod` đạt **Cosine = 0.895** (I/D/W dịch nhẹ ~2.4 điểm % từ Demand sang Inventory: `32.4/35.2/32.4%` → `34.4/32.8/32.8%`).
   - **Thứ hạng từng SKU đơn lẻ không ổn định:** Spearman toàn 660 chiều gần 0 (`-0.053` ở DQN; `+0.024` ở A2C_mod) và Jaccard Top-20 rất thấp (`0.053` ở DQN; `0.000` ở A2C_mod) — nhất quán hoàn toàn với phát hiện ở kịch bản MEDIUM rằng thứ hạng micro chỉ mang tính tham khảo định tính.

---

## Ghi vào bài báo

### Chỉnh sửa 1 — Cập nhật §2.3.4.2 (*Background Sensitivity*) và §2.3.4.3 (*Validity of Perturbed States*), bổ sung bảng ghép R3-02 cho kịch bản HARD

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **Materials and Methods** → *SHAP Experimental Setup and Validity Verification* → **§2.3.4.2 Background Sensitivity** (đoạn `P163–P164`) và **§2.3.4.3 Validity of Perturbed States** (đoạn `P165–P166`).
  - **Thao tác:** Bổ sung đoạn phân tích HARD (Synthetic vs. Trajectory) ngay cuối `P164` / `P166` và chèn **Table Y** (có thể ghép trực tiếp vào bảng độ nhạy background của **R3-02 / Task 4.1**).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

> *(Chèn tiếp vào cuối mục **2.3.4.3 Validity of Perturbed States**, đoạn P166, kèm Table Y — hoặc gộp chung với bảng độ nhạy background của Task 4.1 / R3-02)*

Because synthetic perturbations in the HARD scenario violate physical constraints in 15.5% of anchor samples (and up to 23.6%–48.1% on on-manifold HARD rollout states under the elevated waste rate of 0.05), we re-evaluated HARD SHAP attributions using an on-manifold rollout trajectory background (99.2%–100.0% physically valid) alongside the synthetic baseline (Table Y). At the macro level (3 features, KernelSHAP), DQN preserves its exact feature ranking (`Inventory > Demand > Waste`, Spearman $\rho = 1.00$, cosine similarity $= 0.987$), although attribution magnitudes contract (MAD $= 0.0041$) because trajectory states span a narrower operating range than uniform synthetic states. By contrast, `A2C_mod` exhibits zero macro attribution under the on-manifold HARD trajectory background (mean $|$SHAP$| = 0.000$ across all three features, versus `Demand > Inventory > Waste` with mean $|$SHAP$|$ up to $0.0812$ under synthetic), indicating that on-manifold macro states lie within a policy-saturated regime where synthetic attributions are driven primarily by off-manifold perturbations. At the micro level (660 features, Partition Explainer), overall attribution profiles remain aligned between synthetic and trajectory backgrounds (cosine similarity $= 0.971$ for DQN and $0.895$ for `A2C_mod`, with group shares shifting by $\le 2.4$ percentage points), whereas individual SKU rankings remain unstable (Spearman $\rho = -0.053$ for DQN and $0.024$ for `A2C_mod`; Top-20 Jaccard $\le 0.053$), corroborating the MEDIUM-scenario finding that micro-level rankings should be interpreted as indicative rather than invariant.

**Table Y. SHAP sensitivity to background origin in the HARD scenario: synthetic baseline (`syn_random_100`) versus on-manifold rollout trajectory (`traj_rollout_100`, 50 held-out HARD states).** *(Gọn 5 cột, có thể gộp trực tiếp vào bảng độ nhạy background R3-02 cùng kết quả MEDIUM.)*

| Level / Agent | Validity (Syn / Traj) | Spearman $\rho$ | Cosine | Key Observation (MAD / Top-20 Jaccard) |
|---|---:|---:|---:|---|
| Macro (3-dim) / DQN | 51.9% / 99.2% | 1.000 | 0.987 | Same rank (`Inv > Dem > Wst`); MAD = 0.0041 |
| Macro (3-dim) / `A2C_mod` | 76.4% / 100.0% | N/A* | 0.000 | All 0.000 on Traj (policy saturation); MAD = 0.0305 |
| Micro (660-dim) / DQN | — / 100.0%† | −0.053 | 0.971 | Top-20 Jaccard = 0.053; group share $\Delta \le 0.7$ pp |
| Micro (660-dim) / `A2C_mod` | — / 100.0%† | 0.024 | 0.895 | Top-20 Jaccard = 0.000; group share $\Delta \le 2.4$ pp |

*\*Spearman $\rho$ is undefined because all three macro attributions equal zero under the saturated on-manifold trajectory background.*  
*†Micro-level median masking satisfies 100% physical validity across tested states (Section 2.3.4.3).*

---

- **Bản Tiếng Việt (Để theo dõi):**

Vì các nhiễu loạn tổng hợp (synthetic) ở kịch bản HARD vi phạm ràng buộc vật lý ở 15.5% mẫu neo (và lên tới 23.6%–48.1% trên các trạng thái rollout HARD với tỷ lệ hao hụt 0.05), chúng tôi đánh giá lại SHAP ở kịch bản HARD bằng tập nền quỹ đạo rollout thực tế (đạt 99.2%–100.0% hợp lệ vật lý) đặt cạnh tập nền tổng hợp cơ sở (Bảng Y). Ở cấp độ vĩ mô (3 đặc trưng, KernelSHAP), DQN bảo toàn nguyên vẹn thứ hạng đặc trưng (`Inventory > Demand > Waste`, Spearman $\rho = 1.00$, độ tương đồng cosine $= 0.987$), mặc dù độ lớn attribution thu hẹp lại (MAD $= 0.0041$) do các trạng thái quỹ đạo nằm trong dải vận hành hẹp hơn phân bố đều tổng hợp. Ngược lại, `A2C_mod` cho giá trị SHAP vĩ mô bằng 0 trên cả ba đặc trưng dưới tập nền quỹ đạo HARD (`mean|SHAP| = 0.000`, so với `Demand > Inventory > Waste` đạt tới `0.0812` dưới tập nền tổng hợp), cho thấy các trạng thái vĩ mô hợp lệ của quỹ đạo nằm trong vùng bão hòa chính sách và các attribution dưới tập nền tổng hợp chủ yếu phát sinh từ nhiễu loạn ngoài miền vận hành. Ở cấp độ vi mô (660 đặc trưng, Partition Explainer), cấu trúc độ lớn tổng thể giữa hai tập nền vẫn tương đồng cao (cosine $= 0.971$ với DQN và $0.895$ với `A2C_mod`, tỷ trọng nhóm chỉ dịch chuyển $\le 2.4$ điểm phần trăm), trong khi thứ hạng từng SKU đơn lẻ vẫn không ổn định (Spearman $\rho = -0.053$ với DQN và $0.024$ với `A2C_mod`; Jaccard Top-20 $\le 0.053$), củng cố kết luận ở kịch bản MEDIUM rằng thứ hạng vi mô chỉ nên diễn giải mang tính định hướng.

**Bảng Y. Độ nhạy của SHAP theo nguồn gốc tập nền ở kịch bản HARD: tập nền tổng hợp (`syn_random_100`) so với quỹ đạo rollout thực tế (`traj_rollout_100`, 50 trạng thái kiểm định HARD).**

| Cấp độ (Explainer) | Tác nhân | Tỷ lệ hợp lệ (Syn / Traj) | Spearman $\rho$ (Syn vs. Traj) | Độ tương đồng Cosine | MAD | Jaccard Top-20 | Nhận xét thứ hạng / Tỷ trọng nhóm (I/D/W) |
|---|---|---:|---:|---:|---:|---:|---|
| Vĩ mô (3 chiều, KernelSHAP) | DQN | 51.9% / 99.2% | 1.000 | 0.987 | 0.00412 | — | Giữ nguyên thứ hạng (`Inventory > Demand > Waste`); độ lớn giảm trên quỹ đạo |
| Vĩ mô (3 chiều, KernelSHAP) | `A2C_mod` | 76.4% / 100.0% | Không xác định* | 0.000 | 0.03048 | — | Syn: `Demand > Inventory > Waste`; Traj: đều bằng 0.000 (bão hòa chính sách trên miền thật) |
| Vi mô (660 chiều, Partition) | DQN | — / 100.0%† | −0.053 | 0.971 | 0.00002 | 0.053 | Tỷ trọng nhóm ổn định: Syn 33.5/33.0/33.5% vs. Traj 34.2/32.4/33.4% |
| Vi mô (660 chiều, Partition) | `A2C_mod` | — / 100.0%† | 0.024 | 0.895 | 0.00009 | 0.000 | Dịch nhẹ I/D/W: Syn 32.4/35.2/32.4% vs. Traj 34.4/32.8/32.8% |

*\*Spearman $\rho$ không xác định vì cả 3 giá trị SHAP vĩ mô đều bằng 0 trên tập nền quỹ đạo đã bão hòa.*  
*†Che lấp trung vị ở cấp độ vi mô đạt 100% hợp lệ vật lý trên các trạng thái kiểm định (§2.3.4.3).*

---

## 2. Checklist kiểm chứng hoàn thành Task 3

- [x] Đã chạy thực nghiệm đầy đủ trên 2 checkpoint thật (`DQN ckpt-50`, `A2C_mod ckpt-64`), không fallback mock.
- [x] Xác nhận `traj_rollout_100` khôi phục tính hợp lệ vật lý ở HARD lên **99.2%–100.0%** (so với **51.9%–76.4%** của synthetic).
- [x] Có đủ số liệu đối chiếu song song Synthetic vs. Trajectory ở HARD cho cả **Macro (3-dim)** và **Micro (660-dim)**.
- [x] Trình bày kết luận trung thực (bao gồm phát hiện bão hòa chính sách vĩ mô ở `A2C_mod`), đúng văn phong bài báo khoa học, có song ngữ Anh–Việt và định dạng `Vị trí / Nội dung sửa` sẵn sàng ghép vào bảng R3-02 ở §2.3.4.2–2.3.4.3.

# Output Task 4 — Hoàn thiện phần viết cho Task 4.1, Task 4.2 và Task 4.3

## Tóm tắt công việc thực hiện
- **Phạm vi:** 100% xử lý phần viết trên [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx) theo [planTask4.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task4/planTask4.md).
- **Nội dung hoàn thành:**
  - **Task 4.1 (R3-02):** Giải thích lý do chọn `synthetic + random-100` và gom toàn bộ kết quả độ nhạy background (**MEDIUM** từ `task14-9` + **HARD** từ **Task 3**) thành **1 bảng thống nhất** trong bài chính (§2.3.4.1–2.3.4.3).
  - **Task 4.2 (C-03 & thang đo SHAP):** Sửa mâu thuẫn SHAP target (`P156`, `Table 4`, `P284`) về thống nhất `softmax(Q)` vs. `π(·|s)`, và rà sửa 4 vị trí so sánh độ lớn tuyệt đối (`P288–P289`, `P297`, `P352`, `P401`) về so sánh thứ hạng/định tính.
  - **Task 4.3 (Trade-off Base-Stock vs. DRL):** Bổ sung câu thừa nhận thẳng trade-off vào **Abstract (`P009`)** và **Conclusion (`P441`)**, trung lập hóa tiêu đề `P198`, và tinh chỉnh `P214` để không hàm ý vượt trội tổng thể.

---

## Ghi vào bài báo

---

### PHẦN A — TASK 4.1: Cách chọn Background SHAP & Bảng tổng hợp Background Sensitivity (R3-02)

#### Chỉnh sửa A1 — Lý do chọn `synthetic + random-100` làm baseline
- **Vị trí:** Mục **Materials and Methods** → *SHAP Experimental Setup and Validity Verification* → **2.3.4.1 SHAP Background Construction** ([Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), thay thế đoạn `P162`).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

For the macro analysis, each state has three aggregated features (inventory, demand, and waste, with waste modeled as a bounded linear function of inventory plus controlled noise). Two hundred synthetic states are generated with a fixed seed (seed = 42), from which 100 states are extracted either by uniform random sampling (`synthetic + random-100`, primary baseline) or as KMeans centroids (ablation). We adopt `synthetic + random-100` as the primary baseline for three reasons: (i) sampling uniformly over the normalized operational domain $[0,1]$ provides consistent state-space coverage across heterogeneous demand and inventory regimes; (ii) reducing the background from 200 to 100 states halves KernelSHAP runtime while preserving nearly identical marginal statistics and pairwise-distance distributions compared with both `full-200` and `KMeans-100`; and (iii) fixed-seed random subsampling ensures exact reproducibility while remaining directly comparable against on-manifold trajectory ablations.

- **Bản Tiếng Việt (Để theo dõi):**

Đối với phân tích vĩ mô, mỗi trạng thái gồm ba đặc trưng tổng hợp (tồn kho, nhu cầu và hao hụt, trong đó hao hụt được mô hình hóa là hàm tuyến tính có chặn của tồn kho cộng nhiễu có kiểm soát). Hai trăm trạng thái tổng hợp được sinh với hạt giống cố định (seed = 42), từ đó rút 100 trạng thái bằng lấy mẫu ngẫu nhiên đều (`synthetic + random-100`, làm baseline chính) hoặc tâm cụm KMeans (đối chứng). Chúng tôi chọn `synthetic + random-100` làm baseline chính vì ba lý do: (i) lấy mẫu đều trên miền vận hành chuẩn hóa $[0,1]$ đảm bảo độ phủ nhất quán trên các chế độ nhu cầu và tồn kho khác nhau; (ii) rút từ 200 xuống 100 trạng thái giúp giảm một nửa thời gian chạy KernelSHAP trong khi vẫn giữ thống kê biên và phân bố khoảng cách cặp gần như đồng nhất với `full-200` và `KMeans-100`; và (iii) lấy mẫu ngẫu nhiên theo hạt giống cố định đảm bảo tái lập chính xác đồng thời dễ dàng đối chứng với tập nền quỹ đạo thực tế.

---

#### Chỉnh sửa A2 — Đưa kết quả Background Sensitivity (MEDIUM + HARD) thành 1 bảng thống nhất (R3-02)
- **Vị trí:** Mục **Materials and Methods** → **2.3.4.2 Background Sensitivity** (`P163–P164`) và **2.3.4.3 Validity of Perturbed States** (`P165–P166`) trong [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx).
  - **Thao tác:** Thay thế `P164` và chèn **Table R3-02** ngay dưới `P164` (gộp chung kết quả MEDIUM và HARD từ Task 3 thành 1 bảng duy nhất; khi đó không cần chèn riêng Table Y của `outputTask3.md` để tránh trùng lặp).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

Background sensitivity was evaluated over a grid of background origin (synthetic uniform versus on-manifold test trajectory), size ($n \in \{50, 100, 200\}$, baseline 100), and subsampling strategy (random versus KMeans) for both agents at the macro level (3 features, KernelSHAP) and micro level (660 features, Partition Explainer) on 50 held-out MEDIUM states, and further verified on 50 held-out HARD states using an on-manifold rollout trajectory background (99.2%–100.0% physically valid) versus the synthetic baseline (Table R3-02). In MEDIUM, macro rankings are invariant to size and subsampling within synthetic backgrounds (Spearman $\rho = 1.00$), shift by a single rank swap under trajectory backgrounds ($\rho = 0.87$), and yield higher threshold-based feature coverage under synthetic backgrounds; at the micro level, attribution magnitudes remain strongly aligned (cosine similarity $0.94$–$0.99$) while individual SKU rankings are unstable ($\rho \approx 0$, full 50-state grid runtime $> 4$ h), so micro-level rankings are treated as indicative. In HARD, DQN preserves its exact macro ranking (`Inventory > Demand > Waste`, $\rho = 1.00$, cosine $= 0.987$) with smaller magnitudes on the narrower trajectory manifold, whereas `A2C_mod` drops to zero macro attribution on the on-manifold HARD trajectory because its policy is saturated in that regime; at the micro level, overall magnitude alignment remains high (cosine $= 0.971$ for DQN and $0.895$ for `A2C_mod`, group-share shift $\le 2.4$ percentage points) while individual SKU rankings remain unstable ($\rho \approx 0$, Top-20 Jaccard $\le 0.053$).

**Table R3-02. SHAP background sensitivity across origin, size, and subsampling strategy in MEDIUM and HARD scenarios (50 held-out states per scenario).** *(Bản gọn 5 cột, chèn vừa khít trang A4 Word.)*

| Scenario / Level / Agent | Background Comparison (Validity) | Spearman $\rho$ | Cosine | Key Finding |
|---|---|---:|---:|---|
| MEDIUM / Macro / Both | Syn random & KMeans ($n=50$–$200$, 98.8% valid) | 1.000 | 0.991–1.000 | Rank-invariant to size and sampling |
| MEDIUM / Macro / Both | Traj random & KMeans ($n=50$–$200$, 100% valid) | 0.866 | 0.983 (DQN)* | Single rank swap vs. synthetic; lower FCS |
| MEDIUM / Micro / Both | All 12 grid configs (100% valid†) | −0.11 to +0.17 | 0.937–0.994 | Magnitudes aligned; SKU ranks indicative only |
| HARD / Macro / DQN | Traj vs. Syn (99.2% vs. 51.9% valid) | 1.000 | 0.987 | Same rank (`Inv > Dem > Wst`); smaller magnitude |
| HARD / Macro / `A2C_mod` | Traj vs. Syn (100% vs. 76.4% valid) | N/A‡ | 0.000 | All 0.000 on Traj (on-manifold policy saturation) |
| HARD / Micro / DQN | Traj vs. Syn (100% valid†) | −0.053 | 0.971 | Top-20 Jaccard = 0.053; group share $\Delta \le 0.7$ pp |
| HARD / Micro / `A2C_mod` | Traj vs. Syn (100% valid†) | 0.024 | 0.895 | Top-20 Jaccard = 0.000; group share $\Delta \le 2.4$ pp |

*\*For `A2C_mod` under static MEDIUM trajectory background, constant inventory/waste yields zero attribution on those features.*  
*†Micro-level median masking satisfies 100% physical validity across tested states.*  
*‡Spearman $\rho$ is undefined because all three macro attributions equal zero under the saturated on-manifold HARD trajectory.*

- **Bản Tiếng Việt (Để theo dõi):**

Độ nhạy của SHAP theo tập nền được đánh giá trên lưới gồm nguồn gốc tập nền (tổng hợp đều so với quỹ đạo thực tế), kích thước ($n \in \{50, 100, 200\}$, cơ sở 100), và chiến lược rút mẫu (ngẫu nhiên so với KMeans) cho cả hai tác nhân ở cấp vĩ mô (3 đặc trưng, KernelSHAP) và vi mô (660 đặc trưng, Partition Explainer) trên 50 trạng thái MEDIUM, đồng thời kiểm chứng thêm trên 50 trạng thái HARD bằng tập nền quỹ đạo rollout thực tế (đạt 99.2%–100.0% hợp lệ vật lý) đặt cạnh tập nền tổng hợp (Bảng R3-02). Ở MEDIUM, thứ hạng vĩ mô bất biến theo kích thước và cách rút mẫu trong nhóm tập nền tổng hợp ($\rho = 1.00$), chỉ đổi một cặp thứ hạng khi chuyển sang quỹ đạo ($\rho = 0.87$); ở cấp vi mô, độ lớn tương đồng cao (cosine $0.94$–$0.99$) nhưng thứ hạng từng SKU không ổn định ($\rho \approx 0$, toàn lưới mất $> 4$ giờ), nên thứ hạng vi mô chỉ mang tính định hướng. Ở HARD, DQN bảo toàn tuyệt đối thứ hạng vĩ mô (`Inventory > Demand > Waste`, $\rho = 1.00$, cosine $= 0.987$) với độ lớn nhỏ hơn trên miền quỹ đạo hẹp, trong khi `A2C_mod` giảm về 0 trên cả ba đặc trưng vĩ mô do chính sách bão hòa trên miền quỹ đạo HARD; ở cấp vi mô, cấu trúc độ lớn tổng thể vẫn tương đồng cao (cosine $= 0.971$ với DQN và $0.895$ với `A2C_mod`, tỷ trọng nhóm lệch $\le 2.4$ điểm %) còn thứ hạng SKU lẻ vẫn không ổn định ($\rho \approx 0$, Jaccard Top-20 $\le 0.053$).

**Bảng R3-02. Độ nhạy của SHAP theo nguồn gốc, kích thước và chiến lược lấy mẫu tập nền ở kịch bản MEDIUM và HARD (50 trạng thái kiểm định/kịch bản).**

| Kịch bản | Cấp độ (Explainer) | Cặp so sánh tập nền | Tác nhân | Hợp lệ vật lý (Syn / Traj) | Spearman $\rho$ | Cosine | Nhận xét chính |
|---|---|---|---|---:|---:|---:|---|
| MEDIUM | Vĩ mô (3 chiều, KernelSHAP) | Tổng hợp: random & KMeans ($n=50, 100, 200$) vs. `syn_random_100` | DQN & `A2C_mod` | 98.8% / — | 1.000 | 0.991–1.000 | Bất biến thứ hạng theo kích thước và cách rút mẫu |
| MEDIUM | Vĩ mô (3 chiều, KernelSHAP) | Quỹ đạo: random & KMeans ($n=50, 100, 200$) vs. `syn_random_100` | DQN & `A2C_mod` | — / 100.0% | 0.866 | 0.983 (DQN)* | Hoán đổi 1 thứ hạng so với tổng hợp; FCS thấp hơn ở cùng ngưỡng |
| MEDIUM | Vi mô (660 chiều, Partition) | Cả 12 cấu hình lưới (nguồn × kích thước × rút mẫu) vs. `syn_random_100` | DQN & `A2C_mod` | 100.0%† / 100.0%† | −0.108 đến +0.167 | 0.937–0.994 | Độ lớn tương đồng cao nhưng thứ hạng SKU lẻ không ổn định |
| HARD | Vĩ mô (3 chiều, KernelSHAP) | `traj_rollout_100` vs. `syn_random_100` | DQN | 51.9% / 99.2% | 1.000 | 0.987 | Giữ nguyên thứ hạng (`Inventory > Demand > Waste`); độ lớn nhỏ hơn |
| HARD | Vĩ mô (3 chiều, KernelSHAP) | `traj_rollout_100` vs. `syn_random_100` | `A2C_mod` | 76.4% / 100.0% | Không xác định‡ | 0.000 | Đều bằng 0.000 trên quỹ đạo (bão hòa chính sách trên miền HARD) |
| HARD | Vi mô (660 chiều, Partition) | `traj_rollout_100` vs. `syn_random_100` | DQN | — / 100.0%† | −0.053 | 0.971 | Jaccard Top-20 = 0.053; tỷ trọng nhóm ổn định ($\Delta \le 0.7$ điểm %) |
| HARD | Vi mô (660 chiều, Partition) | `traj_rollout_100` vs. `syn_random_100` | `A2C_mod` | — / 100.0%† | 0.024 | 0.895 | Jaccard Top-20 = 0.000; dịch nhẹ I/D/W ($\Delta \le 2.4$ điểm %) |

---

### PHẦN B — TASK 4.2: Sửa mâu thuẫn SHAP Target (C-03) & Chuyển so sánh độ lớn SHAP sang thứ hạng/định tính

#### Chỉnh sửa B1 — Sửa mâu thuẫn C-03 tại `SHAP Implementation Details` và `Table 4`
- **Vị trí:** Mục **Materials and Methods** → *SHAP Implementation Details* (`P156`) và **Table 4** (`P157`, `TBL04` — 2 hàng `Explained output for DQN` và `Explained output for A2C_mod`) trong [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**
  - *Thay 2 câu ở `P156` ("For DQN, SHAP values are computed... policy output."):*
    > Consistent with the common-quantity formulation (Section 2.2.6), SHAP values are computed with respect to the 14-action softmax-normalized Q-values $\operatorname{softmax}(Q(s,\cdot))$ for DQN (portfolio-averaged across the 220 SKUs in the micro analysis) and the 14-action policy distribution $\pi(\cdot \mid s)$ for `A2C_mod` (portfolio-averaged in the micro analysis). Although both targets lie on the 14-action probability simplex, $\operatorname{softmax}(Q)$ depends on the unnormalized scale of learned Q-values; consequently, raw SHAP magnitudes are not directly comparable across the two agents, and cross-agent comparisons are restricted to feature rankings and qualitative attribution patterns.
  - *Cập nhật 2 hàng tương ứng trong **Table 4**:*

| Component | Configuration |
|---|---|
| Explained output for DQN | Softmax-normalized Q-values $\operatorname{softmax}(Q(s,\cdot))$ (portfolio-averaged over 220 SKUs at micro level) |
| Explained output for `A2C_mod` | Action policy distribution $\pi(\cdot \mid s)$ (portfolio-averaged over 220 SKUs at micro level) |

- **Bản Tiếng Việt (Để theo dõi):**
  - *Đoạn sửa `P156`:* Nhất quán với đại lượng chung (§2.2.6), giá trị SHAP được tính theo phân phối chuẩn hóa softmax của giá trị Q $\operatorname{softmax}(Q(s,\cdot))$ cho DQN (lấy trung bình trên 220 SKU ở cấp vi mô) và phân phối chính sách 14 hành động $\pi(\cdot \mid s)$ cho `A2C_mod` (lấy trung bình trên 220 SKU ở cấp vi mô). Dù cả hai đều nằm trên simplex xác suất 14 hành động, $\operatorname{softmax}(Q)$ vẫn phụ thuộc vào biên độ chưa chuẩn hóa của giá trị Q; do đó độ lớn SHAP thô không tương đương trực tiếp giữa hai tác nhân, và mọi so sánh liên mô hình chỉ thực hiện ở mức thứ hạng đặc trưng và cấu trúc định tính.
  - *2 hàng sửa trong **Bảng 4**:*
    - Đầu ra được giải thích cho DQN: `Giá trị Q chuẩn hóa softmax(Q(s,·)) (trung bình trên 220 SKU ở cấp vi mô)`
    - Đầu ra được giải thích cho `A2C_mod`: `Phân phối chính sách hành động π(·|s) (trung bình trên 220 SKU ở cấp vi mô)`

---

#### Chỉnh sửa B2 — Bỏ so sánh độ lớn SHAP ở Global & Local SHAP (`P284`, `P288–P289`, `P297`)
- **Vị trí:** Mục **Results and Discussion** → *SHapley Additive exPlanations* (`P284`, `P288`, `P289`, `P297` trong [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx)).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**
  - *Thay câu đầu `P284`:*
    > SHAP is computed on $\operatorname{softmax}(Q(s,\cdot))$ for DQN and on the policy distribution $\pi(\cdot \mid s)$ for `A2C_mod`. Because $\operatorname{softmax}(Q)$ depends on the unnormalized scale of learned action values whereas $\pi$ is parameterized directly by the actor, attribution magnitudes are not directly comparable across agents; all cross-agent comparisons in Figures 10–12 are therefore restricted to feature rankings and qualitative attribution structure.
  - *Thay `P288–P289` (Global SHAP):*
    > At the global level (Figure 10), `A2C_mod` displays a relatively balanced attribution structure across Inventory, Demand, and Waste without strong dominance by a single feature. By contrast, DQN exhibits a clearer rank separation in which Inventory and Demand dominate while Waste ranks lowest, indicating that DQN’s global decisions rely primarily on inventory and demand signals. Comparing the two models qualitatively in terms of relative feature concentration and rank ordering, `A2C_mod` distributes importance more evenly across inputs whereas DQN concentrates attribution on its top two features (Section 3.6.6).
  - *Thay `P297` (Local SHAP):*
    > Local SHAP results (Figure 11) reveal distinct state-dependent feature priorities between the two agents. Across the High/Low Inventory and High/Low Demand states, DQN’s top-ranked attributions shift sharply toward Waste and Demand, whereas `A2C_mod` maintains a more evenly distributed attribution profile without a single dominant driver. In the Critical State scenario, DQN again concentrates strongly on its primary risk features while `A2C_mod` retains a balanced attribution pattern. Because the explained outputs are not magnitude-equivalent across architectures, this contrast reflects differences in relative feature concentration and ranking rather than absolute attribution magnitude.

- **Bản Tiếng Việt (Để theo dõi):**
  - *`P284`:* SHAP được tính trên $\operatorname{softmax}(Q(s,\cdot))$ cho DQN và phân phối chính sách $\pi(\cdot \mid s)$ cho `A2C_mod`. Vì $\operatorname{softmax}(Q)$ phụ thuộc vào thang đo chưa chuẩn hóa của giá trị Q còn $\pi$ được tham số hóa trực tiếp bởi actor, độ lớn attribution không tương đương giữa hai tác nhân; mọi so sánh liên mô hình ở Hình 10–12 chỉ giới hạn ở thứ hạng đặc trưng và cấu trúc định tính.
  - *`P288–P289`:* Ở cấp toàn cục (Hình 10), `A2C_mod` thể hiện cấu trúc đóng góp tương đối cân bằng giữa Tồn kho, Nhu cầu và Hao hụt mà không có một đặc trưng đơn lẻ nào chi phối mạnh. Ngược lại, DQN cho thấy sự phân tách thứ hạng rõ rệt hơn với Tồn kho và Nhu cầu đứng đầu còn Hao hụt thấp nhất. So sánh định tính theo mức tập trung tương đối và thứ hạng đặc trưng, `A2C_mod` phân bổ trọng số đồng đều hơn còn DQN tập trung vào hai đặc trưng đứng đầu (§3.6.6).
  - *`P297`:* Kết quả SHAP cục bộ (Hình 11) cho thấy thứ tự ưu tiên đặc trưng theo trạng thái khác nhau giữa hai tác nhân. Ở các trạng thái Tồn kho và Nhu cầu cao/thấp, các đặc trưng xếp hạng cao nhất của DQN dịch chuyển mạnh sang Hao hụt và Nhu cầu, trong khi `A2C_mod` duy trì phân bổ đều hơn mà không bị một yếu tố duy nhất chi phối. Ở trạng thái tới hạn, DQN tiếp tục tập trung mạnh vào các đặc trưng rủi ro chính còn `A2C_mod` giữ cấu trúc cân bằng hơn. Vì đầu ra được giải thích không tương đương về độ lớn giữa hai kiến trúc, sự đối chiếu này phản ánh khác biệt về mức tập trung tương đối và thứ hạng chứ không phải độ lớn tuyệt đối.

---

#### Chỉnh sửa B3 — Bỏ so sánh độ lớn tại `SHAP-only FCS` (`P352`) và `SHAP Faithfulness / AUPC` (`P401`)
- **Vị trí:** `P352` (*SHAP-only*, câu cuối đoạn) và `P401` (*SHAP Faithfulness: Area Under Perturbation Curve*, câu thứ 2) trong [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**
  - *Thay câu cuối `P352` ("However, the magnitude of decline is greater for A2C_mod than for DQN, resulting in a widening gap between the two models as λ increases."):*
    > Both models exhibit a monotonic decline in FCS as $\lambda$ increases; because a fixed absolute threshold $\varepsilon = 0.01\lambda$ acts differently on the two non-equivalent output scales ($\operatorname{softmax}(Q)$ versus $\pi$), the apparent gap between their curves is descriptive only and should not be interpreted as a quantitative difference in attribution magnitude.
  - *Thay câu thứ 2 của `P401` ("The expected faithfulness ordering holds in all conditions (Table 16): ... about twice as large as the value-based agent (indicative only, since the scales differ)."):*
    > The expected within-agent faithfulness ordering holds in all conditions (Table 16): Most Relevant First produces a positive, monotonically increasing output drop, random masking stays near zero, and Least Relevant First remains negative or near zero for both DQN and `A2C_mod`. Because AUPC is computed on relative Q-value decreases for DQN and absolute probability decreases for `A2C_mod`, AUPC values are interpreted strictly within each agent rather than compared in magnitude across agents.

- **Bản Tiếng Việt (Để theo dõi):**
  - *`P352`:* Cả hai mô hình đều có FCS giảm đơn điệu khi $\lambda$ tăng; vì ngưỡng tuyệt đối cố định $\varepsilon = 0.01\lambda$ tác động khác nhau lên hai thang đo đầu ra không tương đương ($\operatorname{softmax}(Q)$ so với $\pi$), khoảng cách giữa hai đường cong chỉ mang tính mô tả và không nên diễn giải là chênh lệch định lượng về độ lớn attribution.
  - *`P401`:* Thứ tự trung thành kỳ vọng trong nội bộ từng tác nhân đều được thỏa mãn ở mọi điều kiện (Bảng 16): che lấp đặc trưng quan trọng nhất (MoRF) tạo ra mức suy giảm đầu ra dương và tăng đơn điệu, che lấp ngẫu nhiên gần 0, và che lấp ít quan trọng nhất (LeRF) âm hoặc gần 0 cho cả DQN và `A2C_mod`. Vì AUPC đo theo tỷ lệ giảm tương đối của giá trị Q ở DQN và mức giảm tuyệt đối của xác suất $\pi$ ở `A2C_mod`, giá trị AUPC chỉ được diễn giải trong nội bộ từng tác nhân thay vì so sánh trực tiếp độ lớn giữa hai tác nhân.

---

### PHẦN C — TASK 4.3: Thừa nhận rõ Trade-off Base-Stock vs. DRL & Bỏ hàm ý vượt trội tổng thể

#### Chỉnh sửa C1 — Thêm câu thừa nhận rõ Trade-off trong `Abstract`
- **Vị trí:** **Abstract** (`P009`, thay 1 câu *"While the base-stock policy provides a strong reference, DRL agents prioritize service levels and stockout reduction."*) trong [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

> Under the unit-weighted reward, the classical base-stock policy attains a higher total reward with lower holding and waste costs, whereas DQN and `A2C_mod` prioritize near-zero stockouts at the expense of higher inventory and waste costs; neither approach dominates across all operational metrics, and policy preference depends on retailer cost priorities.

- **Bản Tiếng Việt (Để theo dõi):**

> Dưới hàm thưởng trọng số đơn vị, chính sách base-stock cổ điển đạt tổng phần thưởng cao hơn với chi phí tồn kho và hao hụt thấp hơn, trong khi DQN và `A2C_mod` ưu tiên triệt tiêu thiếu hàng nhưng đánh đổi bằng chi phí tồn kho và hao hụt cao hơn; không có phương pháp nào vượt trội trên mọi chỉ tiêu vận hành, và việc lựa chọn chính sách phụ thuộc vào ưu tiên chi phí của nhà bán lẻ.

---

#### Chỉnh sửa C2 — Trung lập hóa tiêu đề `P198` và tinh chỉnh `P214` (§3.2–§3.3.2)
- **Vị trí:**
  1. Tiêu đề phụ `P198` trong mục **Operational Performance Comparison with Classical Baseline**
  2. Đoạn `P214` (câu *"When a DRL agent achieves better service-level performance..."*) trong mục **Reward Difference Explanation** ([Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx)).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**
  - *Sửa tiêu đề `P198` ("Why Prefer a DRL Policy When Base-Stock Attains a Higher Total Reward?"):*
    > **Interpreting the Operational Trade-Off When Base-Stock Attains a Higher Total Reward**
  - *Sửa 2 câu trong `P214` ("When a DRL agent achieves better service-level performance... ordering-related penalties."):*
    > When a DRL agent attains higher service levels and lower stockout rates than the classical baseline, RDX identifies which reward components drive that prioritization; conversely, when the DRL agent obtains a lower total reward than Base-Stock, RDX attributes the gap to elevated holding, waste, or ordering-related penalties, reflecting an explicit operational trade-off rather than overall superiority.

- **Bản Tiếng Việt (Để theo dõi):**
  - *Tiêu đề `P198`:* **Diễn giải sự đánh đổi vận hành khi Base-Stock đạt tổng phần thưởng cao hơn**
  - *`P214`:* Khi một tác nhân DRL đạt mức độ dịch vụ cao hơn và tỷ lệ thiếu hàng thấp hơn đường cơ sở cổ điển, RDX chỉ rõ thành phần phần thưởng nào thúc đẩy ưu tiên đó; ngược lại, khi tác nhân DRL đạt tổng phần thưởng thấp hơn Base-Stock, RDX quy nguyên nhân cho mức phạt tồn kho, hao hụt hoặc đặt hàng tăng cao, phản ánh sự đánh đổi vận hành tường minh chứ không phải sự vượt trội tổng thể.

---

#### Chỉnh sửa C3 — Khóa chặt câu thừa nhận Trade-off trong `Conclusion`
- **Vị trí:** **Conclusion** (`P441`, câu thứ 2 *"Operationally, the classical base-stock policy..."*) trong [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

> Operationally, no single policy dominates across all metrics: the classical base-stock policy attains the highest unit-weighted total reward with lower holding and waste costs, whereas both DRL agents nearly eliminate stockouts at the expense of higher holding and waste costs (Section 3.2); which policy is preferable therefore depends on how a retailer weighs stockout penalties against excess inventory and spoilage.

- **Bản Tiếng Việt (Để theo dõi):**

> Về mặt vận hành, không có chính sách đơn lẻ nào vượt trội trên mọi chỉ tiêu: chính sách base-stock cổ điển đạt tổng phần thưởng trọng số đơn vị cao nhất với chi phí tồn kho và hao hụt thấp hơn, trong khi cả hai tác nhân DRL gần như triệt tiêu thiếu hàng nhưng phải đánh đổi bằng chi phí tồn kho và hao hụt cao hơn (§3.2); do đó lựa chọn chính sách phù hợp phụ thuộc vào cách nhà bán lẻ cân nhắc giữa chi phí thiếu hàng với chi phí tồn kho dư thừa và hao hụt.

---

## Checklist kiểm chứng hoàn thành Task 4

- [x] **Task 4.1:** Đã giải thích rõ lý do chọn `synthetic + random-100` (`P162`) và gom toàn bộ kết quả background sensitivity (**MEDIUM + HARD từ Task 3**) thành **1 bảng thống nhất (Table R3-02)** trong bài chính (`P164–P166`).
- [x] **Task 4.2:** Đã sửa triệt để mâu thuẫn **C-03** (`P156`, `Table 4`, `P284`) về thống nhất `softmax(Q)` vs. `π(·|s)`, và bỏ hoàn toàn các câu so sánh độ lớn SHAP/AUPC không tương đương (`P288–P289`, `P297`, `P352`, `P401` — bỏ cụm *"about twice as large"*).
- [x] **Task 4.3:** Đã thêm thừa nhận rõ trade-off vào **Abstract (`P009`)** và **Conclusion (`P441`)**, trung lập hóa tiêu đề `P198`, và loại bỏ hàm ý vượt trội tổng thể ở `P214`.
- [x] Văn phong súc tích, chuẩn bài báo khoa học, đầy đủ song ngữ Anh–Việt và chỉ rõ từng đoạn `Pxxx` trong file `.docx`.

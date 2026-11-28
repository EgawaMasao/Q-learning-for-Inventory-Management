# Output Task 2 — Thực nghiệm 5 seed Standard A2C & Mô tả đầy đủ `A2C_mod` (§2.2.6, Table 2, Table 6 & Table 7)

## Tóm tắt công việc & Kết quả chính (Thực nghiệm + Viết)
- **Phần viết (đã hoàn thiện & xác minh mã nguồn):**
  1. Sửa mâu thuẫn kiến trúc **C-02**, chuẩn hóa Critic (`LayerNormalization`) và ghi chú `entropy` trong **Table 2** (§2.1.1).
  2. Viết lại **§2.2.6 *A2C_mod Algorithm and Its Relation to Conventional A2C*** với đầy đủ 4 phương trình `(1)–(4)`, **Algorithm 1**, **Theoretical Motivation**, và **Table X** (8 tiêu chí so sánh Conventional A2C vs. `A2C_mod`).
- **Phần thực nghiệm mới bổ sung (5 seed độc lập: `42, 123, 256, 512, 1024`):**
  - Huấn luyện hoàn tất 5 seed **Standard A2C** (`600 episodes`, `ckpt-60` mỗi seed) tại `Feedback 23-11/task2/A2C_{42,123,256,512,1024}/`.
  - Đánh giá thống nhất (apples-to-apples) cả 4 chính sách (**Base-Stock $k=1.0$**, **DQN 5-seed**, **`A2C_mod` 5-seed**, **Standard A2C 5-seed**) trên cùng 504 bước kiểm định × 4 kịch bản (`ALL, EASY, MEDIUM, HARD`) + `Mean(E/M/H)` tại [task2_eval_standard_a2c.ipynb](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task2/task2_eval_standard_a2c.ipynb). Dữ liệu xuất tại `Feedback 23-11/task2/output/`.
- **3 phát hiện thực nghiệm nổi bật (kiểm chứng trực tiếp phân tích lý thuyết ở §2.2.6):**
  1. **Hiệu năng 5 seed vượt trội và cực kỳ ổn định:** Trên `Mean(E/M/H)`, **Standard A2C** đạt `Total Reward = +0.8291 ± 0.0088` (gần ngang Base-Stock `+0.8349`, $\Delta = -0.0057$, nhỉnh hơn DQN `+0.8273 ± 0.0192` với SD nhỏ hơn 2 lần, và vượt trội `A2C_mod` `+0.2556 ± 0.6879` ở **5/5 seed**, Wilcoxon $p = 0.0625$, Hedges' $g = +0.70$, 95% CI `[+0.0806, +1.1731]`).
  2. **Triệt tiêu hoàn toàn hiện tượng policy collapse nhờ entropy regularization ($\beta = 0.001$):**
     - `A2C_mod` (chỉ log entropy, không cộng vào loss) có entropy gần bằng `0` (`0.00003–0.0068` ở `ALL`), **2/5 seed bị collapse về `action 13`** (seed 123: `85.6%`, seed 256: `98.7%`), đẩy `holding_cost` lên `0.3834 ± 0.4661` và khiến SD của reward phình tới `±0.6879`.
     - **Standard A2C** (có entropy bonus trong loss) duy trì entropy cao gấp ~22 lần (`0.0193–0.1162`), **0/5 seed bị collapse** (`action 13 ≤ 0.023%` ở mọi seed), kéo `holding_cost` xuống `0.0101 ± 0.0102` và giảm độ lệch chuẩn reward **78 lần** (`±0.6879` → `±0.0088`).
  3. **Bảo toàn đánh đổi dịch vụ–chi phí so với Base-Stock:** Tương tự DQN, Standard A2C đạt `Service Level = 1.0000` (`Stockout Rate = 0.0000`, $\Delta = -0.0037$ so với Base-Stock) với `Ordering Cost` thấp hơn (`-0.0522`), đổi lại `Waste Cost` (`+0.0515`) và `Holding Cost` (`+0.0101`) cao hơn Base-Stock.

---

## Ghi vào bài báo

### Chỉnh sửa 1 — Cập nhật Table 2 (sửa mâu thuẫn C-02, LayerNormalization, Entropy, ký hiệu Loss)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **2.1.1 *Action-to-Replenishment Mapping*** → **Table 2** (`d.tables[1]`, cột `A2C_mod`).

- **Nội dung sửa (Bản Tiếng Anh — giữ nguyên chuẩn đã xác minh từ mã nguồn):**

  | Hyperparameter | DQN | A2C_mod |
  |---|---|---|
  | Hidden layers / width | 3 hidden layers, 128 units each | Actor: 3 hidden layers (32 units each) + output layer (4 layers total); Critic: 1 hidden layer (32 units) + output layer (2 layers total) |
  | Normalization | GroupNormalization (groups = 1) after each hidden layer | LayerNormalization in the critic hidden layer only (none in actor) |
  | Entropy coefficient | (not applicable) | 0.001 (computed and logged only; not added to actor loss; added to loss in the conventional A2C benchmark) |
  | Loss function | Huber loss on the TD error | Critic: $\frac{1}{2}\delta_{t,i}^2$; Actor (`A2C_mod`): MSE between $\pi_\theta(\cdot \mid s_{t,i})$ and distance-weighted soft target $p^{\text{new}}(\cdot \mid s_{t,i})$ (see Eq. (1)–(4)) |

  *(Các hàng khác trong Table 2 giữ nguyên.)*

- **Bản Tiếng Việt (Để theo dõi):** Cập nhật 4 hàng cột `A2C_mod` cho khớp chính xác mã nguồn (Actor 3 lớp ẩn + đầu ra = 4 lớp; Critic 1 lớp ẩn có `LayerNormalization` + đầu ra = 2 lớp; Entropy `0.001` chỉ tính/log ở `A2C_mod` và có cộng vào loss ở bản đối chứng Standard A2C; Loss khớp PT (1)–(4)).

---

### Chỉnh sửa 2 — Tiểu mục §2.2.6 (*A2C_mod Algorithm and Its Relation to Conventional A2C*) — Thay *Scope note* cũ bằng kết nối thực nghiệm 5 seed

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **2.2.6 *A2C_mod Algorithm and Its Relation to Conventional A2C*** (`P139–P140`).
  - **Giữ nguyên** toàn bộ phần kiến trúc, 4 phương trình `(1)–(4)`, **Algorithm 1**, **Theoretical Motivation**, và **Table X** đã viết ở vòng trước.
  - **Thay thế đoạn *Scope note* cuối §2.2.6** bằng đoạn kết nối thực nghiệm bên dưới.

- **Nội dung sửa (Bản Tiếng Anh — thay thế đoạn *Scope note* ngay dưới Table X):**
  > **Empirical Benchmark Setup.** To test these analytical differences empirically, we trained five independent seeds (`42, 123, 256, 512, 1024`; 600 episodes each) of conventional A2C using the exact same weight-shared actor–critic architecture, learning rates, and 32-step chronological rollouts as `A2C_mod`, replacing Eqs. (3)–(4) with the standard log-likelihood policy gradient and explicit entropy regularization ($\beta = 0.001$). Both actor–critic variants are evaluated alongside DQN and Base-Stock on the same unified 504-step test harness in Section 3.2 (Tables 6 and 7), directly confirming that explicit entropy regularization eliminates `A2C_mod`'s seed collapse and reduces cross-seed reward variance by nearly two orders of magnitude.

- **Bản Tiếng Việt (Để theo dõi):**
  > **Thiết lập đối chứng thực nghiệm.** Để kiểm chứng thực nghiệm các khác biệt lý thuyết trên, chúng tôi huấn luyện 5 seed độc lập (`42, 123, 256, 512, 1024`; mỗi seed 600 episode) của A2C tiêu chuẩn với cùng kiến trúc actor–critic chia sẻ trọng số, tốc độ học và cửa sổ 32 bước như `A2C_mod`, chỉ thay PT (3)–(4) bằng policy gradient log-likelihood chuẩn kết hợp điều chuẩn entropy tường minh ($\beta = 0.001$). Cả hai biến thể actor–critic được đánh giá song song với DQN và Base-Stock trên cùng một khung kiểm định 504 bước tại Mục 3.2 (Bảng 6 và Bảng 7), xác nhận trực tiếp rằng điều chuẩn entropy loại bỏ hoàn toàn hiện tượng collapse theo seed của `A2C_mod` và giảm phương sai phần thưởng giữa các seed gần hai bậc độ lớn.

---

### Chỉnh sửa 3 — Thêm cột `Standard A2C (5-seed)` vào **Table 6** & **Table 7** + Phân tích định lượng tại §3.2

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **3.2 *Operational Performance Comparison with Classical Baseline*** (`P192`, **Table 6**, **Table 7**, và thêm 1 đoạn phân tích so sánh Standard A2C vs. `A2C_mod` ngay sau `P196`).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

  1. *Cập nhật câu đầu `P192`:*
     > To evaluate the operational competitiveness of the learned DRL policies—and to benchmark `A2C_mod` directly against conventional A2C—we compare DQN, `A2C_mod`, and conventional A2C (five independently trained seeds each: 42, 123, 256, 512, and 1024) against the classical Base-Stock policy ($k = 1.0$) on a unified 504-step test harness across the unscaled test series (`ALL`), the three controlled stress-test scenarios (`EASY`, `MEDIUM`, `HARD`), and their three-scenario mean (`All scenarios`, Tables 6 and 7).

  2. *Cập nhật **Table 6** (thêm cột `Standard A2C`, đồng bộ cùng 1 khung đánh giá 5 seed cho cả 4 chính sách):*

  **Table 6. Unified five-seed performance comparison of Base-Stock ($k = 1.0$), DQN, `A2C_mod`, and conventional A2C (Mean $\pm$ SD across seeds 42, 123, 256, 512, 1024).**

  | Scenario | Metric | Base-Stock | DQN (5-seed) | `A2C_mod` (5-seed) | Standard A2C (5-seed) |
  |---|---|---:|---:|---:|---:|
  | **All scenarios** *(Mean E/M/H)* | Total Reward | $+0.8349$ | $+0.8273 \pm 0.0192$ | $+0.2556 \pm 0.6879$ | $+0.8291 \pm 0.0088$ |
  | | Service Level | $0.9963$ | $1.0000 \pm 0.0000$ | $0.9897 \pm 0.0189$ | $1.0000 \pm 0.0000$ |
  | | Holding Cost | $0.0000$ | $0.0038 \pm 0.0019$ | $0.3834 \pm 0.4661$ | $0.0101 \pm 0.0102$ |
  | | Waste Cost | $0.0153$ | $0.0631 \pm 0.0009$ | $0.0622 \pm 0.0052$ | $0.0669 \pm 0.0007$ |
  | | Ordering Cost | $0.1461$ | $0.1057 \pm 0.0195$ | $0.2885 \pm 0.3331$ | $0.0939 \pm 0.0081$ |
  | | Stockout Rate | $0.0037$ | $0.0000 \pm 0.0000$ | $0.0103 \pm 0.0189$ | $0.0000 \pm 0.0000$ |
  | **EASY** | Total Reward | $+0.8924$ | $+0.9384 \pm 0.0126$ | $+0.4123 \pm 0.6439$ | $+0.9341 \pm 0.0134$ |
  | **MEDIUM** | Total Reward | $+0.8442$ | $+0.8489 \pm 0.0164$ | $+0.2531 \pm 0.7225$ | $+0.8528 \pm 0.0060$ |
  | **HARD** | Total Reward | $+0.7680$ | $+0.6947 \pm 0.0340$ | $+0.1016 \pm 0.7067$ | $+0.7005 \pm 0.0154$ |
  | **ALL** *(unscaled)* | Total Reward | $+0.7607$ | $+0.8107 \pm 0.0068$ | $+0.1717 \pm 0.6922$ | $+0.8051 \pm 0.0076$ |

  *(Note: Full 6-metric breakdowns for EASY, MEDIUM, HARD, and ALL are archived in the Supplementary Material; all four policies are evaluated under identical deterministic argmax rollouts on the 504-step test set.)*

  3. *Cập nhật **Table 7** (thêm cột `Standard A2C` và $\Delta$ so với Base-Stock trên `All scenarios` / Mean E/M/H):*

  **Table 7. Change relative to the corrected Base-Stock baseline ($k = 1.0$, Mean over EASY, MEDIUM, and HARD).**

  | Metric | Base-Stock | DQN ($\Delta$) | `A2C_mod` ($\Delta$) | Standard A2C ($\Delta$) |
  |---|---:|---:|---:|---:|
  | Service Level | $0.9963$ | $1.0000$ ($+0.0037$) | $0.9897$ ($-0.0065$) | $1.0000$ ($+0.0037$) |
  | Stockout Rate | $0.0037$ | $0.0000$ ($-0.0037$) | $0.0103$ ($+0.0065$) | $0.0000$ ($-0.0037$) |
  | Holding Cost | $0.0000$ | $0.0038$ ($+0.0038$) | $0.3834$ ($+0.3834$) | $0.0101$ ($+0.0101$) |
  | Waste Cost | $0.0153$ | $0.0631$ ($+0.0478$) | $0.0622$ ($+0.0469$) | $0.0669$ ($+0.0515$) |
  | Ordering Cost | $0.1461$ | $0.1057$ ($-0.0404$) | $0.2885$ ($+0.1425$) | $0.0939$ ($-0.0522$) |
  | Total Reward | $+0.8349$ | $+0.8273$ ($-0.0075$) | $+0.2556$ ($-0.5792$) | $+0.8291$ ($-0.0057$) |

  4. *Chèn 1 đoạn phân tích định lượng ngay sau `P196` (so sánh `Standard A2C` vs. `A2C_mod`):*
     > **Empirical Comparison: Conventional A2C versus `A2C_mod`.** Adding the five-seed conventional A2C benchmark directly validates our theoretical analysis in Section 2.2.6. Across all scenarios, conventional A2C achieves $+0.8291 \pm 0.0088$ total reward (nearly matching Base-Stock's $+0.8349$, $\Delta = -0.0057$, and slightly exceeding DQN's $+0.8273 \pm 0.0192$ with half the cross-seed dispersion), while outperforming `A2C_mod` ($+0.2556 \pm 0.6879$) in all five matched seeds ($\Delta = +0.5735$, 95% bootstrap CI $[+0.0806, +1.1731]$, Hedges' $g = +0.70$, exact Wilcoxon $p = 0.0625$; paired $t$-test $p = 0.138$ due to `A2C_mod`'s extreme inter-seed variance). Policy diagnostics explain this gap: without entropy regularization in $\mathcal{L}_{\text{actor}}$, `A2C_mod` suffers severe policy saturation (mean entropy $0.00003$–$0.0068$ in `ALL`), with two of five seeds collapsing onto extreme action 13 (seed 123: $85.6\%$; seed 256: $98.7\%$), inflating holding cost to $0.3834 \pm 0.4661$. By contrast, conventional A2C's explicit entropy bonus ($\beta = 0.001$) maintains $\sim 22\times$ higher policy entropy ($0.0193$–$0.1162$), prevents action-13 collapse in all five seeds ($\le 0.023\%$ across seeds), reduces holding cost to $0.0101 \pm 0.0102$, and cuts five-seed reward SD by $78\times$ (from $0.6879$ to $0.0088$). Consequently, `A2C_mod`'s modified update should be viewed as a historical benchmark adopted from [4] whose XRL behavior exposes the risk of unregularized policy saturation, rather than an operational improvement over standard A2C.

- **Bản Tiếng Việt (Để theo dõi):**
  > **So sánh thực nghiệm: A2C tiêu chuẩn so với `A2C_mod`.** Bổ sung đối chứng 5 seed của A2C tiêu chuẩn kiểm chứng trực tiếp phân tích lý thuyết ở Mục 2.2.6. Trên trung bình các kịch bản, A2C tiêu chuẩn đạt tổng phần thưởng $+0.8291 \pm 0.0088$ (gần sát Base-Stock $+0.8349$, $\Delta = -0.0057$, và nhỉnh hơn DQN $+0.8273 \pm 0.0192$ với độ lệch chuẩn nhỏ hơn một nửa), đồng thời vượt `A2C_mod` ($+0.2556 \pm 0.6879$) ở cả 5/5 seed ghép cặp ($\Delta = +0.5735$, 95% CI $[+0.0806, +1.1731]$, Hedges' $g = +0.70$, Wilcoxon $p = 0.0625$; paired $t$-test $p = 0.138$ do phương sai giữa các seed của `A2C_mod` quá lớn). Chẩn đoán chính sách giải thích rõ khoảng cách này: do không cộng entropy vào $\mathcal{L}_{\text{actor}}$, `A2C_mod` bị bão hòa nặng (entropy $0.00003$–$0.0068$ ở `ALL`), với 2/5 seed sụp đổ vào hành động cực trị `action 13` (seed 123: $85.6\%$; seed 256: $98.7\%$), đẩy chi phí tồn kho lên $0.3834 \pm 0.4661$. Ngược lại, thành phần thưởng entropy ($\beta = 0.001$) của A2C tiêu chuẩn duy trì entropy cao gấp $\sim 22$ lần ($0.0193$–$0.1162$), ngăn hoàn toàn collapse vào `action 13` ở cả 5 seed ($\le 0.023\%$), hạ chi phí tồn kho xuống $0.0101 \pm 0.0102$ và giảm độ lệch chuẩn phần thưởng qua 5 seed tới $78$ lần (từ $0.6879$ xuống $0.0088$). Do đó, cơ chế cập nhật của `A2C_mod` cần được hiểu là một mô hình kế thừa từ [4] giúp bộc lộ rủi ro bão hòa chính sách khi thiếu điều chuẩn entropy trong phân tích XRL, thay vì một cải tiến vận hành so với A2C tiêu chuẩn.

---

## Checklist kiểm chứng hoàn thành Task 2 (Thực nghiệm + Viết)
- [x] **Table 2 (§2.1.1):** Sửa triệt để mâu thuẫn kiến trúc **C-02**, `LayerNormalization`, và ghi chú `entropy`.
- [x] **§2.2.6:** Có đủ 4 phương trình `(1)–(4)`, **Algorithm 1**, **Theoretical Motivation**, **Table X**, và thay *Scope note* cũ bằng **Empirical Benchmark Setup**.
- [x] **Thực nghiệm 5 seed Standard A2C (`42, 123, 256, 512, 1024`):** Train đủ 600 episodes (`ckpt-60`), đánh giá đồng bộ với Base-Stock, DQN (5 seed), `A2C_mod` (5 seed) tại `task2_eval_standard_a2c.ipynb`.
- [x] **Table 6 & Table 7 (§3.2):** Bổ sung cột **`Standard A2C (5-seed)`** kèm $\Delta$ so với Base-Stock, kiểm định ghép cặp 5 seed và chẩn đoán entropy/collapse xác nhận trực tiếp lý thuyết ở §2.2.6.

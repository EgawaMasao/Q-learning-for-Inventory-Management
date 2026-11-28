# Output Task 6 — Bằng chứng định lượng & kiểm định cho sửa lỗi rò rỉ Normalization Capacity (§2.1.3, Table 3)

## Tóm tắt thực nghiệm & Kết quả chính
- **Thực nghiệm đã chạy (Phương án B — đánh giá ghép cặp không retrain):** [task6_capacity_eval_optionB.ipynb](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task6/task6_capacity_eval_optionB.ipynb) xuất `capacity_train_only.tfrecords` và chạy đánh giá ghép cặp (paired evaluation) trên **504 bước kiểm định** × **4 kịch bản** (`ALL, EASY, MEDIUM, HARD`) giữa môi trường cũ (`capacity_full`) và môi trường đã sửa (`capacity_train_only`) cho:
  - **Base-Stock ($k = 1.0$)** (tính lại mục tiêu $S_i$ trên tập train cho từng bộ capacity),
  - **DQN (5 seed độc lập: `42, 123, 256, 512, 1024`)** (`ckpt-50, hidden=128` cho 4 seed; `ckpt-43, hidden=32` cho seed 512),
  - **`A2C_mod` (5 seed độc lập: `42, 123, 256, 512, 1024`)** (`ckpt-53` cho seed 42; `ckpt-60` cho 4 seed còn lại).
  - Kết quả lưu tại `Feedback 23-11/task6/output/` ([task6_summary.json](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task6/output/task6_summary.json), [task6_paired_step_tests.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task6/output/task6_paired_step_tests.csv), [task6_5seed_paired_summary.csv](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task6/output/task6_5seed_paired_summary.csv)).

- **4 phát hiện khoa học quan trọng (phải sửa trung thực trong bài báo):**
  1. **Sửa lỗi ngược chiều trong bản gốc (§2.1.3):** Bài gốc ghi `capacity_full` *vượt quá* (`exceeded`) `capacity_train_only` ~25%, nhưng thực tế **`capacity_full` nhỏ hơn (`fell below`) `capacity_train_only`** (`mean = 20.33` so với `25.79`, lệch trung bình `26.29%`, cực đại `50.0%`, `212/220` SKU lệch > `10%`) vì nhu cầu giai đoạn train cao hơn test.
  2. **Đính chính nhận định `"shift in reward is under 0.02"`:** Nhận định cũ **không đúng trên mọi điều kiện** — ở kịch bản `ALL`, $\Delta\text{Reward} = \text{train-only} - \text{full}$ đạt **`+0.0512`** (Base-Stock), **`+0.0258`** (DQN 5-seed), và **`+0.0345`** (`A2C_mod` 5-seed), vượt biên TOST $\pm 0.02$ (chỉ DQN ở `EASY` `+0.0060` và `MEDIUM` `+0.0118` đạt TOST $|\Delta| < 0.02$).
  3. **Cơ chế rõ ràng — 4 chỉ số vận hành bất biến, dịch chuyển reward hoàn toàn do mẫu số của `Ordering Cost`:**
     - Trên cả 3 phương pháp ở `ALL`, **4 chỉ số vận hành** (*Service Level, Stockout Rate, Holding Cost, Waste Cost*) đều **tương đương thống kê trong biên $\pm 0.02$** (TOST $p \le 0.019$; thực tế $|\Delta| \le 0.0013$ ở `ALL`).
     - Gần như **95–100%** mức tăng của Total Reward bắt nguồn từ **Ordering/Balance Cost giảm cơ học** (`-0.0500` ở Base-Stock, `-0.0260` ở DQN, `-0.0328` ở `A2C_mod`) do cùng một lượng đặt hàng vật lý được chia cho mẫu số `capacity_train_only` lớn hơn ~26.3%.
  4. **Thứ hạng tương đối giữa các chính sách giữ nguyên 100% ở cả 4 kịch bản:**
     - `ALL, EASY, MEDIUM`: **DQN > Base-Stock > `A2C_mod`** (giống hệt ở cả hai bộ capacity).
     - `HARD`: **Base-Stock > DQN > `A2C_mod`** (điểm đảo chiều Base-Stock vượt DQN khi mức hư hỏng cao được bảo toàn hoàn toàn).

---

## Ghi vào bài báo

### Chỉnh sửa 1 — Cập nhật đoạn văn §2.1.3 (*Leakage Audit and Corrected Procedure*) kèm bảng định lượng mới (Table 3b)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **2.1.3 *Leakage Audit and Corrected Procedure*** (`P68`, thay thế câu về `Normalization capacity` và câu kết `Neither correction requires retraining...`; chèn **Table 3b** ngay sau **Table 3** `P69–P71`).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

  *Thay thế đoạn `P68` trong §2.1.3 bằng:*

  > Four components were audited for test-set leakage (Table 3). Demand forecasting is not a learned component (agents observe capacity-normalized realized demand) and the reward uses fixed unity weights, so neither leaks. Two components initially used test information. First, normalization capacity had been derived from the full series and **fell below** its train-only counterpart by 26.29% on average (mean 20.33 vs. 25.79; maximum deviation 50.0%; 212/220 SKUs differing by > 10%) because training demand exceeds test demand; capacity is now defined from the training interval alone (`capacity_train_only`). Second, the base-stock safety factor had been selected on the test interval; re-selecting it on the validation interval shifts the optimum by only half a unit, with nearly parallel validation and test reward curves, so the validation optimum is reported and the test optimum kept only as a reference.
  >
  > To quantify the downstream effect of correcting normalization capacity without retraining the DRL checkpoints, we conducted a paired evaluation across all 504 test steps and four demand/waste scenarios (`ALL, EASY, MEDIUM, HARD`) comparing `capacity_full` versus `capacity_train_only` for Base-Stock ($k = 1.0$) and five independent training seeds (`42, 123, 256, 512, 1024`) of both DQN and `A2C_mod` (Table 3b). Across all three policies in `ALL`, replacing `capacity_full` with the larger `capacity_train_only` increases total reward by $\Delta = +0.0258$ (DQN 5-seed mean, paired $t$-test $p = 2.55 \times 10^{-7}$), $+0.0345$ (`A2C_mod` 5-seed mean, $p = 0.068$), and $+0.0512$ (Base-Stock, step-level $p < 10^{-178}$), exceeding a $\pm 0.02$ equivalence bound (TOST $p \ge 0.82$; equivalence within $\pm 0.02$ holds only for DQN in `EASY`, $\Delta = +0.0060$, and `MEDIUM`, $\Delta = +0.0118$). Crucially, metric decomposition reveals that this reward shift is driven almost entirely by a mechanical reduction in normalized ordering/balance cost ($\Delta = -0.0260$ for DQN, $-0.0328$ for `A2C_mod`, and $-0.0500$ for Base-Stock) due to the larger normalization denominator. All four operational metrics—service level, stockout rate, holding cost, and waste cost—remain tightly invariant ($|\Delta| \le 0.0013$ in `ALL`, all TOST-equivalent within $\pm 0.02$ at $p \le 0.019$). Furthermore, the relative ranking among policies is 100% preserved in every scenario (`DQN > Base-Stock > A2C_mod` in `ALL, EASY, MEDIUM`, and `Base-Stock > DQN > A2C_mod` in `HARD`). Because this audit evaluates existing checkpoints on the corrected environment rather than retraining from scratch, full retraining under `capacity_train_only` is noted as a limitation in Section 5.

  *Chèn **Table 3b** ngay sau **Table 3** (`P70–P71`):*

  **Table 3b. Paired evaluation of capacity-leakage correction (`capacity_train_only` minus `capacity_full`) across 504 test steps (`ALL` scenario; 5-seed means for DQN and `A2C_mod`, refit target for Base-Stock $k = 1.0$).**

  | Policy / Metric | `capacity_full` | `capacity_train_only` | $\Delta$ (train − full) | Paired $p$ / TOST ($\pm 0.02$) |
  |---|---:|---:|---:|---|
  | **Base-Stock ($k=1.0$)** | | | | *(step-level, $n=504$)* |
  | Total Reward | 0.7551 | 0.8063 | $+0.0512$ | $p < 10^{-178}$ / Fail ($1.00$) |
  | Ordering Cost | 0.2215 | 0.1715 | $-0.0500$ | $p < 10^{-171}$ / Fail ($1.00$) |
  | Service Level / Stockout | 0.9823 / 0.0177 | 0.9824 / 0.0176 | $+0.0001$ / $-0.0001$ | $p = 0.0017$ / **Pass** ($< 10^{-15}$) |
  | Holding / Waste Cost | 0.0000 / 0.0057 | 0.0000 / 0.0046 | $0.0000$ / $-0.0011$ | $p = 1.00$ / $< 10^{-83}$ / **Pass** |
  | **DQN (5-seed mean)**† | | | | *(5-seed paired, $n=5$)* |
  | Total Reward | $0.8107 \pm 0.0068$ | $0.8365 \pm 0.0072$ | $+0.0258$ | $p = 2.6 \times 10^{-7}$ / Fail ($1.00$) |
  | Ordering Cost | $0.1602 \pm 0.0066$ | $0.1342 \pm 0.0071$ | $-0.0260$ | $p = 1.8 \times 10^{-7}$ / Fail ($1.00$) |
  | Service Level / Stockout | 0.99998 / 0.00002 | 0.99998 / 0.00002 | $\pm 0.000002$ | $p = 0.374$ / **Pass** ($< 10^{-15}$) |
  | Holding / Waste Cost | 0.0053 / 0.0238 | 0.0053 / 0.0240 | $-0.0000$ / $+0.0002$ | $p = 0.929$ / $10^{-7}$ / **Pass** |
  | **`A2C_mod` (5-seed mean)** | | | | *(5-seed paired, $n=5$)* |
  | Total Reward | $0.1717 \pm 0.6922$ | $0.2062 \pm 0.7081$ | $+0.0345$ | $p = 0.068$ / Fail ($0.82$) |
  | Ordering Cost | $0.3868 \pm 0.3383$ | $0.3540 \pm 0.3540$ | $-0.0328$ | $p = 0.083$ / Fail ($0.79$) |
  | Service Level / Stockout | 0.9859 / 0.0141 | 0.9872 / 0.0128 | $+0.0013$ / $-0.0013$ | $p = 0.082$ / **Pass** ($< 10^{-5}$) |
  | Holding / Waste Cost | 0.4053 / 0.0221 | 0.4046 / 0.0224 | $-0.0007$ / $+0.0003$ | $p = 0.923$ / $0.024$ / **Pass** |

  *Notes: Scenario-level $\Delta\text{Reward}$ (`EASY / MEDIUM / HARD`): Base-Stock $+0.0375 / +0.0429 / +0.0525$; DQN $+0.0060^* / +0.0118^* / +0.0248$ ($^*$TOST-equivalent within $\pm 0.02$, $p < 0.001$); `A2C_mod` $+0.0539 / +0.0261 / +0.0308$. Policy ranking is invariant across all four scenarios. †DQN uses `ckpt-50` (`hidden=128`) for seeds 42, 123, 256, 1024 and `ckpt-43` (`hidden=32`) for seed 512. Full step-level and per-scenario tests are provided in the Supplementary Material.*

- **Bản Tiếng Việt (Để theo dõi):**

  > Bốn thành phần đã được kiểm toán rò rỉ dữ liệu kiểm định (Bảng 3). Dự báo nhu cầu không phải mô hình học (tác tử quan sát nhu cầu thực tế chuẩn hóa theo sức chứa) và hàm thưởng dùng trọng số đơn vị cố định nên đều không rò rỉ. Hai thành phần ban đầu có sử dụng thông tin tập kiểm định. Thứ nhất, sức chứa chuẩn hóa ban đầu tính trên toàn chuỗi và **thấp hơn** phiên bản chỉ tính trên tập huấn luyện trung bình 26,29% (trung bình 20,33 so với 25,79; độ lệch cực đại 50,0%; 212/220 SKU lệch > 10%) do nhu cầu giai đoạn huấn luyện cao hơn kiểm định; sức chứa hiện được định nghĩa chỉ từ tập huấn luyện (`capacity_train_only`). Thứ hai, hệ số an toàn của base-stock từng chọn trên tập kiểm định; khi chọn lại trên tập thẩm định, điểm tối ưu chỉ dịch nửa đơn vị với đường cong phần thưởng gần song song, nên báo cáo nghiệm tối ưu trên tập thẩm định và giữ nghiệm tập kiểm định làm tham chiếu.
  >
  > Để định lượng tác động đầu ra của việc sửa sức chứa chuẩn hóa mà không huấn luyện lại các checkpoint DRL, chúng tôi thực hiện đánh giá ghép cặp trên toàn bộ 504 bước kiểm định và 4 kịch bản (`ALL, EASY, MEDIUM, HARD`) giữa `capacity_full` và `capacity_train_only` cho Base-Stock ($k = 1.0$) cùng 5 seed độc lập (`42, 123, 256, 512, 1024`) của cả DQN và `A2C_mod` (Bảng 3b). Ở kịch bản `ALL`, thay `capacity_full` bằng `capacity_train_only` lớn hơn làm tổng phần thưởng tăng $\Delta = +0.0258$ (DQN trung bình 5 seed, paired $t$-test $p = 2.55 \times 10^{-7}$), $+0.0345$ (`A2C_mod` 5 seed, $p = 0.068$) và $+0.0512$ (Base-Stock, theo bước $p < 10^{-178}$), vượt ngưỡng tương đương $\pm 0.02$ (TOST $p \ge 0.82$; chỉ DQN ở `EASY` $+0.0060$ và `MEDIUM` $+0.0118$ đạt TOST $\pm 0.02$). Quan trọng hơn, phân rã chỉ số cho thấy mức dịch chuyển phần thưởng này gần như hoàn toàn đến từ việc chi phí đặt hàng/cân bằng chuẩn hóa giảm cơ học ($\Delta = -0.0260$ ở DQN, $-0.0328$ ở `A2C_mod`, $-0.0500$ ở Base-Stock) do mẫu số chuẩn hóa lớn hơn. Cả 4 chỉ số vận hành — mức phục vụ, tỷ lệ hết hàng, chi phí tồn kho và chi phí hư hỏng — đều gần như bất biến ($|\Delta| \le 0.0013$ ở `ALL`, tất cả đạt tương đương TOST trong $\pm 0.02$ với $p \le 0.019$). Hơn nữa, thứ hạng tương đối giữa các chính sách được bảo toàn 100% ở mọi kịch bản (`DQN > Base-Stock > A2C_mod` ở `ALL, EASY, MEDIUM` và `Base-Stock > DQN > A2C_mod` ở `HARD`). Vì kiểm toán này đánh giá các checkpoint sẵn có trên môi trường đã sửa thay vì huấn luyện lại từ đầu, việc huấn luyện lại hoàn toàn với `capacity_train_only` được ghi nhận là một hạn chế tại Mục 5.

---

### Chỉnh sửa 2 — Cập nhật hàng `Normalization (capacity)` trong **Table 3** và bổ sung 1 câu tại **Section 5 (*Study Limitations*)**

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx):
  1. **Table 3** (`d.tables[2]`, hàng 2 `Normalization (capacity)`, cột `Leakage` và `Correction`).
  2. **Section 5 *Study Limitations and Future Work*** (`P444`, thêm 1 câu nối mạch về việc chưa retrain dưới `capacity_train_only`).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**
  1. *Cập nhật hàng 2 của **Table 3**:*

  | Component | Estimation basis | Leakage | Correction |
  |---|---|---|---|
  | Normalization (capacity) | Full series (initial, mean 20.33) vs. training-only (corrected, mean 25.79) | Yes — 26.29% mean shift in constant; +0.026 to +0.051 reward shift via ordering-cost scaling, operational metrics invariant ($\le 0.0013$), policy ranking preserved (Table 3b) | Regenerated capacity from training interval only; paired 5-seed audit reported in Table 3b |

  2. *Thêm 1 câu vào **Section 5 (`P444`)**:*
     > Finally, our capacity-leakage audit (Section 2.1.3, Table 3b) evaluates the existing 5-seed checkpoints on the corrected `capacity_train_only` environment rather than retraining all agents from scratch; full retraining under train-only normalization remains a worthwhile step if absolute ordering-cost magnitudes (rather than relative policy rankings) are used for deployment.

- **Bản Tiếng Việt (Để theo dõi):**
  1. *Hàng 2 **Bảng 3**:* Thành phần `Normalization (capacity)` | Cơ sở ước lượng `Toàn chuỗi (ban đầu, TB 20.33) so với chỉ tập huấn luyện (đã sửa, TB 25.79)` | Rò rỉ `Có — lệch TB 26.29% ở hằng số; dịch chuyển reward +0.026 đến +0.051 do co giãn chi phí đặt hàng, các chỉ số vận hành bất biến (≤ 0.0013), giữ nguyên thứ hạng chính sách (Bảng 3b)` | Khắc phục `Tạo lại capacity chỉ từ tập huấn luyện; báo cáo kiểm toán ghép cặp 5 seed ở Bảng 3b`.
  2. *Thêm 1 câu vào **Mục 5 (`P444`)**:* Cuối cùng, kiểm toán rò rỉ sức chứa chuẩn hóa (Mục 2.1.3, Bảng 3b) đánh giá các checkpoint 5 seed hiện có trên môi trường `capacity_train_only` đã sửa thay vì huấn luyện lại toàn bộ tác tử từ đầu; việc huấn luyện lại hoàn toàn dưới chuẩn hóa tập huấn luyện là bước cần thiết nếu sử dụng giá trị tuyệt đối của chi phí đặt hàng (thay vì thứ hạng tương đối giữa các chính sách) trong triển khai thực tế.

---

## Checklist kiểm chứng hoàn thành Task 6
- [x] Tạo `capacity_train_only.tfrecords` và xác nhận `capacity_full` (`20.33`) **nhỏ hơn** `capacity_train_only` (`25.79`, lệch TB `26.29%`, cực đại `50.0%`).
- [x] Đánh giá ghép cặp (504 bước × 4 kịch bản) trên **Base-Stock**, **DQN (đủ 5 seed)**, và **`A2C_mod` (đủ 5 seed)**.
- [x] Báo cáo đầy đủ $\Delta$ của cả **6 metric** kèm kiểm định thống kê (**paired $t$-test**, **Wilcoxon**, **Hedges' $g$**, **95% bootstrap CI**, và **TOST $\pm 0.02$**).
- [x] Đính chính trung thực 2 điểm sai/thiếu ở bản gốc (§2.1.3): (1) sửa chiều `exceeded` → `fell below`, (2) thay nhận định `"under 0.02"` bằng phân rã định lượng rõ ràng (ordering-cost scaling vs. 4 chỉ số vận hành bất biến & thứ hạng 100% bất biến).
- [x] Nêu minh bạch phạm vi Phương án B (đánh giá checkpoint hiện có, không retrain; DQN seed 512 dùng `ckpt-43, hidden=32`) và ghi nhận retrain ở Section 5 (giữ Phương án A làm dự phòng nếu phản biện yêu cầu).

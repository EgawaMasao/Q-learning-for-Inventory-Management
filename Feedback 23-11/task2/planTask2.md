# Plan Task 2 (Cập nhật) — Thực nghiệm 5 seed Standard A2C + Hoàn thiện mô tả `A2C_mod` (§2.2.6, Table 2, Table 6 & Table 7)

> **Cập nhật phạm vi:** Đã huấn luyện xong **đủ 5 seed độc lập (`42, 123, 256, 512, 1024`) của Standard A2C** (`600 episodes`, `ckpt-60` mỗi seed). Kế hoạch chuyển từ *"chỉ viết"* sang **đầy đủ Thực nghiệm + Viết** theo đúng [Task2.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task2/Task2.md).

---

## 1. Kết quả kiểm tra 5 seed Standard A2C đã huấn luyện
Đã kiểm tra toàn bộ thư mục `Feedback 23-11/task2/A2C_{42,123,256,512,1024}/`:

| Seed | Số checkpoint | Checkpoint mới nhất | Số episode log | `eval_metrics_seed*.json` | Trạng thái |
|---:|---:|---|---:|---|---|
| **42** | 60 | `ckpt-60` | 600 | Có sẵn | Hoàn tất |
| **123** | 60 | `ckpt-60` | 600 | Có sẵn | Hoàn tất |
| **256** | 60 | `ckpt-60` | 600 | Có sẵn | Hoàn tất |
| **512** | 60 | `ckpt-60` | 600 | Có sẵn | Hoàn tất |
| **1024** | 60 | `ckpt-60` | 600 | Có sẵn | Hoàn tất |

> **Phát hiện quan trọng cần đồng bộ giao thức đánh giá (Apples-to-Apples):**
> - Trong **Table 6 & Table 7** của bài báo gốc, dòng `All scenarios` là **trung bình của 3 kịch bản (`EASY, MEDIUM, HARD`)** (ví dụ `Base-Stock`: `(0.9320 + 0.8621 + 0.7761)/3 = 0.8567`, `waste_cost = 0.0111`).
> - Tuy nhiên, `eval_metrics_seed*.json` vừa sinh từ notebook `A2C_{seed}.ipynb` có `holding_cost ≈ 0.010` trong khi **Table 6** gốc có `holding_cost ≈ 0.111` (DQN) / `0.094` (`A2C_mod`). Do đó, **không được dán thô** số từ `eval_metrics_seed*.json` sang cạnh Table 6 gốc mà phải **đối chiếu chính xác hàm đánh giá đã sinh Table 6 & Table 7** và chạy lại đánh giá ghép cặp cho cả 5 seed `Standard A2C` trên **đúng cùng 1 giao thức** để đảm bảo công bằng tuyệt đối.

---

## 2. Kế hoạch triển khai (Thực nghiệm + Viết)

### Giai đoạn 1 — Thực nghiệm đánh giá & kiểm định thống kê (`task2_eval_standard_a2c.ipynb`)
1. **Xác minh chính xác giao thức sinh Table 6 & Table 7 gốc:**
   - Truy tìm hàm đánh giá gốc trong repo (`Feedback 7-9/` / `Training/`) để xác định chính xác cách tính `holding_cost`, trạng thái khởi tạo `x_init`, và cách tổng hợp `All scenarios` sao cho tái lập khớp **Table 6** (`Base-Stock = +0.8567`, `DQN = +0.7197`, `A2C_mod = +0.6939`).
2. **Tạo & chạy notebook `Feedback 23-11/task2/task2_eval_standard_a2c.ipynb`:**
   - Nạp toàn bộ **5 checkpoint `ckpt-60` của Standard A2C** (`42, 123, 256, 512, 1024`) bằng đường dẫn tuyệt đối.
   - Đánh giá trên **đúng giao thức Table 6 & Table 7** qua 3 kịch bản (`EASY, MEDIUM, HARD`) và `All scenarios` (trung bình 3 kịch bản), tính `Mean ± SD` của cả **6 metric** (*Total Reward, Service Level, Holding Cost, Waste Cost, Ordering Cost, Stockout Rate*).
   - Tính trực tiếp **$\Delta$ so với Base-Stock ($k=1.0$)** cho cột mới trong **Table 7**.
   - So sánh ghép cặp 5 seed **Standard A2C vs. `A2C_mod`** (paired $t$-test / Wilcoxon / Hedges' $g$) và đo thêm **chỉ số độ ổn định chính sách** (*mean policy entropy*, *tỷ lệ action cực trị*, *SD qua 5 seed*) để kiểm chứng thực nghiệm nhận định ở §2.2.6 về tác dụng của entropy regularization.
   - Lưu kết quả chuẩn hóa vào `Feedback 23-11/task2/output/` (`.csv`, `.json`).

### Giai đoạn 2 — Cập nhật phần Viết (`outputTask2.md`)
Giữ nguyên 100% các phần viết đã đạt chuẩn từ vòng trước (**Table 2** fix C-02, **§2.2.6** 4 phương trình `(1)–(4)`, **Algorithm 1**, **Theoretical Motivation**, **Table X**), và **bổ sung/cập nhật 3 chỗ**:
1. **Cập nhật đoạn cuối §2.2.6 (`Scope note`):**
   - **Bỏ câu cũ** *"an empirical multi-seed benchmark between conventional A2C and A2C_mod is left to future work"*.
   - Thay bằng 2–3 câu nối mạch dẫn sang kết quả thực nghiệm 5 seed ở **Table 6 & Table 7** (nêu rõ tác động thực tế của entropy bonus và policy gradient chuẩn lên độ ổn định qua 5 seed so với `A2C_mod`).
2. **Thêm cột `Standard A2C` (5-seed Mean ± SD) vào Table 6 và cột `Standard A2C` + `Δ` vào Table 7 (§3.2):**
   - Cập nhật đoạn dẫn `P192` (*"compares DQN, A2C_mod, and conventional A2C (5 seeds each) with the classical Base-Stock policy"*).
   - Bổ sung 1 đoạn phân tích định lượng ngắn (3–4 câu) ngay dưới Table 6/Table 7 so sánh **Standard A2C vs. `A2C_mod`** (hiệu năng, phương sai giữa 5 seed, và lý do khác biệt khớp với phân tích lý thuyết ở §2.2.6).
3. **Cập nhật §5 (*Study Limitations*):** rà soát đảm bảo không còn câu nào nói chưa so sánh thực nghiệm với Standard A2C.

### Giai đoạn 3 — Đồng bộ vào Word & `note.md` *(khi bạn xác nhận)*
- Cập nhật `apply_tasks1to8_docx.py` để chèn cột `Standard A2C` vào **Table 6**, **Table 7**, cập nhật **§2.2.6** & **§3.2**, xuất lại file Word và cập nhật `note.md`.

---

## 3. Sản phẩm bàn giao dự kiến
- `Feedback 23-11/task2/task2_eval_standard_a2c.ipynb` (notebook đánh giá 5 seed Standard A2C đồng bộ giao thức Table 6 & Table 7).
- `Feedback 23-11/task2/output/` (CSV/JSON tổng hợp 5 seed, $\Delta$ Table 7, kiểm định thống kê & entropy/collapse diagnostics).
- `Feedback 23-11/task2/outputTask2.md` (cập nhật trọn bộ Thực nghiệm + Viết song ngữ Anh–Việt).

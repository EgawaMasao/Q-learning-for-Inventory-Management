# Plan Task 2 — Hoàn thiện mô tả A2C_mod và so sánh với A2C tiêu chuẩn (Chỉ xử lý phần viết)

## 1. Mục tiêu & Phạm vi

Giải quyết đầy đủ yêu cầu của [Task2.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task2/Task2.md) trên bản thảo [Xai_Inventory_Revision_27Sep (1).md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).md) **chỉ ở phần viết** (không huấn luyện A2C tiêu chuẩn với 5 seed và không sửa Table 7):

1. **Mô tả đầy đủ quy trình học của `A2C_mod`**: bổ sung **Algorithm box (pseudo-code)** và hệ phương trình toán học chính xác khớp 100% với mã nguồn `Training/A2C-mod.ipynb` / `Training/A2C-mod-seed.ipynb`.
2. **Giải thích động cơ lý thuyết** của cơ chế cập nhật `A2C_mod` (tính thứ tự của 14 mức bổ sung, làm mượt kiểu Laplacian theo khoảng cách chỉ số hành động, tổn thất bình phương kiểu Brier giúp giới hạn gradient và giảm phương sai, đồng thời thẳng thắn nêu thiên lệch bảo thủ do không tối ưu entropy).
3. **So sánh hệ thống giữa A2C tiêu chuẩn và `A2C_mod`**: đưa bảng so sánh chi tiết cùng phần phân tích ngay trong bài chính (§3.2) thay vì chỉ trỏ mơ hồ sang Supplementary Material.
4. **Sửa toàn bộ mâu thuẫn nội tại & sai lệch so với mã nguồn** (bao gồm **C-02** trong Table 2 vs văn bản §3.2, chuẩn hóa Critic, và ghi chú entropy).

---

## 2. Đối chiếu hiện trạng bài báo ↔ mã nguồn (Các điểm cần sửa)

Qua đối chiếu [Xai_Inventory_Revision_27Sep (1).md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).md), [Xai_Inventory_27Sep_ReadingNotes.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_27Sep_ReadingNotes.md) và `Training/A2C-mod.ipynb`:

| # | Hạng mục | Hiện trạng trong bài báo | Thực tế trong code (`A2C-mod.ipynb`) | Hướng xử lý trong Task 2 |
|---|---|---|---|---|
| 1 | **Kiến trúc Actor & Critic (C-02)** | Table 2 (dòng 145): `3 hidden layers, 32 units each` (không phân biệt Actor/Critic). Văn bản §3.2 (dòng 363): `four-layer feed-forward network` (Actor) và `two-layer critic`. | **Actor** (`:122-145`): 3 lớp ẩn (32 units, ReLU, dropout 0.1) + 1 lớp đầu ra (14 units, softmax) = **4 lớp Dense**. **Critic** (`:147-162`): 1 lớp ẩn (32 units, LayerNorm, ReLU, dropout 0.1) + 1 lớp đầu ra (1 unit) = **2 lớp Dense**. | Thống nhất cả Table 2 và §3.2: ghi rõ *Actor: 3 hidden layers (32 units) + output layer (4 layers total); Critic: 1 hidden layer (32 units) + output layer (2 layers total)*. |
| 2 | **Chuẩn hóa trong Critic** | Table 2 (dòng 151): `GroupNormalization (groups = 1) in the critic only`. | Critic (`:156-157`) dùng `tf.keras.layers.LayerNormalization()` (có comment `# Layer normalization instead of GroupNormalization`). | Sửa Table 2 thành `LayerNormalization in the critic only` (có thể chú thích ngắn gọn tương đương `GroupNormalization(groups=1)` trên tensor 2D). |
| 3 | **Entropy coefficient** | Table 2 (dòng 181): ghi `0.001` không kèm giải thích; §3.2 (dòng 365) nói `entropy is computed but not optimized`. | `A2C-mod.ipynb:446-453`: tính `entropy_adj = 0.001 * entropy_p` để log, **không** cộng vào `actor_loss`. | Sửa Table 2 thành `0.001 (computed and logged only; not included in actor loss)` để khớp §3.2. |
| 4 | **Phương trình cập nhật Actor** | Table 2 (dòng 205) viết dạng inline `p_new = softmax(logπ(a|s) + δ/(|a−a'|+1))`; §3.2 chỉ diễn giải bằng lời, chưa có Algorithm box và công thức đánh số. | `A2C-mod.ipynb:435-453`: $\delta = r + \gamma V(s') - V(s)$; `p_new = softmax(log(max(1e-15, p)) + delta / (|a - k| + 1))`; `actor_loss = mean((p - p_new)^2)`; `critic_loss = 0.5 * mean(delta^2)`. | Bổ sung đầy đủ các phương trình đánh số cho $\delta_{t,i}$, $\mathcal{L}_{\text{critic}}$, $p^{\text{new}}$, $\mathcal{L}_{\text{actor}}$ và Algorithm box hoàn chỉnh. |
| 5 | **So sánh A2C tiêu chuẩn vs `A2C_mod`** | §3.2 (dòng 365) chỉ nêu 1 câu ngắn và đẩy sang Supplementary Material (`the full comparison is provided in the Supplementary Material`). | Chưa có bảng so sánh trực tiếp trong bài chính. | Đưa bảng so sánh đầy đủ giữa **Conventional A2C** và **`A2C_mod`** trực tiếp vào §3.2 kèm phân tích lý thuyết & hàm ý đối với XRL. |

---

## 3. Kế hoạch chi tiết các phần viết mới / chỉnh sửa

### Bước 1: Chỉnh sửa Table 2 (dòng 131–206 trong bản thảo)
- Cập nhật hàng **Hidden layers / width** cột `A2C_mod`:
  - `Actor: 3 hidden layers, 32 units each (+ output layer, 4 layers total); Critic: 1 hidden layer, 32 units (+ output layer, 2 layers total)`
- Cập nhật hàng **Normalization** cột `A2C_mod`:
  - `LayerNormalization in the critic hidden layer only (none in actor)`
- Cập nhật hàng **Entropy coefficient** cột `A2C_mod`:
  - `0.001 (computed and logged only; not added to actor loss)`
- Chuẩn hóa ký hiệu ở hàng **Loss function** cột `A2C_mod` cho thống nhất với các phương trình ở §3.2 (dùng chỉ số hành động $k \in \{0,\dots,13\}$ và hành động đã chọn $a_{t,i}$).

### Bước 2: Viết lại và mở rộng tiểu mục §3.2 *"A2C_mod Algorithm and Its Relation to Conventional A2C"* (dòng 361–366)
Cấu trúc lại tiểu mục thành 4 phần mạch lạc:

1. **Kiến trúc mạng và chia sẻ trọng số (Network Architecture & Weight Sharing)**
   - Nêu rõ đầu vào per-SKU $s_{t,i} \in \mathbb{R}^3$ (tồn kho, nhu cầu, hao hụt) cho 220 SKU.
   - Khớp chính xác kiến trúc Actor (4 lớp: 3 lớp ẩn 32 đơn vị + ReLU + Dropout 0.1, lớp đầu ra 14 đơn vị + softmax) và Critic (2 lớp: 1 lớp ẩn 32 đơn vị + LayerNormalization + ReLU + Dropout 0.1, lớp đầu ra vô hướng $V_\phi(s_{t,i})$).

2. **Công thức toán học và Algorithm Box (Algorithm 1: Training Procedure of `A2C_mod`)**
   - Trình bày các phương trình tường minh:
     - 1-step TD error (advantage chưa chuẩn hóa):
       $$\delta_{t,i} = r_{t,i} + \gamma V_\phi(s'_{t,i}) - V_\phi(s_{t,i})$$
     - Critic loss:
       $$\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{2|\mathcal{B}|N} \sum_{t \in \mathcal{B}} \sum_{i=1}^{N} \delta_{t,i}^2$$
     - Phân phối mục tiêu mềm có trọng số khoảng cách (distance-weighted soft target) cho từng mức hành động $k \in \{0,\dots,13\}$:
       $$p^{\text{new}}(k \mid s_{t,i}) = \operatorname{softmax}_k\!\left(\log \max\!\left(\epsilon,\, \pi_\theta(k \mid s_{t,i})\right) + \frac{\delta_{t,i}}{|a_{t,i} - k| + 1}\right), \quad \epsilon = 10^{-15}$$
     - Actor loss (MSE / Brier-style distillation):
       $$\mathcal{L}_{\text{actor}}(\theta) = \frac{1}{|\mathcal{B}|N|\mathcal{A}|} \sum_{t \in \mathcal{B}} \sum_{i=1}^{N} \sum_{k=0}^{|\mathcal{A}|-1} \left(\pi_\theta(k \mid s_{t,i}) - p^{\text{new}}(k \mid s_{t,i})\right)^2$$
   - Bổ sung **Algorithm 1** (pseudo-code chuẩn bài báo khoa học) mô tả vòng lặp 600 episode, trượt cửa sổ 32 bước (shift 31) trên chuỗi thời gian huấn luyện, rollout đồng bộ cho 220 SKU, tính $\delta$, $p^{\text{new}}$, log entropy (không đưa vào loss), và cập nhật riêng Actor/Critic bằng Adam ($\text{lr}=0.001$).

3. **Động cơ lý thuyết (Theoretical Motivation)**
   - **Khai thác cấu trúc thứ tự của không gian hành động (Ordinal Action Smoothing)**: 14 hành động biểu diễn các mức tỷ lệ đặt hàng tăng dần. Hệ số $1 / (|a_{t,i} - k| + 1)$ hoạt động như hạt nhân làm mượt kiểu Laplacian rời rạc, lan truyền tín hiệu advantage sang các mức đặt hàng lân cận và suy giảm dần với các mức xa hơn, khuyến khích điều chỉnh chính sách mượt mà thay vì nhảy cóc.
   - **Ổn định gradient và giảm phương sai (Bounded Gradients & Variance Reduction)**: Khác với policy gradient chuẩn $-\delta \log \pi_\theta(a\mid s)$ dễ bùng nổ gradient khi $\pi \to 0$, tổn thất bình phương trên simplex xác suất (tương tự Brier score — một *proper scoring rule*) có gradient bị chặn, giúp ổn định huấn luyện với trọng số dùng chung trên 220 SKU có quy mô cầu khác nhau.
   - **Thiên lệch khai thác do không tối ưu entropy (Conservative / Exploitation Bias)**: Nêu minh bạch rằng việc chỉ tính/log entropy mà không đưa vào hàm mất mát khiến chính sách dễ bão hòa về các hành động cực trị — đây là đặc tính/hạn chế của thiết kế gốc [4], không phải cải tiến thuần túy, và giúp giải thích các đặc điểm hội tụ/giải thích ở phần kết quả.

4. **Bảng so sánh trực tiếp: Conventional A2C vs. `A2C_mod` (Table 3 mới hoặc bảng đánh số phù hợp)**
   - So sánh có cấu trúc trên 8 tiêu chí:
     1. Hàm mục tiêu cập nhật Actor (Policy gradient $-\delta \log \pi(a\mid s)$ vs. MSE distillation tới $p^{\text{new}}$)
     2. Lan truyền tín dụng giữa các hành động (chỉ tác động lên hành động được chọn $a$ vs. lan truyền có trọng số khoảng cách $1/(|a-k|+1)$ trên cả 14 mức thứ tự)
     3. Đặc tính gradient & phương sai (không bị chặn khi $\pi \to 0$, phương sai cao hơn vs. bị chặn trên simplex xác suất, cập nhật mượt hơn)
     4. Ước lượng Advantage (thường dùng $n$-step return / GAE có chuẩn hóa vs. 1-step TD error $\delta$ không chuẩn hóa)
     5. Điều chuẩn Entropy (cộng $-\beta H(\pi)$ trực tiếp vào loss vs. chỉ tính và log, không tối ưu)
     6. Kiến trúc mạng & chia sẻ trọng số (thường dùng backbone chung hoặc 2 mạng tách biệt cho toàn state vs. Actor 4 lớp + Critic 2 lớp có LayerNorm chia sẻ trọng số theo từng SKU)
     7. Bộ tối ưu & lịch cập nhật (thường dùng chung optimizer / rollout song song vs. 2 Adam optimizer độc lập trên cửa sổ trượt 32 bước)
     8. Hàm ý đối với khả năng giải thích (XRL) (chính sách tiêu chuẩn phụ thuộc mạnh vào entropy bonus; `A2C_mod` cho biến thiên mượt theo thứ tự hành động nhưng dễ bão hòa hơn khi thiếu điều chuẩn entropy).
   - **Ghi chú phạm vi minh bạch (Scope Note)**: Nêu rõ nghiên cứu này kế thừa trực tiếp `A2C_mod` từ Meisheri et al. [4] với vai trò đối tượng giải thích trong khung XRL; so sánh với A2C tiêu chuẩn được thực hiện ở mức thuật toán và lý thuyết, còn so sánh thực nghiệm đa seed với A2C tiêu chuẩn được để lại cho nghiên cứu tiếp theo.

### Bước 3: Rà soát tính nhất quán toàn bài báo
- Kiểm tra mọi chỗ nhắc đến `A2C_mod`, `Table 2`, `GroupNormalization`, `four-layer`, `two-layer`, `Supplementary Material` trong [Xai_Inventory_Revision_27Sep (1).md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).md) để:
  - Đảm bảo không còn câu nào hàm ý `A2C_mod` vượt trội thực nghiệm so với A2C tiêu chuẩn.
  - Đảm bảo số thứ tự bảng (nếu thêm bảng so sánh mới vào §3.2) hoặc cách đặt tên bảng (ví dụ: Table 2b / Table X) nhất quán, tránh làm lệch toàn bộ đánh số bảng phía sau nếu chưa cần đổi đánh số toàn bài.
  - Đảm bảo mọi ký hiệu toán học ($\pi_\theta$, $V_\phi$, $\delta_{t,i}$, $p^{\text{new}}$, $\mathcal{A}$) thống nhất giữa Table 2, Algorithm 1 và văn bản §3.2.

---

## 4. Sản phẩm bàn giao dự kiến

1. **File kế hoạch**: `Feedback 23-11/task2/planTask2.md` (file này).
2. **File nội dung đã soạn sẵn để ghép vào bài báo**: `Feedback 23-11/task2/outputTask2.md` gồm:
   - Bản sửa hoàn chỉnh **Table 2** (tiếng Anh chuẩn học thuật).
   - Bản viết hoàn chỉnh tiểu mục **§3.2 A2C_mod Algorithm and Its Relation to Conventional A2C** (gồm phương trình đánh số, **Algorithm 1** pseudo-code, động cơ lý thuyết, **Bảng so sánh Conventional A2C vs. A2C_mod**, và Scope Note).
   - Danh sách các vị trí cụ thể trong `Xai_Inventory_Revision_27Sep (1).md` cần thay thế (kèm trước/sau để dễ đối chiếu và cập nhật sang file `.docx`).
3. *(Tùy chọn theo xác nhận của người dùng)*: Cập nhật trực tiếp các đoạn tương ứng vào `Xai_Inventory_Revision_27Sep (1).md`.

---

## 5. Tiêu chí kiểm chứng (Checklist hoàn thành)

- [ ] **Algorithm box (Algorithm 1)** đầy đủ, từng dòng khớp với `Training/A2C-mod.ipynb`.
- [ ] **Phương trình `p_new`, `delta`, `actor_loss`, `critic_loss`** chính xác, nhất quán ký hiệu.
- [ ] **Động cơ lý thuyết** bao quát đủ 3 ý: ordinal smoothing (Laplacian-like), bounded gradient / Brier-style variance reduction, và thiên lệch khai thác do không tối ưu entropy.
- [ ] **Bảng so sánh Conventional A2C vs. A2C_mod** đầy đủ, khách quan, không tuyên bố vượt trội thực nghiệm chưa kiểm chứng.
- [ ] **Khắc phục triệt để C-02** (Actor 3 hidden / 4 layers, Critic 1 hidden / 2 layers) cùng 2 lỗi nhỏ đi kèm (`LayerNormalization` trong Critic, `entropy` chỉ log không tối ưu).
- [ ] **Không thực hiện thực nghiệm / không sửa Table 7** theo đúng yêu cầu.

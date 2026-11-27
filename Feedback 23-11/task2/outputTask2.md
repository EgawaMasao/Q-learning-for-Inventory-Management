# Output Task 2 — Mô tả thuật toán A2C_mod & So sánh với A2C tiêu chuẩn (Phần viết)

## Tóm tắt công việc thực hiện
- **Phạm vi:** Chỉ xử lý phần viết theo [planTask2.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task2/planTask2.md); không chạy code, không huấn luyện A2C tiêu chuẩn 5 seed, không thay đổi Table 7.
- **Nội dung giải quyết:** (1) Sửa mâu thuẫn kiến trúc **C-02**, chuẩn hóa Critic (`LayerNormalization`) và ghi chú `entropy` trong **Table 2**; (2) Viết lại **§3.2** với đầy đủ phương trình, **Algorithm 1**, động cơ lý thuyết, **Bảng so sánh Conventional A2C vs. `A2C_mod`** và ghi chú phạm vi.

---

## Ghi vào bài báo

### Chỉnh sửa 1 — Cập nhật Table 2 (sửa mâu thuẫn C-02, LayerNormalization, Entropy, ký hiệu Loss)

- **Vị trí:** Mục **Materials and Methods** → *Action-to-Replenishment Mapping* → **Table 2** ([Xai_Inventory_Revision_27Sep (1).md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).md#L133-L206), thay thế các hàng `Hidden layers / width`, `Normalization`, `Entropy coefficient`, `Loss function` ở cột `A2C_mod` — hoặc thay cả bảng bên dưới).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

**Table 2. Training hyperparameters for DQN and A2C_mod.**

| Hyperparameter | DQN | A2C_mod |
|---|---|---|
| Hidden layers / width | 3 hidden layers, 128 units each | Actor: 3 hidden layers (32 units each) + output layer (4 layers total); Critic: 1 hidden layer (32 units) + output layer (2 layers total) |
| Normalization | GroupNormalization (groups = 1) after each hidden layer | LayerNormalization in the critic hidden layer only (none in actor) |
| Output head | Dense(128→14): one Q-value per action, per product | Actor: Dense(32→14) + softmax; Critic: Dense(32→1) state value |
| Optimizer | Adam | Adam (separate actor/critic instances) |
| Learning rate | 0.001 | 0.001 (actor), 0.001 (critic) |
| Discount factor $\gamma$ | 0.99 | 0.99 |
| Entropy coefficient | (not applicable) | 0.001 (computed and logged only; not added to actor loss) |
| Experience replay buffer | 100,000 transitions (min. 1,000 before training starts) | (on-policy, no replay buffer) |
| Exploration | $\varepsilon$-greedy, $\varepsilon$: 1.0 → 0.01, linear decay over the first 400 episodes | Stochastic softmax policy sampling (no explicit $\varepsilon$) |
| Target network | Separate target Q-network, hard-synced from the online network every 10 episodes (Double-DQN: online net selects the action, target net evaluates it) | (no target network; single critic) |
| Loss function | Huber loss on the TD error | Critic: $\frac{1}{2}\delta_{t,i}^2$; Actor (A2C_mod): MSE between $\pi_\theta(\cdot \mid s_{t,i})$ and distance-weighted soft target $p^{\text{new}}(\cdot \mid s_{t,i})$ (see Eq. (1)–(4)) |

- **Bản Tiếng Việt (Để theo dõi):**

**Bảng 2. Siêu tham số huấn luyện cho DQN và A2C_mod.**

| Siêu tham số | DQN | A2C_mod |
|---|---|---|
| Số lớp ẩn / độ rộng | 3 lớp ẩn, mỗi lớp 128 đơn vị | Actor: 3 lớp ẩn (mỗi lớp 32 đơn vị) + lớp đầu ra (tổng 4 lớp); Critic: 1 lớp ẩn (32 đơn vị) + lớp đầu ra (tổng 2 lớp) |
| Chuẩn hóa | GroupNormalization (groups = 1) sau mỗi lớp ẩn | Chỉ dùng LayerNormalization ở lớp ẩn của Critic (không dùng ở Actor) |
| Lớp đầu ra | Dense(128→14): một giá trị Q cho mỗi hành động, mỗi sản phẩm | Actor: Dense(32→14) + softmax; Critic: Dense(32→1) giá trị trạng thái |
| Bộ tối ưu | Adam | Adam (2 bộ tối ưu tách biệt cho Actor và Critic) |
| Tốc độ học | 0.001 | 0.001 (Actor), 0.001 (Critic) |
| Hệ số chiết khấu $\gamma$ | 0.99 | 0.99 |
| Hệ số Entropy | (không áp dụng) | 0.001 (chỉ tính và ghi log; không cộng vào hàm mất mát của Actor) |
| Bộ đệm kinh nghiệm | 100.000 chuyển tiếp (tối thiểu 1.000 trước khi huấn luyện) | (on-policy, không dùng replay buffer) |
| Khám phá | $\varepsilon$-greedy, $\varepsilon$: 1.0 → 0.01, giảm tuyến tính trong 400 episode đầu | Lấy mẫu ngẫu nhiên từ chính sách softmax (không có $\varepsilon$ tường minh) |
| Mạng mục tiêu | Mạng Q mục tiêu tách biệt, đồng bộ cứng mỗi 10 episode (Double-DQN) | (không có mạng mục tiêu; dùng một Critic duy nhất) |
| Hàm mất mát | Huber loss trên sai số TD | Critic: $\frac{1}{2}\delta_{t,i}^2$; Actor (A2C_mod): MSE giữa $\pi_\theta(\cdot \mid s_{t,i})$ và mục tiêu mềm có trọng số khoảng cách $p^{\text{new}}(\cdot \mid s_{t,i})$ (xem PT (1)–(4)) |

---

### Chỉnh sửa 2 — Viết lại tiểu mục §3.2 (*A2C_mod Algorithm and Its Relation to Conventional A2C*)

- **Vị trí:** Mục **Materials and Methods** → tiểu mục **A2C_mod Algorithm and Its Relation to Conventional A2C** ([Xai_Inventory_Revision_27Sep (1).md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).md#L361-L366), thay thế toàn bộ 2 đoạn văn hiện tại dưới tiêu đề này).

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

#### A2C_mod Algorithm and Its Relation to Conventional A2C

Following Meisheri et al. [4], `A2C_mod` adopts a weight-shared per-product actor–critic architecture over $N = 220$ SKUs. At each step $t$, each SKU $i \in \{1,\dots,N\}$ is represented by its local state $s_{t,i} = (I_{t,i}, D_{t,i}, W_{t,i}) \in \mathbb{R}^3$ and mapped to one of $|\mathcal{A}| = 14$ ordered replenishment levels. The actor $\pi_\theta(\cdot \mid s_{t,i})$ is a four-layer feed-forward network (three hidden layers of 32 units with ReLU activation and dropout rate 0.1, followed by a 14-unit softmax output layer), whereas the critic $V_\phi(s_{t,i})$ is a shallower two-layer network (one 32-unit hidden layer with LayerNormalization, ReLU, and dropout rate 0.1, followed by a scalar linear output). Both networks are trained for 600 episodes using sliding chronological rollout windows of length $B = 32$ (stride $B - 1$) and separate Adam optimizers ($\text{lr} = 0.001$, $\gamma = 0.99$).

For each transition $(s_{t,i}, a_{t,i}, r_{t,i}, s'_{t,i})$ in a rollout window $\mathcal{B}$, the critic computes the unnormalized one-step temporal-difference (TD) advantage:

$$\delta_{t,i} = r_{t,i} + \gamma V_\phi(s'_{t,i}) - V_\phi(s_{t,i}), \tag{1}$$

and is updated by minimizing the mean squared TD error:

$$\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{2|\mathcal{B}|N} \sum_{t \in \mathcal{B}} \sum_{i=1}^{N} \delta_{t,i}^2. \tag{2}$$

Rather than applying the standard log-likelihood policy gradient $-\delta_{t,i}\log \pi_\theta(a_{t,i}\mid s_{t,i})$, `A2C_mod` constructs an advantage-reweighted soft target distribution over all 14 actions $k \in \{0,\dots,|\mathcal{A}|-1\}$:

$$p^{\text{new}}(k \mid s_{t,i}) = \operatorname{softmax}_k\!\left(\log \max\!\left(\epsilon,\, \pi_\theta(k \mid s_{t,i})\right) + \frac{\delta_{t,i}}{|a_{t,i} - k| + 1}\right), \quad \epsilon = 10^{-15}, \tag{3}$$

and updates the actor by minimizing the mean squared error (Brier-type loss) between the current policy and this soft target:

$$\mathcal{L}_{\text{actor}}(\theta) = \frac{1}{|\mathcal{B}|N|\mathcal{A}|} \sum_{t \in \mathcal{B}} \sum_{i=1}^{N} \sum_{k=0}^{|\mathcal{A}|-1} \left(\pi_\theta(k \mid s_{t,i}) - p^{\text{new}}(k \mid s_{t,i})\right)^2. \tag{4}$$

Algorithm 1 summarizes the complete training procedure.

**Algorithm 1. Training procedure of `A2C_mod`**

```text
Input:  Training demand series; capacity vector; episodes E = 600; window size B = 32;
        discount γ = 0.99; learning rates η_θ = η_φ = 0.001; numerical floor ε = 1e-15.
Initialize: Weight-shared actor π_θ (4 layers) and critic V_φ (2 layers); separate Adam optimizers.

1:  for episode e = 1 to E do
2:      Initialize normalized inventory x_i ~ Uniform(0, 1) and waste q_i = 0.025 * x_i for all i = 1..N
3:      for each chronological window B of length 32 (stride 31) do
4:          Reset rollout buffer D
5:          for each step t in window B do
6:              Form per-SKU states s_{t,i} = (x_{t,i}, d_{t,i}, q_{t,i}) for i = 1..N
7:              Compute action probabilities π_θ(· | s_{t,i}) and sample a_{t,i} ~ π_θ(· | s_{t,i})
8:              Apply replenishment u_{t,i} = A[a_{t,i}], observe r_{t,i} and next state s'_{t,i}
9:              Store (s_{t,i}, a_{t,i}, π_θ(· | s_{t,i}), r_{t,i}, s'_{t,i}) in D
10:         end for
11:         Flatten D across steps and N = 220 SKUs
12:         Compute 1-step TD advantage δ_{t,i} = r_{t,i} + γ V_φ(s'_{t,i}) - V_φ(s_{t,i})     [Eq. (1)]
13:         Compute critic loss L_critic(φ) = (1 / 2|D|) Σ δ_{t,i}^2                           [Eq. (2)]
14:         For each k in {0..13}, form soft target:
                p^new(k | s_{t,i}) = softmax_k( log max(ε, π_θ(k | s_{t,i})) + δ_{t,i} / (|a_{t,i} - k| + 1) )  [Eq. (3)]
15:         Compute actor loss L_actor(θ) = mean_{D, k}( (π_θ(k | s_{t,i}) - p^new(k | s_{t,i}))^2 )            [Eq. (4)]
16:         Compute policy entropy H(π_θ) for logging only (excluded from L_actor)
17:         Update θ ← Adam(θ, ∇_θ L_actor, η_θ) and φ ← Adam(φ, ∇_φ L_critic, η_φ)
18:     end for
19: end for
Output: Trained weight-shared actor π_θ and critic V_φ.
```

**Theoretical Motivation.** The modified actor update is motivated by two structural properties of multi-product replenishment. First, the 14 discrete actions represent strictly ordered ordering-rate fractions. Whereas conventional policy gradient updates only the log-probability of the executed action $a_{t,i}$, the inverse-distance kernel $1/(|a_{t,i} - k| + 1)$ in Eq. (3) acts as a discrete Laplacian-like smoother along the action index, propagating positive or negative TD signals to neighboring replenishment levels while attenuating distant ones. This encourages gradual policy adjustments rather than abrupt jumps across order magnitudes. Second, regressing $\pi_\theta$ toward $p^{\text{new}}$ via squared error on the probability simplex (Eq. (4)) corresponds to a Brier-style proper scoring rule with bounded gradients, avoiding the unbounded $-\log \pi$ gradient spikes that arise in standard policy gradient when action probabilities approach zero, thereby reducing update variance under shared weights across 220 heterogeneous SKUs. Conversely, because policy entropy is computed solely for logging and omitted from $\mathcal{L}_{\text{actor}}$, `A2C_mod` lacks explicit exploration pressure; this induces a conservative exploitation bias (higher tendency toward policy saturation) rather than an unconditional algorithmic improvement.

Table X summarizes the methodological differences between conventional synchronous A2C and `A2C_mod`. *(Ghi chú dàn trang: đổi `Table X` thành số bảng tương ứng trong `.docx`, ví dụ `Table 2b` hoặc cập nhật lại thứ tự bảng.)*

**Table X. Methodological comparison between conventional A2C and `A2C_mod`.**

| Dimension | Conventional A2C | `A2C_mod` (Meisheri et al. [4], used here) |
|---|---|---|
| Actor update objective | Policy gradient: $-\mathbb{E}[\hat{A}_t \log \pi_\theta(a_t \mid s_t)]$ | Squared-error (Brier-style) distillation toward $p^{\text{new}}$ [Eq. (4)] |
| Action credit assignment | Pointwise on executed action $a_t$ only | Distance-weighted smoothing over all 14 ordinal actions via $1/(\|a_{t,i}-k\|+1)$ [Eq. (3)] |
| Gradient behavior & variance | Unbounded as $\pi_\theta(a_t\mid s_t) \to 0$; higher variance | Bounded on the probability simplex; smoother, lower-variance updates |
| Advantage estimation | $n$-step return or GAE, typically batch-normalized | Unnormalized 1-step TD error $\delta_{t,i}$ [Eq. (1)] |
| Entropy regularization | Explicit bonus $-\beta H(\pi_\theta)$ added to actor loss | Computed and logged only; not optimized (exploitation bias) |
| Architecture & weight sharing | Shared trunk or separate networks over full state | Weight-shared per-SKU 4-layer actor + 2-layer critic with LayerNormalization |
| Optimization schedule | Often shared optimizer across parallel synchronous workers | Separate Adam optimizers updated over 32-step chronological windows |
| Implications for XRL | Exploration bonus preserves broader action support | Smooth ordinal transitions, but unregularized entropy can cause sharper policy saturation |

*Scope note:* Because this study focuses on explaining existing DRL replenishment policies rather than proposing a new RL algorithm, `A2C_mod` is adopted directly from [4]; the comparison with conventional A2C is therefore established analytically here, while an empirical multi-seed benchmark between conventional A2C and `A2C_mod` is left to future work.

- **Bản Tiếng Việt (Để theo dõi):**

#### Thuật toán A2C_mod và mối liên hệ với A2C tiêu chuẩn

Kế thừa Meisheri et al. [4], `A2C_mod` sử dụng kiến trúc actor–critic chia sẻ trọng số theo từng sản phẩm trên $N = 220$ SKU. Tại mỗi bước thời gian $t$, mỗi SKU $i \in \{1,\dots,N\}$ được biểu diễn bởi trạng thái cục bộ $s_{t,i} = (I_{t,i}, D_{t,i}, W_{t,i}) \in \mathbb{R}^3$ và ánh xạ tới một trong $|\mathcal{A}| = 14$ mức bổ sung có thứ tự. Mạng actor $\pi_\theta(\cdot \mid s_{t,i})$ là mạng truyền thẳng 4 lớp (3 lớp ẩn 32 đơn vị với hàm kích hoạt ReLU và dropout 0.1, tiếp theo là lớp đầu ra softmax 14 đơn vị), trong khi mạng critic $V_\phi(s_{t,i})$ nông hơn với 2 lớp (1 lớp ẩn 32 đơn vị kèm LayerNormalization, ReLU và dropout 0.1, tiếp theo là lớp đầu ra tuyến tính vô hướng). Cả hai mạng được huấn luyện trong 600 episode trên các cửa sổ trượt theo trình tự thời gian độ dài $B = 32$ (bước trượt $B - 1$) với hai bộ tối ưu Adam độc lập ($\text{lr} = 0.001$, $\gamma = 0.99$).

Với mỗi chuyển tiếp $(s_{t,i}, a_{t,i}, r_{t,i}, s'_{t,i})$ trong cửa sổ $\mathcal{B}$, critic tính sai số sai phân thời gian một bước (1-step TD advantage) không chuẩn hóa theo PT (1) và cập nhật bằng cách tối thiểu hóa trung bình bình phương sai số TD theo PT (2). Thay vì dùng policy gradient chuẩn $-\delta_{t,i}\log \pi_\theta(a_{t,i}\mid s_{t,i})$, `A2C_mod` xây dựng phân phối mục tiêu mềm có trọng số khoảng cách $p^{\text{new}}(k \mid s_{t,i})$ trên cả 14 mức hành động $k \in \{0,\dots,13\}$ theo PT (3), rồi cập nhật actor bằng cách tối thiểu hóa sai số bình phương trung bình (dạng tổn thất Brier) giữa chính sách hiện tại và mục tiêu mềm này theo PT (4). Quy trình huấn luyện đầy đủ được tóm tắt trong **Thuật toán 1**.

**Thuật toán 1 (bản dịch theo dõi):**
- Khởi tạo actor $\pi_\theta$ (4 lớp) và critic $V_\phi$ (2 lớp) dùng chung trọng số trên 220 SKU, cùng 2 bộ tối ưu Adam riêng biệt.
- Lặp qua $E = 600$ episode: khởi tạo tồn kho $x_i \sim \text{Uniform}(0,1)$ và hao hụt $q_i = 0.025 x_i$.
- Trượt từng cửa sổ thời gian 32 bước (shift 31): thu thập chuyển tiếp $(s_{t,i}, a_{t,i}, \pi_\theta, r_{t,i}, s'_{t,i})$ cho cả 220 SKU.
- Trải phẳng batch theo bước và SKU; tính $\delta_{t,i}$ [PT (1)], $\mathcal{L}_{\text{critic}}$ [PT (2)], mục tiêu mềm $p^{\text{new}}$ [PT (3)], và $\mathcal{L}_{\text{actor}}$ [PT (4)].
- Tính entropy $H(\pi_\theta)$ chỉ để ghi log (không đưa vào $\mathcal{L}_{\text{actor}}$); cập nhật riêng $\theta$ và $\phi$ bằng Adam.

**Động cơ lý thuyết.** Quy tắc cập nhật actor cải tiến xuất phát từ hai đặc tính cấu trúc của bài toán bổ sung hàng đa sản phẩm. Thứ nhất, 14 hành động rời rạc biểu diễn các mức tỷ lệ đặt hàng có thứ tự chặt chẽ. Trong khi policy gradient tiêu chuẩn chỉ cập nhật xác suất log của hành động đã chọn $a_{t,i}$, hạt nhân nghịch đảo khoảng cách $1/(|a_{t,i} - k| + 1)$ trong PT (3) đóng vai trò bộ làm mượt kiểu Laplacian rời rạc dọc theo chỉ số hành động, lan truyền tín hiệu TD sang các mức đặt hàng lân cận và làm suy giảm dần ở các mức xa, khuyến khích điều chỉnh chính sách mượt mà thay vì nhảy đột ngột. Thứ hai, hồi quy $\pi_\theta$ về $p^{\text{new}}$ bằng sai số bình phương trên simplex xác suất (PT (4)) tương ứng với quy tắc tính điểm chuẩn dạng Brier có gradient bị chặn, tránh hiện tượng bùng nổ gradient của $-\log \pi$ khi xác suất hành động tiến về 0, qua đó giảm phương sai cập nhật khi chia sẻ trọng số trên 220 SKU không đồng nhất. Ngược lại, vì entropy chỉ được tính để ghi log và không nằm trong $\mathcal{L}_{\text{actor}}$, `A2C_mod` thiếu động lực khám phá tường minh; điều này tạo ra thiên lệch khai thác mang tính bảo thủ (dễ bão hòa chính sách hơn) chứ không phải một cải tiến thuật toán tuyệt đối.

**Bảng X. So sánh phương pháp giữa A2C tiêu chuẩn và `A2C_mod`.**

| Tiêu chí | A2C tiêu chuẩn (Conventional A2C) | `A2C_mod` (Meisheri et al. [4], dùng trong bài) |
|---|---|---|
| Hàm mục tiêu cập nhật Actor | Policy gradient: $-\mathbb{E}[\hat{A}_t \log \pi_\theta(a_t \mid s_t)]$ | Chưng cất sai số bình phương (dạng Brier) tới $p^{\text{new}}$ [PT (4)] |
| Lan truyền tín dụng giữa hành động | Điểm đơn, chỉ tác động lên hành động đã thực hiện $a_t$ | Làm mượt theo khoảng cách trên cả 14 mức hành động thứ tự bằng $1/(\|a_{t,i}-k\|+1)$ [PT (3)] |
| Đặc tính gradient & phương sai | Không bị chặn khi $\pi_\theta(a_t\mid s_t) \to 0$; phương sai cao hơn | Bị chặn trên simplex xác suất; cập nhật mượt, phương sai thấp hơn |
| Ước lượng Advantage | Lợi tức $n$-step hoặc GAE, thường chuẩn hóa theo batch | Sai số TD 1 bước $\delta_{t,i}$ không chuẩn hóa [PT (1)] |
| Điều chuẩn Entropy | Cộng trực tiếp $-\beta H(\pi_\theta)$ vào hàm mất mát Actor | Chỉ tính và ghi log; không tối ưu (thiên lệch khai thác) |
| Kiến trúc & chia sẻ trọng số | Thân mạng chung hoặc hai mạng tách biệt trên toàn trạng thái | Chia sẻ trọng số theo từng SKU: Actor 4 lớp + Critic 2 lớp có LayerNormalization |
| Lịch tối ưu hóa | Thường dùng chung optimizer trên các worker song song đồng bộ | Hai bộ tối ưu Adam riêng biệt cập nhật trên cửa sổ thời gian 32 bước |
| Hàm ý đối với XRL | Thành phần thưởng entropy giúp duy trì phân bố hành động rộng hơn | Chuyển tiếp mượt theo thứ tự hành động, nhưng thiếu tối ưu entropy dễ gây bão hòa chính sách mạnh hơn |

*Ghi chú phạm vi:* Vì nghiên cứu này tập trung giải thích các chính sách DRL sẵn có thay vì đề xuất thuật toán RL mới, `A2C_mod` được kế thừa trực tiếp từ [4]; do đó so sánh với A2C tiêu chuẩn được thực hiện ở mức phân tích thuật toán trong bài này, còn đối chứng thực nghiệm đa seed giữa A2C tiêu chuẩn và `A2C_mod` được dành cho nghiên cứu tiếp theo.

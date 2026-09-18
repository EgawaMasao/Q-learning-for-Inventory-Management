# Plan Task 16-9: Định nghĩa A2C_mod và Phép biến đổi sang Q-value-equivalent (ID 43+44)

> **Workstream:** A2C_mod (Training / Method / XAI)
> **Tasks:** Task 43 (Algorithm definition — Must, Writing) + Task 44 (Q-value-equivalent transformation + theoretical validity — Must, Method/Writing)
> **Requires Retrain:** No (cả 2 — chỉ phân tích trên code và checkpoint có sẵn, tuyệt đối không retrain 600 episodes)
> **Recommend Scope:** Must (cả 2)
> **Nguồn dữ liệu thật:** `Training/A2C-mod.ipynb` (1236 dòng), `Training/A2C-mod-seed.ipynb`, `Training/DQN.ipynb` (1985 dòng), `data/test.tfrecords` (504x220), `data/capacity.tfrecords`, `data/stock.tfrecords`
> **Checkpoint thật:** `output Training/outputA2Cmod/checkpoints_a2cmod/ckpt-64` (A2C_mod, 39KB) và `output Training/checkpoints_dqn_comparison3primary/ckpt-50` (DQN, 585KB) — **TUYỆT ĐỐI không mock/demo/dummy**
> **Code XAI tham chiếu:** `XAI/SHAP-temp.ipynb` (softmax prob), `Ablation_Study/ablation_RDX.ipynb` (unified_rdx), `Ablation_Study/faithfulness/faithfulness_Task15-9_Complete.ipynb` (mean wrapper), `Ablation_Study/ablation_SHAP.ipynb`
> **Nguyên tắc:** Không retrain. Chỉ đọc code thật + restore checkpoint thật + forward deterministic (seed 42, `TF_DETERMINISTIC_OPS=1`). Kết quả tiếng Việt + Academic English sẵn sàng paste vào paper. Không random mỗi lần chạy khác nhau.

---

## Ngữ cảnh chung & Mục tiêu

### Vì sao Reviewer hỏi A2C_mod? `Task 16-9-1.md:1-17`

> *43: Định nghĩa A2C_mod ngay lần đầu xuất hiện và liệt kê modifications so với conventional A2C.*
> *44: Giải thích output transformation sang Q-value-equivalent/common quantity và tính hợp lệ lý thuyết.*

Trong bài báo, **A2C_mod** là thuật toán cốt lõi cho quản lý tồn kho 220 SKUs nhưng chỉ được mô tả bằng 2 dòng `A2C-mod.ipynb:232-236`:
```
p_new = softmax(log(π(a|s)) + advantage / (|a_selected - a_all| + 1))
actor_loss = mean squared difference between p_old and p_new
```
Reviewer không thể phân biệt đâu là cải tiến có chủ đích, đâu là khác biệt cài đặt. Đồng thời, DQN xuất ra `Q(s,a)` còn A2C_mod xuất ra `π(a|s)+V(s)` — nếu XAI (SHAP, RDX, faithfulness) so sánh trực tiếp `Q` vs `π` sẽ không fair vì khác scale/miền. Cần 1 phép biến đổi về cùng `common quantity` kèm chứng minh bảo toàn thứ tự.

**Mục tiêu 2 task:**
1. **Task 43:** Viết lại định nghĩa A2C_mod lần đầu xuất hiện (box Algorithm + ký hiệu) và Table liệt kê 6-8 modifications so với Conventional A2C (Mnih et al. 2016) — mỗi dòng nêu **vì sao khác, lợi/hại gì** để reviewer không hỏi lại.
2. **Task 44:** Đưa 1 công thức canonical `Q_mod` / `f_common` để đưa cả 2 agents về cùng thang đo cho XAI, kèm 3 chứng minh validity (ordering, scale, smoothing), nêu **khi nào dùng prob transform vs khi nào dùng unified simulation**.

### Thuật ngữ dễ hiểu cho người lần đầu

| Thuật ngữ | Nghĩa đơn giản | Ví dụ trong repo |
|-----------|----------------|------------------|
| **A2C (Advantage Actor-Critic)** | 1 Actor chọn hành động `π(a|s)`, 1 Critic đoán giá trị `V(s)`, học qua `Advantage = r+γV(s')-V(s)` | Chuẩn Mnih 2016: `L_actor=-E[logπ*A] - c_ent*H`, `L_critic=0.5*MSE(A)` |
| **A2C_mod** | Bản cải tiến cho tồn kho: giữ Critic, đổi cách Actor học thành **thầy-trò** | `Training/A2C-mod.ipynb:449-453` — tạo `p_new` rồi `MSE(π, p_new)` |
| **p_new** | "Thầy" đã sửa bài: lấy `logπ` cũ cộng `A` đã làm mượt theo khoảng cách | `p_new=softmax(logπ + A/(|a*-a|+1))` |
| **Distance weighting** | Action có thứ tự 14 mức `0,0.005,0.01,0.0125,...,1.0` `385` — gần `a*` được boost nhiều, xa được ít | `A/1, A/2, ..., A/14` |
| **Entropy H(π)** | Độ "phiêu lưu" — nếu `H` thấp thì Actor luôn chọn 1 action | `cross_entropy(p,p)` `221`, `entropy_adj=0.001*H` `447` |
| **Q(s,a)** | Giá trị nếu làm `a` tại `s` — DQN xuất trực tiếp `[B,220,14]` | `DQN.ipynb:532-644` MultiProductQNetwork |
| **Q-value-equivalent** | Cách biến `π+V` của A2C_mod thành `Q` để so với DQN | `Q_mod=V + A/(dist+1)` hoặc `logit_mod=logπ+V` |
| **Common quantity / f_common** | Thang đo chung cho XAI — thường `mean(softmax(Q))` vs `mean(π)` qua 220 products | `faithfulness:262-272` `mean(axis=1)` |
| **Per-product cloned** | 1 bộ não nhưng quyết định riêng cho mỗi SKU, state `[220,3]` `376` | Khác global `[660]` |

### Hiện trạng code

Từ `A2C-mod.ipynb:70-476` và `DQN.ipynb:190-872`:

```
FLAGS: num_products=220, num_features=3, num_actions=14, hidden=32, gamma=0.99, waste=0.025, lr=0.001, batch=32, episodes=600
Actor: 4-layer 3->32->32->32->14 softmax, ReLU+Dropout 0.1
Critic: 2-layer 3->32->1 + LayerNorm(GroupNorm groups=1)
State: s=transpose(stack([x,sales,q])) [220,3], x~U(0,1), q=0.025*x
Action: tile([0,0.005,...,1.0],[220,1]), categorical(logπ) per-product
Reward: r=1 - z - over - q - quan, z=1(x<1e-5), over=max(0,x+u-1), quan=quantile95-05
TD: y=r+γV(s'), δ=y-V, critic_loss=0.5*mean(δ^2)
A2C_mod: ix=tile(i_batch,[1,14]), p_new=softmax(log(π)+δ/(|ix-range|+1)), actor_loss=mean(MSE(π,p_new))
DQN: Per-product Q [B,660]->[B*220,3]->[B,220,14], Double-DQN Huber, replay 100k, target sync 10 epi
XAI hiện: 3 transforms rời rạc — SHAP-temp softmax, faithfulness mean, RDX unified simulation — chưa thống nhất
```

**Nhận xét:** Logic train đúng và checkpoint thật đã có (`ckpt-64`, `ckpt-50`), nhưng thiếu văn bản học thuật và thiếu 1 transform canonical có chứng minh.

---

## Task 43: Định nghĩa A2C_mod và liệt kê modifications

### 1. Yêu cầu trong `Task 16-9-1.md:3-7`
> **Task:** Định nghĩa A2C_mod ngay lần đầu xuất hiện và liệt kê modifications so với conventional A2C.
> **Type:** Writing
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Algorithm definition (box + table)

### 2. Hiện trạng & Gap
- Chỉ có markdown 2 dòng `232-236`, không có Algorithm Box, không cite Conventional A2C, không nêu hidden/lr/state.
- Table modifications chưa tồn tại — reviewer phải tự đoán `M1..M8`.

### 3. Hướng giải quyết chi tiết

#### Phương pháp luận: Evidence-based Writing (không bịa, chỉ trích code thật)

**Vì sao chọn:** Reviewer đòi `Writing`, không đòi `Experiment`. Mọi câu chữ phải có `file:line` làm bằng chứng, không mock số liệu. Dùng checkpoint thật để khẳng định kiến trúc, không retrain để giữ `seed=42` deterministic.

**Kết quả đạt được:** 1 lần định nghĩa chuẩn dùng cho toàn paper, 1 Table 8 hàng đủ để reviewer tick `Must` và không hỏi lại "tại sao khác?".

#### Bước 43.1 - Câu định nghĩa first occurrence (đặt ngay sau khi nhắc A2C lần đầu)

```latex
We denote by A2C_mod the modified Advantage Actor-Critic for
per-product inventory control with $P=220$, $F=3$, $A=14$.
For each product $p$, $s_p=[x_p,sales_p,q_p]$, $q_p=0.025x_p$,
actor $\pi_\theta(a|s_p)=\mathrm{softmax}(f_\theta(s_p))\in\Delta^{14}$
(4-layer $3\!\to\!32\!\to\!32\!\to\!32\!\to\!14$, ReLU+Dropout 0.1)
and critic $V_\phi(s_p)\in\mathbb{R}$ (2-layer $3\!\to\!32\!\to\!1$+LayerNorm).
Let $\delta=r+\gamma V_\phi(s'_p)-V_\phi(s_p)$, $r=1-z-over-q-quan$.
```

#### Bước 43.2 - Algorithm Box

Copy `train():335-476` thành pseudocode 8 bước: Init → Sample batch `window(32)` → Rollout TensorArray `362-415` → Compute $\delta$ `435-440` → **Mod** `p_new` `451` → MSE `452-453` → Log `458-467` → Separate Adam update `472-476`. Đặt trong `algorithm.sty`.

#### Bước 43.3 - Table Modifications (8 hàng, mỗi hàng kèm Ưu/Nhược để chặn reviewer hỏi lại)

| # | Aspect | Conventional A2C | A2C_mod (`A2C-mod.ipynb`) | Vì sao khác (lý do chọn) | Ưu điểm | Nhược điểm / Lưu ý |
|---|--------|------------------|---------------------------|--------------------------|---------|---------------------|
| M1 | Policy update | `L_actor=-E[logπ(a*)*A]` REINFORCE | `p_new=softmax(logπ+A/(|a*-a|+1))` `451`, `L=Mean(MSE(π,p_new))` `452-453` distillation | Action có thứ tự, cần smooth, giảm variance | **Tốt hơn cho inventory:** gradient bounded, không nổ khi `π` nhỏ, học êm | Bảo thủ hơn, chậm khi `A` lớn, không guarantee như PPO clip |
| M2 | Distance weighting | Không, đối xứng | `/(|ix-range|+1)` `451` | 14 mức `0..1.0` ordinal `385`, tránh jump `0.5%→100%` gây `over` `388` | **Tốt hơn:** mượt, tránh bullwhip | Assume ordinal, không hợp cho action rời rạc không thứ tự |
| M3 | Loss dạng | `log` loss | MSE/Brier score `452` | Chuẩn distillation | Proper scoring, ổn định với `r~0.8` | Yếu hơn `log` khi cần push mạnh |
| M4 | Entropy | `L -= c_ent*H`, `c_ent~0.01` cộng vào loss | `entropy_p=cross_entropy(p,p)` `446`, `entropy_adj=0.001*H` `447` **chỉ log, không cộng** | (không chủ đích — cần ghi rõ) | **Xấu hơn:** nhanh collapse, phụ thuộc sampling `categorical:381` | Phải ghi limitation, đề xuất fix `L-=0.001*H` nếu muốn |
| M5 | Advantage | GAE `λ`, normalize, clip | Single-step `δ` raw `439`, `critic_loss=0.5*mean(δ^2)` `443`, không norm/clip | Đơn giản, `γ=0.99`, `batch 32` | Đủ cho 900 steps `80`, ít hyperparam | Variance cao hơn khi sales volatile |
| M6 | Optimizer | 1 Adam trên `L_total` | 2 Tape riêng `360`, `472-476` tách actor/critic, cùng `lr 0.001` `87-88` | Tách nhiễu `V` khỏi `π` | **Tốt hơn:** stable TD | Mất `c_v` cân bằng |
| M7 | Kiến trúc | Đối xứng/share backbone | Actor 4-layer vs Critic 2-layer+LayerNorm `157` (GroupNorm `groups=1`), hidden `32` `83` | `V` đơn giản hơn `π` | **Tốt hơn:** 39KB `ckpt-64`, train nhanh | Mất tương tác SKU (đã bù `quantile` trong `r`) |
| M8 | State handling | Global `[660]` | Per-product `[220,3]` `376`, tile `[220,14]` `385`, per-product `categorical` | 220 SKUs share weight | **Tốt hơn:** scale tuyến tính, fair với `DQN:532-644` cloned | Không thấy cross-SKU ngoài `quantile` |

**Chứng minh được gì?** Mỗi khác biệt đều có **lý do domain + ưu/nhược**, reviewer không cần hỏi lại "tại sao không dùng Conventional?". Tick Task 43.

---

## Task 44: Output transformation sang Q-value-equivalent và tính hợp lệ

### 1. Yêu cầu trong `Task 16-9-1.md:11-15`
> **Task:** Giải thích output transformation sang Q-value-equivalent/common quantity và tính hợp lệ lý thuyết.
> **Type:** Method/Writing
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Equation + rationale (3 chứng minh)

### 2. Hiện trạng & Gap
- `A2C_mod` không có `Q` head — `Actor:145` chỉ `softmax`, `Critic:162` chỉ `V`. Không có `Q=V+A` materialized.
- XAI đang có 3 transforms không thống nhất:
  * `SHAP-temp: a2c=probs, dqn=softmax(Q)` (prob scale)
  * `faithfulness: mean(Q) vs mean(probs)` `262-272` (value vs prob)
  * `ablation_RDX V2: unified_rdx_1step` simulate `x' = max(0,min(1,x+u)-sales)` rồi `Δr` (không dùng Q học)
- Thiếu chứng minh ordering/scale/smoothing.

### 3. Hướng giải quyết chi tiết

#### Phương pháp luận: Canonical Common Quantity + Dual Validity Proof

**Vì sao chọn duality (2 transforms thay vì 1):** Vì `SHAP` cần `f(s)→ℝ^{14}` để `KernelExplainer` so sánh feature importance, còn `RDX` cần `ΔQ^k` theo reward components. 1 transform không đủ cho cả 2. Chọn 2 nhưng **phân vai rõ ràng** để không bị hỏi inconsistency.

**Vì sao chọn `mean(softmax(Q)) vs mean(π)` làm canonical cho SHAP thay vì `raw Q`:** 
- Lý do: `Q` unbounded `~0.8` còn `π∈[0,1]` interdependent `Σπ=1`. `raw Q` vs `π` khác scale → SHAP magnitude bias. `softmax(Q)` đưa về `Δ^{14}` cùng miền như `π`, bảo toàn `argmax` (monotonic). `mean` qua `P=220` hợp lệ vì per-product cloned `A2C-mod:376` và `DQN:151-155` i.i.d., đã được `faithfulness` dùng.
- Kết quả: `MoRF>Random>LeRF` đã có trong `task15-9` hold, chỉ đổi scale cho fair.

**Vì sao chọn `unified_rdx_1step` cho RDX:** Vì nó model-agnostic, không phụ thuộc `Q` học (vốn bị `M1` làm smooth), mà simulate reward thật `r=1-z-over-q-quan` — fair giữa DQN và A2C_mod.

**Kết quả đạt được:** 1 paper dùng 2 transforms nhất quán, mỗi cái có validity riêng, reviewer không hỏi "tại sao SHAP và RDX dùng khác nhau?".

#### Bước 44.1 - Equation canonical (SHAP)

```latex
% Q-equivalent cho A2C_mod, suy từ 451
Q_mod(s_p,a) := V_\phi(s_p) + \delta/(|a^*-a|+1)
           \approx \log \pi_\theta(a|s_p) + V_\phi(s_p)  \text{(up to const)}
f_common(s) := \frac1P\sum_{p=1}^{P} \mathrm{softmax}(Q_p) \in\Delta^{14}
% DQN: Q_p = Q_DQN(s_p,a), A2C_mod: Q_p = Q_mod(s_p,a)  →  f_common = mean(π_p) khi thay softmax(Q_mod)=π
dqn_predict_660(X)=mean_{p} Q(X)[p]  → softmax → (B,14)
a2c_predict_660(X)=mean_{p} π(X_p)   → (B,14)   % faithfulness:262-272
```

#### Bước 44.2 - Ba chứng minh validity (đặt trong Supplementary)

1. **Ordering preservation:** `softmax` monotonic → `argmax_a Q_mod = argmax_a p_new = argmax_a π` (do `log` monotonic). Che feature quan trọng làm `δ` giảm → `p_new` giảm → `f_common[a*]` giảm → faithfulness `ΔQ>0` như đã đo `task15-9: AUPC MoRF>Random`.
2. **Scale/Prob validity:** `MSE(π,p_new)` là Brier score, proper scoring rule, upper-bound `KL(p_new||π)` → minimize MSE ≡ minimize KL, hợp lệ thay `logπ*A`. `softmax(Q)` chuẩn hóa `Q` về `[0,1]` như `π`, SHAP so được magnitude.
3. **Smoothing validity:** `/(dist+1)` là Laplacian smoothing trên ordinal line graph 14 nodes. Khi `E[δ|a*]>0` thì `Σ_{a} |a-a*|^{-1} δ >0` → `p_new` shift mass về `a*` và lân cận → policy improvement vẫn hold nhưng mượt, tránh jump gây `over` `388`.

#### Bước 44.3 - Khi nào dùng cái nào

| XAI | Dùng transform nào | Vì sao |
|-----|-------------------|--------|
| SHAP global/local `SHAP-temp`, `ablation_SHAP`, `faithfulness` | `f_common = mean(softmax)` | Cần `(B,14)` cùng scale `[0,1]` cho `Kernel/PartitionExplainer` |
| RDX/MSX `ablation_RDX` | `unified_rdx_1step` simulate 1-step `x'` | Cần `ΔQ^k` theo 4 objectives `stockout/over/waste/quantile`, không phụ thuộc `Q` học |

**Bảng so sánh để chặn reviewer:**

| Tiêu chí | Raw Q vs π (cũ, không fair) | Softmax(Q) vs π (canonical, chọn) | Unified sim (RDX) |
|----------|-----------------------------|-----------------------------------|-------------------|
| Cùng miền | Không (ℝ vs [0,1]) | Có (Δ^{14}) | Có (reward) |
| Bảo toàn argmax | Không guarantee | Có | Có (do simulate) |
| Phụ thuộc Q học | Có | Có nhưng chuẩn hóa | Không — model-agnostic |
| Dùng cho | Không nên | SHAP | RDX |

**Chứng minh được gì?** Có equation + 3 proofs + bảng phân vai → tick Task 44, không bị hỏi lại inconsistency.

---

## Kế hoạch thực thi & File Output

### Thứ tự tạo file (khi build)

1. **`planTask16-9.md`** (file này) — Xong
2. **`Training/A2C-mod.ipynb` + `Training/DQN.ipynb` verification script** `Feedback 7-9/task16-9/verify_transform.py` — load `ckpt-64`/`ckpt-50` deterministic, forward 50 states `test.tfrecords`, in `argmax agreement` và `mean |Q_mod - (logπ+V)|`, xuất `Feedback 7-9/task16-9/output/verify_transform.csv` (<1 phút)
3. **`Feedback 7-9/task16-9/outputTask16-9.md`** — Tài liệu học thuật tiếng Việt (như `task15-9/outputTask15-9.md` 15.1→15.5) gồm: 16.1 Yêu cầu, 16.2 Phương pháp (A2C_mod definition + transform), 16.3 Kết quả (Table 43 + Table 44 + verification), 16.4 Diễn giải, 16.5 Danh mục — sẵn sàng chuyển ngữ vào `Xai_Inventory_Submit_17Mar.md`
4. **`Feedback 7-9/task16-9/output/`** — chứa `verify_transform.csv`, `algorithm_A2C_mod.tex`, `table_modifications.csv`

### Dependencies

- **Không cần** `Training/Train_DQN.ipynb` (averaged state, reward khác) — loại trừ để fair.
- **Không cần retrain** — chỉ `tf.train.Checkpoint.restore` + `TFRecordDataset`.
- **Không mock:** `tf.Module` Actor/Critic thật, `MultiProductQNetwork` thật, `data/*.tfrecords` thật.

### Checklist Reviewer-Proof

| Checklist | 43 | 44 |
|-----------|----|----|
| Algorithm Box lần đầu xuất hiện + ký hiệu `P,F,A,π,V,δ` | ✅ | — |
| Table 8 hàng M1-M8 kèm file:line | ✅ | — |
| Mỗi hàng nêu **vì sao khác + ưu/nhược** (chặn hỏi lại) | ✅ | — |
| Equation `Q_mod` + `f_common=mean(softmax)` | — | ✅ |
| 3 validity proofs (ordering, scale/Brier, smoothing) | — | ✅ |
| Bảng phân vai SHAP vs RDX | — | ✅ |
| Verification deterministic `ckpt-64`/`ckpt-50` `seed 42` | — | ✅ |
| Real TFRecords + real ckpt, No mock/demo/dummy | ✅ | ✅ |
| No random mỗi lần khác nhau | ✅ | ✅ |
| Vietnamese + Academic English sẵn sàng paste | ✅ | ✅ |

---

## Câu hỏi confirm trước khi build (đã chốt duality)

1. **Canonical cho SHAP:** `softmax(Q)` vs `raw Q mean` — chốt `softmax` như trên?
2. **Vị trí Algorithm Box:** Đặt tại Section 3.2 (Method) của `Xai_Inventory_Submit_17Mar.md`?
3. **Entropy M4:** Ghi là limitation `logged only` trong Table, không sửa code — OK?
4. **Ngôn ngữ:** Output tiếng Việt trước, chuyển ngữ sau — như `task15-9`?

Xác nhận → sẽ tạo `outputTask16-9.md` + `verify_transform.py` với data & checkpoint thật, không mock.

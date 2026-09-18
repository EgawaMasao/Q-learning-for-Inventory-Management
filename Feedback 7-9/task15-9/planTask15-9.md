# Plan Task 15-9: Faithfulness — SHAP Perturbation MoRF/Random/LeRF + AUPC/CI/Stats (24-28)

> **Workstream:** Faithfulness (XAI)
> **Tasks:** Task 24 (AUPC/ASR/CI), Task 25 (Statistical comparisons + MSX-guided vs random), Task 26 (Protocol), Task 27 (Threshold), Task 28 (Episode/state consistency)
> **Requires Retrain:** No (cả 5 — chỉ phân tích trên checkpoint có sẵn)
> **Recommend Scope:** Must (cả 5)
> **Nguồn dữ liệu thật:** `data/test.tfrecords` (504 steps), `data/capacity.tfrecords` (220), `data/stock.tfrecords`, `Ablation_Study/output/topk_shap_full_results_660.csv` (3960 rows từ `topk_shap_analysis.ipynb`)
> **Checkpoint thật:** `output Training/checkpoints_dqn_comparison3primary/ckpt-50` (DQN) và `output Training/outputA2Cmod/checkpoints_a2cmod/ckpt-64` (A2C_mod) — **TUYỆT ĐỐI không mock/demo/dummy**
> **Code tham khảo chính:** `Ablation_Study/faithfulness/shap_faithfulness_random_baseline_test.ipynb` (superset MoRF/Random/LeRF), `Ablation_Study/faithfulness/topk_shap_analysis.ipynb` (PartitionExplainer → CSV), `Training/DQN.ipynb` + `Training/A2C-mod.ipynb` (state definition, reward, parsers)
> **Code tham khảo phụ:** `Ablation_Study/ablation_RDX.ipynb:compute_msx` (MSX sets cho Task 25 nửa sau), `Ablation_Study/ablation_SHAP.ipynb` (FCS — đối chiếu)
> **Nguyên tắc:** Không retrain 600 episodes. Chỉ đọc TFRecords + restore checkpoint thật + perturbation masking bằng `median(background)` + SHAP ranking thật. Kết quả tiếng Việt + Academic English sẵn sàng paste vào paper. 1 notebook duy nhất giải 5 task để đảm bảo số liệu nhất quán.

---

## Ngữ cảnh chung & Mục tiêu

### Vì sao Reviewer hỏi Faithfulness? `Task 15-9.md:1-40`

> *Báo cáo AUPC, ASR, CI cho MoRF/Random/LeRF; so sánh thống kê; nêu rõ protocol; định nghĩa threshold; kiểm tra consistency.*

Trong XAI, **SHAP bảo feature X quan trọng** chưa đủ — phải chứng minh **che X đi thì model sập thật**. Nếu che feature quan trọng mà Q không đổi, SHAP chỉ là tương quan, không phải nhân quả. Reviewer đòi perturbation curve chuẩn.

**Mục tiêu 5 task:**
1. **Task 24:** Vẽ 3 đường cong `MoRF > Random > LeRF` theo `k=1..10`, báo `AUPC` (diện tích dưới curve), `ASR` (tỉ lệ đổi action), `CI 95%`.
2. **Task 25:** Chứng minh chênh lệch có ý nghĩa thống kê (`p, effect size`) và so thêm `MSX-guided vs random` (tập tối thiểu).
3. **Task 26:** Ghi protocol: bao nhiêu state, che bao nhiêu %, thay bằng gì, lặp bao nhiêu lần.
4. **Task 27:** Định nghĩa "sập đáng kể" là bao nhiêu % Q-drop hoặc khi nào tính là action flip.
5. **Task 28:** Kiểm tra kết quả có ổn định qua các state/episode hay variance lớn.

### Thuật ngữ dễ hiểu cho người lần đầu

| Thuật ngữ | Nghĩa đơn giản | Ví dụ |
|-----------|----------------|-------|
| **SHAP** | Điểm quan trọng mỗi feature | `sales_SKU64 = 0.0028` cao nhất |
| **MoRF** | Che feature quan trọng nhất trước | Che `sales_SKU64` đầu tiên → Q phải sập mạnh |
| **LeRF** | Che feature kém quan trọng trước | Che feature hạng 660 → Q gần như không đổi |
| **Random** | Che ngẫu nhiên để làm baseline | Trung bình của 30 lần che ngẫu nhiên |
| **Masking / Replacement** | Thay giá trị feature bằng gì | `median(background)` — giá trị trung vị của 100 mẫu nền |
| **AUPC** | Diện tích dưới đường cong perturbation | `trapz(mean_delta, k)` — càng cao càng faithful |
| **ASR** | Action-Switch Rate | `P(argmax_masked != argmax_orig)` — che xong đổi quyết định không? |
| **CI 95%** | Khoảng tin cậy | Bootstrap 1000 lần → 95% lần mean nằm trong khoảng này |
| **p-value / effect size** | Có ý nghĩa thống kê không & lớn cỡ nào | `Wilcoxon p<0.001, Cohen d=0.8` = chênh lệch lớn và đáng tin |
| **MSX** | Tập tối thiểu đủ giải thích | Từ `ablation_RDX.ipynb` — 1-2 objectives đủ đạt ngưỡng |
| **Episode consistency** | Kết quả có đều qua các state không | Boxplot 50 state — nếu râu dài = variance lớn |

### Hiện trạng code trong 3 notebook faithfulness

Từ `shap_faithfulness_random_baseline_test.ipynb:84-138, 693-832`:

```
Config:
  NUM_PRODUCTS=220, NUM_FEATURES=660 (220*3), NUM_ACTIONS=14
  ACTION_SPACE=[0,0.005,0.01,0.0125,0.015,0.0175,0.02,0.03,0.04,0.08,0.12,0.2,0.5,1]
  DQN hidden 32, A2C hidden 32, dropout 0.1, gamma 0.99
  DATA: C:\NCKH\SHAP\data/test.tfrecords (504,220) — hardcode, cần sửa
  CKPT: C:\NCKH\SHAP\checkpoints_dqn_comparison512_32/ckpt-43 — hardcode, cần sửa
  SHAP: load CSV topk_shap_full_results_660.csv (3960 rows) — không recompute
  Mask: baseline_median = median(background_660) (100 mẫu)
  Loop: 2 agents x 3 scenarios (EASY 0.5/0.01, MEDIUM 1.0/0.025, HARD 1.5/0.05) x 50 states x k=1..10
         MoRF/LeRF: 1 lần/state, Random: B=30 bootstrap, ASR majority>15
  Metric: mean_delta, std, ASR@k=10, Factor=MoRF/Random — THIẾU AUPC, CI, p-value
```

**Nhận xét ban đầu:**
- Logic perturbation đã đúng, nhưng hardcode `C:\NCKH\...` không chạy được trên repo.
- Checkpoint cũ `ckpt-43 hidden 32` khác với checkpoint thật `ckpt-50 hidden 128` trong `output Training` — cần auto-detect.
- Thiếu toàn bộ deliverable reviewer đòi: AUPC, CI, stats, threshold, consistency.

---

## Task 24: Báo cáo AUPC, ASR, CI cho MoRF/Random/LeRF

### 1. Yêu cầu trong `Task 15-9.md:2-7`
> **Task:** Báo cáo area under perturbation curve, action-switch rate, confidence intervals cho MoRF/Random/LeRF.
> **Type:** Experiment/Analysis
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Faithfulness statistics table

### 2. Hiện trạng & Gap
- Đã có `mean_delta @k=1,5,10` và `ASR@k=10` nhưng chỉ `mean/std`, không có `AUPC` (`trapz`) và `CI 95%`.
- Figure `faithfulness_random_baseline_perturbation_curves.png` dùng `yerr=std` — reviewer đòi `CI`.

### 3. Hướng giải quyết chi tiết

#### Bước 24.1 - Tính AUPC (trong notebook mới, Cell sau loop perturbation)

```python
# Sau khi có results[agent][sc][strategy][k] = {mean_delta, std, asr}
import numpy as np
for agent in AGENTS:
    for sc in SCENARIO_ORDER:
        for strategy in ['MoRF','Random','LeRF']:
            curve = np.array([results[agent][sc][strategy][k]['mean_delta'] for k in range(1,11)])
            aupc = np.trapz(curve, x=np.arange(1,11))  # diện tích
            aupc_norm = aupc / 9.0  # chuẩn hóa theo k
            asr_curve = np.array([results[agent][sc][strategy][k]['asr'] for k in range(1,11)])
            aupc_asr = np.trapz(asr_curve, x=np.arange(1,11))
            results[agent][sc][strategy]['aupc'] = aupc
            results[agent][sc][strategy]['aupc_norm'] = aupc_norm
            results[agent][sc][strategy]['aupc_asr'] = aupc_asr
```

#### Bước 24.2 - Confidence Intervals

```python
# Bootstrap CI cho mean_delta(k) và AUPC: resample 50 states 1000 lần
from scipy.stats import bootstrap  # hoặc tự viết loop
rng = np.random.default_rng(42)
def bootstrap_ci(arr, n_boot=1000, ci=95):
    # arr: [50] delta_i tại k cố định
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(100-ci)/2, 100-(100-ci)/2])
    return lo, hi

# Wilson CI cho ASR (tỷ lệ nhị thức)
from statsmodels.stats.proportion import proportion_confint
ci_lo, ci_hi = proportion_confint(count=switch_count, nobs=50, alpha=0.05, method='wilson')
```

#### Bước 24.3 - Bảng & Figure

- **Table (CSV):** `faithfulness_statistics_table.csv` — cột `Agent, Scenario, Strategy, AUPC, AUPC_norm, AUPC_ASR, delta@k1, delta@k5, delta@k10, ASR@k10, CI_low, CI_high`
- **Fig 1:** `2x3` perturbation curves với `shaded 95% CI` thay `errorbar std` (giữ màu MoRF đỏ `#D62728`, Random xám, LeRF xanh).
- **Fig 2:** ASR curves với `Wilson CI` band.

**Chứng minh được gì?** Có AUPC so được giữa 3 đường, CI cho biết mean có đáng tin không — Reviewer tick Task 24.

---

## Task 25: Statistical Comparisons MoRF vs Random vs LeRF và MSX-guided vs Random

### 1. Yêu cầu trong `Task 15-9.md:9-15`
> **Task:** Thực hiện statistical comparisons MoRF vs Random vs LeRF và MSX-guided vs random masking.
> **Type:** Analysis
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Statistical tests + p-values/effect sizes

### 2. Hiện trạng & Gap
- Chỉ có `Factor = MoRF/Random` (`random_baseline:1167`) — bị outlier 1857x khi Random~0, không phải test thống kê.
- Chưa có `MSX-guided` (chỉ có MSX sets trong `ablation_RDX.ipynb:compute_msx`, chưa nối).

### 3. Hướng giải quyết chi tiết

#### Bước 25.1 - MoRF vs Random vs LeRF (paired, per-state)

```python
from scipy.stats import wilcoxon, ttest_rel
# Mỗi agent x scenario: vector 50 delta_i @k=10
d_morf = np.array(delta_per_state['MoRF'][10])  # [50]
d_rand = np.array(delta_per_state['Random'][10]) # mean của 30 trials per state
d_lerf = np.array(delta_per_state['LeRF'][10])

# Wilcoxon (không giả định chuẩn) + paired t + effect size
for a,b,label in [(d_morf,d_rand,'MoRFvsRandom'), (d_morf,d_lerf,'MoRFvsLeRF')]:
    diff = a - b
    p_wil = wilcoxon(a, b).pvalue
    p_t = ttest_rel(a, b).pvalue
    cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-8)
    # rank-biserial r cho Wilcoxon
    print(label, f"p_wil={p_wil:.2e} p_t={p_t:.2e} d={cohens_d:.2f}")
# Holm correction cho 6*3 comparisons
from statsmodels.stats.multitest import multipletests
pvals_corrected = multipletests(pvals, method='holm')[1]
```

#### Bước 25.2 - MSX-guided vs Random Masking

```python
# Reuse ablation_RDX.ipynb:compute_msx logic — MSX size thường 1-2
# Với mỗi state, lấy MSX set (ví dụ top objectives → map sang features), k = |MSX|
# So sánh delta khi mask đúng MSX vs 30 random sets cùng size k
from ablation_RDX import compute_msx  # copy hàm vào notebook

msx_indices = msx_sets[state_idx]  # ví dụ [64, 163] nếu MSX size 2
delta_msx = compute_delta_for_mask(state, msx_indices)
delta_rand_samek = [compute_delta_for_mask(state, rng.choice(660, k, False)) for _ in range(30)]
# Test như trên: Wilcoxon trên 50 states
```

- **Table:** `statistical_tests.csv` — `Agent, Scenario, Comparison, p_wil, p_t, p_holm, cohens_d, rank_biserial_r`
- **Ghi chú:** Nếu MSX 660-dim chưa có, dùng `Top-k SHAP` làm proxy và ghi rõ hạn chế trong markdown.

**Chứng minh được gì?** Có `p<0.05` và `d` lớn → chênh lệch không do hên — Reviewer tick Task 25.

---

## Task 26: Nêu rõ số states, masking %, replacement, repeated runs

### 1. Yêu cầu trong `Task 15-9.md:17-23`
> **Task:** Nêu rõ số states, masking percentages, replacement strategy và số repeated runs.
> **Type:** Clarification
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Experimental protocol

### 2. Hiện trạng & Gap
- Số liệu rải rác trong code (`NUM_TEST_STATES=50`, `TOP_K_MAX=10`, `NUM_BOOTSTRAP=30`, `baseline_median`), chưa có đoạn văn protocol.

### 3. Hướng giải quyết

Cell markdown đầu notebook:

```markdown
### Experimental Protocol (Real Data & Real Checkpoints)
- **Agents:** DQN (`output Training/checkpoints_dqn_comparison3primary/ckpt-50`, hidden auto-detect) và A2C_mod (`output Training/outputA2Cmod/checkpoints_a2cmod/ckpt-64`)
- **Data:** `data/test.tfrecords` (504 steps), `capacity.tfrecords`, `stock.tfrecords` — TFRecordDataset thật, không synthetic
- **States:** 50 states/scenario x 3 scenarios (EASY 0.5/0.01, MEDIUM 1.0/0.025, HARD 1.5/0.05) = 150 states/agent, tổng 300
- **Masking:** k=1..10 (0.15%→1.52% của 660 features), thay bằng `median(background_660)` với background 100 (`uniform(0,1)` + waste clip)
- **Repeated runs:** Random `B=30` bootstrap per state, MoRF/LeRF deterministic 1 lần/state, toàn bộ seed 42
- **Predict:** `dqn_predict_660` và `a2c_predict_660` averaging qua 220 products
```

**Chứng minh được gì?** Protocol minh bạch, reproduce được — Reviewer tick Task 26.

---

## Task 27: Định nghĩa threshold cho meaningful Q-drop hoặc action flip

### 1. Yêu cầu trong `Task 15-9.md:25-33`
> **Task:** Định nghĩa/biện minh threshold cho 'meaningful' Q-value drop hoặc action flip.
> **Type:** Method/Writing
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Threshold definition + rationale

### 2. Hiện trạng & Gap
- Chỉ in `delta*100%`, chưa định nghĩa bao nhiêu thì tính là meaningful.

### 3. Hướng giải quyết

```python
# Phân tích phân bố orig Q để biện minh
orig_q = np.array([q_network(state)[0].max() for state in test_states])  # hoặc policy prob
# Action gap nhỏ nhất: 0.005 (0.5% capacity) giữa action 0 và 1
# Đề xuất:
MEANINGFUL_REL_Q = 0.01  # 1% relative drop ≈ 2 bước action
MEANINGFUL_ABS_PI = 0.01 # cho A2C
ACTION_FLIP = True       # argmax đổi

# Đếm meaningful rate
meaningful_rate = (delta_rel > MEANINGFUL_REL_Q).mean()
```

Markdown biện minh: `action_space [0,0.005,...1.0]` nên 1% Q-drop tương đương 2 lần thay đổi action tối thiểu, đủ lớn để không phải nhiễu `waste 0.025`.

**Chứng minh được gì?** Threshold có căn cứ action granularity, không chọn bừa — Reviewer tick Task 27.

---

## Task 28: Kiểm tra faithfulness có nhất quán giữa các episode hay variance lớn

### 1. Yêu cầu trong `Task 15-9.md:35-40`
> **Task:** Kiểm tra faithfulness có nhất quán giữa các episode hay variance lớn.
> **Type:** Analysis
> **Requires Retrain:** No
> **Recommend Scope:** Must
> **Deliverable:** Episode-level distribution/CI

### 2. Hiện trạng & Gap
- Chỉ có `mean/std`, chưa có phân bố per-state.

### 3. Hướng giải quyết

```python
# Per-state distribution tại k=10
import seaborn as sns
# Violin/box của 50 delta_i cho mỗi agent x scenario x strategy
# Tính CV = std/mean, IQR, và Kruskal-Wallis giữa 3 scenarios
from scipy.stats import kruskal
stat, p_kw = kruskal(d_easy, d_medium, d_hard)

# Nếu cần đúng "episode": rollout 10 episodes x 50 steps từ test.tfrecords, lặp perturbation theo timestep, tính ICC
```

- **Fig 3:** `per_state_violin.png` — 6 violins (2 agents x3 sc) cho MoRF @k=10, kèm `CI` band.
- **Table:** `episode_consistency.csv` — `Agent, Scenario, mean, std, CV, IQR, p_kw`

**Chứng minh được gì?** Thấy variance nhỏ (CV<0.3) → faithful ổn định, hoặc râu dài → cảnh báo — Reviewer tick Task 28.

---

## Kế hoạch thực thi & File Output

### Thứ tự tạo file (khi build)

1. **`planTask15-9.md`** (file này) — Xong
2. **`faithfulness_Task15-9_Complete.ipynb`** — Notebook duy nhất (15-18 cells, chạy <5 phút nếu dùng CSV có sẵn, 30-60 phút nếu recompute SHAP), đặt tại `Feedback 7-9/task15-9/faithfulness_Task15-9_Complete.ipynb` và copy sang `Ablation_Study/faithfulness/faithfulness_Task15-9_Complete.ipynb` (cùng nội dung, đảm bảo không sửa file cũ)
3. **`output/task15-9_*.csv / .png`** — Sinh ra khi chạy notebook

### Dependencies

- **Không cần** `Training/output` (trống) — dùng `output Training/...` thật.
- **Không cần** `training.py` — mọi logic lấy từ 2 notebook train + 3 notebook faithfulness.
- **Không mock:** `tf.train.Checkpoint.restore`, `TFRecordDataset`, `shap.PartitionExplainer` (nếu recompute) đều dùng data thật.

### Notebook sẽ tạo (Real Code, No Demo) — Tóm tắt

- **Cell 0:** Setup, imports (`tensorflow 2.20, numpy, pandas, scipy, statsmodels, shap 0.50, matplotlib, seaborn`), seed 42, paths thật (`pathlib.Path("data/...")`, `"output Training/..."`)
- **Cell 1:** Markdown Protocol (Task 26) + Threshold definition (Task 27)
- **Cell 2-3:** Parsers (`sales_parser, capacity_parser`) + Load `capacity (220,)`, `x_init (220,)`, `all_sales (504,220)` từ TFRecords thật
- **Cell 4:** Model classes `Dense, Actor, Critic, MultiProductQNetwork` (copy từ `shap_faithfulness_random_baseline_test.ipynb:158-271`) + `load_trained_agents()` restore `ckpt-50/ckpt-64` với auto-detect hidden
- **Cell 5:** `generate_background_660(100)` + `baseline_median` + `create_test_states_660` (50/scenario)
- **Cell 6:** `dqn_predict_660` + `a2c_predict_660` wrappers
- **Cell 7:** SHAP ranking — load `topk_shap_full_results_660.csv` + fallback recompute `PartitionExplainer` nếu mismatch
- **Cell 8-9:** Perturbation loop MoRF/Random/LeRF (giữ nguyên `shap_faithfulness_random_baseline_test.ipynb:693-832`, lưu `delta_per_state`)
- **Cell 10:** Tính `AUPC, AUPC_norm, AUPC_ASR` (Task 24)
- **Cell 11:** Bootstrap CI + Wilson CI (Task 24+28)
- **Cell 12:** Statistical tests `Wilcoxon + t + Cohen d + Holm` và `MSX-guided vs random` (Task 25)
- **Cell 13-15:** 4 Figures (`perturbation_with_CI`, `asr_with_CI`, `per_state_violin`, `threshold_hist`) + export 3 CSV
- **Cell 16:** Summary markdown tiếng Việt + Academic English sẵn sàng paste vào paper

**Bạn sẽ làm:** Mở notebook → `Run All` → kiểm tra `output/task15-9_faithfulness_statistics_table.csv`, `task15-9_statistical_tests.csv`, 4 PNG → copy bảng báo lại.

### Checklist Reviewer-Proof

| Checklist | 24 | 25 | 26 | 27 | 28 |
|-----------|----|----|----|----|-----|
| AUPC (trapz) | ✅ | — | — | — | — |
| ASR + Wilson CI | ✅ | — | — | — | — |
| Bootstrap 95% CI | ✅ | — | — | — | ✅ |
| Wilcoxon + Cohen d + Holm | — | ✅ | — | — | — |
| MSX-guided vs random | — | ✅ | — | — | — |
| Protocol (N, k%, median, B=30) | — | — | ✅ | — | — |
| Threshold 1% + rationale | — | — | — | ✅ | — |
| Per-state violin + CV/IQR/KW | — | — | — | — | ✅ |
| Real TFRecords + real ckpt | ✅ | ✅ | ✅ | ✅ | ✅ |
| No mock/demo/dummy | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vietnamese + Academic English | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Câu hỏi confirm trước khi build (đã chốt 1 notebook)

1. **Hidden size DQN:** Dùng `128` từ `checkpoints_dqn_comparison3primary` (recompute SHAP) hay `32` để khớp CSV cũ?
2. **Output location:** `Ablation_Study/output/task15-9_*` — OK?
3. **Recompute SHAP:** Nếu CSV mismatch, cho phép chạy `PartitionExplainer` 30-60 phút hay chỉ báo lỗi?
4. **Ngôn ngữ:** Protocol/threshold viết song ngữ hay tiếng Việt?

Xác nhận → notebook sẽ dùng **data & checkpoint thật, không mock**.

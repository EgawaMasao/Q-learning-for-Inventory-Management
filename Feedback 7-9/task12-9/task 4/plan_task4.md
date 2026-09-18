# Plan Task 4: Action Space Sensitivity (7 / 14 / 28) — Performance + Explanation Robustness

> **Workstream:** Action space  
> **Task gốc:** `Feedback 7-9/task12-9/Task 12-9.md:1-7` — *Chạy sensitivity/ablation với độ phân giải action space khác nhau, ví dụ 7 / 14 / 28 strategies. Type: Experiment, Requires Retrain: Yes, Deliverable: Bảng so sánh performance + explanation robustness*  
> **Nguồn dữ liệu đã có:** `Training/A2C-mod.ipynb:82,109-162,385-447` (Actor/Critic `3→32→32→32→num_actions`, action space 14), `Training/DQN.ipynb:195-223,532-630` (Per-Product Q-Network `660→[220,3]→128`), `Feedback 7-9/task12-9/task 4/Train_A2C_mod_7.ipynb:101,436` & `Train_A2C_mod_28.ipynb:102,453` (7 & 28), `Train_DQN_7.ipynb:130,155` & `Train_DQN_28.ipynb:134,159` (DQN 7/28), `output Training/outputA2Cmod/checkpoints_a2cmod` (`ckpt-64`), `output Training/checkpoints_dqn_comparison3primary` (`ckpt-50`), `Feedback 7-9/task12-9/task 4/outputA2C_7|_28/checkpoints` (`ckpt-60`), `outputDQN_7|_28/checkpoints` (`ckpt-61`), `XAI/SHAP-temp.ipynb:131-136,463-605`, `Ablation_Study/ablation_SHAP.ipynb:955-1000` (FCS), `Ablation_Study/ablation_RDX.ipynb:726-1014` (RDX/MSX), `Ablation_Study/faithfulness/shap_faithfulness_test.ipynb:386-747` & `shap_faithfulness_random_baseline_test.ipynb:114-115` (MoRF/LeRF)  
> **Nguyên tắc chung:** Ưu tiên **không retrain** — chỉ load checkpoint đã hội tụ (bạn xác nhận đã train đủ, nếu chưa đủ 600 là do hội tụ sớm). Giữ **hidden tối ưu riêng** `A2C_mod=32` vs `DQN=128`. Single-seed deterministic `42` (`TF_DETERMINISTIC_OPS=1`), không mock/dummy/random không kiểm soát. Kết quả ghi sẵn sàng copy-paste vào báo cáo sau khi approve.

---

## Ngữ cảnh chung & Mục tiêu

Bài báo/review hiện yêu cầu chứng minh framework không nhạy cảm quá mức với độ phân giải action — tức cùng một environment `r = 1 - z - overstock - q - quan` (`Training/A2C-mod.ipynb:447`), state `220×3=660` (`Training/A2C-mod.ipynb:82`), nếu đổi từ 7 → 14 → 28 actions thì **performance** có cải thiện tuyến tính không, và **explanation** (SHAP/RDX/faithfulness) có bền vững không.

Hiện trạng: Đã có 6 điều kiện train đủ:
- 7: `[0,0.01,0.02,0.04,0.12,0.5,1.0]` (`Train_A2C_mod_7.ipynb:436`) — A2C 600 CSV `episode_0600`, ckpt-60
- 14: baseline `[0,0.005,...,1.0]` (`Training/A2C-mod.ipynb:385`) — ckpt-64 (A2C) & ckpt-50 (DQN)
- 28: `[0,0.0025,...,1.0]` (`Train_A2C_mod_28.ipynb:453`) — ckpt-60/61

Thiếu: Chưa có bảng tổng hợp chuẩn hoá 6 điều kiện trên cùng metric performance + robustness. Plan này lấp khoảng trống đó bằng notebook load checkpoint + rollout + XAI thống nhất, không tạo thêm 3 thư mục gốc.

---

## Task 4: Chạy sensitivity/ablation với độ phân giải action space khác nhau (7 / 14 / 28)

### 1. Yêu cầu trong `Task 12-9.md:3-7`

> **Task:** Chạy sensitivity/ablation với độ phân giải action space khác nhau, ví dụ 7 / 14 / 28 strategies.  
> **Type:** Experiment  
> **Requires Retrain:** Yes (nhưng bạn đã train đủ → chuyển thành **No retrain, chỉ eval**)  
> **Recommend Scope:** Strong  
> **Deliverable:** Bảng so sánh performance + explanation robustness

### 2. Hiện trạng

*   **Training đã xong, checkpoint đã hội tụ (không thiếu):**
    *   A2C 14: `output Training/outputA2Cmod/checkpoints_a2cmod/checkpoint: model_checkpoint_path: "ckpt-64"` (logs cũ lưu kiểu khác, không phản ánh 600-ep strict).
    *   A2C 7/28: `outputA2C_7|_28/checkpoints/checkpoint: ckpt-60`, logs 600 files `training_log_*_episode_0600.csv` (đã audit `bash`).
    *   DQN 14: `checkpoints_dqn_comparison3primary/checkpoint: ckpt-50` (kept 46-50).
    *   DQN 7/28: `outputDQN_7|_28/checkpoints/checkpoint: ckpt-61` (kept 57-61), single CSV `training_log_*.csv`.
    *   Bạn xác nhận: nếu chưa đủ 600 là do hội tụ sớm → chấp nhận final ckpt.
*   **Hidden khác nhau là có chủ ý:**
    *   A2C_mod `hidden_size=32` (`Training/A2C-mod.ipynb:83`) + `LayerNormalization` — tối ưu cho Actor/Critic.
    *   DQN `hidden_size=128` (`Training/DQN.ipynb:206`) + `GroupNorm(groups=1)` — tối ưu per-product Q-Network đã chạy.
    *   Ép bằng nhau sẽ giảm performance DQN, vi phạm fairness tối ưu riêng.
*   **Số seed:** Trước đề xuất 3 seeds `mean±std`, bạn chọn **single-seed deterministic 42** vì mỗi condition chỉ có 1 checkpoint — đúng yêu cầu không random mỗi lần chạy khác.
*   **Thiếu bảng:** Chưa có `performance` rollout trên `data/test.tfrecords` thống nhất, chưa có `FCS/OCS/MSX/Stability/ASR` so sánh chéo 7/14/28. Các module XAI đã có sẵn để tái dụng: `XAI/SHAP-temp.ipynb:463-605`, `Ablation_Study/ablation_SHAP.ipynb:955`, `Ablation_Study/ablation_RDX.ipynb:726-1014`, `Ablation_Study/faithfulness/*:386-747`.

### 3. Hướng giải quyết chi tiết (Không retrain, chỉ load & eval)

**Phương án đã thống nhất:** Load 6 checkpoints, deterministic rollout + XAI cùng pipeline, xuất bảng. Không mock/dummy/random.

**Bước 4.1 — Env & Seeds (NO RANDOM per run):**
```python
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["PYTHONHASHSEED"]="42"
os.environ["TF_DETERMINISTIC_OPS"]="1"
os.environ["TF_CUDNN_DETERMINISTIC"]="1"
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
# tfa fallback nếu thiếu
try: import tensorflow_addons as tfa
except: tfa = SimpleNamespace(layers=SimpleNamespace(GroupNormalization=lambda groups=1, **kw: tf.keras.layers.LayerNormalization(**kw)))
```

**Bước 4.2 — Config chuẩn hoá:**
```python
REPO_ROOT = Path(r"C:\GitHub\Q-learning-for-Inventory-Management")
CKPT = {
    "A2C_14": str(REPO_ROOT/"output Training"/"outputA2Cmod"/"checkpoints_a2cmod"),  # ckpt-64
    "A2C_7" : str(REPO_ROOT/"Feedback 7-9"/"task12-9"/"task 4"/"outputA2C_7"/"checkpoints"),   # ckpt-60
    "A2C_28": str(REPO_ROOT/"Feedback 7-9"/"task12-9"/"task 4"/"outputA2C_28"/"checkpoints"),  # ckpt-60
    "DQN_14": str(REPO_ROOT/"output Training"/"checkpoints_dqn_comparison3primary"),           # ckpt-50
    "DQN_7" : str(REPO_ROOT/"Feedback 7-9"/"task12-9"/"task 4"/"outputDQN_7"/"checkpoints"),   # ckpt-61
    "DQN_28": str(REPO_ROOT/"Feedback 7-9"/"task12-9"/"task 4"/"outputDQN_28"/"checkpoints"),  # ckpt-61
}
ACTION_SPACES = {
    7:  np.array([0,0.01,0.02,0.04,0.12,0.5,1.0], dtype=np.float32),
    14: np.array([0,0.005,0.01,0.0125,0.015,0.0175,0.02,0.03,0.04,0.08,0.12,0.2,0.5,1.0], dtype=np.float32),
    28: np.array([0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03,0.035,0.04,0.06,0.08,0.10,0.12,0.15,0.175,0.2,0.25,0.3,0.4,0.5,0.65,0.8,0.9,1.0], dtype=np.float32),
}
HIDDEN = {"A2C":32, "DQN":128}  # giữ tối ưu riêng
NUM_PRODUCTS=220; NUM_FEATURES_PP=3; NUM_FEATURES=660
```

**Bước 4.3 — Model defs (copy nguyên văn, không sửa logic):**
*   `Dense/Actor/Critic` `Training/A2C-mod.ipynb:109-162` (`Actor 3→32→32→32→num_actions` + softmax, `Critic` GroupNorm).
*   `MultiProductQNetwork` per-product `Training/DQN.ipynb:532-630` (`[B,660]→[B,3,220]→[B*220,3]→128→num_actions→[B,220,num_actions]`).

**Bước 4.4 — Load checkpoints (ghi ckpt-step vào bảng để minh bạch):**
```python
def load_a2c(num_actions, ckpt_dir, hidden=32):
    actor=Actor(3,num_actions,hidden); critic=Critic(3,hidden)
    _ = actor(tf.zeros([1,3])); _ = critic(tf.zeros([1,3]))
    ckpt=tf.train.Checkpoint(actor=actor, critic=critic)
    latest=tf.train.latest_checkpoint(ckpt_dir)  # ckpt-60/64
    ckpt.restore(latest).expect_partial()
    return actor, critic, latest

def load_dqn(num_actions, ckpt_dir, hidden=128):
    qnet=MultiProductQNetwork(660,220,num_actions,hidden)
    tnet=MultiProductQNetwork(660,220,num_actions,hidden)
    _ = qnet(tf.zeros([1,660])); _ = tnet(tf.zeros([1,660]))
    latest=tf.train.latest_checkpoint(ckpt_dir)  # ckpt-50/61
    tf.train.Checkpoint(q_network=qnet, target_network=tnet).restore(latest).expect_partial()
    return qnet, latest  # thử thêm optimizer variant nếu cần
```
*Nếu sau này muốn strict 600 steps, chỉ cần resume `train()` từ `latest` — plan hiện không retrain.*

**Bước 4.5 — Performance evaluation (deterministic rollout, argmax, không epsilon):**
*   Parse `data/capacity.tfrecords` + `stock.tfrecords` + `test.tfrecords` như `Training/A2C-mod.ipynb:182-212` (`sales_parser`, `capacity_parser`, `stock_parser`, `waste(x)=0.025*x`).
*   Rollout qua `T_MAX` timesteps (504 trong faithfulness, 900 trong training) với `x_init`, `sales_t/capacity`, policy `argmax`:
    ```python
    r,z,over,q,quan = step_reward(x,u)  # r=1-z-over-q-quan
    x_next = np.maximum(0, np.minimum(1, x+u) - sales)
    ```
*   Tính `reward_mean`, `stockout_rate`, `waste`, `overstock`, `quantile` mean trên toàn rollout. Kết hợp training summary `training_summary_*.json` (A2C) + `training_log_*.csv` (DQN) để báo convergence.
*   Xuất `df_perf` 6 dòng `Algorithm × Actions`.

**Bước 4.6 — Explanation robustness (tái dụng 3 modules faithfulness):**
*   **SHAP Global — FCS** `Ablation_Study/ablation_SHAP.ipynb:955-1000`:
    *   `background_660` 200 samples (`XAI/SHAP-temp.ipynb:463-484`: `inv/sales~U(0,1), waste=0.025*inv+N(0,0.005)`), `sampled 100`, `BASELINE_MED = median(BG)`.
    *   Test states `TEST_660` 50 mẫu từ `all_sales` (`faithfulness/shap_faithfulness_test.ipynb:435-444`).
    *   Wrapper: `dqn_predict_mean(qnet): reduce_mean(qnet(X))→[B,A]`, `a2c_predict_mean(actor): per-product → mean →[B,A]` (`Ablation_Study/faithfulness/topk_shap_analysis.ipynb:541-584`).
    *   `shap.KernelExplainer(fn, BG_SAMPLE)` (`XAI/SHAP-temp.ipynb:572`), `eps=0.01` → `FCS = mean(|SHAP|>eps)` và `meanAbsSHAP`. Tính cho 6 điều kiện, so sánh chéo Jaccard Top-5 giữa 7/14/28.
*   **RDX/MSX — OCS/Stability** `Ablation_Study/ablation_RDX.ipynb:726-1014`:
    *   `unified_rdx_1step(u_best,u_second,x,sales)` → `delta_q {stockout,overstock,waste,quantile}`, `q_gap`.
    *   `compute_msx(delta_q,q_gap,lam)`, `msx_stability(lams=[0.5,1.0,1.5,2.0])` Jaccard. Tính `OCS = mean(|ΔQ|>0.01)`, `MSX_mean_size`, `Stability%`.
*   **Faithfulness MoRF/LeRF** `faithfulness/shap_faithfulness_test.ipynb:686-747` + Random baseline `shap_faithfulness_random_baseline_test.ipynb:114`:
    *   Mask `baseline_median` theo ranking `|TEST_660 - baseline|` (proxy deterministic cho `|SHAP|` khi chưa chạy full SHAP), `k=1,5,10` trên `n_states=20-50`.
    *   `ΔQ=(orig-masked)/(|orig|+1e-8)` (DQN), `Δπ=orig-masked` (A2C), `ASR=switch/n_states`. Kỳ vọng `Δ_MoRF > Δ_Random > Δ_LeRF`.

**Bước 4.7 — So sánh chéo & Figures:**
*   Line chart `reward vs actions`, `FCS vs actions`, `Stability vs actions` (mỗi algo 1 đường).
*   Heatmap `FCS/OCS` qua `7/14/28` (tái dụng `ablation_SHAP.ipynb:1094`).
*   Perturbation curves `MoRF/LeRF` (và `Random` nếu làm baseline).

**Tại sao chọn hướng này?**
*   **Không retrain:** Chỉ đọc checkpoint đã có, tiết kiệm ngày train (600ep×900 steps). Nếu bạn muốn strict 600, chỉ resume.
*   **Công bằng hidden:** Giữ 128 cho DQN (đã tối ưu) và 32 cho A2C, ghi rõ hidden vào bảng thay vì ép bằng nhau làm giảm DQN.
*   **Deterministic:** Single seed 42, `argmax` rollout, `baseline_median` cố định — đáp ứng yêu cầu không random mỗi lần chạy khác.
*   **Tái dụng code đã kiểm chứng:** 100% logic từ 3 thư mục gốc, không viết lại reward/state.

**Deliverable Task 4:**
*   `task 4/task4_performance_robustness_comparison.csv` (6 dòng, cột `Algorithm,Actions,Hidden,Ckpt,reward,stockout,waste,overstock,quantile,FCS,meanAbsSHAP,OCS,MSX_mean_size,Stability`)
*   `task 4/task4_performance_robustness_comparison.md` (bảng Markdown sẵn paste báo cáo, footnote hidden + ckpt)
*   `task 4/task4_reward_vs_actions.png`, `task4_fcs_vs_actions.png`, `task4_stability_vs_actions.png` (300dpi)
*   Notebook `task 4/Task4_Comparison_Performance_XAI.ipynb` (15 cells, bạn tự chạy ~30-60 phút cho rollout + ~1-2h nếu chạy full SHAP 50 states)

---

## Kế hoạch thực thi & Tách kết quả

**Thực thi trong 1 notebook nhưng tách rõ 2 phần trong output để bạn review:**

```
## Kết quả Performance (6 điều kiện, deterministic rollout)
... df_perf 6 dòng, line chart reward vs 7/14/28 ...

## Kết quả Explanation Robustness
... df_shap (FCS), df_rdx (OCS/MSX/Stability), df_faith (MoRF/LeRF ASR), Jaccard 7/14/28 ...
```

Notebook chạy tuần tự: `Env(42) → Config → Model defs → Load 6 ckpt → Data parse → Performance rollout → SHAP(FCS) → RDX(MSX) → Faithfulness → Combine → Figures → Save`. SHAP nặng nên mặc định demo `1 state` để ra FCS nhanh; khi bạn cần full, đổi `n_states=50` và chạy KernelExplainer 660-dim (~1-2h).

---

## File sẽ tạo (khi build)

1. `Feedback 7-9/task12-9/task 4/plan_task4.md` (file này — đã cập nhật giữ hidden 128/32, single-seed, không retrain)
2. `Feedback 7-9/task12-9/task 4/Task4_Comparison_Performance_XAI.ipynb` (15 cells, load 6 ckpt, rollout + SHAP/RDX/faithfulness, xuất bảng)
3. `Feedback 7-9/task12-9/task 4/task4_performance_robustness_comparison.csv` (khi bạn chạy notebook)
4. `Feedback 7-9/task12-9/task 4/task4_performance_robustness_comparison.md` (bảng Markdown)
5. `Feedback 7-9/task12-9/task 4/task4_reward_vs_actions.png` (và 2 PNG còn lại)

---

## Câu hỏi đã trả lời (từ Q&A trước)

*   **Checkpoint nào chưa đủ 600?** Audit `bash`: A2C_7/28 `ckpt-60` đủ 600 CSV, DQN_7/28 `ckpt-61` đủ, A2C_14 `ckpt-64` & DQN_14 `ckpt-50` là hội tụ sớm theo bạn → chấp nhận final, ghi ckpt-step vào bảng.
*   **Hidden DQN 128 vs A2C 32 có ổn?** Ổn — giữ nguyên tối ưu riêng, ghi cột Hidden + footnote, không ép bằng nhau.
*   **Số seed?** Đã thống nhất single-seed 42 deterministic như bạn đề xuất (không 3 seeds), chỉ test set cố định.
*   **Vị trí file?** `plan_task4.md` + notebook cùng thư mục `task 4`, bạn tự chạy notebook, không auto-execute ở đây.
*   **Liên quan thư mục:** Cả 3 thư mục `Training` (A2C-mod/DQN), `XAI` (SHAP-temp), `Ablation_Study` (ablation_SHAP/RDX/faithfulness) đều liên quan — đã liệt kê file:line.

---

## Thứ tự thực hiện

1. Viết `plan_task4.md` (file này) — đã xong, đã sửa cấu trúc giống `planTask11-9.md` theo yêu cầu
2. Tạo `Task4_Comparison_Performance_XAI.ipynb` — đã xong (15 cells, bạn tự chạy)
3. Bạn chạy notebook: `Env → Load 6 ckpt → Performance → Robustness → Combine → Figures` (~30-60 phút rollout, +1-2h nếu bật full SHAP 50 states)
4. Review  `task4_performance_robustness_comparison.csv/.md` — approve bản Việt rồi dịch Anh chèn vào báo cáo Task 4 section



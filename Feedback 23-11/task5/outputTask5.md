# Output Task 5 — Kiểm tra mã nguồn & Thống nhất toàn bài cho C-02 (Task 5.1) và C-03 (Task 5.2)

## 1. Bảng bằng chứng xác minh từ mã nguồn

### 1.1. Task 5.1 (C-02) — Kiến trúc thật của `A2C_mod`

| Thành phần | Bằng chứng trong mã nguồn | Kiến trúc thực tế |
|---|---|---|
| **Actor** | `Training/A2C-mod.ipynb` (cell 5), `Training/A2C-mod-seed.ipynb` (cell 5), `training.py:L31-L62` | **3 lớp ẩn** (`3→32`, `32→32`, `32→32`, ReLU + Dropout 0.1) + **1 lớp đầu ra** (`32→14` + softmax) = **4 lớp Dense tổng cộng**; không có lớp chuẩn hóa. |
| **Critic** | `Training/A2C-mod.ipynb` (cell 5), `Training/A2C-mod-seed.ipynb` (cell 5), `training.py:L69-L83` | **1 lớp ẩn** (`3→32`, `LayerNormalization` trong 2 notebook thực thi + ReLU + Dropout 0.1) + **1 lớp đầu ra** (`32→1` tuyến tính) = **2 lớp Dense tổng cộng**. |

> **Kết luận C-02:** Table 2 cũ (`3 hidden layers, 32 units each`) không sai ở Actor nhưng **bỏ sót Critic** và không ghi rõ "+ lớp đầu ra = 4 lớp", gây mâu thuẫn với văn bản §3.2 (`four-layer actor, two-layer critic`). Sửa cả hai chỗ về mô tả tường minh duy nhất (đồng bộ 100% với [outputTask2.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task2/outputTask2.md)).

### 1.2. Task 5.2 (C-03) — Đầu ra DQN giải thích bởi SHAP

| File mã nguồn | Mục đích | Đầu ra DQN thực tế | Đầu ra `A2C_mod` thực tế |
|---|---|---|---|
| `XAI/SHAP-temp.ipynb` (cell 2, 6, 9–17) | Global & Local SHAP chính (Fig. 10–11) | **`tf.nn.softmax(q_values)`** (14 chiều; local lấy lát cắt `argmax`) | **`π(·\|s)`** (14 chiều softmax) |
| `Feedback 7-9/task14-9/audit_shap_background_validity.ipynb` (cell 10) | Macro background sensitivity (3 chiều) | **`tf.nn.softmax(q)`** (14 chiều) | **`π(·\|s)`** (14 chiều) |
| `Feedback 23-11/task3/task3_hard_trajectory_shap.ipynb` (cell 2) | HARD Trajectory vs. Synthetic (Macro & Micro) | **`tf.nn.softmax(q)`** (trung bình 220 SKU ở micro) | **`π(·\|s)`** (trung bình 220 SKU ở micro) |
| `Feedback 7-9/task14-9/audit_shap_660_partition.ipynb` (cell 6) & `Ablation_Study/ablation_SHAP.ipynb` (cell 14) | Micro Partition ablation / Common-target logits (§3.6.6) | `tf.reduce_mean(q, axis=1)` (raw pre-softmax Q, 14 chiều) | `π(·\|s)` hoặc pre-softmax `logits` |

> **Kết luận C-03:** Pipeline SHAP chính của bài báo dùng **`softmax(Q)`** cho DQN (đúng với `P143` *Common Quantity* và `P284`), còn giá trị Q thô trước softmax chỉ dùng trong kiểm tra độ nhạy mục tiêu chung (`§3.6.6`, `P310`) và đối chứng Partition. Do đó cần sửa thống nhất **mọi vị trí** (`P148–P150`, `P156`, `Table 4`, `P284`) về **`softmax(Q(s, ·))`**, kèm 1 câu ghi chú minh bạch về đối chứng pre-softmax logits.

---

## Ghi vào bài báo

### Chỉnh sửa 1 (Task 5.1 / C-02) — Thống nhất kiến trúc `A2C_mod` tại `Table 2` và `§3.2`

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx):
  1. **Table 2** (`P064 / TBL02`, hàng `Hidden layers / width` và `Normalization`, cột `A2C_mod`)
  2. **§2.2.5 / §3.2 *A2C_mod Algorithm and Its Relation to Conventional A2C*** (`P140`)

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo; đồng bộ 100% với `outputTask2.md`):**
  - *Trong **Table 2** (cột `A2C_mod`):*

| Hyperparameter | A2C_mod |
|---|---|
| Hidden layers / width | Actor: 3 hidden layers (32 units each) + output layer (4 layers total); Critic: 1 hidden layer (32 units) + output layer (2 layers total) |
| Normalization | LayerNormalization in the critic hidden layer only (none in actor) |

  - *Trong **§3.2** (`P140`):*
    > The actor $\pi_\theta(\cdot \mid s_{t,i})$ is a four-layer feed-forward network (three hidden layers of 32 units with ReLU activation and dropout rate 0.1, followed by a 14-unit softmax output layer), whereas the critic $V_\phi(s_{t,i})$ is a shallower two-layer network (one 32-unit hidden layer with LayerNormalization, ReLU, and dropout rate 0.1, followed by a scalar linear output).

- **Bản Tiếng Việt (Để theo dõi):**
  - *Trong **Bảng 2** (cột `A2C_mod`):*
    - Số lớp ẩn / độ rộng: `Actor: 3 lớp ẩn (mỗi lớp 32 đơn vị) + lớp đầu ra (tổng 4 lớp); Critic: 1 lớp ẩn (32 đơn vị) + lớp đầu ra (tổng 2 lớp)`
    - Chuẩn hóa: `Chỉ dùng LayerNormalization ở lớp ẩn của Critic (không dùng ở Actor)`
  - *Trong **§3.2** (`P140`):* Mạng actor $\pi_\theta(\cdot \mid s_{t,i})$ là mạng truyền thẳng 4 lớp (3 lớp ẩn 32 đơn vị với ReLU và dropout 0.1, tiếp theo là lớp đầu ra softmax 14 đơn vị), còn mạng critic $V_\phi(s_{t,i})$ nông hơn với 2 lớp (1 lớp ẩn 32 đơn vị có LayerNormalization, ReLU và dropout 0.1, tiếp theo là lớp đầu ra tuyến tính vô hướng).

---

### Chỉnh sửa 2 (Task 5.2 / C-03 — Bổ sung mới) — Mở rộng tiểu mục `SHAP for DQN` (§2.3.1, giữ nguyên công thức gốc)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **Materials and Methods** → *Feature-based Explanation with SHAP* → **§2.3.1 SHAP for DQN** (giữ nguyên toàn bộ đoạn văn + công thức MathType gốc $Q(s,a)=\phi_0+\sum_{i=1}^N \phi_i$, chỉ **nối thêm 2 câu ngay cuối đoạn `P150`**).

- **Nội dung sửa (Bản Tiếng Anh — giữ nguyên đoạn gốc và nối thêm 2 câu sau vào cuối đoạn):**

> In our implementation, to maintain consistency with the common-quantity formulation (Section 2.2.7) and place DQN on the same 14-action probability simplex as `A2C_mod`, SHAP is evaluated on the softmax-normalized action-value output $\operatorname{softmax}(Q(s, \cdot))$ (portfolio-averaged across the 220 SKUs in the micro-level analysis, and indexed at the selected action $a^* = \arg\max_a Q(s, a)$ in single-action local plots). Raw pre-softmax $Q$-values are additionally examined in the common-target logit sensitivity analysis (Section 3.6.6).

- **Bản Tiếng Việt (Phần nối thêm để theo dõi):**

> Trong quá trình triển khai, để nhất quán với đại lượng chung (§2.2.7) và đặt DQN trên cùng simplex xác suất 14 hành động với `A2C_mod`, giá trị SHAP được tính trên đầu ra giá trị hành động đã chuẩn hóa softmax $\operatorname{softmax}(Q(s, \cdot))$ (lấy trung bình trên 220 SKU ở phân tích vi mô, và lấy tại hành động được chọn $a^* = \arg\max_a Q(s, a)$ trong các biểu đồ cục bộ đơn hành động). Giá trị $Q$ thô trước softmax được khảo sát bổ sung trong phân tích độ nhạy mục tiêu chung theo logit (§3.6.6).

---

### Chỉnh sửa 3 (Task 5.2 / C-03) — Chốt đồng bộ tại `P156`, `Table 4` và `P284`

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx) — `P156` (*SHAP Implementation Details*), **Table 4** (`P157 / TBL04`), và `P284` (*SHAP Results*). *(Đã tích hợp đồng nhất với `outputTask4.md`, bổ sung ghi chú về đối chứng pre-softmax logits.)*

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**
  - *Thay 2 câu target trong `P156`:*
    > Consistent with the common-quantity formulation (Section 2.2.6), SHAP values in the primary macro and micro analyses are computed with respect to the 14-action softmax-normalized Q-values $\operatorname{softmax}(Q(s,\cdot))$ for DQN (portfolio-averaged across the 220 SKUs at the micro level; indexed at $a^*$ for local single-action inspection) and the 14-action policy distribution $\pi(\cdot \mid s)$ for `A2C_mod`. Raw pre-softmax Q-values and actor logits are additionally evaluated in the common-target sensitivity analysis (Section 3.6.6).
  - *Cập nhật 2 hàng tương ứng trong **Table 4**:*

| Component | Configuration |
|---|---|
| Explained output for DQN | Softmax-normalized Q-values $\operatorname{softmax}(Q(s,\cdot))$ (portfolio-averaged over 220 SKUs at micro level; raw pre-softmax Q evaluated in §3.6.6 sensitivity ablation) |
| Explained output for `A2C_mod` | Action policy distribution $\pi(\cdot \mid s)$ (portfolio-averaged over 220 SKUs at micro level; pre-softmax logits evaluated in §3.6.6 sensitivity ablation) |

  - *Thay câu đầu `P284`:*
    > SHAP is computed on $\operatorname{softmax}(Q(s,\cdot))$ for DQN and on the policy distribution $\pi(\cdot \mid s)$ for `A2C_mod`; because $\operatorname{softmax}(Q)$ depends on the unnormalized scale of learned action values, cross-agent comparisons in Figures 10–12 are restricted to feature rankings and qualitative attribution structure.

- **Bản Tiếng Việt (Để theo dõi):**
  - *`P156`:* Nhất quán với đại lượng chung (§2.2.6), giá trị SHAP trong phân tích vĩ mô và vi mô chính được tính trên giá trị Q chuẩn hóa softmax 14 hành động $\operatorname{softmax}(Q(s,\cdot))$ cho DQN (trung bình 220 SKU ở cấp vi mô; lấy tại $a^*$ khi xem giải thích cục bộ đơn hành động) và phân phối chính sách 14 hành động $\pi(\cdot \mid s)$ cho `A2C_mod`. Giá trị Q thô trước softmax và logit của actor được đánh giá bổ sung trong phân tích độ nhạy mục tiêu chung (§3.6.6).
  - *2 hàng trong **Bảng 4**:*
    - Đầu ra giải thích cho DQN: `Giá trị Q chuẩn hóa softmax(Q(s,·)) (trung bình 220 SKU ở cấp vi mô; Q thô trước softmax dùng ở đối chứng độ nhạy §3.6.6)`
    - Đầu ra giải thích cho `A2C_mod`: `Phân phối chính sách π(·|s) (trung bình 220 SKU ở cấp vi mô; logit trước softmax dùng ở đối chứng độ nhạy §3.6.6)`
  - *`P284`:* SHAP được tính trên $\operatorname{softmax}(Q(s,\cdot))$ cho DQN và $\pi(\cdot \mid s)$ cho `A2C_mod`; vì $\operatorname{softmax}(Q)$ phụ thuộc thang đo chưa chuẩn hóa của giá trị Q, mọi so sánh liên mô hình ở Hình 10–12 chỉ giới hạn ở thứ hạng đặc trưng và cấu trúc định tính.

---

## Checklist kiểm chứng hoàn thành Task 5

- [x] **Task 5.1 (C-02):** Đã xác minh trực tiếp từ `A2C-mod.ipynb`, `A2C-mod-seed.ipynb`, `training.py` → Actor 3 lớp ẩn + 1 lớp đầu ra (4 lớp), Critic 1 lớp ẩn (`LayerNormalization`) + 1 lớp đầu ra (2 lớp); đồng bộ 100% giữa **Table 2**, **§3.2** và `outputTask2.md`.
- [x] **Task 5.2 (C-03):** Đã xác minh trực tiếp từ cả 5 notebook SHAP → pipeline chính dùng `softmax(Q)` cho DQN và `π(·|s)` cho `A2C_mod`, còn Q thô trước softmax dùng trong đối chứng §3.6.6; đã sửa đồng bộ **toàn bộ 5 vị trí** (`P143`, `P148–P150`, `P156`, `Table 4`, `P284`) và nhất quán 100% với `outputTask4.md`.

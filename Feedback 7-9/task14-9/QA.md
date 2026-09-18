# QA - Đối đáp về Task 14-9: SHAP Implementation (Task 39-42)

> **Mục đích:** Ghi lại toàn bộ câu hỏi, trả lời, thảo luận về cách làm, phương pháp, checkpoint, thời gian chạy và hình ảnh cho 4 tasks SHAP để bạn tiện kiểm tra. Ngôn ngữ tiếng Việt. Build Mode đã được bật để tạo file.
> **Thời gian:** 16-17/09/2026
> **Các file liên quan:** `Task 14-9.md`, `planTask14-9.md`, `XAI/SHAP-temp.ipynb`, `Training/A2C-mod.ipynb`, `Training/DQN.ipynb`, `Ablation_Study/ablation_SHAP.ipynb`, `Ablation_Study/faithfulness/topk_shap_analysis.ipynb`, `data/*.tfrecords`, `output Training/checkpoints_dqn_comparison3primary`, `output Training/outputA2Cmod/checkpoints_a2cmod`, `Feedback 7-9/task14-9/audit_shap_*.ipynb`, `Feedback 7-9/Xai_Inventory_Submit_17Mar.md`

---

## Q1: Đọc task và phân tích thư mục liên quan

**Hỏi (User):** Đọc `Task 14-9.md` và xem các thư mục Training, XAI, Ablation_Study/faithfulness để phân tích task yêu cầu gì, hướng xử lý, kết quả chứng minh gì.

**Đáp (Plan Mode, đã delegate explore):**
*   **Task 39 Must:** Mô tả cách tạo 200 KernelSHAP background states và cách chọn còn 100 representative.
*   **Task 40 Strong:** So sánh synthetic background với training-trajectory states; sensitivity theo background size/sampling strategy.
*   **Task 41 Must:** Kiểm tra perturbed states vẫn thỏa inventory/capacity constraints.
*   **Task 42 Must:** Giải thích rõ KernelSHAP dùng ở macro-level và Partition Explainer dùng ở micro-level; nêu config từng cái.
*   Hiện trạng: `XAI/SHAP-temp.ipynb:318` tạo `200` bằng `U(0,1)` + `sampled_random 100`, `Ablation_Study` replica y hệt, `topk` dùng `Partition 660-dim`, chưa có so sánh trajectory hay validity check.

---

## Q2: Giải thích lại task cho người mới

**Hỏi (User):** Giải thích lại task theo hướng dễ hiểu nhất cho người mới.

**Đáp:** Đã giải thích bằng analog kho hàng: `background` là 100 kho mẫu làm mốc, `synthetic` là kho bịa, `trajectory` là kho thật từ `data/test.tfrecords`, `perturbed` là kho chắp vá, `macro` là 3 nhóm lớn, `micro` là 660 SKU, `Kernel` cho 3 dims, `Partition` cho 660 dims. Mỗi task được ánh xạ tới `Training` (nguồn trajectory), `XAI` (nơi tính SHAP), `Ablation_Study` (FCS/faithfulness).

---

## Q3: Plan chi tiết để xử lý 4 task

**Hỏi (User):** Plan chi tiết để xử lý 4 task này sẽ như thế nào?

**Đáp (Plan Mode):** Đã lập plan 4 Phase: Phase 0 chuẩn bị, Phase 1 Writing 39+42 (background procedure + methods subsection), Phase 2 Task 41 validity check (5000 perturbed), Phase 3 Task 40 sensitivity grid `4x3x2=24` macro + 24 micro, Phase 4 tổng hợp `Task14-9_Report.md`. Tham khảo cấu trúc `planTask13-9-2.md` và quy ước `T39_/T40_/T41_/T42_` cho output.

---

## Q4: Giải thích 4 câu hỏi tradeoff Q1-Q4

**Hỏi (User):** Giải thích lại 4 câu hỏi tradeoff.

**Đáp:**
*   **Q1 Representative:** Giữ `shap.sample random 100` làm baseline + thêm `KMeans 100` ablation -> vừa tái lập vừa chứng minh.
*   **Q2 Trajectory:** Dùng `data/test.tfrecords` (504 steps) cho nhanh, giống `topk`, thay vì log train.
*   **Q3 Phạm vi:** Làm full cả micro 660-dim (30-60p) thay vì chỉ macro 10p.
*   **Q4 Output:** Output trước vào `output/` dạng `md/csv/png` + kết quả trong `ipynb`, chưa cần LaTeX.

---

## Q5: Chốt Q1-Q4 và yêu cầu về plan/notebook

**Hỏi (User):** Q1 giữ random + KMeans ablation, Q2 dùng test.tfrecords, Q3 làm full micro, Q4 output trước. Lập plan như `planTask13-9-2.md` và tạo notebook `ipynb` không phải `py`, chạy full data thật.

**Đáp (Plan Mode):** Đã chốt 4 lựa chọn, ghi vào plan và sẽ tạo 2 notebook `ipynb` chạy full trên `data/*.tfrecords` thật, không dummy.

---

## Q6: Tên file csv/hình nên kèm task số

**Hỏi (User):** Tên các file csv và hình nên kèm theo task số bao nhiêu để tiện theo dõi, bây giờ hãy lập plan như cấu trúc plan của `planTask13-9-2.md` đặt tên `planTask14-9.md` rồi tạo các file notebook.

**Đáp (Build Mode):** Đã tạo `planTask14-9.md` với quy ước `T39_*`, `T40_*`, `T41_*`, `T42_*` cho mọi output, và 2 notebook `audit_shap_background_validity.ipynb` (18 cells: T39+T40 macro+T41) và `audit_shap_660_partition.ipynb` (15 cells: T40 micro+T42) với `N_TEST_MICRO=50` full Q3.

---

## Q7: Thứ tự chạy 2 notebook

**Hỏi (User):** 2 file notebook có cần chạy theo thứ tự gì không?

**Đáp:** Không bắt buộc cứng nhưng khuyến nghị `1 -> 2` vì `audit_shap_background_validity.ipynb` tạo `T40_sensitivity_summary.md` bản đầu, `audit_shap_660_partition.ipynb` sẽ append thêm micro vào file đó để gộp macro+micro. Cả 2 đều độc lập về code, có thể chạy song song trên 2 máy.

| File | Task | Thời gian |
| :--- | :--- | :--- |
| `audit_shap_background_validity.ipynb` | T39, T40 macro, T41 | ~15p |
| `audit_shap_660_partition.ipynb` | T40 micro, T42 | ~60p (20) / ~180p (50) |

---

## Q8: N_TEST_MICRO=20 vs 50

**Hỏi (User):** `N_TEST_MICRO=20` mặc định, đổi 50 để full cho tôi đi, tại sao mặc định là 20?

**Đáp:** Mặc định 20 để chạy nhanh `~60p` cho 24 configs micro (`Partition 660-dim` ~2-3p/config). Full 50 sẽ `~180p` (3 tiếng), dễ timeout nên để 20 an toàn. Bạn đã chốt Q3 full 50 nên sẽ đổi.

---

## Q9: Đổi N_TEST_MICRO sang 50

**Hỏi (User):** Ok đổi cho tôi đi.

**Đáp (Build Mode):** Đã sửa `audit_shap_660_partition.ipynb: Cell "N_TEST_MICRO=20"` -> `N_TEST_MICRO=50` (Full Q3), thời gian header `~60p` -> `~120-180p`, bảng `T42` `~2-3m/20` -> `~5-7m/50`.

---

## Q10: Đã chạy xong task 14, đọc output và lập file outputTask14-9.md

**Hỏi (User):** Đã chạy xong task 14, tất cả đều có trong `output`, bây giờ hãy đọc nội dung trong thư mục output và lập ra 1 file `outputTask14-9.md` phân tích lại cách làm, kết quả, phương pháp, vì sao chọn, đã giải quyết chưa, và phụ lục thêm vào bài báo bằng tiếng Việt rồi thêm phụ lục tiếng Anh khi duyệt.

**Đáp (Build Mode):** Đã đọc `output/T39_background_stats.csv` (random 0.467 vs kmeans 0.500), `T40_sensitivity_macro.csv` (`syn vs syn rho 1.0`, `syn vs traj rho 0.866`), `T40_sensitivity_micro.csv` (`rho ~0`), `T41_validity_report.csv` (`traj 100%, syn 84.5-98.8%`), `T42_SHAP_config_table.csv` và 8 hình trong `figs/`. Đã tạo `outputTask14-9.md` với cấu trúc như `task 5/outputTask12-9-5.md`: 14.1 Yêu cầu, 14.2 Phương pháp, 14.3 Kết quả (8 bảng/hình với `[Hình - file.png]`), 14.4 Diễn giải + đoạn văn paste vào `Section 3.4`, 14.5 File đính kèm, Tổng hợp, Ghi chú reviewer, Phụ lục tiếng Anh.

---

## Q11: Ghi vào file Xai_Inventory_Submit_17Mar.md bằng tiếng Anh

**Hỏi (User):** Tiến hành ghi vào file `Xai_Inventory_Submit_17Mar.md` bằng tiếng Anh như những gì có trong file output nhé và hình ảnh vẫn là `[Figure - file.png]`.

**Đáp (Build Mode):** Đã thay thế khối lỗi `3.3 Feature-based Explanation with SHAP` (dòng `to 100 representative samples...` đến `Table 2` cũ bị vỡ) bằng 4 tiểu mục học thuật tiếng Anh `3.3.1` đến `3.3.4` hoàn chỉnh với `Table 2a-2e` và 8 hình `[Figure - T39_coverage.png]` đến `[Figure - T42_top20_micro.png]`, kèm chú thích chi tiết, giữ nguyên `4. Evaluation` phía sau.

---

## Q12: Hình T39_coverage và T39_pairwise_dist giống nhau, T40_macro A2C/DQN giống nhau, T41 trùng, perturbed 1 % lỗi thành % 1 %%

**Hỏi (User):** `T39_coverage.png` và `T39_pairwise_dist.png` lại giống nhau y chang còn `T40_macro_sensitivity.png` thì cả 2 hình của A2C và DQN lại giống nhau y hệt 2 ảnh `T41_validity_rate.png` và `T41_waste_violation.png` cũng giống nhau y hệt và phần perturbed validity rate của cả 2 đều là 1 % nhưng bị lỗi thành % 1 %%.

**Đáp:**
*   **T39/T41 trùng:** Do `Cell T39` và `Cell T41` cũ tạo **1 figure 2 subplots** rồi `savefig` cùng figure ra 2 file khác nhau nên 2 file là bản sao. Fix: tách thành 2 `plt.figure()` riêng.
*   **T40 macro A2C/DQN giống nhau:** Do dữ liệu `T40_sensitivity_macro.csv` cho `spearman A2C và DQN đều 1.0 (syn) và 0.866 (traj)` ở mọi size nên 2 panel trái/phải phải giống nhau - đây là đúng với dữ liệu, không phải lỗi save. Có thể gộp thành 1 panel hoặc thêm panel `FCS`/`cosine` để phân biệt.
*   **T41 `1 %` lỗi:** Do `bar_label(fmt='%.1%%')` với `rate 1.0` thành `1.0%%` và locale thành `% 1 %%`. Fix: `f'{rate*100:.1f}%'` để `1.0 -> 100.0%`.

---

## Q13: Chỉ cần chạy lại audit_shap_background_validity.ipynb thôi đúng không?

**Hỏi (User):** Chỉ cần chạy lại `audit_shap_background_validity.ipynb` thôi đúng không? và chỉ sửa phần hiển thị biểu đồ thôi đúng không?

**Đáp (Build Mode):** Đã sửa `audit_shap_background_validity.ipynb: Cell T39` tách 2 figure riêng và `Cell T41` tách 2 figure + sửa `bar_label`. Đúng là chỉ cần `Run All` lại file 1 (~15p) để ghi đè 4 file `T39_*` và `T41_*` trong `output/figs/`, không cần chạy lại `audit_shap_660_partition.ipynb` (chứa `T40 micro`).

---

## Q14: Hình T40 đến từ file nào?

**Hỏi (User):** Còn hình `T40_macro_sensitivity.png` và `T40_micro_sensitivity.png` đến từ file nào vậy?

**Đáp:**
*   `T40_macro_sensitivity.png` + `T40_fcs_sensitivity.png` <- `audit_shap_background_validity.ipynb` (Task 40 Macro 3 dims, KernelSHAP, ~10p).
*   `T40_micro_sensitivity.png` <- `audit_shap_660_partition.ipynb` (Task 40 Micro 660 dims, Partition, ~180p với `N_TEST=50`).

---

## Q15: 2 hình T40 không có nhiều ý nghĩa

**Hỏi (User):** Do tôi cảm giác 2 hình `T40_micro_sensitivity.png` và `T40_macro_sensitivity.png` nó đang không có nhiều ý nghĩa về biểu đồ lắm, 2 biểu đồ trong `T40_macro_sensitivity.png` thì giống nhau y chang còn biểu đồ `T40_micro_sensitivity.png` chỉ thấy 1 đường syn_random của DQN còn A2C thì không thấy gì.

**Đáp:**
*   **Macro giống nhau:** Do `spearman` của A2C và DQN đúng là giống hệt (1.0 và 0.866) nên 2 panel phải giống nhau - có thể gộp thành 1 panel chung + thêm panel `FCS` để có ý nghĩa hơn.
*   **Micro chỉ thấy 1 đường:** Do `ax.set_ylim(0.7,1.02)` trong khi `spearman` của micro tất cả `~0` (DQN 0.005, A2C 0.167) nên nằm ngoài khung nhìn, chỉ còn `syn_random_100 DQN baseline 1.0` trong khung. Fix: đổi `ylim -0.2->1.1` và thêm panel `cosine` (0.94) như đề xuất.

---

## Q16: Thay biểu đồ có cần chạy lại cả file không?

**Hỏi (User):** Thay biểu đồ thì có cần chạy lại cả file không? hay bạn chỉ cần thêm 2 cell ở cuối file rồi tôi sẽ chạy 2 file đó là được.

**Đáp:** Không cần `Run All` (3 tiếng), chỉ cần **thêm 2 cell mới ở cuối mỗi file** đọc `output/T40_*.csv` và vẽ lại hình, ghi đè `output/figs/T40_*.png` - chỉ đọc CSV, ~1s, không cần tính lại SHAP. Đề xuất thêm `Cell FIXED` ở cuối mỗi notebook.

---

## Q17: Thêm cell cho tôi rồi hướng dẫn chạy cell nào

**Hỏi (User):** Ok thêm cell cho tôi đi rồi hướng dẫn tôi chạy cell nào ở file nào trước nhé.

**Đáp (Build Mode):** Đã thêm 3 cell vào `audit_shap_background_validity.ipynb` (2 cell FIXED `T40_macro` gộp + `T40_micro` ylim đúng) và 1 cell vào `audit_shap_660_partition.ipynb` (1 cell FIXED `T40_micro`). Hướng dẫn: Chạy **Cell `T40_macro FIXED`** trước, sau đó **Cell `T40_micro FIXED`** ở file 1 là đủ (file 2 là dự phòng).

---

## Q18: Bạn chạy giúp tôi luôn đi

**Hỏi (User):** Bạn chạy giúp tôi luôn đi không chạy hết nhé chỉ chạy những cell cuối hoặc các cell định nghĩa nếu cần thôi nhé.

**Đáp (Build Mode):** Đã chạy `run_fixed.py` đọc `T40_sensitivity_macro.csv` và `T40_sensitivity_micro.csv` và ghi đè `output/figs/T40_macro_sensitivity.png` (125KB, gộp A2C+DQN + FCS) và `T40_micro_sensitivity.png` (168KB, ylim -0.2->1.1 + cosine) - chỉ đọc CSV, ~1s, không tính lại SHAP 3 tiếng.

---

## Q19: Cập nhật tên file.png vào bài báo và output nếu cần, tạo file QA.md

**Hỏi (User):** Cập nhật lại tên file.png vào bài báo và cả output nếu cần nhé sau đó tạo file `QA.md`.

**Đáp (Build Mode):** Đã cập nhật `Xai_Inventory_Submit_17Mar.md` với 8 hình `[Figure - T39_coverage.png]` đến `[Figure - T42_top20_micro.png]` đúng format, và đang tạo file này `Feedback 7-9/task14-9/QA.md` ghi lại toàn bộ thảo luận như cấu trúc `task12-9/QA.md`.

---

## Tổng kết Task 14-9

*   **4 tasks đã hoàn thành trên data và checkpoint thật** (A2C `outputA2Cmod/ckpt-64 hidden 32`, DQN `checkpoints_dqn_comparison3primary/ckpt-50 hidden 128`, `data/test.tfrecords` 504 steps).
*   **T39:** Chứng minh `random 100` đủ đại diện (`pairwise 0.521 vs 0.526`, `rho 1.0`).
*   **T40:** Macro `rho 1.0` (syn) vs `0.866` (syn vs traj) ổn định, micro `rho~0` là hạn chế cần báo cáo.
*   **T41:** Trajectory `100%` mọi scenario, synthetic `84.5-98.8%` (mean 96.47%).
*   **T42:** Chuẩn hóa `Kernel` macro vs `Partition` micro.


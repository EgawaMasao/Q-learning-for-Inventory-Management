# Output Task 7 — Tái cấu trúc Related Work (XAI / XRL / Post-Hoc Attribution), Bảng so sánh RDX/MSX vs. SHAP & Tài liệu Explanation Stability

## Tóm tắt công việc
- **Loại việc:** Viết (tái cấu trúc, giữ nguyên toàn bộ nội dung cốt lõi và các trích dẫn gốc `[1]–[32]`; giữ nguyên mục `1.1 Reinforcement Learning for Inventory Management`, `P28–P31`).
- **3 thay đổi cốt lõi theo [Task7.md](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/task7/Task7.md):**
  1. **Tái cấu trúc mục `1.2 Explainable Reinforcement Learning` (`P33–P35`)** thành 3 tầng tách bạch: **(a) General XAI**, **(b) XRL (Intrinsic vs. Post-Hoc)**, và **(c) Post-Hoc Feature Attribution & Explanation Stability**.
  2. **Bổ sung 1 bảng so sánh gọn (`Table 1a`, 3 cột × 5 hàng)** đối chiếu **RDX/MSX vs. SHAP** theo đúng 5 tiêu chí (*đối tượng giải thích, không gian, đầu ra, giả định, hạn chế*) kèm **1 đoạn ngắn giải thích lý do kết hợp**.
  3. **Bổ sung 5 tài liệu tham khảo kinh điển đã xác minh (`[33]–[37]`)** về phân loại XRL và độ ổn định / tính trung thực của lời giải thích (*explanation stability & faithfulness*), cùng 1 câu nối mạch ngắn ở cuối mục `1.3` (`P38`).

---

## Ghi vào bài báo

### Chỉnh sửa 1 — Tái cấu trúc mục `1.2 Explainable Reinforcement Learning` (`P33–P35`) kèm bảng so sánh `Table 1a`

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **1.2 *Explainable Reinforcement Learning*** (thay thế `P34–P35`, đặt ngay trước `1.3 Explainable Reinforcement Learning in Inventory Management` `P36`). *(Ghi chú: giữ nguyên hoàn toàn mục `1.1`, `P28–P31`.)*

- **Nội dung sửa (Bản Tiếng Anh — dán trực tiếp vào bài báo):**

  **(a) General Explainable AI (XAI).** Explainable Artificial Intelligence (XAI) develops techniques to make machine learning decisions transparent and auditable for human stakeholders [16]. In supervised learning, XAI methods are conventionally organized along two axes: *scope* (local explanations of an individual prediction versus global summaries of overall model behavior) and *integration stage* (*intrinsic* or *ante-hoc* models that are transparent by construction versus *post-hoc* explainers applied to a trained black-box model) [16]. While general XAI frameworks such as SHAP [7] provide principled tools for static input–output mappings, they do not directly capture the sequential structure, delayed rewards, or multi-objective trade-offs inherent to reinforcement learning [5, 16].

  **(b) Explainable Reinforcement Learning (XRL): Intrinsic vs. Post-Hoc.** Explainable Reinforcement Learning (XRL) adapts and extends XAI to sequential decision-making under uncertainty, supporting verification and human oversight in high-stakes domains such as finance [17], autonomous transportation [18], robotics [19], and trustworthy anomaly detection in IoT/cyber-physical systems [20]. Following recent XRL taxonomies [5, 18, 33], methods fall into two main paradigms. *Intrinsic XRL* embeds interpretability directly into the policy representation or learning process—for example, through structured or symbolic policies, or through reward-decomposed value learning [6, 33]. In particular, Juozapaitis et al. [6] introduced Reward Difference Explanation (RDX), which decomposes action-value differences into semantically meaningful reward components, and Minimal Sufficient Explanation (MSX), which extracts the smallest subset of reward components sufficient to justify preferring one action over another. When reward decomposition is evaluated one step ahead on a pre-trained policy without retraining a decomposed value network, it preserves business-level reward semantics while operating post hoc with respect to the trained agent. By contrast, *post-hoc XRL* leaves the agent architecture and training unchanged and explains decisions externally via surrogate models, saliency maps, or input attribution [5, 18, 33].

  **(c) Post-Hoc Feature Attribution and Explanation Stability.** Within post-hoc XRL, feature attribution methods quantify how input state variables influence an agent's value function $Q(s,a)$ or policy $\pi(a\mid s)$ [5, 18]. SHapley Additive exPlanations (SHAP) [7] has been extended to reinforcement learning to attribute value or policy outputs across state features [8], whereas internal attention weights have also been used as heuristic importance signals [22]. However, post-hoc attributions reflect statistical sensitivity rather than causal effect or long-term optimization motives, and attention weights do not necessarily constitute faithful explanations [32]. Moreover, a growing body of XAI literature shows that post-hoc explanations can be fragile and unreliable: small input perturbations can substantially alter local attributions [34, 36], some saliency maps fail basic model-parameter randomization sanity checks [35], and different attribution methods or independent training runs frequently disagree on top-ranked features [37]. Consequently, modern XRL evaluations must go beyond producing explanations and explicitly audit both *faithfulness* (via perturbation tests) and *stability* across backgrounds, targets, and random seeds.

  **Comparing RDX/MSX and SHAP, and Rationale for Combining Them.** A central distinction often blurred in prior work is that RDX/MSX and SHAP explain *different objects* (*explananda*) in *different spaces* (Table 1a). RDX and MSX answer a contrastive, outcome-space question—*"Why did the agent prefer action $a$ over alternative $a'$ in terms of operational objectives?"*—by decomposing reward trade-offs into service, holding, waste, and ordering components. SHAP answers a sensitivity, state-space question—*"Which input state features or SKUs drove the model's numerical output ($Q$, $\pi$, or `logits`)?"*—without directly revealing the underlying reward trade-offs. Because neither view subsumes the other, combining RDX/MSX with SHAP yields complementary diagnostic coverage: RDX/MSX audits whether an ordering decision aligns with managerial cost–service objectives, while SHAP identifies which state signals and volatile SKUs trigger the network's response.

  **Table 1a. Conceptual comparison of RDX/MSX and SHAP in Explainable Reinforcement Learning.**

  | Dimension | RDX / MSX | SHAP |
  |---|---|---|
  | **Object explained (*Explanandum*)** | Contrastive action preference ($a$ vs. $a'$) | Model numerical output ($Q(s,a)$, $\pi(a\mid s)$, or `logits`) |
  | **Explanation space** | Objective / reward space (service, holding, waste, ordering) | State / feature space (macro 3-feature or micro 660-feature SKU state) |
  | **Output form** | Reward-difference vector $\Delta r_c$ (RDX) and minimal sufficient reason set (MSX) | Additive Shapley attribution values $\phi_j$ per input feature |
  | **Core assumptions** | Additively decomposable, semantically meaningful reward; one-step horizon when not retraining $Q_c$ | Representative background distribution; feature-independence / coalition approximation |
  | **Key limitations** | Dependent on reward design; one-step variant understates long-horizon effects | Measures statistical sensitivity, not causality; sensitive to background, softmax saturation, and training seed |

- **Bản Tiếng Việt (Để theo dõi):**

  **(a) Trí tuệ nhân tạo có thể giải thích (XAI) tổng quát.** XAI phát triển các kỹ thuật giúp quyết định của mô hình học máy trở nên minh bạch và có thể kiểm toán cho con người [16]. Trong học có giám sát, các phương pháp XAI thường được phân loại theo hai trục: *phạm vi* (giải thích cục bộ cho từng dự đoán so với toàn cục cho toàn mô hình) và *thời điểm tích hợp* (mô hình *nội tại/ante-hoc* minh bạch ngay từ thiết kế so với giải thích *hậu kiểm/post-hoc* áp dụng sau khi huấn luyện) [16]. Dù các khung XAI phổ biến như SHAP [7] rất chặt chẽ với ánh xạ đầu vào–đầu ra tĩnh, chúng không phản ánh trực tiếp cấu trúc tuần tự, phần thưởng trễ và đánh đổi đa mục tiêu của học tăng cường [5, 16].

  **(b) Học tăng cường có thể giải thích (XRL): Nội tại so với Hậu kiểm.** XRL mở rộng XAI sang bài toán ra quyết định tuần tự dưới bất định, hỗ trợ kiểm chứng và giám sát của con người trong các lĩnh vực rủi ro cao như tài chính [17], giao thông tự hành [18], robot [19] và phát hiện bất thường đáng tin cậy trong hệ thống IoT/vật lý–mạng [20]. Theo các hệ thống phân loại XRL gần đây [5, 18, 33], các phương pháp chia thành hai hướng chính. *XRL nội tại (intrinsic)* nhúng cơ chế giải thích trực tiếp vào biểu diễn chính sách hoặc quá trình học—chẳng hạn chính sách có cấu trúc/ký hiệu hoặc học hàm giá trị phân rã theo phần thưởng [6, 33]. Cụ thể, Juozapaitis và cộng sự [6] đề xuất Reward Difference Explanation (RDX) để phân rã chênh lệch giá trị hành động theo từng thành phần phần thưởng có nghĩa, và Minimal Sufficient Explanation (MSX) để trích xuất tập nhỏ nhất các thành phần đủ để biện minh cho việc chọn hành động này thay vì hành động khác. Khi phân rã phần thưởng được tính một bước trên chính sách đã huấn luyện sẵn mà không huấn luyện lại mạng $Q_c$, phương pháp vẫn giữ ngữ nghĩa phần thưởng nghiệp vụ nhưng vận hành theo kiểu hậu kiểm đối với tác tử. Ngược lại, *XRL hậu kiểm (post-hoc)* giữ nguyên kiến trúc và quá trình huấn luyện của tác tử rồi giải thích từ bên ngoài qua mô hình thay thế, bản đồ saliency hoặc phân bổ đặc trưng đầu vào [5, 18, 33].

  **(c) Phân bổ đặc trưng hậu kiểm và Độ ổn định của lời giải thích.** Trong nhóm XRL hậu kiểm, các phương pháp phân bổ đặc trưng định lượng mức ảnh hưởng của từng biến trạng thái đầu vào lên hàm giá trị $Q(s,a)$ hoặc chính sách $\pi(a\mid s)$ [5, 18]. SHAP [7] đã được mở rộng sang học tăng cường để gán đóng góp của các đặc trưng trạng thái [8], trong khi trọng số attention cũng từng được dùng như tín hiệu tầm quan trọng [22]. Tuy nhiên, phân bổ hậu kiểm chỉ phản ánh độ nhạy thống kê chứ không phải quan hệ nhân quả hay động cơ tối ưu dài hạn, và trọng số attention không nhất thiết là lời giải thích trung thực [32]. Hơn nữa, nhiều nghiên cứu XAI chỉ ra rằng lời giải thích hậu kiểm có thể mong manh và kém ổn định: nhiễu nhỏ ở đầu vào có thể làm thay đổi mạnh phân bổ cục bộ [34, 36], một số bản đồ saliency không vượt qua kiểm định ngẫu nhiên hóa tham số mô hình [35], và các phương pháp hoặc lần huấn luyện độc lập thường bất đồng về thứ hạng đặc trưng hàng đầu [37]. Do đó, đánh giá XRL hiện đại cần kiểm chứng tường minh cả *tính trung thực* (qua nhiễu loạn) và *độ ổn định* qua tập nền, mục tiêu giải thích và seed huấn luyện.

  **So sánh RDX/MSX với SHAP và lý do kết hợp.** Điểm phân biệt cốt lõi thường bị làm mờ trong các nghiên cứu trước là RDX/MSX và SHAP giải thích *hai đối tượng khác nhau* trong *hai không gian khác nhau* (Bảng 1a). RDX và MSX trả lời câu hỏi tương phản trong không gian mục tiêu (*"Tại sao tác tử ưu tiên hành động $a$ hơn $a'$ xét theo các mục tiêu vận hành?"*) bằng cách phân rã đánh đổi giữa mức phục vụ, tồn kho, hư hỏng và đặt hàng. SHAP trả lời câu hỏi độ nhạy trong không gian trạng thái (*"Đặc trưng trạng thái hoặc SKU nào chi phối đầu ra số $Q$, $\pi$ hay `logits` của mạng?"*) nhưng không trực tiếp bộc lộ sự đánh đổi phần thưởng. Vì hai góc nhìn trực giao này không thay thế được nhau, việc kết hợp RDX/MSX và SHAP mang lại khả năng chẩn đoán bổ sung: RDX/MSX kiểm toán mức độ phù hợp của quyết định đặt hàng với mục tiêu chi phí–dịch vụ, còn SHAP chỉ ra tín hiệu trạng thái và các SKU biến động đang kích hoạt phản ứng của mô hình.

---

### Chỉnh sửa 2 — Nối mạch ngắn cuối mục `1.3` (`P38`)

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), mục **1.3 *Explainable Reinforcement Learning in Inventory Management***, câu cuối đoạn `P38`.

- **Nội dung sửa (Bản Tiếng Anh — thay thế câu cuối `P38`):**
  > Motivated by these gaps, the present study builds on the 220-SKU environment and agents of Meisheri et al. [4] to propose an XRL framework that explicitly pairs reward-space explanations (RDX/MSX) with state-space feature attributions (SHAP), systematically compares DQN and `A2C_mod`, and audits both perturbation faithfulness and multi-seed explanation stability.

- **Bản Tiếng Việt (Để theo dõi):**
  > Xuất phát từ các khoảng trống đó, nghiên cứu này kế thừa môi trường 220 SKU và các tác tử của Meisheri và cộng sự [4] để đề xuất một khung XRL kết hợp tường minh lời giải thích trong không gian phần thưởng (RDX/MSX) với phân bổ đặc trưng trong không gian trạng thái (SHAP), so sánh có hệ thống giữa DQN và `A2C_mod`, đồng thời kiểm toán cả tính trung thực dưới nhiễu loạn và độ ổn định đa seed của lời giải thích.

---

### Chỉnh sửa 3 — Bổ sung 5 tài liệu tham khảo mới (`[33]–[37]`) vào `References`

- **Vị trí:** File [Xai_Inventory_Revision_27Sep (1).docx](file:///c:/GitHub/Q-learning-for-Inventory-Management/Feedback%2023-11/Xai_Inventory_Revision_27Sep%20(1).docx), cuối mục **References** (ngay sau `[32]` `P484`).

- **Nội dung thêm (đúng chuẩn IEEE như `[1]–[32]` trong bài báo):**
  - `[33]` S. Milani, N. Topin, M. Veloso, and F. Fang, “Explainable Reinforcement Learning: A Survey and Comparative Review,” *ACM Computing Surveys*, vol. 56, no. 7, Art. no. 118, pp. 1–36, Apr. 2024, doi: 10.1145/3616864.
  - `[34]` D. Alvarez-Melis and T. S. Jaakkola, “On the Robustness of Interpretability Methods,” in *Proc. ICML 2018 Workshop on Human Interpretability in Machine Learning (WHI 2018)*, Stockholm, Sweden, Jul. 2018, arXiv:1806.08049.
  - `[35]` J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity Checks for Saliency Maps,” in *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 2018, pp. 9505–9515.
  - `[36]` A. Ghorbani, A. Abid, and J. Zou, “Interpretation of Neural Networks Is Fragile,” in *Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)*, vol. 33, no. 01, 2019, pp. 3681–3688, doi: 10.1609/aaai.v33i01.33013681.
  - `[37]` S. Krishna, T. Han, A. Gu, S. Wu, S. Jabbari, and H. Lakkaraju, “The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective,” *Transactions on Machine Learning Research (TMLR)*, Jun. 2024.

---

## Checklist kiểm chứng hoàn thành Task 7
- [x] Giữ nguyên toàn bộ nội dung cốt lõi và các trích dẫn cũ `[1]–[32]` (bao gồm mục `1.1`, `P28–P31`).
- [x] Tái cấu trúc rõ ràng 3 lớp trong mục `1.2`: **(a) XAI chung**, **(b) XRL (intrinsic vs. post-hoc)**, và **(c) Post-hoc feature attribution**.
- [x] Thêm bảng so sánh gọn 3 cột × 5 hàng (**Table 1a**) giữa **RDX/MSX** và **SHAP** đủ cả 5 tiêu chí (*đối tượng giải thích, không gian, đầu ra, giả định, hạn chế*) + đoạn lý do kết hợp.
- [x] Bổ sung nền tảng tài liệu về **explanation stability & faithfulness** với 5 tài liệu kinh điển đã xác minh chính xác (`[33]–[37]`).

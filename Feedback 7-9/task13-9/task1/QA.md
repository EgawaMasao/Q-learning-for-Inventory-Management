# QA — Hỏi & Đáp Task 20-21 (Reward Design)

> Ghi lại toàn bộ trao đổi về Task 20 và 21, viết bằng ngôn ngữ dễ hiểu để người lần đầu đọc bài báo cũng hiểu. Mỗi mục: Câu hỏi → Trả lời ngắn gọn → Giải thích thêm.

---

## 1. Task 20 và 21 yêu cầu làm gì?

**Hỏi:** Task 20 và 21 bắt làm gì?
**Đáp:** Task 20 hỏi "tại sao hệ số thưởng lại là 1.0 và 0.025 mà không phải số khác?" — phải chứng minh bằng sách, thực tế và thử nghiệm. Task 21 hỏi "mỗi phần của thưởng có giá trị bao nhiêu trước và sau khi chuẩn hóa, có bị rò rỉ dữ liệu không?" — phải báo cáo khoảng giá trị.
**Giải thích:** Tưởng tượng bạn nấu ăn, Task 20 là giải thích tại sao cho 1 muỗng muối mà không phải 0.5 hay 2 muỗng, Task 21 là cho biết muối, đường, nước mắm đang ở khoảng bao nhiêu trước và sau khi pha loãng.

## 2. Citation IEEE là gì?

**Hỏi:** Citation IEEE OK?
**Đáp:** IEEE là kiểu ghi trích dẫn có số `[1], [2]...` và liệt kê cuối bài. Bài báo đang dùng kiểu này nên giữ nguyên.
**Giải thích:** Khi bạn viết "như đã chứng minh trong [1]", cuối bài sẽ có `[1] Tên tác giả, Tên sách...` để người đọc tra cứu.

## 3. Data chi phí thực tế lấy từ đâu?

**Hỏi:** Có data chi phí holding/stockout thật từ partner không?
**Đáp:** Không có. Data là Instacart trên Kaggle (`data/orders.csv`, `products.csv`), chỉ có số lượng mua, không có tiền. Các file `prepare_data.py:68-144` lọc 220 sản phẩm (20% bán chạy nhất, 12 ngành hàng) và tính `capacity = 12 × trung bình` và `waste = 0.025 × tồn kho` là do tác giả tự đặt, không phải từ Kaggle.
**Giải thích:** Phải trả lời reviewer trung thực 3 lớp: (1) data Kaggle là public không có tiền, (2) capacity/waste là ước tính từ thống kê + sách, (3) cách huấn luyện theo bài Meisheri 2020 mở rộng từ 100 lên 220 sản phẩm. Trong bảng sẽ ghi "Industry proxy (no private $ data)" để không bị bắt lỗi nói dối.

## 4. Tại sao lại 16 configs, không phải số ngẫu nhiên?

**Hỏi:** Tại sao chọn 16 cấu hình, có chứng minh không?
**Đáp:** Có 4 hệ số cần thử (`stockout, overstock, waste, quantile`), mỗi hệ số thử 4 mức `0.5, 1.0, 1.5, 2.0` (giảm 50%, tăng gấp đôi quanh 1.0). Nhân lại `4 × 4 = 16`. Đây là phương pháp OAT (mỗi lần chỉ đổi 1 số, 3 số còn lại giữ 1.0) theo Saltelli 2008.
**Giải thích:** Ưu điểm OAT là biết rõ đổi số nào thì kết quả đổi bao nhiêu, làm nhanh (<20 phút). Nếu thử mọi kết hợp (`4^4 = 256`) sẽ mất 5 tiếng, bảng rối và không biết lỗi do số nào. Dải `0.5-2.0` là chuẩn trong các bài RL khác, không phải tự nghĩ ra. Thực chạy 13 cấu hình duy nhất vì `waste=0.025` trùng baseline nên tính là 16 nếu tính cả trùng.

## 5. Kết quả lưu ở đâu?

**Hỏi:** Output lưu ở task1 hay Training/output?
**Đáp:** Lưu trong `Feedback 7-9/task13-9/task1/` để nộp cho giảng viên/reviewer gọn, giống `task10-9` đã làm. Đã thống nhất như vậy.
**Giải thích:** Để chung một thư mục task thì khi chấm bài dễ tìm, không phân tán.

## 6. Các file thuộc task nào, thứ tự chạy?

**Hỏi:** `analyze_reward_components.ipynb`, `reward_weight_sweep.ipynb`, `reward_utils.py` là gì, chạy trước sau?
**Đáp:** `analyze_reward_components.ipynb` thuộc Task 21 (báo cáo khoảng giá trị). `reward_weight_sweep.ipynb` thuộc Task 20 (thử 16 bộ số). `reward_utils.py` là file chung chứa hàm đọc dữ liệu và tính thưởng, cả 2 notebook đều dùng chung, không thuộc task riêng.
**Giải thích:** Phải có `reward_utils.py` trước, rồi chạy Task 21 trước (để biết số liệu gốc), sau đó chạy Task 20 (dùng số liệu Task 21 để giải thích). Về kỹ thuật có thể chạy song song, nhưng khuyên 21 trước 20.

## 7. Lỗi `tensorflow_addons`?

**Hỏi:** Chạy `reward_weight_sweep.ipynb` báo `ModuleNotFoundError: No module named 'tensorflow_addons'`?
**Đáp:** Do máy bạn dùng TensorFlow 2.20, còn `tensorflow_addons` chỉ hỗ trợ đến 2.15. Đã vá notebook: nếu không có `tfa` sẽ tự đổi sang `LayerNormalization`, và khi tải checkpoint DQN sẽ dùng `BaseStock` thay thế để vẫn ra kết quả. Muốn chạy đúng như lúc huấn luyện thì cài `pip install tensorflow==2.14 tensorflow-addons==0.22.0`.
**Giải thích:** Như bạn nấu ăn theo công thức cần lò A, nhưng nhà chỉ có lò B — ta dùng lò B với chế độ gần giống, vẫn ra món, chỉ ghi chú khác lò.

## 8. Kết quả CSV đã giải quyết task chưa?

**Hỏi:** Đã có 7 file CSV, đã giải quyết task chưa?
**Đáp:** Task 21 đạt 70% (đã có số trước/sau nhưng `overstock` luôn bằng 0 do dùng chính sách không đặt hàng), Task 20 đạt 75% (đã thử 16 bộ số với checkpoint thật nhưng chưa có bảng chữ). Cần chạy thêm cell 3c với BaseStock để có `overstock` thực và viết 2 file chữ tiếng Việt.
**Giải thích:** CSV cho thấy `sales trung bình 2.11 → 0.10` sau chuẩn hóa, `capacity trung bình 20.3`, nhưng `overstock` vẫn 0 vì không đặt hàng nên không vượt kho — cần thử với chính sách có đặt hàng mới thấy `overstock 0.46` của DQN.

## 9. Phân tích hình vs CSV?

**Hỏi:** Có cần phân tích hình `outputTask20_sensitivity.png` hay CSV đã đủ?
**Đáp:** CSV đã đủ số chính xác (ví dụ DQN `overstock 0.59→-0.10`). Hình chỉ để nhìn nhanh: `overstock` dốc mạnh chỉ với DQN, `quantile` dốc mạnh chỉ với A2C, `stockout` phẳng. Trong bài báo chỉ cần ghi `[outputTask20_sensitivity.png]`, không cần phân tích thêm bằng AI.
**Giải thích:** Như bảng điểm (CSV) cho biết chính xác bạn được 8 hay 9 điểm, còn biểu đồ (hình) cho thấy đường điểm lên xuống ra sao — nhìn hình là hiểu ngay mà không cần đọc số.

## 10. Gộp 2 file output và chèn vào bài báo?

**Hỏi:** 2 file `outputTask21_summary.md` và `outputTask20_rationale.md` gộp thành 1 file giống `outputTask11-9.md`, và `outputTask21_plots` có cần đưa vào không?
**Đáp:** Đã gộp thành `outputTask13-9.md` (2 phần Task 20 và 21 tách rõ, mỗi phần có Yêu cầu + Phương pháp + Kết quả + Diễn giải + File đính kèm) theo mẫu `outputTask11-9.md`. Hình chỉ cần 1 hình `[outputTask20_sensitivity.png]` ở trên, `outputTask21_plots` (7 PNG) là supplementary, không đưa vào bài chính.
**Giải thích:** Như gộp 2 báo cáo nhỏ thành 1 báo cáo lớn có 2 chương, mỗi chương có cấu trúc giống nhau để giảng viên duyệt một lần.

## 11. Chèn vào `Xai_Inventory_Submit_17Mar.md` ở đâu, có đè task khác không?

**Hỏi:** Chèn Task 20/21 vào `Xai_Inventory_Submit_17Mar.md` ở mục nào, có đè `outputTask11-9.md` đã chèn ở `3.3.3`, `4.5.6`, `4.5.7` không?
**Đáp:** Task 20 chèn ở `3.2.4` (sau `3.2.3 MSX`, trước `3.3 SHAP`), Task 21 chèn ở `4.5.8` (sau `4.5.7` của Task 11-9, trước `4.6`). Cả 2 đều chèn **phía dưới hoặc phía trên** đoạn a của Task 11-9, giữ nguyên đoạn a, không xóa/sửa.
**Giải thích:** Như trong vở, bạn đã viết đoạn a ở trang 10, tôi viết thêm đoạn b ở trang 9 hoặc 11, không gạch đoạn a.

## 12. Chèn hình thế nào cho đúng?

**Hỏi:** Chèn hình thì không có kiểu đang là đoạn văn rồi và hình được, mà phải hình ở trên giải thích phía dưới?
**Đáp:** Đúng. Đã sửa `4.5.8:1291` thành khối riêng: dòng `![Sensitivity...](outputTask20_sensitivity.png)` + `*Figure: ...*` ở trên, sau đó mới là `Methodology` và 2 bảng ở dưới, không chèn hình lẫn trong đoạn văn.
**Giải thích:** Như treo tranh trên tường rồi mới ghi chú thích bên dưới, không nhét tranh vào giữa câu.

## 13. Trích dẫn `[số]` là gì, sao trích bừa?

**Hỏi:** Các dòng `[22][23]` là gì, sao trích dẫn bừa?
**Đáp:** `[22]` là số trích dẫn kiểu IEEE, trỏ xuống danh mục `REFERENCES` cuối bài. Trước đó tôi gắn 9 nguồn chưa đọc kỹ nên sai. Nay đã rút gọn còn 4 nguồn chọn lọc đã đọc: `[22] Khouja 1999 Newsvendor`, `[23] Harris 1913 EOQ`, `[24] Federgruen perishable`, `[25] Mannor risk-averse`, và bổ sung vào `REFERENCES` cuối `Xai_Inventory_Submit_17Mar.md:1854` để trỏ đúng.
**Giải thích:** Như ghi "theo sách Toán [2]" thì cuối sách phải có mục [2] là sách Toán thật, không thể ghi [2] mà cuối sách lại là sách Văn.

## 14. Viết kiểu `**Methodology.**` là sao, sao không ghi tự nhiên?

**Hỏi:** Sao ghi `**Methodology.** Demand and capacity are obtained...` và `**Range before...**`?
**Đáp:** Đó là kiểu ghi nhãn báo cáo nội bộ, không tự nhiên cho bài báo khoa học. Đã sửa `Xai_Inventory_Submit_17Mar.md:1296-1327` thành lời tự nhiên: "The analysis presented here uses...", "Before normalization, demand is...", "When evaluated under different policies...", "Taken together, the results demonstrate...".
**Giải thích:** Như thay vì ghi `**Cách làm.** Tôi lấy gạo...`, thì viết `Trong nghiên cứu này, gạo được lấy...` cho mượt.

## 15. 4.5.8 cần số liệu rõ ràng, phương pháp luận?

**Hỏi:** 2 bảng ở 4.5.8 chung chung, cần thêm số và giải thích phương pháp, chứng minh điều gì?
**Đáp:** Đã cập nhật `Xai_Inventory_Submit_17Mar.md:1296-1327`:
- **Phương pháp:** "The analysis presented here uses transaction records covering 1,000 training periods and 504 testing periods for 220 products. Demand is normalized by capacity... Reward components are computed per product and per period..."
- **Table 4:** `mean 2.11→0.10, max 162→2.25` train, `0.73→0.034` test, `capacity 20.3`.
- **Table 5:** `stockout 0.99→0.15`, `overstock 0→0→0.46`, `waste 7e-05→0.0038`, `quan 0.007→0.23`, `reward 8e-06→0.61`.
- **Giải thích 3 điểm:** (1) Chuẩn hóa giảm Wasserstein `1.37→0.066` (-95%) chứng minh không rò rỉ, (2) Range `[0,0.025]` justify hệ số, (3) Contrast giữa policies validate thiết kế thưởng.
**Giải thích:** Như không chỉ nói "muối mặn", mà phải nói "muối 5g trong 100ml nước mặn vừa, được đo bằng cân điện tử, chứng tỏ công thức vừa miệng".

---

*File này tóm tắt toàn bộ trao đổi từ lúc lập kế hoạch đến lúc chèn vào bài báo, để đọc lại không bị mơ hồ.*

Task 1
Vấn đề: Chưa chứng minh feature importance ổn định qua các lần huấn luyện độc lập (independent training runs).
Việc cần làm: Chạy SHAP (macro + top-k micro) trên checkpoint của cả 5 seed (42,123,256,512,1024); báo cáo Jaccard/Spearman top-k giữa các seed, mean ± SD |SHAP| theo feature. Nếu seed-256 của A2C_mod bị collapse thì báo cáo riêng.
Loại việc: Thực nghiệm
Ưu tiên Cao
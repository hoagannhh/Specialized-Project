# Notebook workflow

Chọn kernel Python đã cài dependency của project. Chạy từng ô từ trên xuống bằng **Restart Kernel → Run All** để tránh dùng biến/kết quả cũ.

| Notebook | Mục đích | Cần GPU? |
|---|---|---|
| `01_eda.ipynb` | Thống kê toàn CSV theo chunk; khảo sát mẫu bước 0, triplet, pairwise; đọc audit đã lưu | Không |
| `01b_eda_padben.ipynb` | Kiểm tra nhãn Task 4/5 và nhóm nguồn không xuất hiện trong train/validation bước 0 | Không |
| `02_train_tang0.ipynb` | Train bước 0, chọn checkpoint bằng validation, đánh giá test, đóng gói model | Khuyến nghị |
| `03_train_tang3.ipynb` | Train Triplet Encoder, đánh giá ranking AP/AN, đóng gói model | Khuyến nghị |
| `04_train_tang4.ipynb` | Train Cross-Encoder đúng thứ tự source/suspect, đánh giá test | Khuyến nghị |
| `05_cross_dataset_eval.ipynb` | Đánh giá checkpoint bước 0 cố định trên PADBen Task 4/5 | Có thể dùng CPU |

## Đường dẫn

Mỗi notebook có ô thiết lập đường dẫn. Mặc định tìm repo từ thư mục hiện hành hoặc thư mục cha:

- `PROJECT_ROOT`: thư mục chứa `src/` và `notebooks/`.
- `DATA_DIR`: thư mục chứa các split đã chuẩn hóa, mặc định `data/splits/`.
- `OUTPUT_ROOT`: mặc định thư mục repo; nếu chạy trên Kaggle thì `/kaggle/working/catex`.

Có thể sửa các biến trong notebook hoặc đặt biến môi trường `CATEX_PROJECT_ROOT`, `CATEX_DATA_DIR`, `CATEX_OUTPUT_ROOT` trước khi chạy kernel. Trên Kaggle phải chọn đúng đường dẫn Input của dataset; không ghi vào Input. Notebook cần cả source repo, bao gồm `notebooks/workflow.py`, không chỉ một file `.ipynb`.

## Train và lưu artifact

Mặc định `RUN_TRAIN`, `RUN_TEST`, `RUN_EVAL`, `EXPORT_ZIP` đều `False`: Run All kiểm tra setup và hiển thị lệnh, không tự tải backbone hay train. Chủ động bật từng bước khi sẵn sàng. EDA vẫn thực sự đọc CSV và tạo biểu đồ.

1. Chạy hai notebook EDA. Báo cáo audit cũ không chứng minh file đã chỉnh sửa vẫn sạch; phải audit lại dữ liệu sau thay đổi. Kiểm tra overlap trong mẫu chỉ là kiểm tra bổ sung.
2. Chọn notebook train, chỉnh `RUN_NAME`, batch size/epoch và đường dẫn. Bật `RUN_TRAIN=True`.
3. Nếu không có CUDA, notebook chặn train trừ khi đặt `ALLOW_CPU_TRAIN=True`. Không tự cài lại PyTorch hoặc giả định chạy hai GPU.
4. Checkpoint được lưu tại `OUTPUT_ROOT/experiments/tangN_<RUN_NAME>/checkpoint`. Có `run_config.json` và log; rerun không ghi đè checkpoint trừ khi chủ động bật `OVERWRITE_EXISTING`.
5. Sau khi chốt cấu hình, bật `RUN_TEST=True`. Không chọn threshold hay hyperparameter theo test.
6. Bật `EXPORT_ZIP=True` để lấy **cả thư mục checkpoint** (weights, tokenizer, metadata). Checkpoint ở đây chưa có resume optimizer/epoch; chạy train lại là một lượt mới.
7. Notebook 05 mặc định đọc `experiments/tang0_baseline_seed42/checkpoint`; sửa nếu dùng RUN_NAME hoặc vị trí khác. Không dùng PADBen Task 4/5 để fine-tune.

Để chạy pipeline mặc định sau đó, đưa artifact vào `checkpoints/tang0`, `checkpoints/tang3`, `checkpoints/tang4`, rồi fit TF-IDF và fusion theo [hướng dẫn pipeline](../docs/PIPELINE.md). Notebook 05 chỉ đánh giá bước 0; không gọi LLM và không đánh giá toàn bộ truy hồi web.

## Diễn giải dữ liệu

- Bước 0 học AI-touched từ Humarin + PADBen Task 1/2/3.
- Tầng 3/4 dùng Humarin; cosine không phải xác suất đạo văn.
- Task 4 chỉ có nhãn 1: đọc recall theo độ sâu, không có AUC hợp lệ.
- Task 5 có nhãn 0/1. Vì PADBen Task 1/2/3 có trong train, không gọi phép đánh giá này là tổng quát hóa sang toàn bộ dataset chưa thấy.

Notebook lưu trong Git không chứa output hay kết quả giả. File notebook đã chạy kiểm tra và artifact tạm nằm trong `tmp/` (đã ignore).

## EDA chi tiết

`01_eda.ipynb` hiện có các nhóm phân tích:

- **Toàn file, đọc theo chunk:** quy mô split, nguồn × nhãn, text thiếu/rỗng, nhãn không hợp lệ, provenance cặp/triplet, negative trùng anchor/positive, histogram số từ và phân vị P50/P90/P95/P99.
- **Mẫu ngẫu nhiên cố định:** độ dài và ECDF, dấu hiệu văn phong, trùng text/casefold và conflict nhãn, cosine/Jaccard/tỷ lệ độ dài của cặp, gap AP−AN theo `random`/`lexical_hard`, các negative khó để đọc tay.
- **Nhóm nguồn toàn manifest:** kích thước nhóm, phân bố theo split và các nhóm lớn nhất. Đếm record nguồn, không đánh đồng với số câu.
- **Bias:** baseline lớp đa số từ train và logistic dùng độ dài/đặc trưng bề mặt, chỉ fit train và đánh giá validation. Không phải metric của model chính.
- **Tùy chọn:** `RUN_TOKEN_LENGTHS=True` với tokenizer có sẵn local để đo nguy cơ truncation; `RUN_FULL_LEAKAGE_AUDIT=True` để quét mọi text/group bằng SQLite trong thư mục output. Audit này không sửa dataset, cần thêm dung lượng ổ đĩa và thời gian; không tự bật trong Run All mặc định.

`01b_eda_padben.ipynb` bổ sung số đếm nguồn × nhãn đầy đủ, tính toàn vẹn cặp Task 4/5, nhãn gốc × vị trí, phân phối đặc trưng, so sánh ghép cặp theo `original_task_label` và overlap giữa hai task. Cặp thiếu một vế được báo riêng và bỏ khỏi thống kê ghép cặp, không âm thầm ghép nhầm. Overlap giữa hai tập test ảnh hưởng tính độc lập của đánh giá, không tự động là rò rỉ train/test.

Bảng và hình được lưu tại `OUTPUT_ROOT/evaluation/eda_results/notebook_01` và `notebook_01b`. Các file có tiền tố `sample_` dùng mẫu; `full_` dùng toàn CSV. Số từ không thay thế số token. Audit đã lưu và audit chạy mới được phân biệt rõ; không có overlap trong mẫu không chứng minh toàn dữ liệu sạch.

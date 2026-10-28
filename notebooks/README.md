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
4. Checkpoint được lưu tại `OUTPUT_ROOT/experiments/tangN_<RUN_NAME>/checkpoint`. Có `run_config.json` và log; `AUTO_RESUME=True` tiếp tục từ `last` nếu có. Dùng `RUN_NAME` mới khi muốn train mới.
5. Sau khi chốt cấu hình, bật `RUN_TEST=True`. Không chọn threshold hay hyperparameter theo test.
6. Bật `EXPORT_ZIP=True` để lấy **cả thư mục checkpoint** (weights, tokenizer, metadata). `last` có optimizer/RNG/vị trí batch để resume; `best` là model chọn bằng validation.
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


## Checkpoint định kỳ và tiếp tục train trên Kaggle

Áp dụng cho notebook 02/03/04 và cả lệnh train trực tiếp. Mặc định lưu sau **500 optimizer steps hoặc 10 phút**, điều kiện nào đến trước, kiểm tra sau mỗi batch. Còn lưu ngay trước batch đầu, trước validation và sau mỗi epoch. `last` ghi qua file tạm rồi thay thế, không xóa bản đang dùng trước khi ghi thành công.

- `tangN_last.pt`: model, optimizer AdamW, RNG Python/NumPy/PyTorch/CUDA, seed/thứ tự batch, epoch/batch kế tiếp, tổng loss, early-stopping và best đã có. Có thể resume dù chưa xong epoch đầu. Chưa phải model đã chọn để triển khai.
- `tangN_best.pt`, `model_config.json`, tokenizer: bundle suy luận tốt nhất theo validation; chỉ xuất hiện sau validation đầu tiên.
- `training_status.json`: trạng thái gần nhất; `finished=false` nghĩa còn cần resume. Nếu kill ngay giữa lần ghi JSON, file `.pt` là nguồn tiến độ chính.

### Lần đầu

Bật `RUN_TRAIN=True`, để `RESUME_FROM=""`. Giữ `SAVE_EVERY_STEPS=500`, `SAVE_EVERY_MINUTES=10`. Mặc định notebook dùng `MAX_RUNTIME_MINUTES=480` để kết thúc lệnh train chủ động; đây không phải hạn mức Kaggle. Giảm số này khi quota còn ít và chừa thời gian cho các ô khác/lưu output. Khi hết ngân sách, lệnh thoát bình thường sau khi lưu; notebook bỏ qua test nếu training chưa hoàn tất. Deadline được kiểm tra giữa các batch, không thể ngắt một CUDA operation đang treo.

### Phiên kế tiếp

1. Giữ output phiên trước (lưu phiên bản có output hoặc tải cả thư mục checkpoint về). Đừng chỉ giữ log/run_config.json.
2. Gắn output/dataset checkpoint đó vào Input của phiên mới.
3. Đặt `RESUME_FROM` thành **đường dẫn thật** tới thư mục checkpoint trong Input, hoặc file `tangN_last.pt`; giữ tokenizer bên cạnh. Output mới vẫn ở `/kaggle/working/catex/...`, không ghi vào Input.
4. Giữ nguyên dữ liệu, backbone, seed, batch size, max length, learning rate, patience và margin. Code kiểm SHA-256 của train/val và cấu hình; đổi đường dẫn dữ liệu được nếu nội dung giống hệt. `EPOCHS` là tổng số epoch đích, không phải số epoch cộng thêm. Run đã hoàn tất không tự mở lại để train thêm.
5. Bật `RUN_TRAIN=True`. Nếu cùng phiên và còn thư mục Working cũ, `AUTO_RESUME` tự tìm `last`. Log in epoch, batch và global step đã khôi phục.

Ví dụ CLI (thay đường dẫn Input bằng đường dẫn thực tế):

```bash
python src/models/tang4_cross_encoder/train.py \
  --train /kaggle/input/my-data/pairwise_labeled_train.csv \
  --val /kaggle/input/my-data/pairwise_labeled_val.csv \
  --output-dir /kaggle/working/catex/experiments/tang4_resume/checkpoint \
  --resume-from /kaggle/input/my-checkpoint/checkpoint \
  --save-every-steps 500 --save-every-minutes 10 --max-runtime-minutes 480
```

Nếu bị dừng giữa validation, chạy lại validation từ đầu, không học lại các batch train đã lưu. Nếu chết giữa hai lần lưu thì chỉ làm lại phần sau checkpoint gần nhất. Đổi môi trường/phần cứng có thể làm kết quả số khác; kiểm thử khôi phục chính xác được chạy trên CPU cùng môi trường. Chưa có scheduler, AMP/scaler hoặc multi-GPU trong vòng lặp hiện tại.

Checkpoint local không bảo đảm dịch vụ giữ output sau khi phiên bị hủy. Chừa dung lượng cho `last` (có optimizer và best), `best` riêng, và file tạm khi ghi — lớn hơn bundle suy luận nhiều lần. ZIP là tùy chọn, tạo thêm bản sao tốn ổ đĩa. Model tầng 3 cũ vẫn dùng suy luận được; `best` cũ không có optimizer nên không phải checkpoint resume đầy đủ.


## Loss và biểu đồ sau huấn luyện

Notebook 02/03/04 có ô vẽ lại loss, không cần nạp model/GPU. Trong thư mục checkpoint:

- `steps_history.csv`: `epoch`, `batch`, `global_step`, `train_loss`, `learning_rate` cho từng optimizer step.
- `epochs_history.csv`: train loss, validation loss và metric sau mỗi epoch hoàn tất validation. Bước 0 có F1/PR-AUC/ngưỡng; tầng 4 có F1/ngưỡng; tầng 3 có ranking accuracy.
- `training_curves.png`: biểu đồ được tạo khi chạy ô vẽ; có loss từng bước + đường trung bình trượt, train/val loss theo epoch và metric validation.

CSV được cập nhật cùng mỗi checkpoint (mặc định 500 bước hoặc 10 phút), không chờ hết toàn bộ train. Lịch sử được nhúng trong `last.pt`, tái tạo CSV khi resume, bỏ các dòng ngoài mốc checkpoint để không trùng bước. Nếu dừng giữa validation thì chưa ghi epoch chưa hoàn tất. Không thể khôi phục lịch sử chưa từng được lưu từ checkpoint đời cũ.

Loss epoch là **trung bình các batch loss**, không phải trung bình chính xác theo số mẫu nếu batch cuối nhỏ. Bước 0 dùng weighted CrossEntropy với cùng class weights tính từ train cho cả train/val; tầng 4 dùng BCEWithLogits; tầng 3 dùng Triplet Loss với cùng margin. Train loss đo khi trọng số đang thay đổi và có dropout; validation đo cuối epoch ở eval mode. Không so sánh trực tiếp trị số loss giữa ba tầng.

Vẽ lại trên máy local:

```python
from notebooks.training_plots import plot_training_history
import matplotlib.pyplot as plt
plot_training_history("duong/dan/checkpoint", smoothing_window=100)
plt.show()
```

Loss train giảm và val tăng kéo dài có thể gợi ý overfitting, nhưng cần xem đồng thời metric validation. Loss giảm không chứng minh không có rò rỉ dữ liệu hoặc tổng quát hóa tốt.

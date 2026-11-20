# Demo app CaTE-X

Chạy từ thư mục project (đã cài requirements.txt):

```powershell
.\.venv\Scripts\python.exe demo_app/app.py
```

Mở http://127.0.0.1:8501. Đổi cổng bằng `--port 8502`. Giao diện chỉ lắng nghe localhost; đây là demo local, chưa phải server production/public. Dừng bằng Ctrl+C trong terminal. Không cần cài thêm framework giao diện.

## Hai chế độ kiểm tra

- **Kiểm tra thường:** tài liệu đi qua Bước 0 để chọn câu cần tìm nguồn trên web. Mỗi cặp câu với nguồn ứng viên sau đó vẫn chạy đủ 5 tầng.
- **Kiểm tra sâu:** tìm nguồn trên web cho mọi câu trong tài liệu; mỗi cặp nguồn cũng chạy đủ 5 tầng.
- **So sánh hai đoạn đã có nguồn:** cả hai chế độ đều chạy cùng pipeline 5 tầng. Khác biệt thường/sâu chỉ thay đổi phạm vi tìm kiếm khi đầu vào là tài liệu.

Tải TXT UTF-8 / DOCX / PDF có lớp văn bản hoặc dán nội dung. Báo cáo HTML, DOCX có highlight và JSON được lưu trong `output/demo/<job-id>/`. DOCX demo được tái tạo từ văn bản trích xuất, không giữ nguyên bố cục file gốc. PDF scan cần OCR bên ngoài. Tập train hiện chủ yếu tiếng Anh; chưa khẳng định chất lượng tiếng Việt.

Mỗi file tối đa 8 MB, mỗi tài liệu 40.000 ký tự và tối đa 200 câu. Giới hạn này giúp giữ thời gian/RAM trong mức phù hợp cho demo CPU. Lượt phân tích xử lý tuần tự.

## Bật pipeline đầy đủ

Checkpoint tầng 0, 3 và 4 được đọc từ `checkpoints/tang0/`, `checkpoints/tang3/`, `checkpoints/tang4/`. Tầng 1 và bộ fusion là các artifact riêng, không được tạo chỉ bằng cách chép checkpoint tầng 0/4. Trong workspace này, chạy các lệnh sau một lần để tạo chúng:

```powershell
.\.venv\Scripts\python.exe src\models\tang1_tfidf\filter.py --train data\splits\triplet_data_train.csv
.\.venv\Scripts\python.exe evaluation\fit_fusion.py --validation data\splits\pairwise_labeled_val.csv
```

Lệnh thứ hai tính điểm của các tầng trên tập validation và có thể mất thời gian trên CPU; không dùng tập test để fit fusion. Fusion lưu dấu chữ ký của vectorizer và các checkpoint nên cần fit lại sau khi thay những artifact này. Lần đầu chạy có thể cần tải model pretrained tầng 2.

Tạo `.env` từ `.env.example`, điền API key cho LLM và Google Search, rồi khởi động lại app. Khóa chỉ được đọc trong tiến trình local, giao diện chỉ báo đã cấu hình chứ không hiện giá trị. Cặp câu cần tầng 1–5, fusion và API LLM; tài liệu cần thêm bước 0 và Google Search. Mở **Các thành phần cần cho luồng đầy đủ** để xem chính xác thành phần nào còn thiếu. App bật chế độ tương ứng khi các điều kiện có đủ.

Cặp câu chạy nguyên `CaTEXPipeline`; tài liệu chạy nguyên `SingleFilePipeline`. Cả hai chế độ đều chạy đủ 5 tầng cho từng ứng viên. Trạng thái giao diện kiểm tra file và cấu hình, còn quota API, kết nối và tương thích model chỉ được xác nhận lúc chạy.

## Lưu kết quả

Mỗi lượt có thư mục `output/demo/<job-id>/`: file nhập và báo cáo. Thư mục output đã được ignore; nội dung còn trên máy sau khi dừng app, tự quản lý/xóa khi không cần. Các link tải từ UI dùng trong phiên server hiện tại; sau khởi động lại có thể lấy file trực tiếp từ thư mục output. Lưu cả checkpoint ở nơi khác để sao lưu; app không thay đổi hoặc train lại model.

CLI xử lý tài liệu cũ vẫn dùng được qua:

```bash
python src/pipeline/single_file_pipeline.py --input document.docx --output output/report.html
```

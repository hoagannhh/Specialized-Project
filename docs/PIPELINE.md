# Luồng vận hành hiện tại

## Hai chế độ, cùng kiến trúc 5 tầng

- `full`: bước 0 chấm điểm nhưng không loại câu; tìm tối đa 5 trang/câu, chọn tối đa 3 ứng viên mỗi kênh từ vựng/ngữ nghĩa trên một trang.
- `economy`: bỏ truy hồi cho câu có decision `skip` ở bước 0; tối đa 3 trang/câu và 1 ứng viên mỗi kênh. Câu bị bỏ qua mang trạng thái `skipped_budget`, không được tính là không paraphrase.
- Mỗi cặp ứng viên được chọn đều chạy tầng 1, 2, 3, 4 và gọi tầng 5. Không có early exit theo cosine. Nếu một tầng lỗi, kết quả được ghi là chưa hoàn tất; không thể bảo đảm API luôn thành công.
- API `predict(suspect, source)` nhận câu cần kiểm tra trước; nội bộ đảo thành `(source, suspect)` cho model để khớp cột `sentence_1=anchor` lúc train. LLM vẫn nhận đúng vai trò nguồn và câu cần kiểm tra.
- Tầng 1–4 cung cấp bốn đặc trưng cho logistic fusion. Tầng 5 chỉ giải thích, không thay đổi nhãn. Điểm đầu ra chưa được chứng minh calibration; không gọi là “xác suất đạo văn”.
- Web search trả nguồn đối chiếu, không chứng minh đó là tác giả/nguồn xuất bản đầu tiên. Các tầng so cặp nhận diện quan hệ paraphrase; riêng quan hệ này không chứng minh AI là người viết.

## Chuẩn bị và chạy

Chạy tại thư mục gốc repo, dùng Python môi trường đã cài requirements. Train bước 0, tầng 3, tầng 4 theo README trước. Placeholder checkpoint rỗng không dùng được.

```powershell
python src/models/tang1_tfidf/filter.py --train data/splits/triplet_data_train.csv
python evaluation/fit_fusion.py --validation data/splits/pairwise_labeled_val.csv
python src/pipeline/single_file_pipeline.py --input document.docx --output output/report.docx --mode full
python src/pipeline/single_file_pipeline.py --input document.docx --output output/report.html --mode economy
```

Fusion đọc `sentence_1,sentence_2,label,group_id,candidate_group_id` từ validation. Chia theo nguồn thành phần fit trọng số và phần chọn ngưỡng, bỏ cặp nối hai nhóm khác phần. Không fit trên test. Validation này đã dùng để chọn checkpoint nên không phải tập đánh giá độc lập; chất lượng cuối phải đo trên test. `--limit` chỉ phục vụ thử chạy, không dùng artifact từ mẫu nhỏ để công bố chất lượng. Nếu đổi TF-IDF hoặc checkpoint 3/4, phải fit lại fusion; chữ ký artifact được kiểm tra khi nạp.

Trong `.env` đặt `GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`, cùng `OPENAI_API_KEY` hoặc `ANTHROPIC_API_KEY`; dùng `--provider openai` hoặc `--provider anthropic`. Lệnh xử lý tài liệu thực tế gọi dịch vụ ngoài. Script fit fusion và đánh giá model không gọi LLM; lần đầu nạp backbone có thể cần tải model.

`demo_app/app.py` là điểm chạy CLI tương đương, chưa phải giao diện web.

## Đọc file, tìm nguồn và báo cáo

- TXT đọc UTF-8/UTF-8 BOM; không âm thầm bỏ ký tự sai encoding.
- DOCX đọc thân bài và bảng, lưu offset gốc; xuất DOCX giữ các thành phần tài liệu và định dạng run. Header/footer, footnote và chữ trong ảnh không thuộc phạm vi kiểm tra, được cảnh báo.
- PDF đọc lớp text, cảnh báo trang không đọc được; chưa có OCR. Xuất Word từ PDF không tái tạo bố cục PDF.
- Tách câu dùng heuristic; cần đánh giá riêng với viết tắt và ngôn ngữ dữ liệu thực tế. Câu ngắn và câu lặp vẫn giữ vị trí riêng.
- Trong trang nguồn, lấy hợp ứng viên theo từ vựng và ngữ nghĩa, gồm cửa sổ một/hai câu. Không loại paraphrase chỉ vì TF-IDF thấp. Cache tìm kiếm/trang/cặp lặp trong một lần chạy.
- Chỉ trạng thái `suspicious` được highlight. HTML giữ toàn bộ text và escape nội dung; DOCX highlight theo offset, không replace toàn chuỗi.
- Kèm JSON cùng tên để lưu các nguồn ứng viên, điểm bốn tầng, trace năm tầng và lỗi. Bảng chi tiết trong báo cáo hiển thị nguồn tốt nhất; các nguồn còn lại nằm trong JSON.

Các trạng thái không được gộp thành âm tính: `skipped_budget`, `no_source`, `search_failed`, `extraction_failed`, `retrieval_failed`, `no_candidate`, `model_failed`, `incomplete`. `checked_negative` chỉ có nghĩa chưa phát hiện quan hệ trên những ứng viên đã xử lý. Lỗi tầng 5 có `explanation_status=failed` và `complete=false`, vẫn giữ nhãn định lượng.

## Đánh giá

```powershell
python evaluation/evaluate_tang0.py --input data/splits/tang0_test.csv
python evaluation/evaluate_tang3.py --input data/splits/triplet_data_test.csv
python evaluation/evaluate_tang4.py --input data/splits/pairwise_labeled_test.csv
python evaluation/ablation_study.py --validation data/splits/pairwise_labeled_val.csv --test data/splits/pairwise_labeled_test.csv
python evaluation/evaluate_full_pipeline.py --report output/report.json --labels sentence_labels.csv
python -X utf8 -m unittest discover -s tests -v
```

`cross_dataset_eval.py --input <PADBen-test.csv>` đánh giá bước 0 với nhãn đã chuẩn hóa; Task 4 chỉ có một lớp nên AUC/AP trả null, cần đọc recall và kết quả theo `original_task_label`. Không chọn lại ngưỡng trên test. Tầng 3 báo tỷ lệ similarity(anchor,positive) lớn hơn similarity(anchor,negative), không biến cosine thành xác suất.

Đánh giá tài liệu dùng CSV `sentence_id,label` với ID từ JSON, đủ mọi câu. Báo riêng coverage và các câu chưa quyết định; recall trên toàn bộ câu dương tính tính cả những câu bị bỏ sót. Đây chưa đo riêng Recall@K truy hồi nguồn; muốn đo cần nhãn URL/đoạn nguồn chuẩn. Ablation chạy offline, không thay đổi quy tắc đủ 5 tầng của hệ thống vận hành.

Các kiểm thử dùng model nhỏ hoặc dịch vụ giả lập. Chúng kiểm tra logic, không chứng minh độ chính xác trên dữ liệu thực tế hay độ sẵn sàng của API.

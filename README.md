# Specialized-Project

> Hướng dẫn vận hành hiện tại: [docs/PIPELINE.md](docs/PIPELINE.md).
> Cây thư mục đầu trang là phác thảo ban đầu; tên split và lệnh chạy thực tế xem bên dưới.
plagiarism-detection-catex/
│
├── data/
│   ├── raw/                              # Dataset gốc, không sửa
│   │   ├── humarin_chatgpt_paraphrases.csv     # Dataset chính — dùng để TRAIN
│   │   └── padben_sentence_pair_task1.json     # Dataset phụ — dùng để CROSS-TEST
│   │
│   ├── processed/                        # Dữ liệu đã xử lý, chia theo mục đích dùng
│   │   ├── triplet_data.csv              # (Anchor, Positive, Negative) — cho Tầng 3
│   │   ├── pairwise_labeled.csv          # (Câu 1, Câu 2, Nhãn) — cho Tầng 4
│   │   └── single_labeled.csv            # (Câu, Nhãn 0/1) — cho Bước 0
│   │
│   ├── splits/                           # Chia từ dataset CHÍNH (humarin)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── cross_eval/                       # Dữ liệu PADBen, chỉ dùng để đánh giá
│       └── padben_test_processed.csv     # Đã convert format khớp với Tầng 3/4
│
├── src/
│   ├── data_prep/
│   │   ├── download_dataset.py           # Tải humarin/chatgpt-paraphrases
│   │   ├── download_padben.py            # tải PADBen từ HuggingFace
│   │   ├── build_triplet.py              # Tạo dữ liệu Triplet + hard negative mining
│   │   ├── build_pairwise.py             # Tạo dữ liệu Cặp-có-nhãn cho Tầng 4
│   │   ├── build_single_label.py         # Tạo dữ liệu đơn-câu cho Bước 0
│   │   ├── convert_padben_format.py      # convert PADBen về cùng format với Tầng 3/4
│   │   └── split_dataset.py              # Chia train/val/test (chỉ trên humarin)
│   │
│   ├── models/
│   │   ├── tang0_style_classifier/
│   │   │   ├── train.py
│   │   │   ├── model.py
│   │   │   └── inference.py
│   │   │
│   │   ├── tang1_tfidf/
│   │   │   └── filter.py
│   │   │
│   │   ├── tang2_sbert_pretrained/
│   │   │   └── encode.py
│   │   │
│   │   ├── tang3_triplet_encoder/
│   │   │   ├── train.py
│   │   │   ├── model.py
│   │   │   └── inference.py
│   │   │
│   │   ├── tang4_cross_encoder/
│   │   │   ├── train.py
│   │   │   ├── model.py
│   │   │   └── inference.py
│   │   │
│   │   └── tang5_llm_judge/
│   │       ├── prompts.py
│   │       └── call_api.py
│   │
│   ├── retrieval/
│   │   ├── web_search.py
│   │   └── extract_page_text.py
│   │
│   ├── document_processing/
│   │   ├── file_reader.py
│   │   ├── sentence_splitter.py
│   │   └── highlight_exporter.py
│   │
│   ├── pipeline/
│   │   ├── cascade_pipeline.py
│   │   ├── single_file_pipeline.py
│   │   └── calibration.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── evaluation/
│   ├── evaluate_tang0.py
│   ├── evaluate_tang3.py
│   ├── evaluate_tang4.py
│   ├── evaluate_full_pipeline.py
│   ├── ablation_study.py
│   ├── cross_dataset_eval.py              # MỚI — đánh giá model (train trên humarin) 
│   │                                       #        trên tập PADBen, đo độ generalize
│   └── results/
│       ├── in_domain_results.csv          # Kết quả trên test-set humarin
│       └── cross_domain_results.csv       # MỚI — kết quả trên PADBen
│
├── checkpoints/
│   ├── tang0_classifier.pt
│   ├── tang3_encoder.pt
│   └── tang4_cross_encoder.pt
│
├── demo_app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 01b_eda_padben.ipynb               # MỚI — khám phá riêng dữ liệu PADBen
│   ├── 02_train_tang0.ipynb
│   ├── 03_train_tang3.ipynb
│   ├── 04_train_tang4.ipynb
│   └── 05_cross_dataset_eval.ipynb        # MỚI — chạy đánh giá chéo, vẽ biểu đồ so sánh
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

# CaTE-X — Hệ thống phát hiện đạo văn dạng diễn giải bởi AI

**CaTE-X** (*Cascade Triplet Encoder with eXplanation*) là hệ thống phát hiện văn bản bị diễn đạt lại (paraphrase) bởi AI, kết hợp pipeline đa tầng (TF-IDF → SBERT pretrained → SBERT fine-tuned/Triplet Loss → Cross-Encoder → LLM-as-a-Judge) với khả năng mở rộng xử lý trực tiếp 1 file văn bản (tự động tìm nguồn qua Web Search).

## Kiến trúc tổng quan

```
Model lõi CaTE-X (so sánh 2 câu cho sẵn)
  Tầng 1 — TF-IDF + Cosine Similarity      (điểm từ vựng, fit corpus train)
  Tầng 2 — Sentence-BERT pretrained         (so sánh ngữ nghĩa sơ bộ, không cần train)
  Tầng 3 — Sentence-BERT fine-tuned         (mô hình chính, Triplet Loss)
  Tầng 4 — Cross-Encoder                    (tinh chỉnh, Binary Cross-Entropy)
  Tầng 5 — LLM-as-a-Judge                   (giải thích, không cần train)

Hệ thống mở rộng (input 1 file)
  Bước 0 — bộ phân loại đơn-câu             (lọc câu đáng nghi, Cross-Entropy)
  Web Search — tự động tìm nguồn nghi ngờ
  -> đưa cặp (câu nghi ngờ, nguồn tìm được) qua model lõi ở trên
```

## Cấu trúc thư mục

```
├── data/                   # Dataset (raw, processed, splits, cross_eval)
├── notebooks/              # Notebook EDA, thử nghiệm
├── src/
│   ├── data_prep/          # Script chuẩn bị dữ liệu cho từng tầng
│   ├── models/              # Định nghĩa + huấn luyện + suy luận từng tầng
│   │   ├── tang0_style_classifier/
│   │   ├── tang1_tfidf/
│   │   ├── tang2_sbert_pretrained/
│   │   ├── tang3_triplet_encoder/
│   │   ├── tang4_cross_encoder/
│   │   └── tang5_llm_judge/
│   ├── retrieval/           # Web Search + trích xuất nội dung trang web
│   ├── pipeline/            # Ghép toàn bộ hệ thống
│   │   ├── cascade_pipeline.py        # Model lõi (5 tầng, nhận 2 câu)
│   │   ├── single_file_pipeline.py    # Hệ thống mở rộng (nhận 1 file)
│   │   └── calibration.py             # Hiệu chỉnh similarity -> xác suất
│   └── utils/                # Cấu hình, logging dùng chung
├── checkpoints/              # Model đã train (không commit lên Git)
└── evaluation/                # Script đánh giá, kết quả EDA
```

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env   # rồi điền API key thật vào .env
```

## Quy trình sử dụng

### 1. Chuẩn bị dữ liệu

```bash
# Chỉ tải nếu data/raw chưa có dữ liệu:
python src/data_prep/download_dataset.py
python src/data_prep/download_padben.py

# Một lệnh tạo cả ba bộ train/val/test, dùng chung nhóm nguồn:
python src/data_prep/prepare_datasets.py --strategy mixed
```

Luồng mới gom câu gốc, paraphrase và các câu trùng (NFC + khoảng trắng) thành
connected components, sau đó chia nhóm 70/15/15 trước khi chọn negative.
Tỷ lệ theo số dòng là xấp xỉ. Cả ba model dùng chung phép chia này.

- Bước 0: `data/splits/tang0_{train,val,test}.csv`, nhãn 1 = AI-generated **hoặc**
  AI-paraphrased (AI-touched), nhãn 0 = human theo nhãn/giả định nguồn dữ liệu.
  Dùng Humarin + PADBen Task 1/2/3; Task 3 có cả hai phía mang nhãn AI.
  Câu có nhãn mâu thuẫn bị loại khỏi bộ đơn câu và ghi vào `label_conflicts.csv`.
- Tầng 3: `triplet_data_{train,val,test}.csv`; negative chỉ lấy trong cùng split,
  khác toàn bộ nhóm nguồn. `mixed` trộn random với lexical-hard TF-IDF trong block
  (mặc định 512); đây là negative heuristic, cần rà soát mẫu gần nghĩa trước nghiên cứu.
- Tầng 4: `pairwise_labeled_{train,val,test}.csv`; tái sử dụng positive/negative
  của tầng 3. Nhãn 1 là quan hệ paraphrase, không phải nhãn AI hay kết luận đạo văn.
- `padben_task4_test.csv`, `padben_task5_test.csv`: chỉ chứa nhóm thuộc test.
  Giữ `original_task_label` để đánh giá riêng độ sâu, còn `label` là AI-touched.
  Task 4 chỉ có positive theo nhãn AI, nên không dùng riêng nó để đo specificity.
- `source_manifest.csv`, `data_report.json`, `audit_report.json` ghi phép chia và
  kiểm tra. TF-IDF mining chỉ fit trên train. Các bản split cũ được lưu ở
  `data/splits_legacy_*` sau khi bản mới tạo và kiểm tra thành công.

Không chạy lại chuỗi convert → merge → split cũ: các CLI đó đã bị chặn để tránh
sinh lại nhãn sai/rò rỉ. File trung gian cũ trong `data/processed` không được luồng
mới đọc. Hai bộ tầng 3/4 hiện dùng Humarin; chưa tự suy ra cặp paraphrase từ PADBen.
Nếu đã train checkpoint bằng split cũ, cần train lại để báo cáo kết quả trên split mới.

Kiểm tra chất lượng dữ liệu trước khi train (**bắt buộc**, xem `notebooks/01_eda.ipynb`).

### 2. Huấn luyện (cần GPU — khuyến nghị Google Colab/Kaggle)

```bash
python src/models/tang0_style_classifier/train.py --train data/splits/tang0_train.csv --val data/splits/tang0_val.csv
python src/models/tang3_triplet_encoder/train.py --train data/splits/triplet_data_train.csv --val data/splits/triplet_data_val.csv
python src/models/tang4_cross_encoder/train.py --train data/splits/pairwise_labeled_train.csv --val data/splits/pairwise_labeled_val.csv
```

### Hợp đồng model sau khi sửa

- Tầng 2 và tầng 3 mặc định cùng khởi tạo từ
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  Tầng 2 giữ nguyên trọng số; tầng 3 fine-tune bằng cosine Triplet Loss.
- Tầng 1–3 trả cosine score và tiếp tục sang tầng sau. Không diễn giải cosine
  thành xác suất, không kết luận đạo văn hoặc dừng sớm trong các model này.
- Bước 0 dùng nhãn AI-touched; tầng 4 dùng nhãn quan hệ paraphrase. Softmax/sigmoid
  là xác suất dự đoán **chưa hiệu chỉnh**, không phải kết luận đạo văn.
- Train 0/3/4 có `--seed` (mặc định 42), gradient clipping, lưu checkpoint tốt nhất.
  Ngưỡng 0/4 tối ưu F1 trên validation và được cố định khi inference/test.
  Full kiểm tra mọi câu; economy dùng ngưỡng bước 0 lưu trong checkpoint.
  Ngưỡng tối ưu F1 chưa phải ngưỡng bảo đảm recall cao.
- Mỗi thư mục checkpoint có `tangN_best.pt` (weights + metadata),
  `model_config.json`, tokenizer. Metadata gồm cấu hình encoder, backbone,
  max_length, threshold, seed và metric/epoch. Nạp inference không cần tải lại
  backbone; cấu hình lấy từ checkpoint, cache phân biệt đường dẫn/phiên bản file.
- Checkpoint cũ chỉ có state_dict không tự đoán cấu hình: cần train lại bằng
  script mới. Các file placeholder rỗng trong `checkpoints/` không phải model.
- Các API batch có `batch_size`, xử lý input rỗng nhất quán. Câu rỗng báo lỗi,
  không bị gán thành kết quả âm tính.
- Tầng 5 chỉ trả giải thích và trích cụm nguồn → câu nghi vấn, không đổi nhãn
  hoặc sinh confidence. JSON sai schema/cụm trích không tồn tại bị đánh dấu lỗi.
  API tương thích `tang5_judge(sentence_a, sentence_b, ...)` nhận **câu nghi vấn
  trước, câu nguồn sau**, đúng thứ tự của pipeline hiện có.

Fit riêng TF-IDF tầng 1 (vectorizer mining ở data prep không phải artifact này):

```bash
python src/models/tang1_tfidf/filter.py --train data/splits/triplet_data_train.csv
```

Kiểm thử offline với Transformer nhỏ tạo tại máy (không tải model, không gọi API):

```bash
python -X utf8 -m unittest discover -s tests -p test_models.py -v
```

Pipeline đã nối đủ 5 tầng trong cả hai chế độ. Xem hướng dẫn chạy và đánh giá tại
[docs/PIPELINE.md](docs/PIPELINE.md). Trước khi chạy cần fit thêm bộ kết hợp điểm:

```bash
python evaluation/fit_fusion.py --validation data/splits/pairwise_labeled_val.csv
```

### 3. Chạy thử model lõi (so sánh 2 câu)

```python
from src.pipeline.cascade_pipeline import CaTEXPipeline

pipeline = CaTEXPipeline()
result = pipeline.predict(
    "The economy grew rapidly last year.",
    "Last year, the economy expanded at a fast pace.",
)
print(result["label"], result["probability"])
```

### 4. Chạy hệ thống mở rộng (kiểm tra 1 file)

```bash
python src/pipeline/single_file_pipeline.py \
    --input path/to/document.docx \
    --output result_highlighted.html
```

## Lưu ý quan trọng

- Tầng 1, 2, 5 **không cần huấn luyện**. Chỉ Tầng 0, 3, 4 cần train.
- `PADBen` (Task 1,2,3) dùng để tăng cường dữ liệu train Tầng 0; `PADBen` (Task 4,5) giữ riêng để đánh giá cross-dataset.
- Tầng 5 bắt buộc trong pipeline vận hành; cần API key trong `.env`. Lỗi API được ghi nhận, không sửa nhãn của bộ kết hợp điểm.
- Web Search dùng Google Custom Search API. Hạn mức và chi phí theo tài khoản; code phân biệt lỗi API với không có kết quả, chưa có bộ đếm hạn mức toàn tài khoản.

## Dataset

- Chính: [`humarin/chatgpt-paraphrases`](https://huggingface.co/datasets/humarin/chatgpt-paraphrases)
- Phụ (cross-eval + tăng cường Tầng 0): [`JonathanZha/PADBen`](https://huggingface.co/datasets/JonathanZha/PADBen)

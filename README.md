# Specialized-Project
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
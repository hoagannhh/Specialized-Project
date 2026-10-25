# CHƯƠNG 2
# PHƯƠNG PHÁP ĐỀ XUẤT CaTE X

Chương này trình bày phương pháp CaTE-X (Cascade Triplet Encoder with eXplanation) nhằm phát hiện các đoạn văn bản nghi vấn diễn đạt lại từ một nguồn ứng viên, tập trung vào trường hợp AI-paraphrase. Mô hình chính là bi-encoder được huấn luyện bằng Triplet Loss với chiến lược khai thác mẫu âm khó. Các thành phần còn lại hỗ trợ sàng lọc, đối chiếu chi tiết và giải thích bằng chứng. Hệ thống được thiết kế để tiếp nhận một tệp văn bản, tìm nguồn đối chiếu và xuất báo cáo có đánh dấu các đoạn nghi vấn.

Nội dung chương xác định thiết kế phương pháp và quy trình thực nghiệm. Các cấu hình khởi tạo được nêu để triển khai và kiểm chứng; hiệu quả, ngưỡng cuối cùng, quy mô dữ liệu sau xử lý và chi phí vận hành được xác định qua thực nghiệm, không được xem là kết quả đã đạt được.

## 2.1 Kiến trúc tổng thể của hệ thống
### 2.1.1 Phát biểu bài toán và phạm vi nghiên cứu

Đầu vào của mô hình lõi là cặp văn bản (A, B), trong đó A là đoạn nguồn ứng viên và B là đoạn cần kiểm tra. Mô hình đánh giá liệu hai đoạn có quan hệ diễn đạt lại cùng nội dung hay không. Trong dữ liệu huấn luyện, mẫu dương được xây dựng từ nguồn và bản paraphrase tương ứng; mẫu âm gồm các cặp không diễn đạt lại cùng nội dung, kể cả những cặp cùng chủ đề. Đầu ra là nhãn dự đoán quan hệ nội dung, điểm của mô hình và bằng chứng đối chiếu khi có.

Quan hệ paraphrase, dấu hiệu sử dụng AI và hành vi đạo văn là ba khái niệm khác nhau. Một cặp tương đồng ngữ nghĩa không tự chứng minh việc sử dụng nội dung thiếu dẫn nguồn; mô hình đối chiếu cũng không xác định chắc chắn công cụ đã tạo ra văn bản. Do đó, hệ thống cung cấp cảnh báo nghi vấn và hỗ trợ người dùng xem xét nguồn, ngữ cảnh và thông tin trích dẫn. Điểm phát hiện dấu hiệu AI được trình bày riêng với điểm đối chiếu nội dung.

Nghiên cứu có hai hình thức đầu vào. Hình thức so sánh cặp sử dụng hai đoạn cho sẵn để huấn luyện và đánh giá mô hình lõi. Hình thức kiểm tra tệp sử dụng một tài liệu TXT, DOCX hoặc PDF, bổ sung quá trình trích xuất, tìm nguồn và tổng hợp báo cáo. Trong hình thức kiểm tra tệp, người dùng lựa chọn chế độ đầy đủ hoặc tiết kiệm. Hai chế độ này khác nhau về mức độ bao phủ và cơ chế phân bổ tài nguyên, không làm thay đổi định nghĩa nhãn học.

Phạm vi thực nghiệm ban đầu tập trung vào văn bản tiếng Anh theo các bộ dữ liệu được lựa chọn. Việc sử dụng checkpoint đa ngôn ngữ không đồng nghĩa hệ thống đã được kiểm chứng trên tiếng Việt. Khả năng xử lý tiếng Việt cần một tập đánh giá riêng trước khi công bố chất lượng.

### 2.1.2 Luồng xử lý tài liệu và hai chế độ kiểm tra

Hệ thống trích xuất văn bản, lưu ánh xạ vị trí, tách câu và tạo đơn vị đối chiếu có ngữ cảnh. Bộ phân loại bước 0 ước lượng dấu hiệu văn bản do AI sinh hoặc biến đổi. Tiếp theo, hệ thống truy xuất nguồn, tạo các đoạn ứng viên, áp dụng pipeline năm tầng và tổng hợp kết quả về tài liệu gốc. Luồng tổng thể được mô tả trong Hình 2.1.

@figure overall|Hình 2.1 Luồng xử lý tài liệu và hai chế độ kiểm tra

@table Bảng 2.1 Hai chế độ kiểm tra tệp
Tiêu chí|Chế độ đầy đủ|Chế độ tiết kiệm
Phạm vi tìm nguồn|Mọi đơn vị đủ thông tin để đối chiếu|Ưu tiên theo điểm bước 0 và ngân sách
Vai trò bước 0|Chỉ số bổ sung và thứ tự xử lý|Điều phối các đơn vị được tìm nguồn
Lọc tại tầng 1 và 2|Không loại cứng theo hai điểm sơ bộ|Có thể loại khi cả hai điểm đều thấp
Phần chưa xử lý|Ghi rõ nếu lỗi hoặc vượt giới hạn vận hành|Ghi rõ chưa kiểm tra nguồn
Mục tiêu|Ưu tiên độ bao phủ trong phạm vi truy xuất|Giảm chi phí với mức bỏ sót được đo lường
@end

Trong chế độ đầy đủ, điểm bước 0 không được dùng để loại câu khỏi quá trình tìm nguồn. Tuy nhiên, đầy đủ không có nghĩa là bao phủ toàn bộ Internet: kết quả vẫn phụ thuộc số truy vấn, số nguồn ứng viên, khả năng truy cập trang và chất lượng trích xuất. Những giới hạn này phải được lưu trong cấu hình và báo cáo.

Trong chế độ tiết kiệm, các đơn vị có điểm bước 0 cao được ưu tiên. Ngưỡng và ngân sách được xác định trên validation nhằm giảm số truy vấn trong khi duy trì recall mục tiêu. Khi ngân sách đã hết, phần còn lại mang trạng thái chưa kiểm tra nguồn, không được gán nhãn không đạo văn. Tỷ lệ nội dung thực sự được kiểm tra được báo cáo cùng kết quả.

### 2.1.3 Trích xuất văn bản và chuẩn bị nguồn đối chiếu

Hệ thống duy trì hai biểu diễn: văn bản gốc để hiển thị và văn bản chuẩn hóa để phân tích. Mỗi câu có mã định danh, mã đoạn, khoảng ký tự và vị trí tương ứng trong tài liệu. Với DOCX, vị trí được ánh xạ tới đoạn và các run; với PDF, hệ thống lưu trang và vùng chữ nếu công cụ trích xuất hỗ trợ. Các phép chuẩn hóa khoảng trắng phải bảo toàn ánh xạ này để tránh đánh dấu sai vị trí.

Đơn vị hiển thị có thể là một câu, nhưng đơn vị phân tích gồm câu trung tâm và ngữ cảnh lân cận phù hợp. Cửa sổ ngữ cảnh được giới hạn theo độ dài đầu vào của mô hình. Việc mở rộng cửa sổ không đồng nghĩa mọi câu trong cửa sổ đều bị đánh dấu: vị trí nghi vấn cần được xác định lại trên phần nội dung có bằng chứng. Các câu quá ngắn, tiêu đề hoặc phần không trích xuất được được ghi nhận riêng theo quy tắc tiền xử lý, không bị bỏ qua âm thầm.

Đối với PDF dạng ảnh, cần thành phần OCR trước khi phân tích. Nếu OCR chưa được triển khai hoặc chất lượng chữ không đủ, hệ thống thông báo phần không đọc được thay vì coi tài liệu đã được kiểm tra đầy đủ. Phần tài liệu tham khảo, câu trích dẫn trực tiếp và ký hiệu dẫn nguồn được nhận diện để hỗ trợ diễn giải kết quả, không mặc định bị loại khỏi mọi phép đối chiếu.

Truy vấn tìm nguồn được tạo từ cụm từ đặc trưng, thuật ngữ, thực thể và ngữ cảnh. Có thể dùng câu đầy đủ như một biến thể truy vấn, nhưng không phụ thuộc duy nhất vào so khớp nguyên câu. Nhà cung cấp tìm kiếm, số truy vấn và số kết quả tối đa là tham số cấu hình; chính sách giá hoặc hạn mức dịch vụ không được dùng như một thuộc tính cố định của phương pháp.

Với mỗi URL ứng viên, hệ thống tải và trích xuất nội dung chính, loại trang trùng, chia văn bản thành câu hoặc đoạn có chồng lấn. Danh sách đoạn ứng viên được lấy bằng hợp các kết quả từ truy hồi từ vựng và embedding, sau đó loại trùng. Không chỉ giữ một câu có TF-IDF cao nhất vì paraphrase có thể thay đổi phần lớn từ ngữ hoặc gộp nhiều câu nguồn.

Hệ thống lưu URL, tiêu đề, thời điểm truy xuất, đoạn trích và mã nguồn nội bộ. Trường hợp chỉ có snippet, không tải được trang hoặc không tìm thấy ứng viên được ghi thành các trạng thái riêng. Một URL phù hợp được gọi là nguồn đối chiếu; hệ thống không mặc định đó là nơi xuất bản đầu tiên của nội dung.

### 2.1.4 Pipeline năm tầng và quy tắc chuyển tiếp

CaTE-X tổ chức xử lý theo các tầng có nhiệm vụ khác nhau. Tầng 1 và 2 cung cấp tín hiệu sơ bộ; tầng 3 là mô hình học chính; tầng 4 xử lý cặp khó; tầng 5 sinh giải thích từ bằng chứng. Các mô hình nhận văn bản đầu vào tương ứng, không nối trực tiếp các hàm mất mát vào nhau. Trong suy luận, không tính Triplet Loss hoặc Cross-Entropy.

@table Bảng 2.2 Chức năng và quy tắc chuyển tiếp của các tầng
Tầng|Phương pháp|Chức năng và quy tắc
1|TF-IDF và cosine|Tính điểm từ vựng; không loại ngay tại tầng này
2|SBERT pretrained|Tính điểm ngữ nghĩa; chỉ chế độ tiết kiệm mới cho phép loại khi cả hai tín hiệu thấp
3|Bi-Encoder với Triplet Loss|Chấm quan hệ nội dung; chuyển vùng mập mờ sang tầng 4
4|Cross-Encoder|Đánh giá chi tiết cặp khó; xuất điểm phân loại và trạng thái quyết định
5|Giải thích bằng LLM|Giải thích cặp được đưa vào báo cáo; không thay đổi nhãn hoặc tự tạo nguồn
@end

Gọi s_lex và s_pre lần lượt là điểm tầng 1 và 2. Trong chế độ tiết kiệm, một cặp chỉ được lọc tại giai đoạn này nếu s_lex < τ_lex đồng thời s_pre < τ_pre. Hai ngưỡng được lựa chọn chung trên validation dưới ràng buộc recall. Trong chế độ đầy đủ, các cặp đã được lấy vào danh sách ứng viên vẫn chuyển tới tầng 3. Việc loại một cặp không loại cả câu nghi vấn nếu câu đó còn nguồn ứng viên khác.

Gọi s_tri là điểm tầng 3 và τ_L < τ_H là hai ngưỡng định tuyến. Nếu s_tri < τ_L, cặp được ghi nhận chưa đủ bằng chứng tương ứng; nếu s_tri ≥ τ_H, cặp được ghi nhận nghi vấn có bằng chứng mô hình; các trường hợp còn lại chuyển sang tầng 4. Nhánh thấp phải được kiểm tra tỷ lệ bỏ sót, còn nhánh cao phải được kiểm tra precision. Nếu validation chưa hỗ trợ độ tin cậy cần thiết, tắt nhánh tương ứng và chuyển các cặp đó sang tầng 4.

Tầng 4 có thể dùng hai ngưỡng quyết định để giữ một vùng cần xem xét. Trong đánh giá phân loại nhị phân, sử dụng một ngưỡng đã cố định trên validation; trong báo cáo người dùng, giữ trạng thái chưa chắc chắn khi cần. Hai cách trình bày phải được phân biệt để không loại các ca khó khỏi phép tính chất lượng mà không công bố.

@figure cascade|Hình 2.2 Quy tắc chuyển tiếp trong mô hình lõi CaTE X

Không cộng hoặc lấy giá trị lớn nhất trực tiếp giữa cosine và điểm sigmoid. Khi cần xếp hạng các nguồn cuối cùng, hệ thống chấm lại danh sách rút gọn bằng cùng một mô hình, chẳng hạn Cross-Encoder, và ghi nhận chi phí bổ sung. Điểm AI, điểm tương đồng và điểm phân loại được lưu tách biệt. Màu đánh dấu dựa trên trạng thái kết luận cùng bằng chứng, không dựa trên một thang phần trăm chung cho mọi tầng.

## 2.2 Dữ liệu và tiền xử lý
### 2.2.1 Bộ dữ liệu và định nghĩa nhãn

Dữ liệu chính được khai thác từ humarin/chatgpt-paraphrases, gồm trường văn bản gốc và danh sách paraphrase, với các nguồn như Quora, SQuAD và CNN [1]. Mỗi nhóm nguồn gốc và các biến thể tương ứng là một đơn vị quản lý dữ liệu. Nhãn dương của mô hình đối chiếu biểu thị quan hệ paraphrase theo nguồn dữ liệu; nhãn này không tự biểu thị hành vi thiếu trích dẫn. Số lượng bản ghi cuối cùng được thống kê lại sau khi cố định phiên bản dữ liệu, làm sạch và chia tập.

Đối với bước 0, văn bản gốc được dùng làm lớp tham chiếu người viết theo giả định dữ liệu; các bản do AI sinh hoặc biến đổi được dùng làm lớp dương. Giả định nguồn gốc văn bản cần được ghi nhận vì nhãn không phải chứng cứ pháp y về tác giả. Các mẫu trùng nội dung nhưng mang nhãn mâu thuẫn phải được loại hoặc đưa vào tập kiểm tra thủ công, không giữ đồng thời như các mẫu độc lập.

PADBen được dùng làm nguồn bổ sung để khảo sát độ bền trước paraphrase và biến đổi nhiều vòng [2]. Trước khi sử dụng, cần cố định revision, cấu hình dữ liệu và bảng ánh xạ nhãn theo loại văn bản. Không sao chép trực tiếp nhãn 0/1 giữa các task vì ý nghĩa lớp có thể khác nhau. Dữ liệu human-paraphrase, AI-generated và AI-paraphrase được lưu với loại nguồn riêng để đánh giá từng nhóm.

Thiết kế ưu tiên giữ PADBen làm benchmark bên ngoài cho bước 0. Nếu thực hiện thêm thí nghiệm tăng cường bằng một phần PADBen, các nhóm có chung nguồn với tập đánh giá phải được loại khỏi phần huấn luyện và validation. Khi đó, kết quả được gọi là đánh giá trên nhóm nguồn hoặc dạng biến đổi chưa thấy, thay vì khẳng định toàn bộ PADBen là dataset hoàn toàn chưa tham gia huấn luyện.

Các tập một câu và nhãn nguồn gốc AI phù hợp với bước 0, nhưng không tự tạo thành benchmark cho mô hình đối chiếu. Để đánh giá tầng 3 và 4 trên dữ liệu ngoài miền, cần có cặp nguồn–paraphrase được xác minh và các cặp âm tương ứng. Không dùng nhãn phân loại AI đơn câu thay cho nhãn quan hệ giữa hai đoạn.

### 2.2.2 Chia tập trước khi xây dựng mẫu học

Quy trình chống rò rỉ bắt đầu từ dữ liệu nhóm. Trước hết, gán group_id cho nguồn và mọi biến thể đã biết, kiểm tra trùng và gần trùng giữa các nhóm, sau đó chia train, validation và test theo tỷ lệ mục tiêu 70/15/15. Nếu dữ liệu có nguồn tài liệu cấp cao hơn câu, ưu tiên chia ở cấp tài liệu đó. Tỷ lệ thực tế theo số mẫu có thể khác tỷ lệ nhóm và được công bố sau khi chia.

Chỉ sau bước chia nhóm mới tạo triplet, cặp phân loại và dữ liệu đơn câu trong từng tập. Negative của train chỉ được lấy từ train; quy tắc tương tự áp dụng cho validation và test. Mọi vai trò anchor, positive, negative và hai vế của cặp đều được kiểm tra nguồn gốc nhóm. Kiểm tra chỉ riêng anchor hoặc sentence_1 là không đủ để ngăn nội dung test xuất hiện làm negative của train.

@figure split|Hình 2.3 Chia nhóm nguồn trước khi xây dựng dữ liệu cho các mô hình

Một bản kê chia tập được dùng chung cho bước 0, tầng 3, tầng 4 và các baseline. Bản kê lưu ID nhóm, tập được gán, seed và phiên bản xử lý. TF-IDF của tầng 1 chỉ fit trên train. Các mô hình phục vụ khai thác negative không được fine-tune trên test. Việc khai thác negative trong test chỉ sử dụng pool test theo một quy tắc được cố định trước, không thay đổi để ưu ái mô hình đang đánh giá.

Validation được tách theo nhóm thành phần chọn mô hình và phần chọn ngưỡng khi số lượng dữ liệu cho phép. Nếu cần hiệu chỉnh xác suất, sử dụng thêm phần calibration độc lập hoặc quy trình dự đoán ngoài fold. Test chỉ được dùng để báo cáo cuối cùng sau khi cố định mô hình, ngưỡng và quy tắc truy xuất.

### 2.2.3 Xây dựng dữ liệu cho từng thành phần

@table Bảng 2.3 Định dạng dữ liệu và ý nghĩa nhãn
Thành phần|Định dạng|Quy tắc xây dựng
Bước 0|(x, y_AI)|0 là lớp tham chiếu người viết; 1 là AI sinh hoặc biến đổi theo metadata
Tầng 3|(a, p, n)|a là nguồn; p là paraphrase cùng nguồn; n không tương đương nội dung
Tầng 4|(A, B, y_pair)|1 là cặp paraphrase; 0 là cặp không tương đương nội dung
Đánh giá tệp|Tài liệu và vùng nguồn đã gán nhãn|Lưu đoạn nghi vấn, đoạn nguồn và vị trí tương ứng
@end

Với tầng 3, mỗi anchor kết hợp với một positive cùng nhóm và một negative thuộc nhóm khác trong cùng split. Loại trừ toàn bộ paraphrase cùng nhóm trước khi khai thác negative. Khác group_id là điều kiện cần nhưng không đủ: hai nhóm có thể chứa diễn đạt tương đương. Những ứng viên nghi ngờ là false negative được kiểm tra hoặc loại khỏi pool khai thác.

Với tầng 4, mẫu âm gồm cả negative ngẫu nhiên và negative khó đã kiểm tra. Tỷ lệ phối hợp được chọn trên validation. Cách này giúp dữ liệu huấn luyện gần hơn với các cặp mập mờ mà tầng 3 chuyển tới. Ngoài tập đánh giá cặp chung, tầng 4 được đánh giá riêng trên tập cặp thực sự đi tới nó theo quy tắc định tuyến đã cố định.

Với bước 0, loại trùng ở mức văn bản và giữ metadata gồm group_id, nguồn, loại biến đổi và số vòng paraphrase nếu có. Nhiều paraphrase từ cùng nguồn không được xem như nhiều nguồn độc lập khi chia tập hoặc ước lượng độ bất định. Việc có nhiều mẫu dương hơn mẫu âm được xử lý trong huấn luyện và đánh giá, không làm thay đổi định nghĩa nhãn.

### 2.2.4 Chuẩn hóa và xử lý mất cân bằng

Tiền xử lý loại giá trị rỗng, bản ghi lỗi và các cặp trùng hoàn toàn không phù hợp mục tiêu paraphrase. Các cặp sao chép nguyên văn có thể được lưu riêng cho thí nghiệm kiểm tra khả năng phát hiện trùng lặp. Văn bản đưa vào Transformer giữ dấu câu và chữ hoa theo yêu cầu tokenizer; TF-IDF dùng quy tắc chuẩn hóa riêng. Văn bản nguồn và bản chuẩn hóa được lưu tách biệt.

Độ dài đầu vào phải được chọn sau khi khảo sát phân bố token. Cấu hình khởi tạo sử dụng tối đa 128 token cho câu đơn và 256 token cho cặp. Với đoạn dài, ưu tiên chia thành cửa sổ có chồng lấn thay vì cắt bỏ phần cuối mà không ghi nhận. Báo cáo thực nghiệm cần nêu tỷ lệ mẫu bị cắt hoặc chia cửa sổ.

Bước 0 sử dụng Cross-Entropy có trọng số lớp, tính chỉ từ tập train. Với N mẫu, N_c mẫu thuộc lớp c và hai lớp, trọng số khởi tạo là w_c = N/(2N_c). Trọng số không tự bảo đảm điểm softmax được hiệu chỉnh tốt. Ngưỡng vận hành vẫn phải được chọn riêng theo mục tiêu recall và số truy vấn tiết kiệm được.

## 2.3 Mô hình chính dựa trên Triplet Loss
### 2.3.1 Kiến trúc bi encoder chia sẻ trọng số

Mô hình sử dụng một sentence encoder f_θ dùng chung cho anchor, positive và negative. Cách biểu diễn hai văn bản độc lập kế thừa hướng Sentence-BERT [3]. Mỗi đầu vào được token hóa bằng tokenizer của checkpoint, đi qua Transformer và pooling để tạo vector câu. Kiến trúc chia sẻ trọng số cho phép biểu diễn nguồn và paraphrase trong cùng không gian, đồng thời hỗ trợ lưu embedding để tái sử dụng.

Với vector token h_i và attention mask m_i, biểu diễn mean pooling được tính như công thức (2.1). Vector sau đó được chuẩn hóa L2 như công thức (2.2), trong đó ε là hằng số nhỏ để tránh chia cho không.

@eq e = (Σᵢ mᵢhᵢ) / (Σᵢ mᵢ)                                      (2.1)
@eq ê = e / max(‖e‖₂, ε)                                           (2.2)
@eq s_tri(A, B) = ê_Aᵀ ê_B                                         (2.3)

Ở thời điểm suy luận, hệ thống chỉ mã hóa A và B rồi tính cosine theo công thức (2.3). Không có nhánh negative và không tính hàm mất mát. Cosine thể hiện quan hệ trong không gian embedding, không phải phần trăm xác suất đạo văn.

Để so sánh có kiểm soát, tầng 2 và tầng 3 bắt đầu từ cùng checkpoint sentence embedding, dự kiến sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. Tầng 2 giữ nguyên trọng số, còn tầng 3 được fine-tune. Cùng tokenizer, pooling, chuẩn hóa và độ dài đầu vào được áp dụng cho các cấu hình so sánh. Checkpoint được cố định revision trước khi chạy thực nghiệm.

TriPlaNet là một công trình tham chiếu về học tương phản cho machine-paraphrasing, nhưng dùng biểu diễn phong cách với các encoder riêng và hướng tiếp cận khác [4]. CaTE-X lấy cảm hứng ở việc học quan hệ qua các mẫu đối chiếu; không xem mô hình một encoder là bản tái hiện TriPlaNet và không suy ra chất lượng tương đương khi chưa có thực nghiệm.

### 2.3.2 Hàm mất mát và đóng góp nghiên cứu

Với bộ ba (a, p, n), khoảng cách được định nghĩa d(x, y) = 1 − cos(f_θ(x), f_θ(y)). Triplet Loss được sử dụng theo công thức (2.4), với m > 0 là margin. FaceNet là một công trình tiêu biểu sử dụng triplet learning và khai thác triplet trong học embedding [5].

@eq L_triplet = max{0, d(a, p) − d(a, n) + m}                       (2.4)

Khi negative xa anchor hơn positive ít nhất margin, loss bằng không. Khi ràng buộc bị vi phạm, tối ưu hóa điều chỉnh embedding để tăng khoảng cách tương đối giữa negative và anchor so với positive. Loss của mini-batch là trung bình loss của các triplet hợp lệ trong lô.

Đóng góp được khảo sát là cải tiến chiến lược xây dựng triplet bằng mẫu âm khó, kết hợp mô hình vào pipeline có điều kiện. Đề tài không tuyên bố đề xuất một công thức Triplet Loss mới. Giả thuyết nghiên cứu là negative gần về từ vựng hoặc ngữ nghĩa nhưng khác nội dung cụ thể tạo tín hiệu học hữu ích hơn negative ngẫu nhiên, giúp giảm nhầm lẫn giữa cùng chủ đề và thực sự paraphrase.

### 2.3.3 Khai thác mẫu âm khó và kiểm soát nhiễu nhãn

Phiên bản cơ sở chọn negative ngẫu nhiên từ nhóm khác. Phiên bản cải tiến tạo pool ứng viên bằng TF-IDF và embedding pretrained, loại cùng nhóm và trùng lặp, rồi chọn negative có mức tương đồng cao nhưng không tương đương nội dung. Tập ứng viên có thể được truy hồi theo block hoặc chỉ mục lân cận để tránh tạo ma trận tương đồng đầy đủ N × N.

Độ khó từ vựng và độ khó ngữ nghĩa được xem là hai tín hiệu riêng. Một câu có cùng thuật ngữ nhưng khác thực thể, số liệu, quan hệ hoặc kết luận có thể là negative hữu ích. Tuy nhiên, thay đổi một chi tiết không tự động biến toàn bộ đoạn thành negative; nhãn phải phản ánh tiêu chí tương đương nội dung được thống nhất. Các ca không rõ được loại khỏi tập huấn luyện nhị phân hoặc lưu để phân tích lỗi.

Trong phạm vi nguồn lực cho phép, kiểm tra thủ công một mẫu ngẫu nhiên của negative được khai thác để ước lượng nhiễu nhãn; tập test cần được rà soát chặt hơn. Không dùng duy nhất dự đoán của mô hình đang được đánh giá làm nhãn chuẩn. Khi pool không có negative hợp lệ, bỏ triplet hoặc dùng negative ngẫu nhiên hợp lệ và ghi lại tỷ lệ thay thế.

Các phương án random negative, lexical hard negative và hybrid hard negative được so sánh với cùng số triplet, positive, checkpoint khởi tạo và ngân sách huấn luyện. Nếu không đủ nguồn lực cho cả ba, so sánh tối thiểu random negative với một chiến lược hard negative đã xác định trước.

### 2.3.4 Quy trình huấn luyện và chọn mô hình

Mỗi mini-batch được đưa qua shared encoder để thu ba tập embedding, tính loss và cập nhật bằng AdamW. Tham số mô hình, trạng thái optimizer và cấu hình preprocessing được lưu cùng checkpoint. Seed được cố định cho quá trình chia dữ liệu, chọn negative và huấn luyện; các lần chạy lặp sử dụng danh sách seed được công bố.

@table Bảng 2.4 Cấu hình khởi tạo cho thực nghiệm tầng 3
Tham số|Thiết kế khởi tạo
Checkpoint|Cùng checkpoint SBERT pretrained với tầng 2
Pooling và chuẩn hóa|Mean pooling có mask và chuẩn hóa L2
Độ dài câu|128 token; khảo sát tỷ lệ cắt trước khi chốt
Mini-batch|16 triplet; điều chỉnh theo bộ nhớ thực tế
Bộ tối ưu|AdamW; learning rate khởi tạo 2 × 10⁻⁵
Margin|Khảo sát 0,2; 0,3; 0,5 trên validation
Ngân sách huấn luyện|Tối đa 3 epoch; early stopping theo validation
Chọn checkpoint|AP trên tập cặp validation cố định; triplet accuracy là chỉ số phụ
@end

Triplet accuracy đo tỷ lệ s(a, p) > s(a, n), nhưng không bảo đảm tồn tại ngưỡng phân loại cặp tốt. Vì mục tiêu cuối cùng là đối chiếu cặp, checkpoint được chọn theo AP trên tập cặp validation cố định, có cả positive, negative dễ và negative khó. Sau khi chọn checkpoint, mới chọn ngưỡng quyết định trên phần validation dành cho ngưỡng.

Các cấu hình phải dùng cùng lịch tối ưu và cùng ngân sách khi so sánh tác dụng của negative mining. Learning rate, warm-up nếu dùng, số bước, gradient accumulation và phần cứng được ghi lại trong chương thực nghiệm. Không suy ra tỷ lệ bộ nhớ GPU chính xác chỉ từ số nhánh của triplet vì chi phí còn phụ thuộc activation, optimizer và cách thực thi.

## 2.4 Các thành phần bổ trợ
### 2.4.1 Tín hiệu từ vựng và ngữ nghĩa sơ bộ

Tầng 1 sử dụng TF-IDF với word unigram và bigram, chuẩn hóa L2 rồi tính cosine. Vocabulary và IDF được fit trên train, sau đó giữ cố định. Văn bản không có token trong vocabulary được đánh dấu thiếu tín hiệu từ vựng; trường hợp này không tự tạo bằng chứng không tương ứng. Quy tắc chuẩn hóa, min_df và max_df được lưu trong cấu hình.

Tầng 2 dùng SBERT pretrained, biểu diễn mỗi đoạn độc lập rồi tính cosine [3]. Embedding của đoạn nguồn có thể được cache theo nội dung và revision mô hình. Tầng 2 không kết luận dương tính sớm, nhằm bảo đảm mọi cảnh báo nghi vấn đi qua tầng 3 hoặc 4. Nếu hai tầng sơ bộ không mang lại lợi ích chi phí trong thực nghiệm, phiên bản triển khai có thể tắt lọc sơ bộ nhưng vẫn giữ các phương pháp này làm baseline.

### 2.4.2 Cross Encoder cho các cặp khó

Cross-Encoder nhận đồng thời A và B qua tokenizer dành cho cặp câu. Với checkpoint BERT, dạng đầu vào là [CLS] A [SEP] B [SEP]. Vector biểu diễn cuối được đưa qua dropout và lớp tuyến tính để tạo logit z. Khả năng tương tác giữa token hai phía giúp mô hình xem xét quan hệ chi tiết mà hai embedding độc lập có thể không giữ lại đầy đủ. Việc kết hợp bi-encoder truy hồi và cross-encoder chấm lại được sử dụng trong mô hình retrieve-and-rerank [6].

@eq p_pair = σ(z) = 1 / (1 + exp(−z))                              (2.5)
@eq L_BCE = −y log σ(z) − (1 − y) log(1 − σ(z))                    (2.6)

Huấn luyện dùng BCEWithLogitsLoss để ổn định số học. Cấu hình khởi tạo có thể sử dụng bert-base-multilingual-cased, đầu vào tối đa 256 token, dropout 0,1, batch 16 và learning rate 2 × 10⁻⁵. Đây là tham số ban đầu, không phải cấu hình tối ưu đã được chứng minh. Mẫu âm kết hợp random và hard negative; tỷ lệ lớp và nguồn mẫu được báo cáo.

Điểm sigmoid là xác suất ước lượng theo mô hình, không mặc nhiên được hiệu chỉnh tốt [7]. Nếu cần trình bày xác suất, áp dụng hiệu chỉnh trên dữ liệu calibration độc lập và đánh giá Brier score, biểu đồ reliability cùng lỗi hiệu chỉnh. Nếu chưa kiểm chứng, giao diện gọi giá trị này là điểm phân loại. Việc thay đổi tỷ lệ lớp ngoài thực tế cũng có thể làm xác suất ước lượng lệch so với tập huấn luyện.

### 2.4.3 Bộ sàng lọc ở bước 0

Bước 0 nhận một câu hoặc cửa sổ ngữ cảnh đã được xác định thống nhất giữa huấn luyện và suy luận. Mô hình gồm encoder, biểu diễn phân loại, dropout và lớp tuyến tính hai đầu ra. Softmax tạo điểm cho lớp tham chiếu người viết và lớp AI sinh hoặc biến đổi. Cross-Entropy có trọng số lớp được dùng theo công thức (2.7).

@eq L_AI = −Σ_c w_c y_c log p_c                                   (2.7)

Do dữ liệu có thể gồm cả AI-generated và AI-paraphrase, tên tác vụ vận hành là sàng lọc dấu hiệu AI. Khả năng phát hiện AI-paraphrase được báo cáo riêng trên nhóm đó. Một điểm cao không chứng minh đạo văn và một điểm thấp không bảo đảm văn bản không qua AI.

Ngưỡng tiết kiệm được chọn bằng cách tối thiểu hóa tỷ lệ đơn vị cần tìm nguồn dưới ràng buộc recall ≥ R_target trên validation. Giá trị R_target được xác định trước khi đánh giá test; có thể báo cáo nhiều điểm vận hành để thấy đánh đổi giữa recall và chi phí. Nếu không đạt ràng buộc, hệ thống không tuyên bố bộ lọc đáp ứng mục tiêu. Chế độ đầy đủ vẫn tìm nguồn bất kể điểm bước 0.

### 2.4.4 Giải thích dựa trên bằng chứng bằng LLM

Tầng 5 sử dụng LLM để diễn giải bằng chứng cho các cặp nghi vấn từ tầng 3 hoặc 4. Tầng này không phải một bộ phân loại quyết định cuối cùng. Prompt cung cấp A, B, ID nguồn và kết quả mô hình; yêu cầu chỉ sử dụng nội dung đã cung cấp, không suy đoán ý định người viết và không tạo URL mới.

Đầu ra có cấu trúc gồm evidence_pairs, transformation_types, differences, explanation và evidence_status. Mỗi evidence_pair chứa đoạn trích nguyên văn từ hai phía; hệ thống kiểm tra chúng tồn tại trong đầu vào. Nếu bằng chứng không đủ hỗ trợ nhãn, LLM được phép trả về trạng thái chưa đủ cơ sở giải thích. Trạng thái này tạo dấu hiệu cần xem xét, không tự thay đổi nhãn mô hình.

Không yêu cầu LLM sinh một confidence mới để gộp với điểm tầng 3 hoặc 4. Lỗi API, JSON không hợp lệ hoặc trích dẫn không tồn tại được ghi nhận và xử lý bằng cách xuất kết quả không có giải thích. Nội dung tài liệu và trang web được xem là dữ liệu, không phải chỉ dẫn cho LLM. Chất lượng giải thích được đánh giá riêng theo độ trung thành với bằng chứng, tính đúng của kiểu biến đổi và mức hữu ích đối với người đọc.

### 2.4.5 Tổng hợp nguồn và xuất báo cáo

Đối với mỗi đơn vị nghi vấn, hệ thống giữ các nguồn phù hợp cùng đoạn trích và provenance. Những URL trùng nội dung được nhóm để tránh tạo cảm giác có nhiều bằng chứng độc lập. Nếu cần chọn nguồn đại diện, dùng điểm từ cùng một bộ chấm cuối; không lấy max giữa các điểm thuộc tầng khác nhau.

@table Bảng 2.5 Các trạng thái trong báo cáo người dùng
Trạng thái|Ý nghĩa
Nghi vấn có bằng chứng|Có đoạn nguồn và kết quả đối chiếu vượt ngưỡng đã chọn
Cần xem xét|Mô hình chưa chắc chắn hoặc bằng chứng và giải thích chưa nhất quán
Chưa tìm được nguồn phù hợp|Đã thực hiện truy xuất trong ngân sách nhưng chưa có nguồn đủ căn cứ
Nguồn không đọc được|Có ứng viên nhưng không trích xuất được nội dung để xác minh
Chưa kiểm tra nguồn|Chưa xử lý do ngân sách hoặc chính sách chế độ tiết kiệm
Không có cặp đạt ngưỡng|Đã đối chiếu nguồn ứng viên nhưng chưa có cặp đủ điều kiện cảnh báo
@end

Thông tin dẫn nguồn trong tài liệu là một thuộc tính đi kèm. Có ký hiệu trích dẫn không tự chứng minh trích dẫn phù hợp; ngược lại, không tìm thấy ký hiệu bằng bộ phân tích cũng không chứng minh hoàn toàn thiếu dẫn nguồn. Báo cáo hiển thị kết quả nhận diện này để người dùng kiểm tra.

Đầu ra gồm DOCX hoặc HTML có highlight và bảng chi tiết chứa đoạn nghi vấn, đoạn nguồn, URL, loại điểm, mô hình chấm, trạng thái và giải thích. Các đoạn highlight chồng lấn được hợp nhất khi tính mức bao phủ. Nếu báo cáo tỷ lệ ký tự bị đánh dấu, tỷ lệ đó được gọi là mức bao phủ cảnh báo, không gọi là xác suất hoặc tỷ lệ đạo văn chắc chắn.

## 2.5 Phương pháp so sánh và thiết kế thực nghiệm
### 2.5.1 Baseline và câu hỏi nghiên cứu

Thiết kế thực nghiệm tách hiệu quả biểu diễn, chiến lược khai thác negative và cơ chế cascade. Các mô hình dùng cùng bản chia dữ liệu, cùng tập cặp test và cùng tiêu chí chọn ngưỡng phù hợp tác vụ. Khi so sánh fine-tuning, giữ nguyên checkpoint khởi tạo, positive và ngân sách huấn luyện.

@table Bảng 2.6 Các cấu hình thực nghiệm
Ký hiệu|Cấu hình|Mục tiêu
B1|TF-IDF và cosine|Mốc tham chiếu từ vựng
B2|SBERT pretrained|Mốc trước fine-tuning
B3|Cùng SBERT và Triplet với random negative|Đo hiệu quả huấn luyện theo miền
B4|Cùng SBERT và Triplet với hard negative|Đo đóng góp negative mining
B5|Cross-Encoder trên mọi cặp|Mốc đối chiếu chi tiết và chi phí
C1|B4 và Cross-Encoder theo vùng mập mờ|Đo lợi ích phân tuyến
C2|Cascade tầng 1 đến 4 chế độ đầy đủ|Đánh giá lõi không lọc sơ bộ
C3|Cascade tầng 1 đến 4 chế độ tiết kiệm|Đánh đổi chất lượng và tính toán
@end

Trong C2, tầng 1 và 2 chỉ tạo điểm phục vụ phân tích; vì vậy cần báo cáo cả chi phí giữ hai tầng này và cấu hình tắt chúng nếu không ảnh hưởng nhãn. C3 chỉ được xem là hữu ích khi giảm tài nguyên với mức giảm chất lượng nằm trong giới hạn đã công bố. Thêm tầng 5 không làm đổi nhãn nên không được diễn giải là nguyên nhân tăng F1 phân loại.

### 2.5.2 Thí nghiệm loại bỏ thành phần và tái lập

Các ablation ưu tiên gồm: random so với hard negative; chỉ lexical so với hybrid mining nếu đủ nguồn lực; bật và tắt lọc sơ bộ; bật và tắt nhánh thoát sớm tầng 3; có và không Cross-Encoder. Đánh giá tệp bổ sung so sánh bước 0 điều phối với kiểm tra đầy đủ. Mỗi thí nghiệm thay đổi một thành phần chính để diễn giải được nguyên nhân chênh lệch.

Tập ứng viên nguồn được cố định khi so sánh mô hình để biến động Web Search không che khuất đóng góp của Triplet. Sau đó thực hiện đánh giá end-to-end riêng với nhật ký truy vấn và thời điểm truy xuất. Mọi phương pháp được đo trên cùng phần cứng, chế độ batch, mức cache và điều kiện warm-up. Báo cáo riêng độ trễ cold-start nếu đây là chi phí người dùng thực sự gặp.

Nếu nguồn lực cho phép, chạy ít nhất ba seed huấn luyện và báo cáo trung bình cùng độ phân tán. Nếu chỉ chạy một seed, nêu giới hạn đó và không diễn giải một chênh lệch nhỏ như cải thiện chắc chắn. Cấu hình, phiên bản thư viện, checksum dữ liệu, số mẫu và checkpoint được lưu để có thể tái lập.

## 2.6 Phương pháp đánh giá
### 2.6.1 Đánh giá phân loại cặp và bước 0

Các độ đo chính gồm Precision, Recall, F1 cho từng lớp, Macro-F1, Balanced Accuracy và MCC. Với điểm liên tục có cùng ý nghĩa trong một mô hình, báo cáo thêm ROC-AUC và Average Precision. Các công thức Precision, Recall và F1 cho lớp dương được trình bày trong (2.8) đến (2.10).

@eq Precision = TP / (TP + FP)                                    (2.8)
@eq Recall = TP / (TP + FN)                                       (2.9)
@eq F1 = 2TP / (2TP + FP + FN)                                    (2.10)

Mỗi độ đo đi kèm số mẫu, phân bố nhãn và quy tắc xử lý mẫu số bằng không. Accuracy hoặc F1 lớp dương không được dùng riêng khi lớp dương chiếm đa số. AP được đọc cùng tỷ lệ lớp dương và baseline tương ứng; AP không đồng nhất với diện tích PR tính bằng nội suy hình thang.

Với tập chỉ có mẫu dương, báo cáo tập trung vào recall. Precision và F1 có thể vẫn tính được trong một số trường hợp nhưng không đo được khả năng phân biệt đầy đủ vì thiếu mẫu âm; ROC-AUC nhị phân không xác định khi nhãn thật chỉ có một lớp. Quy tắc đánh giá phải phân biệt thiếu ý nghĩa thống kê với không xác định về mặt công thức, thay vì gán chung mọi chỉ số thành không xác định [8].

Điểm dừng của cascade có thể đến từ nhiều thang đo. Vì vậy báo cáo phân loại cascade bằng nhãn cuối; không ghép các điểm thô để tính ROC-AUC hoặc AP. Nếu triển khai bộ chấm thống nhất cho tất cả cặp, có thể đánh giá ranking riêng trên điểm đó, đồng thời tính đủ chi phí phát sinh.

### 2.6.2 Chọn ngưỡng và đánh giá các nhánh thoát sớm

Ngưỡng của baseline và cascade đều được chọn trên validation, không để một phía tối ưu còn phía kia dùng ngưỡng đặt tay. Với phân loại cặp độc lập, có thể chọn ngưỡng tối đa Macro-F1 theo tiêu chí đã xác định trước. Với bước 0 và lọc sơ bộ, ưu tiên giảm lượng xử lý dưới ràng buộc recall. Với nhánh dương tính sớm, đặt ràng buộc precision. Các mục tiêu này phản ánh chi phí lỗi khác nhau của từng vị trí trong pipeline.

Ngưỡng cosine được tìm trên miền điểm quan sát hợp lệ, không mặc định chỉ nằm trong khoảng 0 đến 1. Toàn bộ ngưỡng được đánh giá lại cùng nhau trên validation khi ghép cascade vì phân bố đầu vào mỗi tầng phụ thuộc quyết định trước đó. Test và tập ngoài miền giữ nguyên ngưỡng đã chọn; không chọn lại để cải thiện số liệu báo cáo.

Với từng tầng dừng, báo cáo tỷ lệ cặp dừng, precision, recall đóng góp và số false negative không được các tầng sau xử lý. Tỷ lệ dương tính bị mất tại một bộ lọc được tính trên toàn bộ mẫu dương đi vào bộ lọc. Đồng thời báo cáo recall cuối trên tập đầu vào ban đầu để tránh làm đẹp kết quả bằng cách bỏ các mẫu đã lọc ra khỏi mẫu số.

### 2.6.3 Đánh giá truy xuất nguồn và kết quả tài liệu

Đánh giá truy xuất sử dụng tập có nguồn đúng đã biết. Recall@K đo tỷ lệ đơn vị có ít nhất một nguồn đúng trong K ứng viên; đánh giá đoạn nguồn được thực hiện riêng với đánh giá URL. Nguồn là bản sao ở URL khác được xử lý theo quy tắc tương đương nguồn đã định trước. Trường hợp nguồn không có trên web công khai cần được tách khi phân tích giới hạn của tìm kiếm.

Đánh giá end-to-end yêu cầu hệ thống phát hiện đúng vùng nghi vấn và cung cấp nguồn đối chiếu đúng. Các độ đo mức ký tự hoặc span sử dụng phần giao giữa vùng dự đoán và vùng gán nhãn, với quy tắc ghép và hợp nhất đoạn được cố định. Ngoài precision và recall vùng, báo cáo tỷ lệ cảnh báo có nguồn đúng và tỷ lệ nội dung thực sự đã kiểm tra.

So sánh hai chế độ bằng chất lượng cuối, số truy vấn, số trang đọc thành công, số cặp qua từng tầng, thời gian và chi phí. Recall của chế độ tiết kiệm được tính trên toàn bộ mẫu dương trong tài liệu đánh giá, kể cả phần chưa kiểm tra; chỉ số trên phần đã xử lý được báo cáo bổ sung và ghi rõ điều kiện.

### 2.6.4 Độ bền ngoài miền và độ bất định

Đánh giá ngoài miền phân tách theo nguồn văn bản, mô hình sinh, loại biến đổi và số vòng paraphrase khi metadata cho phép. Tất cả nhóm dùng cùng mô hình và ngưỡng đã cố định. Nếu PADBen đã được dùng một phần trong train, chỉ các nhóm nguồn độc lập mới được dùng để đánh giá tổng quát hóa, và phạm vi này phải được nêu đúng.

Với các quan sát dương độc lập, có thể dùng khoảng Wilson 95% cho recall. Khi nhiều biến thể cùng nguồn được đánh giá, ưu tiên bootstrap theo nhóm nguồn để phản ánh sự phụ thuộc. Nếu so sánh vòng paraphrase 1 và 3 của cùng nguồn, giữ cấu trúc ghép cặp và ước lượng khoảng tin cậy của chênh lệch recall.

Không dùng việc hai khoảng tin cậy riêng chồng lấn hay không như phép kiểm định duy nhất. Hai khoảng chồng lấn vẫn có thể đi kèm chênh lệch có ý nghĩa; phân tích phải nhắm trực tiếp vào đại lượng chênh lệch [9]. Các nhóm ít mẫu được công bố cùng cỡ mẫu và độ bất định, không kết luận mạnh chỉ từ điểm ước lượng.

### 2.6.5 Đánh giá giải thích và hiệu quả vận hành

Một tập cặp được chọn phân tầng theo nhãn, độ khó và tầng quyết định để đánh giá giải thích. Tiêu chí gồm: đoạn trích tồn tại trong đầu vào; phần tương ứng thực sự hỗ trợ nhận xét; kiểu biến đổi được mô tả đúng; không tạo nguồn hoặc chi tiết ngoài bằng chứng; lời giải thích giúp người dùng hiểu lý do cảnh báo. Khi có nhiều người đánh giá, báo cáo mức đồng thuận và cách giải quyết bất đồng.

Thời gian xử lý được báo cáo bằng trung bình, trung vị và phân vị 95; thông lượng được đo riêng khi xử lý theo batch. Chi phí mạng, truy xuất nguồn, mô hình và LLM được tách để thấy thành phần chi phối. Điều kiện cache, số lần khởi động và lỗi dịch vụ được lưu. Không so sánh một mô hình chạy batch với cascade chạy từng cặp rồi quy toàn bộ chênh lệch cho kiến trúc.

Kiểm thử mã đánh giá bao gồm dữ liệu tính tay, trường hợp một lớp, nhãn hằng, mẫu số bằng không và kiểm tra giao nhóm giữa các split. Những kiểm thử này giúp xác minh phép tính và quy trình, không thay thế kết quả trên mô hình và dữ liệu thật.

## 2.7 Tổng kết chương

CaTE-X được đề xuất như một hệ thống đối chiếu có bằng chứng nguồn, trong đó bi-encoder học bằng Triplet Loss là thành phần chính. Chiến lược khai thác mẫu âm khó là trọng tâm cải tiến được kiểm chứng bằng các cấu hình cùng checkpoint khởi tạo. TF-IDF và SBERT pretrained cung cấp tín hiệu sơ bộ, Cross-Encoder xử lý cặp khó, còn LLM diễn giải bằng chứng mà không thay đổi nhãn.

Hai chế độ đầy đủ và tiết kiệm cho phép khảo sát đánh đổi giữa độ bao phủ và tài nguyên. Điểm bước 0 chỉ điều phối quá trình kiểm tra và được tách khỏi kết luận về quan hệ nội dung. Phần chưa kiểm tra, nguồn không đọc được và trường hợp chưa tìm thấy bằng chứng được thể hiện rõ trong báo cáo, thay vì gộp thành không đạo văn.

Quy trình thực nghiệm chia nhóm trước khi tạo mẫu, dùng chung bản chia giữa các thành phần, chọn ngưỡng trên validation và đánh giá test một lần sau khi cố định cấu hình. Kết quả cần chứng minh đồng thời chất lượng đối chiếu, khả năng tìm nguồn và độ chính xác của vùng highlight. Các giới hạn về nguồn web, phạm vi ngôn ngữ, nhiễu nhãn và độ tin cậy của giải thích được xem xét khi diễn giải kết quả ở chương thực nghiệm.

## Tài liệu tham khảo sử dụng trong chương

[1] V. Vorobev và M. Kuznetsov. ChatGPT paraphrases dataset. Hugging Face, 2023. https://huggingface.co/datasets/humarin/chatgpt-paraphrases

[2] Nhóm phát triển PADBen. PADBen Paraphrase Attack Detection Benchmark. Kho mã và tài liệu dữ liệu. https://github.com/JonathanZha47/PadBen-Paraphrase-Attack-Benchmark ; https://huggingface.co/datasets/JonathanZha/PADBen

[3] N. Reimers và I. Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 2019. https://arxiv.org/abs/1908.10084

[4] D. Meng, Z. Wang, T. Phuntsho và T. Gonsalves. TriPlaNet: Enhancing machine-paraphrasing plagiarism detection through triplet network and contrastive learning. Egyptian Informatics Journal, tập 31, bài 100752, 2025. https://doi.org/10.1016/j.eij.2025.100752

[5] F. Schroff, D. Kalenichenko và J. Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR, 2015. https://arxiv.org/abs/1503.03832

[6] Sentence Transformers. Retrieve and Re-Rank. Tài liệu kỹ thuật. https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html

[7] C. Guo, G. Pleiss, Y. Sun và K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML, 2017. https://arxiv.org/abs/1706.04599

[8] scikit-learn developers. precision_recall_fscore_support. Tài liệu API. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

[9] M. E. Payton, M. H. Greenstone và N. Schenker. Overlapping confidence intervals or standard error intervals: What do they mean in terms of statistical significance? Journal of Insect Science, 2003. https://pmc.ncbi.nlm.nih.gov/articles/PMC524673/

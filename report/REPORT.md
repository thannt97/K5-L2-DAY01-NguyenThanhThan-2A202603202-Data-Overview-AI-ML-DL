# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh
- Vì:
+ Sample_id (một ảnh) sinh ra đúng 5 record — chỉ khác nhau ở rank, class_id, class_name, score. Các trường image_width, image_height, coco_image_id lặp y hệt nhau cho cả 5 record
+ Không có trường nào mang thông tin vị trí (bounding box, mask, tọa độ) hay định danh vật thể (object_id, track_id). Nếu là per-object thì bắt buộc phải có ít nhất một trong những trường đó để phân biệt "vật thể nào".
+ Task: "image_classification" và taxonomy_name: "ImageNet-1K" — ImageNet-1K là bài toán một nhãn cho toàn ảnh, mỗi ảnh được giả định chủ đề chính.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Đi kèm checkpoint/model, không phải model tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Nếu chỉ ghi class_name: mất khả năng join với logits/nhãn gốc, và nếu sau này đổi sang model khác có tên lớp khác thì không phân biệt được "tên đổi do model đổi" hay "tên đổi do người sửa". Nếu chỉ ghi class_id: không ai đọc được, và không biết ID thuộc bảng nào. Cả ba cùng có là để truy vết được từ kết quả về đúng checkpoint + đúng bảng nhãn + đúng lớp.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ quy tắc chọn nhãn duy nhất (single-label decision rule), tối thiểu gồm:
 + Tiêu chí "chủ thể chính" — ví dụ: vật/chủ đề chiếm diện tích lớn nhất, hoặc là tâm điểm của bố cục, hoặc là đối tượng được lấy nét. Phải định nghĩa được bằng lời, không chỉ "cái nào nổi bật".
 + Thứ tự ưu tiên khi hòa — nếu hai đối tượng ngang nhau về tiêu chí chính (ví dụ: người và chó cùng kích thước), quy định rõ ai thắng (ví dụ: ưu tiên lớp xuất hiện trước trong danh sách taxonomy, hoặc ưu tiên người, hoặc ưu tiên vật thể ở tiền cảnh).
 + Quy tắc cho ảnh nền / không có đối tượng rõ — có được gán lớp "background"/"other" không, hay bắt buộc phải chọn lớp gần nhất.
 + Quy tắc cho ảnh đa chủ đề cân bằng (ví dụ: ảnh chụp một nhóm bạn + phong cảnh) — chọn theo "chủ đề của ảnh" hay "vật thể đếm được"? Cần ví dụ mẫu (positive/negative examples) để người gán nhãn áp dụng nhất quán.
 + Cấm đoán mò — nếu không xác định được chủ thể chính sau khi áp dụng quy tắc, phải đánh dấu ambiguous/skip thay vì chọn đại, để không tạo nhãn rác.
 + Ví dụ biên (edge cases) có ảnh minh họa — quan trọng hơn cả câu chữ, vì "chủ thể chính" là khái niệm mơ hồ nếu chỉ mô tả bằng lời.
6. Vì sao không được lấy `score` của model làm ground truth?
- Vì score là đầu ra của model, không phải sự thật về dữ liệu. Lấy nó làm ground truth là circular reasoning — dùng chính thứ cần đánh giá để định nghĩa thứ đúng.
- score là ý kiến của model về chính nó, còn ground truth phải là ý kiến độc lập về dữ liệu. Trộn hai thứ này là mất khả năng đo lường — và trong pipeline của bạn, score chỉ nên xuất hiện ở cột "prediction confidence" để review, còn cột "label" phải đến từ nguồn khác.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="bus" ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên trái nằm ở:
 + Cách mép trái ảnh 93.17 px
 + Cách mép trên ảnh 187.95 px
- Góc dưới-phải của box nằm ở:
 + Cách mép trái ảnh 223.01 px
 + Cách mép trên ảnh 320.91 px
- Kích thước box:
+ Rộng = x2 − x1 = 223.01 − 93.17 = 129.84 px (khớp bbox_width)
+ Cao = y2 − y1 = 320.91 − 187.95 = 132.96 px (khớp bbox_height)
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- ngưỡng 0.45 có 24 prediction giữ lại
- ngưỡng 0.78 có 5 prediction giữ lại
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Kiểm tra các box "yếu" — chủ yếu là false positive
- Kiểm tra trùng lặp / chồng lấn (duplicate & overlap)
- Kiểm tra biên box (localization quality)
- Phân loại lại lớp (class correction)
- Xử lý box ở vùng khó (rìa ảnh, bị che, quá nhỏ)
- Cập nhật "ignore" / "ambiguous" thay vì chỉ giữ/bỏ
- Ưu tiên review
- Đo lại chất lượng và ghi nhận tác động
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Một box = một vật thể nhìn thấy được, thuộc 80 lớp COCO.
- Tạo box khi đủ 3 điều:
 + Xác định được lớp → không xác định được thì skip, không đoán.
 + Nhìn thấy ≥ 50% vật thể.
 + Cạnh ngắn nhất ≥ 10 px.
- Vẽ box:
 + Sát biên vật, bao trọn phần nhìn thấy — không thừa nền, không cắt vật.
 + Vật cắt mép ảnh → box dừng ở mép ảnh.
 + Nhiều vật chồng lấn / đứng nhóm → mỗi vật một box, không gộp.
- Không tạo box cho: vật ngoài 80 lớp, vật < 10 px, vật bị che > 50%, hình ảnh/tranh vẽ, bóng đổ, nền.
- Lớp dễ nhầm (car/truck/bus, cup/bowl/wine glass, person): dựa vào hình dáng tổng thể + ngữ cảnh, không dựa chi tiết nhỏ. Không quyết được → ambiguous.
- Trước khi nộp — tự kiểm 4 bước: đúng lớp? đúng biên? không trùng? không sót?
- Không chắc → ghi ambiguous, không đoán bừa. Nhất quán quan trọng hơn "đúng tuyệt đối".
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Đưa lên hỏi Lab Coach vì:
- Đây là tình huống biên chưa có tiền lệ rõ trong guideline hiện tại
- Ảnh hưởng đến metric, không chỉ đến một box.
- Cần định nghĩa "cắt bao nhiêu thì bỏ"
- Ảnh hưởng đến cả taxonomy và tập huấn luyện.
- Có ít nhất 2 lựa chọn hợp lý, xung đột nhau: Giữ box, dừng ở mép ảnh, skip vật cắt mép quá nhiều
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-002"; `class_name`="bowl" ; `score`=0.735743 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[[
        53.0,
        344.0
      ],
      [
        52.0,
        345.0
      ],
      [
        50.0,
        345.0
      ],
      [
        49.0,
        346.0
      ],
      [
        48.0,
        346.0
      ],
...
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box chỉ cho 4 số — hình chữ nhật bao quanh vật. Polygon cho danh sách điểm theo đường viền thật của vật.
- Hình dạng thật
- Diện tích thật
- Tỷ lệ lấp đầy box
- Định vị chính xác hơn cho vật không lồi
- Ranh giới giữa hai vật chạm nhau
- Hỗ trợ task khác
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id là định danh duy nhất cho một vật thể cụ thể (một instance) trong một ảnh 
- class_id dùng chung cho mọi vật cùng lớp — mọi bus trên mọi ảnh đều là 5.
- track_id đòi hỏi liên kết xuyên ảnh/frame — nó nói "con bus ở frame 1 và con bus ở frame 5 là cùng một chiếc".
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Sát đường viền thật — mask chạm đúng mép ngoài cùng của vật (bánh xe, tay áo, miệng cốc…), không chừa viền nền, không ăn vào trong vật.
- Bao trọn phần nhìn thấy — mọi pixel thuộc vật đều nằm trong mask; không cắt mất mảng nào.
- Không gộp — không mở rộng mask để bao vật khác, bóng đổ, phản chiếu, hay nền, dù chúng chạm vật.
- Vật bị che — mask chỉ bao phần nhìn thấy được, dừng ở ranh giới với vật che. Không vẽ phần bị che.
- Vật cắt mép ảnh — mask dừng ở mép ảnh, không vẽ ra ngoài.
- Vật có lỗ / rỗng (vành bánh xe, tay cầm cốc) — không lấp phần rỗng; đường viền đi theo cả biên ngoài và biên trong nếu phần rỗng nhìn thấy rõ.
- Vật chạm nhau — vẽ hai mask riêng, đường biên đi theo ranh giới thật giữa hai vật; không để mask chồng lấn, không gộp thành một.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Chọn: bowl bbox [155.02, 168.67, 182.38, 184.18] (score 0.701283) và bowl bbox [155.76, 113.58, 173.98, 130.13] (score 0.464904).
- Dùng Escalation  vì:
+ Guideline hiện tại chưa trả lời được câu hỏi cốt lõi: hai box bowl nhỏ, cùng lớp, cách nhau 38 px, không chồng lấn — là hai vật thật hay một vật bị phát hiện hai lần? Guideline có quy tắc cho "hai vật chồng lấn → hai mask riêng" và "vật < 10 px → skip", nhưng không có quy tắc cho "hai vật cùng lớp, nhỏ, gần nhau, không chồng lấn" — đúng tình huống này.
+ Đây là quyết định về ngưỡng, ảnh hưởng toàn dataset. Cần Coach quyết: khoảng cách tối thiểu giữa hai vật cùng lớp là bao nhiêu thì coi là hai instance? Vật nhỏ sát ngưỡng 10 px có nên giữ, hay nâng ngưỡng tối thiểu lên (ví dụ 15 px) để tránh nhiễu? Đây là policy, không phải người gán nhãn tự đặt.
+ Vẽ mask cho vật ~16 px là cực khó và dễ sai. Với vật nhỏ như vậy, biên mask chỉ có vài pixel — sai 1–2 px là đã lệch 10% hình dạng. Nếu không có quy tắc rõ "vật nhỏ thế này có bắt buộc vẽ mask không, hay chỉ cần box", người gán nhãn sẽ làm mỗi người một kiểu → mask không nhất quán, huấn luyện model segmentation bị nhiễu.
+ Có ít nhất 3 lựa chọn hợp lý, xung đột:
Giữ cả hai, vẽ hai mask riêng → giữ recall, nhưng mask nhỏ dễ sai, và nếu thực ra là một vật thì tạo duplicate.
Gộp thành một mask → nếu là hai vật thật thì sai (mất một instance).
skip cả hai vì quá nhỏ → mất vật thật, model bị phạt oan nếu dự đoán đúng.
+ Guideline không phân định được → cần Coach chọn.
Ảnh hưởng đến metric segmentation. Nếu mask nhỏ bị vẽ sai, IoU mask sẽ thấp giả tạo — model tốt vẫn bị điểm thấp. Cần Coach quyết định: vật nhỏ dưới ngưỡng nào thì không tính vào metric (dùng ignore region), thay vì bắt người gán nhãn vẽ mask không chính xác.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho cả ảnh — (sample_id, taxonomy_name, class_id, class_name). Không có tọa độ, không có score. Mỗi ảnh = 1 dòng GT. | Ảnh đa chủ đề / nhiều vật: chọn lớp nào cho cả ảnh? Trong JSON classification, mỗi sample_id sinh ra 5 record top-5 (rank 1–5) — nhưng GT chỉ được phép 1 nhãn, không phải 5. Đây là chỗ dễ nhầm: annotator không được copy top-5 của model thành 5 nhãn. | Áp dụng quy tắc "chủ thể chính" (diện tích lớn nhất / tâm điểm / lấy nét), có thứ tự ưu tiên khi hòa, có quy tắc cho ảnh nền. Ghi ambiguous/skip nếu không quyết được — không copy score của model. | Nhãn có khớp quy tắc "chủ thể chính" không? Có phải là 1 nhãn duy nhất (không phải 5)? Có bị lấy score của model làm nhãn không? Ảnh đa chủ đề có được xử lý nhất quán với các ảnh khác không? |
| Phát hiện vật thể | Danh sách instance, mỗi instance = (instance_id, class_id, bbox_xyxy) + metadata ảnh (image_width/height, coordinate_unit, bbox_format). Một ảnh có nhiều dòng GT (nhiều vật), khác hẳn classification. | Trên kitchen: oven bbox [0.11, 187.54, 195.96, 292.81] cắt mép trái — giữ hay skip? Trên traffic: nhiều car/bus nhỏ ở rìa (score 0.48–0.59) dễ nhầm car/bus/truck và dễ trùng lặp. Trên dining: dining table bbox [0, 318, 478, 638] gần như cả ảnh — có phải một vật hay nhiều vật ghép? | Vẽ box sát biên vật, tách từng vật kể cả khi chồng lấn/đứng nhóm, skip vật < 10 px hoặc bị che > 50%, ghi ambiguous khi nghi hai vật hay một. Không copy bbox/score của model. | Box có sát biên, đúng lớp, không trùng, không sót? Vật cắt mép có được xử lý nhất quán? Box nhỏ sát ngưỡng 10 px có bị vẽ ẩu? instance_id có được gán cho từng vật (không dùng class_id thay thế)? |
| Instance segmentation | Danh sách instance, mỗi instance = (instance_id, class_id, polygon/mask) + metadata ảnh. Mask là đường viền thật của vật, không phải box. Một ảnh nhiều dòng GT, mỗi dòng một mask riêng. | Trên kitchen: hai bowl nhỏ ([155.02, 168.67, 182.38, 184.18] và [155.76, 113.58, 173.98, 130.13]), cùng lớp, cách nhau 38 px, cạnh ngắn ~16 px sát ngưỡng 10 px — hai vật thật hay một vật bị phát hiện hai lần? Vẽ mask cho vật ~16 px là cực khó (sai 1–2 px = lệch 10% hình dạng). Hai cup sát nhau ([143.52, 268.13, …] và [119.95, 272.2, …]) cũng cùng vấn đề. | Vẽ mask sát đường viền thật, không lấp phần rỗng, không gộp vật chạm nhau, mask dừng ở mép ảnh với vật cắt mép, ambiguous khi biên mờ. Không copy mask của model. | Mask có sát biên vật, không thiếu/thừa mảng, không chồng lấn mask khác? Vật nhỏ sát ngưỡng có được vẽ mask hay chỉ box? Hai vật cùng lớp gần nhau có được tách đúng không? instance_id có khớp giữa box và mask không (nếu có cả hai)? |


## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ xử lý ảnh/JSON trong phạm vi bài lab được giao, không copy, chia sẻ, tải lên công cụ ngoài, hay dùng cho mục đích khác ngoài lab — và không đưa dữ liệu thật (ảnh gốc, GT, prediction) vào bất kỳ kênh nào không được Lab cho phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng lại, không xử lý tiếp — không cố gán nhãn, không tự sửa, không bỏ qua âm thầm. Báo ngay cho Lab Coach (người phụ trách lab), kèm mô tả ngắn: ảnh/file nào, sai phạm vi ở đâu (khác taxonomy, khác model, khác định dạng, hay dữ liệu nhạy cảm). Chờ Coach quyết định hướng xử lý trước khi làm tiếp — không tự đặt quy tắc riêng, không tự loại bỏ dữ liệu.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS

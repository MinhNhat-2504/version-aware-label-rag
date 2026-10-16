# Giao diện thẩm định nguồn LabelQA

Giao diện này chạy cục bộ, không cần cài đặt và không chứa câu trả lời của mô hình. Khi được tạo cùng gói dữ liệu riêng, nó hiển thị các cặp phiên bản và toàn bộ section đã đóng băng, tự lưu trên trình duyệt, kiểm tra phiếu và xuất JSONL đúng schema của pipeline. Repository mã nguồn không kèm `review-data.js`, dữ liệu holdout hoặc bản thảo paper.

## Mở giao diện

Sau khi tạo `dist/review-data.js` từ dữ liệu cục bộ, mở `dist/index.html` bằng Chrome, Edge hoặc Firefox. Giao diện hoạt động khi không có Internet.

Nếu chính sách trình duyệt chặn tệp cục bộ, chạy tại thư mục dự án:

```powershell
python -m http.server 8765 --directory reviewer_ui/dist
```

Sau đó mở `http://localhost:8765`.

## Quy trình reviewer

1. Nhập một mã reviewer ẩn danh và giữ nguyên mã đó cho mọi mục trong gói (bản mở rộng hiện tại: 60 cặp, 120 mục). Mã không được bắt đầu bằng `llm_`: tiền tố này dành cho phiếu của mô hình ngôn ngữ, và giao diện từ chối cả mã lẫn file mang tiền tố đó.
2. Đọc câu hỏi và từng phiên bản nguồn độc lập.
3. Chọn quyết định. Với `accept`, điền khả năng trả lời, câu trả lời độc lập và đánh giá thay đổi phiên bản.
4. Nếu nguồn trả lời được, tạo từng claim nguyên tử, chọn claim đang làm việc, tìm section, bôi đen nguyên văn bằng chứng và thêm đoạn đã chọn.
5. Với bằng chứng từ bảng, bôi đen nguyên văn trong một hàng; giao diện tự chọn `table_row` và điền chỉ số bảng/hàng khi đoạn bôi đen chỉ khớp một hàng. Dùng liên kết đúng phiên bản DailyMed trong giao diện để đối chiếu bố cục gốc. Nếu chiều tác động hoặc nhóm thuốc nằm ở một hàng tiêu đề riêng, tách thành hai claim, mỗi claim trích từ hàng nêu ra ý đó.
6. Nút **Xuất JSONL hoàn chỉnh** chỉ tạo file khi mọi mục trong gói đều hợp lệ. Nút **Tải bản nháp** dùng để sao lưu giữa chừng và có thể nhập lại.

Khi thư mục `results/pinned_xml_holdout_review/` có XML đã ghim (tạo bằng `python -m labelqa.fetch_review_xml`), gói dữ liệu kèm các đoạn văn xuôi và hàng bảng do parser tách. Khi bôi đen, giao diện tự chọn loại nguồn, điền chỉ số bảng/hàng nếu đoạn chỉ khớp một hàng, và từ chối trích dẫn không nằm trọn trong một đoạn mà pipeline sau gold có thể ánh xạ.

Tự kiểm tra một phiếu đã xuất: `python -m labelqa.check_holdout_reviews --reader-a <file>`.

Một file reviewer chưa phải gold. Thiết kế hiện hành (Amendment 3–4 của kế hoạch phân tích) gồm một phiếu của mô hình ngôn ngữ có công bố và một phiếu của chuyên gia người. Phiếu chuyên gia được khóa bằng biên nhận (mã băm và thời điểm) trước khi chuyên gia xem phiếu AI. Sau đó chuyên gia tự hòa giải và ký từng mục bằng bảng tính của `labelqa.expert_resolution`. Chủ dự án khóa phiếu bằng:

```powershell
python -m labelqa.check_holdout_reviews --reader-a <file-chuyen-gia> --receipt data/review/<ma>.receipt.json
```

## Tạo lại gói dữ liệu

```powershell
$env:PYTHONPATH='src'
python -m labelqa.export_reviewer_interface
```

Lệnh kiểm tra định danh và SHA-256 của từng nguồn trước khi tạo lại `dist/review-data.js`.

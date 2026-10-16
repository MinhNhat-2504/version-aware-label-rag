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

1. Nhập một mã reviewer ẩn danh và giữ nguyên mã đó cho cả 24 mục.
2. Đọc câu hỏi và từng phiên bản nguồn độc lập.
3. Chọn quyết định. Với `accept`, điền khả năng trả lời, câu trả lời độc lập và đánh giá thay đổi phiên bản.
4. Nếu nguồn trả lời được, tạo từng claim nguyên tử, chọn claim đang làm việc, tìm section, bôi đen nguyên văn bằng chứng và thêm đoạn đã chọn.
5. Với bằng chứng từ bảng, chọn `table_row` và điền chỉ số bảng/hàng không âm. Snapshot hiện lưu văn bản section đã làm phẳng; dùng liên kết đúng phiên bản DailyMed trong giao diện để đối chiếu bố cục. Pipeline vẫn chỉ tải và lưu bản XML cục bộ sau khi gold được khóa.
6. Nút **Xuất JSONL hoàn chỉnh** chỉ tạo file khi cả 24 mục hợp lệ. Nút **Tải bản nháp** dùng để sao lưu giữa chừng và có thể nhập lại.

Một file reviewer chưa phải gold. Cần hai reviewer độc lập, sau đó hòa giải bằng đồng thuận theo pipeline hiện tại.

## Tạo lại gói dữ liệu

```powershell
$env:PYTHONPATH='src'
python -m labelqa.export_reviewer_interface
```

Lệnh kiểm tra định danh và SHA-256 của từng nguồn trước khi tạo lại `dist/review-data.js`.

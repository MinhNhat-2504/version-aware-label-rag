# Giao diện chấm câu trả lời LabelQA

Giao diện chạy cục bộ, không cần cài đặt, dùng ở giai đoạn 3: chuyên gia chấm từng câu trả lời của mô hình trên holdout. Tên phương pháp đã được che, và giao diện không chứa đáp án chuẩn (gold). Repository mã nguồn không kèm `answer-review-data.js`, vì file này chứa câu trả lời của mô hình trên holdout.

## Tạo gói dữ liệu (chủ dự án)

Chạy sau khi đã có gói chấm mù từ `labelqa.prepare_blinded_answer_review`:

```powershell
$env:PYTHONPATH='src'
python -m labelqa.export_answer_review_interface --packet data/review/holdout_answers_blind.jsonl
```

Lệnh này chỉ đọc gói chấm mù và toàn văn nhãn đã đóng băng. Nó không đọc khóa phương pháp và từ chối gói có trường tên phương pháp hay đáp án chuẩn. Gửi cho chuyên gia cả thư mục `answer_review_ui/dist`, **không** gửi khóa phương pháp.

## Cách chấm (chuyên gia)

1. Mở `dist/index.html` bằng Chrome, Edge hoặc Firefox. Nhập đúng mã chuyên gia đã dùng ở giai đoạn thẩm định nguồn. Mã không được bắt đầu bằng `llm_`.
2. Mỗi mục là một câu trả lời của một phương pháp chưa biết tên, thuộc một trong ba loại:
   - **Trả lời:** chấm từng claim. Claim có được đoạn nó trích hỗ trợ không, và trích có đúng vị trí không. Sau đó chấm câu trả lời có đầy đủ không, có đúng phiên bản SPL không, có claim lâm sàng không căn cứ có thể gây hại không. Hai ô tổng hợp (mọi claim được hỗ trợ, mọi trích dẫn đúng vị trí) được tính tự động từ từng claim.
   - **Từ chối:** việc từ chối có phù hợp không (phù hợp khi đúng phiên bản này không đủ thông tin; vắng thông tin không có nghĩa là an toàn), đúng phiên bản, và claim không căn cứ.
   - **Lỗi đầu ra:** mô hình trả về sai định dạng. Chỉ cần ghi chú.
3. Bấm mã `P0001`… để nhảy tới đoạn được trích. Mở **Toàn văn nhãn** để xét câu trả lời có bỏ sót không và từ chối có phù hợp không. Ô lọc chỉ giữ section chứa mọi từ đã gõ.
4. Mọi mục cần một ghi chú chung. Claim không được hỗ trợ, chưa chắc hoặc trích sai vị trí cần thêm ghi chú riêng.
5. Bản nháp tự lưu trên trình duyệt. **Tải bản nháp** để sao lưu giữa chừng; **Nhập bản đã lưu** để tiếp tục trên máy khác. Giao diện không cho nhập phiếu của LLM.
6. **Xuất JSONL hoàn chỉnh** chỉ chạy khi mọi mục hợp lệ. Gửi file này cho chủ dự án và đừng sửa nó nữa.

## Sau khi nhận phiếu (chủ dự án)

```powershell
python -m labelqa.answer_review receipt --packet data/review/holdout_answers_blind.jsonl --reader <phieu-chuyen-gia.jsonl> --output data/review/<ma>_answers.receipt.json
```

Phải khóa phiếu bằng lệnh trên **trước** khi chuyên gia thấy phiếu chấm của LLM. Sau đó hòa giải bằng `labelqa.expert_resolution`, giống giai đoạn 2. Khi muốn sửa một mục, chuyên gia nhập lại file của chính mình vào giao diện, sửa, rồi xuất bản `revised`.

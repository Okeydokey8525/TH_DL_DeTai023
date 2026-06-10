# Đồ án Bài 23: Mạng chẩn đoán ECG hợp nhất đa nguồn có Attention

Thư mục này chứa mã nguồn hoàn chỉnh, độc lập cho **Bài 23** chuyên về xử lý đa nguồn dữ liệu (Multimodal Learning) kết hợp đồng thời Tín hiệu ECG 12 kênh, Ảnh đồ thị biểu diễn nhịp tim, và Thông số chỉ số số bệnh nhân (Tuổi, Giới tính, Chiều cao, Cân nặng).

---

## 💻 Kiến Trúc Mô Hình Đa Nhánh (Multimodal)
1.  **Nhánh Chuỗi (Sequence)**: Mô hình cải tiến Multi-Scale Conv1D + BiLSTM để trích xuất đặc trưng sóng ECG.
2.  **Nhánh Ảnh (Image)**: Mạng 2D-CNN trích xuất đặc trưng hình thái sóng từ ảnh đồ thị.
3.  **Nhánh Số (Tabular)**: Mạng MLP trích xuất đặc trưng nhân khẩu học lâm sàng.
4.  **Hợp nhất Attention**: Lớp Modality Attention Fusion tính toán và nhân trọng số đóng góp động ($\alpha_{seq}, \alpha_{img}, \alpha_{tab}$) cho từng nhánh dựa trên đặc trưng cụ thể từng bệnh nhân.

---

## 📂 Gợi ý Cấu trúc Thư mục
*   `processed/`: Lưu trữ các file checkpoints `.pt`, biểu đồ attention kết quả và file dữ liệu `.npz` của riêng Bài 23.
*   `templates/index.html`: Giao diện Web Dashboard kính mờ tương tác cho Bài 23.
*   `data_preprocess.py`: Đường ống tiền xử lý đa nguồn (đọc dữ liệu thô từ thư mục cha `../`), điền khuyết số, vẽ ảnh thô và lưu mảng NumPy.
*   `models.py`: Định nghĩa kiến trúc mạng đa nguồn.
*   `train.py`: Huấn luyện mô hình đa nguồn, đánh giá phân bổ attention trung bình trên tập kiểm thử và lưu log.
*   `app.py`: Máy chủ Flask chạy ở cổng **5002**.

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh

### Bước 1: Chạy tiền xử lý dữ liệu đa nguồn
Nếu bạn muốn tạo lại dữ liệu kiểm thử đa nguồn sạch cho Bài 23 (sinh ảnh đồ thị, điền khuyết tuổi tác), hãy chạy lệnh sau:
```bash
python data_preprocess.py
```
*Lưu ý: Script sẽ đọc các bản ghi từ thư mục cha và lưu mảng NumPy đã xử lý tại `processed/ptbxl_processed_bai23.npz`.*

### Bước 2: Huấn luyện mô hình (Tùy chọn)
Trọng số mô hình đa nguồn đã được **huấn luyện sẵn** và lưu trong `processed/multimodal_ecg.pt`. Nếu bạn muốn chạy lại huấn luyện:
```bash
python train.py
```
Quá trình huấn luyện mô hình đa nguồn sẽ mất khoảng 15-20 phút trên CPU. Sau đó sẽ tự động cập nhật biểu đồ loss `comparison_bai23.png` và biểu đồ cột trọng số attention `attention_weights.png`.

### Bước 3: Khởi động Web Dashboard
Chạy máy chủ Flask phục vụ giao diện đồ án Bài 23:
```bash
python app.py
```
Sau đó, mở trình duyệt web và truy cập địa chỉ:
👉 **[http://127.0.0.1:5002/](http://127.0.0.1:5002/)**

### 📊 Các tính năng trên Web Dashboard:
*   Chọn danh sách các mẫu nhịp tim ở Sidebar trái.
*   Hiển thị thông tin lâm sàng bệnh nhân và biểu diễn hình thái nhịp tim thô.
*   Đồ thị sóng ECG tương tác 12 kênh động.
*   Nhấn **"Chạy Chẩn Đoán AI"** để mô hình thực hiện chẩn đoán đa nguồn và tự động vẽ động 3 biểu đồ tròn biểu thị tỷ lệ % quan trọng đóng góp của 3 nhánh thông tin.

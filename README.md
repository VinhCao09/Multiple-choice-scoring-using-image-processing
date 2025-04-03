# Multiple choice scoring using image processing
Multiple choice scoring using image processing. Train with machine learning models!

Đoạn code trên triển khai mô hình CNN (Convolutional Neural Network) sử dụng thư viện TensorFlow/Keras.
Mô hình được xây dựng bằng lớp Sequential(), có cấu trúc như sau:
- Ba khối Convolutional + MaxPooling + Dropout:

- Conv2D(32, (3,3)): Lớp tích chập với 32 filters, kernel 3x3.

- Conv2D(64, (3,3)): Tiếp tục với 64 filters.

- MaxPooling2D(pool_size=(2,2)): Giảm kích thước của ảnh xuống 1/2.

- Dropout(0.25): Giúp giảm overfitting.

Lớp Fully Connected (Dense)

- Flatten(): Chuyển ma trận thành vector.

- Dense(512, activation='relu'): Lớp 512 neurons.

- Dense(128, activation='relu'): Lớp 128 neurons.

- Dense(2, activation='softmax'): Lớp đầu ra với 2 lớp (classification 2 classes).

Mô hình dùng để làm gì?
- Nhận diện hình ảnh đã chọn hoặc chưa chọn trên phiếu trả lời trắc nghiệm.

- Dữ liệu đầu vào là ảnh 28x28 grayscale (1 kênh).

Dữ liệu huấn luyện gồm hai thư mục:

- datasets/unchoice/: Ảnh chưa chọn (gán nhãn 0).

- datasets/choice/: Ảnh đã chọn (gán nhãn 1).


![images](https://github.com/VinhCao09/Multiple-choice-scoring-using-image-processing/blob/main/img/1.jpg)



## 🔹 Lưu ý
Giải nén datasets.zip ra thành folder nằm chung với thư mục gốc.

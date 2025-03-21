# Data Mining Web

**Data Mining Web** là một ứng dụng web trực quan giúp sinh viên và người học khai phá dữ liệu có thể **upload, chỉnh sửa dữ liệu (CSV, Excel) và áp dụng các thuật toán Data Mining** một cách dễ dàng. Thay vì chạy các thuật toán trên terminal, ứng dụng này cung cấp giao diện thân thiện giúp người dùng có thể tương tác trực tiếp và quan sát kết quả một cách trực quan.

---
## Công nghệ sử dụng

- **Frontend:** React.js (Giao diện tương tác, xử lý dữ liệu phía client)
- **Backend:** Django (Xử lý yêu cầu, thực thi thuật toán)
- **Thư viện Data Mining:** scikit-learn, Pandas, NumPy, Matplotlib (Xử lý và trực quan hóa dữ liệu)

---
## Tính năng chính

- Upload dữ liệu CSV, Excel trực tiếp từ máy tính.  
- Chỉnh sửa dữ liệu ngay trên web, dễ dàng thao tác trước khi chạy thuật toán.  
- Hỗ trợ nhiều thuật toán phổ biến trong Data Mining:
   - K-Means Clustering (Phân cụm dữ liệu)
   - Decision Trees (Cây quyết định)
   - Apriori Algorithm (Tìm luật kết hợp)
- Hiển thị bảng kết quả và biểu đồ trực quan giúp dễ dàng phân tích.  
- Không lưu trữ dữ liệu, chỉ xử lý và hiển thị kết quả tạm thời.  

---
## Cách cài đặt và chạy dự án

### Cài đặt Backend (Django)
```bash
# Clone repository
git clone https://github.com/quyenptb/data-mining-web.git
cd data-mining-web/backend

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Trên macOS/Linux
venv\Scripts\activate  # Trên Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy server
python manage.py runserver
```

### Cài đặt Frontend (React.js)
```bash
cd ../frontend

# Cài đặt dependencies
npm install

# Chạy ứng dụng React
npm start
```

---
## Hướng dẫn sử dụng

1. **Tải dữ liệu lên:** Người dùng có thể tải file CSV hoặc Excel.  
2. **Chỉnh sửa dữ liệu:** Cung cấp giao diện để thao tác, chỉnh sửa dữ liệu trước khi chạy thuật toán.  
3. **Chạy thuật toán:** Chọn thuật toán mong muốn và nhấn "Run" để xem kết quả ngay lập tức.  
4. **Xem kết quả:** Ứng dụng hiển thị bảng số liệu và biểu đồ trực quan để dễ dàng phân tích.  

---
## Hình ảnh giao diện

### 🔹 Trang chủ
<div align="center">
  <a href="https://github.com/quyenptb/data-mining-web/blob/master/trangchu.png?raw=true">
    <img src="https://github.com/quyenptb/data-mining-web/blob/master/trangchu.png?raw=true" width="600">
  </a>
</div>

### 🔹 Các thuật toán hỗ trợ
<div align="center">
  <a href="https://github.com/quyenptb/data-mining-web/blob/master/thuattoan.png?raw=true">
    <img src="https://github.com/quyenptb/data-mining-web/blob/master/thuattoan.png?raw=true" width="600">
  </a>
</div>

### 🔹 Chỉnh sửa file trực tiếp
<div align="center">
  <a href="https://github.com/quyenptb/data-mining-web/blob/master/chinhsuafile.png?raw=true">
    <img src="https://github.com/quyenptb/data-mining-web/blob/master/chinhsuafile.png?raw=true" width="600">
  </a>
</div>


---



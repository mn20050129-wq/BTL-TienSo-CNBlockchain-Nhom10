# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN (SETUP GUIDE)

> Hỗ trợ chạy trên **macOS** và **Windows** (Yêu cầu Python >= 3.10).

---

## 1. Tải dự án (Clone Repository)

Mở Terminal (macOS) hoặc Command Prompt / PowerShell (Windows) và chạy:

```bash
git clone https://github.com/mn20050129-wq/BTL-TienSo-CNBlockchain-Nhom10.git
cd BTL-TienSo-CNBlockchain-Nhom10
```

---

## 2. Tạo và kích hoạt môi trường ảo (Virtual Environment)

### Trên macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Trên Windows
* **PowerShell:**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi script execution, chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi kích hoạt lại).*

* **Command Prompt (cmd):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

---

## 3. Cài đặt thư viện phụ thuộc

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Thiết lập cấu hình môi trường (.env)

Tạo file `.env` từ file mẫu:

* **macOS / Linux:**
  ```bash
  cp .env.example .env
  ```
* **Windows (PowerShell / cmd):**
  ```cmd
  copy .env.example .env
  ```

*(Tùy chọn: Mở file `.env` và điền `ALCHEMY_URL` nếu muốn kết nối RPC Mainnet trực tiếp; nếu không điền, hệ thống vẫn tự động chạy fallback offline).*

---

## 5. Chạy dự án

### Cách 1: Chạy giao diện tương tác (Streamlit Web App)
```bash
streamlit run app.py
```
*(Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`).*

### Cách 2: Chạy trực tiếp từ Terminal (Simulation Pipeline)
* **macOS / Linux:**
  ```bash
  python3 main.py
  ```
* **Windows:**
  ```cmd
  python main.py
  ```

### Cách 3: Chạy kiểm thử toàn bộ mô hình (Unit Tests)
```bash
pytest -v test_model.py
```
*(Kết quả yêu cầu: 60 passed).*

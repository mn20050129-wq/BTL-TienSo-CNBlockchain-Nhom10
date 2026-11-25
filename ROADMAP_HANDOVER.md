# TÀI LIỆU BÀN GIAO & TỔNG KẾT TOÀN DIỆN (ROADMAP HANDOVER)
## Dự Án: Hệ Thống Đánh Giá & Tối Ưu Hóa Rủi Ro Lending Crypto (Aether Risk Terminal)

---

## 1. Bảng Trạng Thái Tiến Độ Toàn Bộ Dự Án

| Checkpoint | Nội Dung Công Việc | File Trọng Tâm | Trạng Thái |
| :--- | :--- | :--- | :---: |
| **CP1** | Tối ưu hóa logic LTV (tách hàm thuần, Wilson CI95, 4 giới hạn khả thi) | `main.py`, `config.py` | **HOÀN THÀNH** |
| **CP2** | Rà soát Agent Engine (phân rã Realized/Closeout, trượt giá đỉnh, fix wrapper) | `agent_engine.py`, `risk_engine.py` | **HOÀN THÀNH** |
| **Scale** | Tham số quy mô danh mục `borrower_collateral_scale` kích hoạt thanh khoản | `config.py`, `agent_engine.py`, `app.py` | **HOÀN THÀNH** |
| **CP3** | Kiểm định Merton Jump-Diffusion & Hội tụ/Ổn định Monte Carlo | `audit_cp3_merton_mc.py`, `test_model.py` | **HOÀN THÀNH** |
| **CP4** | Thử nghiệm ứng suất Stress Test & Phân tích độ nhạy (Kịch bản 3.A, 3.B, 3.C) | `stress_test_cp4.py`, `test_model.py` | **HOÀN THÀNH** |
| **CP5** | Sơ đồ thuật toán Mermaid, Cẩm nang & Bộ câu hỏi phản biện bảo vệ BTL | `DEFENSE_GUIDE.md` | **HOÀN THÀNH** |
| **Live Setup** | Mẫu biến môi trường `.env.example` và tích hợp Streamlit Dashboard | `.env.example`, `app.py` | **HOÀN THÀNH** |

**Toàn bộ Unit Test hiện tại**: **36/36 test đã PASS hoàn toàn** (`python3 -m unittest -v test_model.py`).

---

## 2. Chi Tiết Các File Trong Hệ Thống

### 2.1 Mã Nguồn Chính
- [`config.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/config.py): Cấu hình tham số mô hình `ModelConfig`, bổ sung `borrower_collateral_scale`, `max_liquidation_probability`, `max_hf_breach_probability`, `use_ci_upper_bound`, cùng bộ validate chặt chẽ.
- [`agent_engine.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/agent_engine.py): Quần thể người vay dị thể (Lognormal), mô phỏng trễ giá Oracle Chainlink, tương tác thanh lý với độ sâu pool Uniswap V3, phân rã `Realized_Shortfall` và `Closeout_Shortfall`.
- [`simulation.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/simulation.py): Sinh đường đi giá Monte Carlo bằng quá trình Merton Jump-Diffusion với bước nhảy Poisson.
- [`data_engine.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/data_engine.py): Kéo dữ liệu giá realtime và 1,095 nến ngày lịch sử từ Binance, hiệu chỉnh tham số Merton sau khi tách cú nhảy đuôi trái $\ge 3\sigma$.
- [`onchain_engine.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/onchain_engine.py): Đọc dữ liệu on-chain trực tiếp từ RPC Alchemy: Round Chainlink ETH/USD và số dư pool Uniswap V3 WETH/USDC.
- [`main.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/main.py): Pipeline đầu cuối, thuật toán tìm kiếm LTV tối ưu dựa trên 4 tiêu chí an toàn đồng thời, tính toán độ nhạy ngưỡng thanh lý.
- [`risk_engine.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/risk_engine.py): Wrapper tương thích ngược cho một vị thế đơn lẻ.
- [`app.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/app.py): Dashboard giao diện tương tác Streamlit cao cấp, hiển thị biểu đồ rủi ro, phân tích độ nhạy và tùy biến quy mô danh mục.

### 2.2 Công Cụ Kiểm Định & Thử Nghiệm Mới
- [`audit_cp3_merton_mc.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/audit_cp3_merton_mc.py): Script thực nghiệm CP3 đo lường hiện tượng dịch chuyển $\mu, \sigma$, bất định Bootstrap do ít cú nhảy lịch sử, và đo lường độ phân tán seed Monte Carlo.
- [`stress_test_cp4.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/stress_test_cp4.py): Ma trận thử nghiệm ứng suất CP4 theo 3 kịch bản toàn diện:
  - **3.A**: Sốc giá thị trường (Normal vs Bear vs Black Swan).
  - **3.B**: Cạn kiệt thanh khoản & Gas tăng vọt (hiện tượng đóng băng thanh lý).
  - **3.C**: Ma trận độ nhạy LTV tối ưu theo APR (0%–20%) và ngưỡng chấp nhận thanh lý (2%–20%).
- [`test_model.py`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/test_model.py): Bộ 36 unit test toàn diện bảo vệ toàn bộ logic, invariant rủi ro, tính tương thích của dashboard và các kịch bản stress test.

### 2.3 Tài Liệu & Hướng Dẫn
- [`DEFENSE_GUIDE.md`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/DEFENSE_GUIDE.md): Sơ đồ thuật toán Mermaid, giải trình cơ sở toán học 4 ngưỡng khả thi, tóm tắt kết quả thực nghiệm và kịch bản câu hỏi phản biện bảo vệ BTL.
- [`.env.example`](file:///Users/ngoquanghiep/Desktop/BTL%20Blockchain/.env.example): Hướng dẫn thiết lập `ALCHEMY_URL` để chạy live dữ liệu on-chain.

---

## 3. Hướng Dẫn Chạy Thực Tế Khi Có Mạng

### Bước 1: Cấu hình khóa RPC Alchemy
Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```
Mở `.env` và dán URL API Alchemy (Ethereum Mainnet):
```env
ALCHEMY_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
```

### Bước 2: Chạy kiểm tra mã nguồn & unit test
```bash
python3 -m py_compile *.py
python3 -m unittest -v test_model.py
```

### Bước 3: Chạy các bài kiểm định & Stress test
```bash
python3 audit_cp3_merton_mc.py
python3 stress_test_cp4.py
```

### Bước 4: Khởi động Streamlit Dashboard
```bash
streamlit run app.py
```
Sau đó truy cập trình duyệt tại `http://localhost:8501`, nhấn **"▶ Chạy phân tích"** và khám phá các biểu đồ rủi ro theo LTV và APR!

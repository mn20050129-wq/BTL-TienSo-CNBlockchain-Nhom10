# Crypto-Collateralized Lending Risk Model — V2.1

Mô hình đánh giá rủi ro cho vay thế chấp ETH, đối chiếu theo hướng **Collateral Factor + Monte Carlo + Agent-Based Liquidation + dữ liệu on-chain**.

## Cấu trúc

- `data_engine.py`: giá ETH/USDT realtime và nến ngày Binance; hiệu chỉnh downside-jump diffusion.
- `simulation.py`: sinh các đường giá Monte Carlo theo Merton-style downside-jump diffusion trong log-price.
- `onchain_engine.py`: Chainlink ETH/USD và trạng thái pool ETH/USDC Uniswap V3 qua Alchemy.
- `agent_engine.py`: borrower agents không đồng nhất, protocol monitoring và liquidator execution rule; thanh lý của nhiều vị thế cùng tác động lên thanh khoản dùng chung.
- `main.py`: quét LTV, đo P(Liquidation), P(Bad Debt), Expected Loss, VaR99, CVaR99 và chọn LTV khả thi cao nhất.
- `app.py`: dashboard Streamlit.
- `config.py`: tất cả tham số cấu hình và giả định chính.

## Cài đặt và chạy trên macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Điền Ethereum Mainnet RPC URL vào `.env`:

```text
ALCHEMY_URL=https://eth-mainnet.g.alchemy.com/v2/your-api-key
```

Chạy kiểm thử kỹ thuật offline (không gọi Binance/Alchemy):

```bash
python -m unittest -v test_model.py
```

Chạy mô hình:

```bash
python main.py
```

Chạy dashboard:

```bash
streamlit run app.py
```

## Những thay đổi quan trọng ở V2.0

1. Loại bỏ nến ngày Binance chưa đóng khỏi dữ liệu hiệu chỉnh.
2. Hiệu chỉnh drift trên các ngày không được phân loại là downside jump; mô phỏng compound Poisson jump đúng với trường hợp một bước có nhiều cú nhảy.
3. Định nghĩa `mu` nhất quán là log-drift của các quan sát không có downside jump; không trừ thêm `0.5*sigma^2` hoặc jump compensator lần thứ hai.
4. Quy đổi oracle delay từ giờ sang bước thời gian và nội suy log-price cho độ trễ không nguyên bước; anchor giá Oracle ban đầu theo Chainlink.
5. Mô phỏng một danh mục borrower agents có quy mô tài sản và utilization khác nhau; thanh khoản bán ra được cộng gộp theo từng bước.
6. Liquidator chỉ thực thi khi proceeds sau slippage đủ bù khoản hoàn trả và execution cost đã cấu hình.
7. Tính VaR99 và CVaR99 theo đúng khối lượng tail 1% của phân phối thực nghiệm.
8. Đổi cách diễn giải Uniswap pool-balance TVL thành **proxy thanh khoản**, không tuyên bố đó là executable depth.
9. API key chỉ được đọc lúc cần kết nối; thiếu `.env` không còn làm lỗi import `main.py` ngay từ đầu.

## Bổ sung ở V2.1

1. Bổ sung khoảng tin cậy Wilson 95% cho `P(HF Breach)`, `P(Liquidation)` và `P(Bad Debt)` để thể hiện sai số lấy mẫu Monte Carlo. Khoảng này không bao gồm bất định tham số hoặc sai lệch mô hình.
2. `Expected Slippage Cost` hiện bao gồm slippage từ thanh lý trong kỳ và slippage theo giả định bán thu hồi tài sản còn lại tại cuối kỳ.
3. Bổ sung kiểm tra hợp lệ cho các tham số borrower, oracle, slippage và ngưỡng rủi ro.
4. Bổ sung unit test cho khoảng tin cậy Wilson và chi phí slippage của horizon closeout.

## Bổ sung ở V2.2 (Checkpoint 1 — thuật toán tối ưu LTV)

1. Thêm giới hạn `max_liquidation_probability` và `max_hf_breach_probability` (đều là **giả định quản trị rủi ro**, xem `config.py`). Trước đây chỉ có Bad Debt và CVaR nên mô hình chọn LTV chạm biên lưới dù xác suất thanh lý rất cao.
2. Điều kiện khả thi dùng **cận trên Wilson CI95** của các xác suất thay vì ước lượng điểm (`use_ci_upper_bound`).
3. Optimal LTV = LTV lớn nhất thỏa đồng thời mọi giới hạn; nếu không có mức nào, trả `None` / `NO_FEASIBLE_LTV`.
4. Đầu ra có bảng quyết định (LTV được chọn + LTV kế tiếp bị loại và lý do), cảnh báo biên lưới, cảnh báo không đơn điệu, và bảng độ nhạy của Optimal LTV theo ngưỡng thanh lý.
5. Sửa `risk_engine.simulate_dynamic_lending_position` (lỗi `n_sims` với config mặc định).

## Bổ sung ở V2.3 (Checkpoint 2 — rà soát agent_engine)

1. Phân rã Bad Debt thành **tổn thất thực tế từ thanh lý** (collateral đã bị tịch thu hết, nợ còn dư) và **thiếu hụt chỉ do giả định đóng vị thế cuối kỳ**. Không thay đổi con số cũ; chỉ thêm cột chẩn đoán.
2. Thêm `Peak Liquidation Slippage` để biết kênh thanh khoản dùng chung có thực sự hoạt động ở quy mô danh mục hiện tại hay không (mặc định: không).
3. Ghi rõ trong docstring/spec định nghĩa từng chỉ số và hai xấp xỉ đã biết (chọn liquidator một lượt, slippage do liquidator chịu).
4. Thêm test bất biến: Liquidation ⊆ HF Breach, tập HF Breach lồng nhau theo LTV, Bad Debt = Realized + Closeout, CVaR ≥ VaR, CI chứa ước lượng điểm; đã kiểm tra bằng mutation test.

## Bổ sung ở V2.4 (Chuẩn hóa ETH/FLEX/USD, Độ nhạy 4 chiều & Vòng xoáy thanh khoản nội sinh)

1. **Chuẩn hóa kiến trúc 3 tài sản ETH / FLEX / USD:**
   - Tách bạch rõ: Thế chấp = `Collateral_ETH` (token), Khoản nợ = `Debt_FLEX` (token), Đơn vị quy chuẩn = `USD`.
   - Hỗ trợ 2 chế độ giá FLEX: Mode A (Pegged $1.00) và Mode B (Scenario Depeg Stress Test $\pm 15\%$).
   - Health Factor, thanh lý và tính lãi được tính nhất quán trên số lượng token FLEX quy đổi sang USD.
2. **Cơ chế Cú sốc thanh khoản nội sinh (Endogenous Liquidity Crunch / Liquidity Death Spiral):**
   - Mô phỏng hiện tượng LP hoảng loạn rút thanh khoản khỏi Uniswap V3 (cắt giảm 40%–60% pool depth) khi giá ETH sụt giảm $\ge 15\%$ trong 24 giờ.
   - Thể hiện sự gia tăng đột biến của rủi ro đuôi (CVaR99 tăng hơn gấp đôi tại LTV 73%), đẩy LTV an toàn tối đa co hẹp từ 73% xuống vùng 65%–68%.
3. **Phân tích độ nhạy 4 chiều độc lập:**
   - Đánh giá riêng biệt tác động của từng giới hạn rủi ro: `P(Liquidation)`, `P(HF Breach)`, `P(Bad Debt)`, và `CVaR99 Ratio`.
4. **Hệ thống kiểm thử toàn diện:**
   - 60 bài unit tests (A, B, C, D, E, F) đạt chuẩn 100% PASS.

## Các giả định cần công khai trong tiểu luận

- ETH/USDT trên Binance được xem là đại diện gần đúng cho ETH/USD; chênh lệch với Chainlink được ghi nhận ở đầu vào.
- Downside jumps được nhận diện bằng ngưỡng 3 độ lệch chuẩn. Đây là quy tắc phân loại, không phải nhãn sự kiện thị trường được xác minh độc lập.
- Borrower collateral size được sinh theo phân phối lognormal và chuẩn hóa trung bình 1 ETH; utilization nằm trong khoảng cấu hình.
- Borrowers không tự nạp thêm tài sản hoặc trả nợ chủ động trong kỳ mô phỏng.
- Liquidator ra quyết định theo lợi nhuận bán tài sản sau slippage; chi phí gas mặc định 0 USD và cần được hiệu chỉnh nếu đưa vào kết luận thực nghiệm.
- Mọi lệnh thanh lý của các borrower agents trong cùng bước thời gian cùng chịu một mức slippage theo tổng notional. Đây là mô hình hóa đơn giản hóa tác động thị trường.
- Slippage dùng hàm lũy thừa trên TVL proxy; chưa phải quote thực thi hoặc tick-by-tick simulation của Uniswap V3.
- APR được kiểm tra theo các kịch bản 0%, 5%, 10%, 20% trong `borrow_apr_scenarios`. Đây là kịch bản minh họa để phân tích độ nhạy, không phải lãi suất thị trường đã được xác minh; cần thay bằng các mức APR phù hợp với giao thức hoặc phạm vi nghiên cứu.
- Ngưỡng Bad Debt 1%, CVaR99/Initial Debt 2%, P(Liquidation) 10%, P(HF Breach) 10%, liquidation threshold 85%, bonus 5%, close factor 50% là tham số cấu hình/giả định của nghiên cứu, không phải mặc nhiên là cấu hình hiện hành của mọi giao thức lending.

## Bảo mật

Không chia sẻ hoặc commit `.env`. Chỉ chia sẻ `.env.example`. Nếu API key từng được gửi trong file ZIP hoặc đưa lên nơi công khai, hãy revoke/rotate key đó tại Alchemy.

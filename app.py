from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from config import ModelConfig
from main import run_v1_6_onchain_analysis_with_inputs


st.set_page_config(
    page_title="Aether | Lending Risk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #e8eef8; --muted: #91a0b8; --panel: #111b2b; --line: #233149; --mint: #62e6bd; }
    .stApp { background: #0a111d; color: var(--ink); font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: #0e1725; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.03em; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    .brand { color: var(--mint); font: 700 0.82rem 'Space Grotesk', sans-serif; letter-spacing: .16em; text-transform: uppercase; }
    .hero { padding: 1.7rem 2rem; margin: .5rem 0 1.5rem; border: 1px solid #263650; border-radius: 18px;
            background: radial-gradient(ellipse at 90% 10%, #17384a 0, transparent 42%), linear-gradient(135deg, #121e30, #101a29); }
    .hero p { color: var(--muted); margin: .35rem 0 0; }
    .eyebrow { color: var(--mint); font-size: .76rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
    div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 14px; }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    div[data-testid="stMetricValue"] > div { color: var(--ink); }
    div[data-testid="stMetricDelta"] { font-size: .8rem; }
    .note { color: var(--muted); font-size: .86rem; }
    .empty { border: 1px dashed #31415a; border-radius: 16px; padding: 2rem; background: #101927; }
    .empty p { color: var(--muted); }
    button[kind="primary"] { background: var(--mint); color: #07131b; border: 0; font-weight: 700; }
    button[kind="primary"]:hover { background: #8af1d1; color: #07131b; }
    </style>
    """,
    unsafe_allow_html=True,
)

config = ModelConfig()
with st.sidebar:
    st.markdown('<div class="brand">◈ AETHER / RISK TERMINAL</div>', unsafe_allow_html=True)
    st.markdown("### Cấu trúc tài sản (ETH/FLEX/USD)")
    st.caption("• **Collateral**: ETH\n• **Debt**: FLEX\n• **Numeraire**: USD")
    
    st.markdown("### Chế độ giá FLEX/USD")
    flex_mode_choice = st.radio(
        "FLEX Price Mode",
        options=["Mode A: Pegged ($1.00)", "Mode B: Scenario Shock"],
        index=0,
        help="Mode A: giả định 1 FLEX = $1 cố định. Mode B: giả định kịch bản giá FLEX biến động (stress test).",
    )
    is_scenario = "Mode B" in flex_mode_choice
    flex_mode = "scenario" if is_scenario else "pegged"
    flex_shock = 0.0
    if is_scenario:
        flex_shock = st.slider(
            "Cú sốc giá FLEX (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=5.0,
            format="%+0.0f%%",
            help="Ví dụ +10%: FLEX tăng giá lên $1.10 làm nợ USD tăng và HF xấu đi; -10%: FLEX giảm giá xuống $0.90.",
        ) / 100.0

    portfolio_scale = st.number_input(
        "Quy mô danh mục (Collateral Scale)",
        min_value=0.1,
        max_value=500.0,
        value=1.0,
        step=1.0,
        help="Hệ số nhân quy mô vị thế borrower. Tăng lên (ví dụ 50x–100x) để kích hoạt tác động trượt giá thanh khoản (stress test).",
    )

    st.markdown("### Cú sốc thanh khoản nội sinh")
    st.caption(
        "**Vòng xoáy tử thần thanh khoản:** Khi ETH sập mạnh, LP rút vốn khỏi "
        "Uniswap V3 → pool depth giảm → slippage tăng → liquidator ngại thực hiện → Nợ xấu tăng."
    )
    haircut_enabled_ui = st.checkbox(
        "Bật Liquidity Haircut",
        value=False,
        help="Nếu bật, khi ETH giảm >= ngưỡng trigger trong 1 bước, pool depth bị cắt giảm ngay lập tức.",
    )
    haircut_pct_ui = 0.0
    haircut_trigger_ui = 0.15
    if haircut_enabled_ui:
        haircut_pct_ui = st.slider(
            "Mức cắt thanh khoản (%)",
            min_value=0,
            max_value=80,
            value=50,
            step=5,
            format="%d%%",
            help="Phần trăm pool depth bị rút khi trigger xảy ra. Tham khảo thực tế: 40–60% trong sự kiện 2024-08-05.",
        ) / 100.0
        haircut_trigger_ui = st.slider(
            "Ngưỡng trigger ETH drop (%)",
            min_value=5,
            max_value=40,
            value=15,
            step=5,
            format="%d%%",
            help="Nếu ETH giảm >= X% trong 1 bước, haircut được kích hoạt.",
        ) / 100.0

    config = ModelConfig(
        borrower_collateral_scale=float(portfolio_scale),
        flex_price_mode=flex_mode,
        flex_scenario_shock_pct=float(flex_shock),
        flex_price_source="stress_scenario" if is_scenario else "assumed_pegged",
        liquidity_haircut_pct=haircut_pct_ui,
        liquidity_haircut_trigger_pct=haircut_trigger_ui,
    )
    run_analysis = st.button(
        "▶  Chạy phân tích",
        type="primary",
        use_container_width=True,
    )
    st.divider()
    st.markdown("#### Cấu hình mô hình")
    st.caption(f"{config.n_sims:,} paths · {config.horizon_days} ngày · bước LTV {config.ltv_step:.0%}")
    st.caption(f"Dải LTV: {config.min_ltv:.0%}–{config.max_ltv:.0%}")
    st.caption(f"Borrower agents: {config.n_borrower_agents} · APR scenarios: {len(config.borrow_apr_scenarios)}")
    st.caption("Nguồn dữ liệu: Binance · Chainlink · Uniswap V3")
    st.divider()
    st.markdown("#### Ngưỡng rủi ro")
    if config.max_hf_breach_probability is not None:
        st.caption(f"P(HF Breach) ≤ {config.max_hf_breach_probability:.2%}")
    st.caption(f"P(Liquidation) ≤ {config.max_liquidation_probability:.2%}")
    st.caption(f"P(Bad Debt) ≤ {config.max_bad_debt_probability:.2%}")
    st.caption(f"CVaR99 / dư nợ ban đầu ≤ {config.max_cvar_ratio:.2%}")
    st.caption(
        ("Xác suất được so với cận trên Wilson CI95. " if config.use_ci_upper_bound else "Xác suất được so theo ước lượng điểm. ")
        + "Các ngưỡng là giả định quản trị rủi ro, không phải dữ liệu quan sát."
    )
    st.markdown(
        '<p class="note">Chạy local · RPC được đọc từ file <code>.env</code></p>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">ON-CHAIN · MONTE CARLO · RISK ANALYTICS</div>
      <h1 style="margin:.45rem 0 0">Crypto Lending Risk Engine</h1>
      <p>Đánh giá rủi ro thanh lý và nợ xấu để tìm tỷ lệ vay trên tài sản thế chấp tối ưu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_analysis:
    try:
        with st.spinner("Đang lấy dữ liệu thị trường/on-chain và chạy mô phỏng Monte Carlo..."):
            results_df, optimal_ltv, analysis_inputs = run_v1_6_onchain_analysis_with_inputs(config=config)
        st.session_state.analysis_results = results_df
        st.session_state.analysis_inputs = analysis_inputs
        st.session_state.optimal_ltv = optimal_ltv
        st.session_state.analysis_timestamp = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S")
    except Exception as error:
        st.error(f"Phân tích không hoàn tất: {error}")
        st.caption("Chi tiết lỗi bên dưới để xử lý:")
        st.exception(error)

if "analysis_results" not in st.session_state:
    st.markdown(
        """
        <div class="empty">
          <div class="eyebrow">SẴN SÀNG</div>
          <h3 style="margin:.5rem 0">Bắt đầu một phiên phân tích</h3>
          <p>Nhấn <b>Chạy phân tích</b> ở thanh bên. Engine sẽ truy vấn giá ETH, Chainlink, thanh khoản Uniswap V3,
          hiệu chỉnh tham số Merton và chạy mô phỏng rủi ro.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
    raise SystemExit(0)

df = st.session_state.analysis_results
inputs = st.session_state.analysis_inputs
market_data = inputs["market"]
oracle_data = inputs["chainlink"]
dex_data = inputs["uniswap"]
merton_params = inputs["merton"]
simulation_data = inputs["simulation"]
timestamp = st.session_state.analysis_timestamp

apr_options = sorted(df["Borrow APR"].dropna().unique().tolist())
selected_apr = st.selectbox(
    "Chọn kịch bản lãi suất vay (APR)",
    options=apr_options,
    format_func=lambda value: f"{value:.0%} APR" + (" · giả định minh họa" if inputs["rate_sensitivity"]["rates_are_illustrative_scenarios"] else ""),
)
view_df = df[np.isclose(df["Borrow APR"], selected_apr)].copy()
feasible_count = int(view_df["Feasible"].sum())
feasible_rows = view_df[view_df["Feasible"]]
optimal_ltv = float(feasible_rows["LTV"].max()) if not feasible_rows.empty else None
optimal_row = view_df.loc[view_df["LTV"] == optimal_ltv].iloc[0] if optimal_ltv is not None else None
st.markdown(f'<p class="note">Lần chạy gần nhất: {timestamp} · APR {selected_apr:.0%} · {len(view_df)} mức LTV được đánh giá</p>', unsafe_allow_html=True)

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Optimal LTV", f"{optimal_ltv:.0%}" if optimal_ltv is not None else "Không có")
with metric_cols[1]:
    st.metric(
        "P(Bad Debt) tại tối ưu",
        f"{optimal_row['P(Bad Debt)']:.2%}" if optimal_row is not None else "—",
        help=(
            f"Xác suất phát sinh nợ xấu. Khoảng tin cậy Wilson 95%: "
            f"{optimal_row['P(Bad Debt) CI95 Low']:.2%}–{optimal_row['P(Bad Debt) CI95 High']:.2%}. "
            "Khoảng này chỉ phản ánh sai số lấy mẫu Monte Carlo."
        ) if optimal_row is not None else "Xác suất phát sinh nợ xấu tại mức LTV tối ưu.",
    )
with metric_cols[2]:
    st.metric(
        "P(Liquidation) tại tối ưu",
        f"{optimal_row['P(Liquidation)']:.2%}" if optimal_row is not None else "—",
        help="Xác suất xảy ra ít nhất một sự kiện thanh lý.",
    )
with metric_cols[3]:
    st.metric("LTV đạt ngưỡng", f"{feasible_count} / {len(view_df)}")

overview_tab, sensitivity_tab, inputs_tab, table_tab, methodology_tab = st.tabs(
    ["Tổng quan", "Độ nhạy rủi ro", "Dữ liệu đầu vào", "Bảng kết quả", "Phương pháp"]
)

with overview_tab:
    chart_col, summary_col = st.columns([1.7, 1], gap="large")
    with chart_col:
        st.subheader("Rủi ro theo mức LTV")
        st.caption("Phân biệt xác suất HF vi phạm, thanh lý thực thi được và phát sinh nợ xấu.")
        probability_chart = view_df[["LTV", "P(HF Breach)", "P(Liquidation)", "P(Bad Debt)"]].copy()
        probability_chart["LTV"] *= 100
        probability_chart[["P(HF Breach)", "P(Liquidation)", "P(Bad Debt)"]] *= 100
        st.line_chart(
            probability_chart,
            x="LTV",
            y=["P(HF Breach)", "P(Liquidation)", "P(Bad Debt)"],
            x_label="LTV (%)",
            y_label="Xác suất (%)",
            height=340,
        )
    with summary_col:
        st.subheader("Ngưỡng khả thi")
        ordered_view = view_df.sort_values("LTV")
        current_decision = inputs["decision"]["by_apr"].get(f"{selected_apr:.4f}", {})
        if optimal_row is not None:
            st.success(f"LTV tối ưu **{optimal_ltv:.0%}** thỏa đồng thời mọi giới hạn rủi ro.")
            st.markdown(f"- P(HF Breach): **{optimal_row['P(HF Breach)']:.2%}**")
            st.markdown(f"- CVaR99 / dư nợ: **{optimal_row['CVaR99/InitialDebt']:.2%}**")
            st.markdown(f"- Expected loss: **${optimal_row['Expected Loss']:,.2f}**")
            st.markdown(f"- CVaR99: **${optimal_row['CVaR99']:,.2f}**")
            next_rows = ordered_view[ordered_view["LTV"] > optimal_ltv + 1e-9]
            if not next_rows.empty:
                nxt = next_rows.iloc[0]
                st.info(f"**Giới hạn quyết định chặn LTV tăng tiếp ({nxt['LTV']:.0%}):** {nxt['Violated Constraints']}")
            else:
                st.warning("Tối ưu nằm đúng biên trên của lưới LTV; kết quả bị chặn bởi `max_ltv`.")
            
            with st.expander("Chỉ số rủi ro cấp Borrower (Agents)"):
                st.caption("Tỷ lệ borrower gặp sự kiện trên tổng số agents:")
                st.write(f"• Từng vi phạm HF: **{optimal_row.get('Borrower HF Breach Rate', 0.0):.1%}**")
                st.write(f"• Bị thanh lý thực tế: **{optimal_row.get('Borrower Liquidation Rate', 0.0):.1%}**")
                st.write(f"• Phát sinh nợ xấu: **{optimal_row.get('Borrower Bad Debt Rate', 0.0):.1%}**")
        else:
            st.warning("Không có mức LTV nào thỏa đồng thời các giới hạn rủi ro.")
            lowest = ordered_view.iloc[0]
            st.caption(f"LTV thấp nhất ({lowest['LTV']:.0%}) vi phạm: {lowest['Violated Constraints']}")
        
        if current_decision.get("infeasible_ltv_below_optimal"):
            lows = ", ".join(f"{x:.0%}" for x in current_decision["infeasible_ltv_below_optimal"])
            st.warning(f"Cảnh báo: Có LTV không khả thi nằm dưới mức tối ưu ({lows}); đường rủi ro không hoàn toàn đơn điệu.")
        hf_text = (
            f"P(HF Breach) ≤ {config.max_hf_breach_probability:.2%}, "
            if config.max_hf_breach_probability is not None else ""
        )
        st.markdown(
            f'<p class="note">Điều kiện: {hf_text}P(Liquidation) ≤ {config.max_liquidation_probability:.2%}, '
            f'P(Bad Debt) ≤ {config.max_bad_debt_probability:.2%} '
            f'và CVaR99 / dư nợ ≤ {config.max_cvar_ratio:.2%}.</p>',
            unsafe_allow_html=True,
        )

    st.subheader("CVaR99 / dư nợ ban đầu")
    cvar_chart = view_df[["LTV", "CVaR99/InitialDebt"]].copy()
    cvar_chart["LTV"] *= 100
    cvar_chart["CVaR99/InitialDebt"] *= 100
    st.area_chart(
        cvar_chart,
        x="LTV",
        y="CVaR99/InitialDebt",
        x_label="LTV (%)",
        y_label="CVaR99 / dư nợ (%)",
        height=240,
    )

    # --- Liquidity Crunch Status ---
    lc_data = inputs.get("liquidity_crunch", {})
    if lc_data.get("haircut_active"):
        st.warning(
            f"⚠️ **Vòng xoáy thanh khoản nội sinh đang bật** — "
            f"Pool depth bị cắt **{lc_data['haircut_pct']:.0%}** khi ETH giảm >= {lc_data['trigger_pct']:.0%} trong 1 bước. "
            f"Thanh khoản cơ sở: ${lc_data['baseline_liquidity_usd']:,.0f} → "
            f"Sau cú sốc: ${lc_data['baseline_liquidity_usd'] * (1 - lc_data['haircut_pct']):,.0f}."
        )
    else:
        st.info(
            "ℹ️ **Thanh khoản cố định (baseline)** — `Liquidity Haircut` đang tắt. "
            "Bật trong sidebar để mô phỏng vòng xoáy tử thần thanh khoản DEX."
        )

with sensitivity_tab:
    st.subheader("Phân tích độ nhạy các ngưỡng rủi ro độc lập")
    st.caption(
        "Mỗi phân tích dưới đây thay đổi duy nhất MỘT ngưỡng rủi ro trên cùng bộ kết quả Monte Carlo baseline, "
        "giữ nguyên ba giới hạn còn lại để đánh giá tác động độc lập."
    )

    sens_dict = inputs["decision"].get("comprehensive_sensitivity_by_apr", {}).get(f"{selected_apr:.4f}", {})
    liq_sens = sens_dict.get("liquidation", [])
    hf_sens = sens_dict.get("hf_breach", [])
    bd_sens = sens_dict.get("bad_debt", [])
    cvar_sens = sens_dict.get("cvar_ratio", [])

    # Policy question
    st.markdown("### Câu hỏi chính sách: Nếu chấp nhận P(Liquidation) cao hơn, Optimal LTV tăng bao nhiêu?")
    if liq_sens:
        feasible_ltvs = [item["optimal_ltv"] for item in liq_sens if item["optimal_ltv"] is not None]
        highest_opt = max(feasible_ltvs) if feasible_ltvs else None
        curr_opt = optimal_ltv
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Optimal LTV hiện tại", f"{curr_opt:.0%}" if curr_opt is not None else "Không có")
        with p2:
            st.metric("Optimal LTV nới lỏng tối đa", f"{highest_opt:.0%}" if highest_opt is not None else "Không đổi")
        with p3:
            diff = f"{(highest_opt - curr_opt)*100:+.1f}%" if (highest_opt is not None and curr_opt is not None) else "0.0%"
            st.metric("Biên độ tăng LTV", diff)

    st.divider()

    def _render_sens_table(items, limit_name):
        records = []
        for it in items:
            records.append({
                f"Ngưỡng {limit_name}": f"{it['limit']:.1%}",
                "Optimal LTV": f"{it['optimal_ltv']:.0%}" if it["optimal_ltv"] is not None else "Không có LTV khả thi",
                "Trạng thái": it["status"],
                "LTV kế tiếp bị loại": f"{it['next_ltv']:.0%}" if it["next_ltv"] is not None else "—",
                "Lý do loại": it["next_ltv_violations"] or "—",
            })
        return pd.DataFrame(records)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("#### 1. Độ nhạy theo P(Liquidation)")
        st.caption("Chỉ thay đổi `max_liquidation_probability`; giữ nguyên P(HF), P(Bad Debt) và CVaR99/Debt.")
        if liq_sens:
            st.dataframe(_render_sens_table(liq_sens, "P(Liquidation)"), use_container_width=True, hide_index=True)
            liq_chart = pd.DataFrame({
                "Giới hạn P(Liq) (%)": [it["limit"] * 100 for it in liq_sens],
                "Optimal LTV (%)": [it["optimal_ltv"] * 100 if it["optimal_ltv"] is not None else None for it in liq_sens],
            }).dropna()
            if not liq_chart.empty:
                st.line_chart(liq_chart, x="Giới hạn P(Liq) (%)", y="Optimal LTV (%)", height=200)

        st.markdown("#### 3. Độ nhạy theo P(Bad Debt)")
        st.caption("Chỉ thay đổi `max_bad_debt_probability`; giữ nguyên P(Liq), P(HF) và CVaR99/Debt.")
        if bd_sens:
            st.dataframe(_render_sens_table(bd_sens, "P(Bad Debt)"), use_container_width=True, hide_index=True)
            bd_chart = pd.DataFrame({
                "Giới hạn P(Bad Debt) (%)": [it["limit"] * 100 for it in bd_sens],
                "Optimal LTV (%)": [it["optimal_ltv"] * 100 if it["optimal_ltv"] is not None else None for it in bd_sens],
            }).dropna()
            if not bd_chart.empty:
                st.line_chart(bd_chart, x="Giới hạn P(Bad Debt) (%)", y="Optimal LTV (%)", height=200)

    with col_b:
        st.markdown("#### 2. Độ nhạy theo P(HF Breach)")
        st.caption("Chỉ thay đổi `max_hf_breach_probability`; giữ nguyên P(Liq), P(Bad Debt) và CVaR99/Debt.")
        if hf_sens:
            st.dataframe(_render_sens_table(hf_sens, "P(HF Breach)"), use_container_width=True, hide_index=True)
            hf_chart = pd.DataFrame({
                "Giới hạn P(HF) (%)": [it["limit"] * 100 for it in hf_sens],
                "Optimal LTV (%)": [it["optimal_ltv"] * 100 if it["optimal_ltv"] is not None else None for it in hf_sens],
            }).dropna()
            if not hf_chart.empty:
                st.line_chart(hf_chart, x="Giới hạn P(HF) (%)", y="Optimal LTV (%)", height=200)

        st.markdown("#### 4. Độ nhạy theo CVaR99 / Initial Debt")
        st.caption("Chỉ thay đổi `max_cvar_ratio`; giữ nguyên P(Liq), P(HF) và P(Bad Debt).")
        if cvar_sens:
            st.dataframe(_render_sens_table(cvar_sens, "CVaR99 Ratio"), use_container_width=True, hide_index=True)
            cvar_chart_data = pd.DataFrame({
                "Giới hạn CVaR99/Debt (%)": [it["limit"] * 100 for it in cvar_sens],
                "Optimal LTV (%)": [it["optimal_ltv"] * 100 if it["optimal_ltv"] is not None else None for it in cvar_sens],
            }).dropna()
            if not cvar_chart_data.empty:
                st.line_chart(cvar_chart_data, x="Giới hạn CVaR99/Debt (%)", y="Optimal LTV (%)", height=200)

with inputs_tab:
    st.subheader("Dữ liệu thị trường & on-chain")
    st.caption(
        f"Dữ liệu được truy vấn cho phiên phân tích lúc {timestamp}. "
        "Các giá trị dưới đây được dùng làm đầu vào cho mô hình."
    )
    asset_info = inputs.get("asset_structure", {})
    st.info(
        f"**Kiến trúc tài sản:** Collateral: **{asset_info.get('collateral_asset', 'ETH')}** | "
        f"Debt: **{asset_info.get('debt_asset', 'FLEX')}** | "
        f"Numeraire: **{asset_info.get('numeraire', 'USD')}** | "
        f"Chế độ giá FLEX: **{asset_info.get('mode_description', 'Mode A')}** "
        f"(Nguồn: `{asset_info.get('flex_price_source', 'assumed')}`)"
    )
    input_cols = st.columns(4)
    with input_cols[0]:
        st.metric("ETH · Binance", f"${market_data['binance_eth_price']:,.2f}")
    with input_cols[1]:
        st.metric("ETH/USD · Chainlink", f"${oracle_data['oracle_price']:,.2f}")
    with input_cols[2]:
        st.metric("Tuổi Chainlink oracle", f"{oracle_data['oracle_age_hours']:.2f} giờ")
    with input_cols[3]:
        st.metric("Uniswap V3 · Pool TVL", f"${dex_data['liquidity_usd']:,.0f}")

    detail_col, model_col = st.columns(2, gap="large")
    with detail_col:
        st.markdown("#### Chainlink ETH/USD")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Thông số": "Giá oracle", "Giá trị": f"${oracle_data['oracle_price']:,.4f}"},
                    {"Thông số": "Oracle age", "Giá trị": f"{oracle_data['oracle_age_hours']:.4f} giờ"},
                    {
                        "Thông số": "Median update interval",
                        "Giá trị": f"{oracle_data['oracle_update_interval_hours']:.4f} giờ",
                    },
                    {
                        "Thông số": "P90 update interval (được dùng làm delay proxy)",
                        "Giá trị": f"{oracle_data['oracle_p90_update_interval_hours']:.4f} giờ",
                    },
                    {"Thông số": "Round ID", "Giá trị": f"{oracle_data['round_id']:,}"},
                    {"Thông số": "Số rounds lấy mẫu", "Giá trị": f"{oracle_data['sampled_rounds']:,}"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Chi tiết pool Uniswap V3"):
            st.write(f"**Pool address:** `{dex_data['pool_address']}`")
            st.write(f"**Token0 / Token1:** `{dex_data['token0']}` / `{dex_data['token1']}`")
            st.write(f"**Tick hiện tại:** {dex_data['tick']:,}")
            st.write(f"**Active liquidity:** {dex_data['active_liquidity']:,}")
            st.write(f"**USDC balance:** {dex_data['usdc_balance']:,.2f} USDC")
            st.write(f"**WETH balance:** {dex_data['weth_balance']:,.4f} WETH")

    with model_col:
        st.markdown("#### Dữ liệu hiệu chỉnh & cấu hình mô phỏng")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Thông số": "μ log-drift (annual)", "Giá trị": f"{merton_params['mu']:.2%}"},
                    {"Thông số": "Merton σ (annual)", "Giá trị": f"{merton_params['sigma']:.2%}"},
                    {"Thông số": "Jump intensity λ", "Giá trị": f"{merton_params['lam']:.4f} / năm"},
                    {"Thông số": "Jump mean μj", "Giá trị": f"{merton_params['mu_j']:.2%}"},
                    {"Thông số": "Jump volatility σj", "Giá trị": f"{merton_params['sigma_j']:.2%}"},
                    {"Thông số": "Số downside jumps", "Giá trị": f"{merton_params['jump_count']:,}"},
                    {"Thông số": "Các APR kịch bản", "Giá trị": ", ".join(f"{rate:.0%}" for rate in simulation_data['borrow_apr_scenarios'])},
                    {"Thông số": "Số borrower agents", "Giá trị": f"{inputs['agents']['n_borrower_agents']:,}"},
                    {"Thông số": "Số đường mô phỏng", "Giá trị": f"{simulation_data['paths']:,}"},
                    {
                        "Thông số": "Thời hạn / bước thời gian",
                        "Giá trị": f"{simulation_data['horizon_days']} ngày · {simulation_data['steps_per_day']} bước/ngày",
                    },
                    {
                        "Thông số": "Khoảng LTV",
                        "Giá trị": f"{simulation_data['ltv_min']:.0%}–{simulation_data['ltv_max']:.0%} (bước {simulation_data['ltv_step']:.0%})",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

with table_tab:
    st.subheader("Kết quả tại APR đang chọn")
    st.caption("Bảng hiển thị các mức LTV của APR đã chọn; file CSV tải xuống chứa tất cả APR và LTV.")
    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "LTV": st.column_config.NumberColumn("LTV", format="percent"),
            "P(HF Breach)": st.column_config.NumberColumn("P(HF Breach)", format="percent"),
            "P(Liquidation)": st.column_config.NumberColumn("P(Liquidation)", format="percent"),
            "P(Bad Debt)": st.column_config.NumberColumn("P(Bad Debt)", format="percent"),
            "Expected Loss": st.column_config.NumberColumn("Expected Loss", format="$%.2f"),
            "Expected Slippage Cost": st.column_config.NumberColumn("Expected Slippage Cost", format="$%.2f"),
            "Mean Minimum HF": st.column_config.NumberColumn("Mean Minimum HF", format="%.3f"),
            "VaR99": st.column_config.NumberColumn("VaR99", format="$%.2f"),
            "CVaR99": st.column_config.NumberColumn("CVaR99", format="$%.2f"),
            "CVaR99/InitialDebt": st.column_config.NumberColumn("CVaR99 / dư nợ", format="percent"),
            "Feasible": st.column_config.CheckboxColumn("Đạt ngưỡng"),
        },
    )
    st.download_button(
        "Tải kết quả CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="lending-risk-results.csv",
        mime="text/csv",
    )

with methodology_tab:
    st.subheader("Mô hình và cách đọc kết quả")
    st.markdown(
        r"""
        ### 1. Kiến trúc định giá & Đơn vị tài sản
        - **Collateral:** ETH (duy trì số lượng ETH trong `collateral_eth`). Giá trị USD: `Collateral_USD(t) = Collateral_ETH(t) * ETH_Oracle_Price(t)`.
        - **Debt:** FLEX (duy trì dư nợ bằng token trong `debt_flex`). Giá trị USD: `Debt_USD(t) = Debt_FLEX(t) * FLEX_Price_USD(t)`.
        - **Numeraire:** USD (dùng làm đơn vị quy đổi chuẩn để tính Health Factor, Bad Debt và tổn thất thanh lý).
        - **Health Factor:** `HF(t) = [Collateral_USD(t) * Liquidation_Threshold] / Debt_USD(t)`.
          - $HF \ge 1$: vị thế an toàn.
          - $HF < 1$: borrower đủ điều kiện xem xét thanh lý.
        - **Chế độ giá FLEX/USD:**
          - *Mode A (Pegged Debt Assumption):* Giả định 1 FLEX = $1.00 cố định (không có biến động giá nợ).
          - *Mode B (Scenario Stress Test):* Giả định kịch bản giá FLEX tăng hoặc giảm so với USD (ví dụ shock $\pm 10\%$).

        ### 2. Dữ liệu thị trường và On-Chain
        - **Dữ liệu thị trường:** Giá ETH hiện tại và lịch sử daily ETH/USDT từ Binance.
        - **Dữ liệu on-chain:** Giá Chainlink ETH/USD, tuổi oracle, và độ trễ P90 update interval làm proxy cho Oracle delay.
        - **Thanh khoản Uniswap V3:** Pool-balance TVL proxy làm mẫu số tính trượt giá (Slippage) theo hàm lũy thừa $k \cdot (\text{Flow} / \text{Liquidity})^\alpha$. TVL này là proxy thanh khoản, không phải executable depth ở mọi mức giá.

        ### 3. Quy trình Thanh lý và 3 Trạng thái phân biệt
        - **Trạng thái A (HF Breach):** Borrower chạm ngưỡng $HF < 1$.
        - **Trạng thái B (Eligibility / Attempt):** Borrower có $HF < 1$, có nợ FLEX và còn collateral ETH.
        - **Trạng thái C (Executed Liquidation):** Liquidator chỉ thực thi khi lợi nhuận bán collateral sau slippage và gas lớn hơn khoản nợ hoàn trả (`Profit > 0`). Nếu gas cao hoặc slippage lớn làm lỗ, liquidator sẽ không thanh lý!

        ### 4. Định nghĩa xác suất rủi ro cấp danh mục
        - **P(HF Breach):** Xác suất một kịch bản Monte Carlo có ít nhất 1 borrower vi phạm $HF < 1$.
        - **P(Liquidation):** Xác suất một kịch bản Monte Carlo có ít nhất 1 giao dịch thanh lý thực tế được thực thi.
        - **P(Bad Debt):** Xác suất tổng tổn thất của toàn bộ danh mục trong một kịch bản $> 0$.
        - **Expected Loss & CVaR99:** Tổn thất kỳ vọng và tổn thất trung bình trong 1% kịch bản xấu nhất (tính theo tail mass).
        - **Khoảng tin cậy Wilson 95%:** Đo lường sai số lấy mẫu Monte Carlo; quyết định khả thi được so với cận trên Wilson CI95.

        ### 5. Phân tích độ nhạy độc lập (Independent Risk Sensitivity)
        - Đánh giá riêng biệt từng giới hạn: P(Liquidation), P(HF Breach), P(Bad Debt), CVaR99 Ratio.
        - Mỗi phân tích chỉ thay đổi duy nhất một giới hạn trên cùng bộ Monte Carlo baseline, giữ nguyên ba giới hạn còn lại để đo lường chính xác tác động của từng chính sách quản trị rủi ro.

        *Lưu ý: Kết quả mô phỏng là công cụ phân tích rủi ro phục vụ nghiên cứu học thuật, không phải khuyến nghị đầu tư tài chính.*
        """
    )

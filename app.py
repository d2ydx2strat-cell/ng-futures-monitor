# 2. Storage
st.subheader("2. US Storage Levels (EIA Weekly)")

# Region selector
region_names = list(EIA_SERIES.keys())
default_region = "Lower 48 Total"
selected_region = st.selectbox("Select Region / South Central Detail", region_names, index=region_names.index(default_region))

series_id = EIA_SERIES[selected_region]
capacity_bcf = REGION_CAPACITY_BCF.get(selected_region)

storage_df = get_eia_series(EIA_API_KEY, series_id)

if storage_df is not None and not storage_df.empty:
    # Compute analytics on FULL history
    storage_df = compute_storage_analytics(storage_df)
    latest_storage = storage_df.iloc[-1]

    # Metrics (still based on full-history stats)
    current_level = latest_storage['value']
    current_delta = latest_storage['delta']
    level_5y_avg = latest_storage['level_5y_avg']
    delta_5y_avg = latest_storage['delta_5y_avg']
    level_deficit = current_level - level_5y_avg
    delta_deficit = current_delta - delta_5y_avg
    level_z = latest_storage['level_zscore']

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric(f"{selected_region} Working Gas (Bcf)",
                  f"{current_level:,.0f}",
                  delta=f"{level_deficit:,.0f} vs 5yr Avg")

    s_col2.metric("Weekly Change (Bcf)",
                  f"{current_delta:,.0f}",
                  delta=f"{delta_deficit:,.0f} vs 5yr Avg")

    s_col3.metric("Storage Level Z-Score",
                  f"{level_z:.2f}" if pd.notna(level_z) else "N/A",
                  delta="vs hist. week-of-year")

    if capacity_bcf is not None:
        pct_full = current_level / capacity_bcf * 100
        s_col4.metric("Utilization (% of Capacity)",
                      f"{pct_full:.1f}%",
                      delta=None)
    else:
        s_col4.metric("Utilization (% of Capacity)", "N/A", delta=None)

    # ---- LIMIT DISPLAY TO LAST 2 YEARS (104 WEEKS) ----
    display_window_weeks = 52 * 2
    display_df = storage_df.tail(display_window_weeks)
    recent = display_df  # for deltas / z-scores

    # --- 2A. Storage Level + Fan Chart (last 2 years only) ---
    fig_store = go.Figure()

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p90'],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p10'],
        fill='tonexty',
        fillcolor='rgba(0, 123, 255, 0.1)',
        line=dict(width=0),
        name='10–90% band',
        hoverinfo='skip'
    ))

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p75'],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p25'],
        fill='tonexty',
        fillcolor='rgba(0, 123, 255, 0.2)',
        line=dict(width=0),
        name='25–75% band',
        hoverinfo='skip'
    ))

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['p50'],
        line=dict(color='rgba(0,0,0,0.4)', dash='dash'),
        name='Median (hist.)'
    ))

    fig_store.add_trace(go.Scatter(
        x=display_df['period'],
        y=display_df['value'],
        line=dict(color='blue', width=2),
        name='Actual Storage'
    ))

    fig_store.update_layout(
        title=f"{selected_region} Storage vs Historical Distribution (Last 2 Years)",
        xaxis_title="Date",
        yaxis_title="Bcf",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_store, use_container_width=True)

    # --- 2B. Weekly Injection/Withdrawal vs 5-Year Avg (last 2 years) ---
    st.markdown("#### Storage Analytics: Weekly Balances vs History (Last 2 Years)")

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(
        x=recent['period'],
        y=recent['delta'],
        name='Actual Weekly Δ (Bcf)',
        marker_color=recent['delta'].apply(lambda x: 'red' if x < 0 else 'steelblue')
    ))
    fig_delta.add_trace(go.Scatter(
        x=recent['period'],
        y=recent['delta_5y_avg'],
        mode='lines',
        name='5yr Avg Weekly Δ',
        line=dict(color='black', dash='dash')
    ))
    fig_delta.update_layout(
        title=f"{selected_region}: Weekly Injection/Withdrawal vs 5-Year Average",
        xaxis_title="Date",
        yaxis_title="Bcf",
        height=400,
        barmode='group',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # --- 2C. Deviation & Z-Score (last 2 years) ---
    c1, c2 = st.columns(2)

    with c1:
        fig_dev = go.Figure()
        fig_dev.add_trace(go.Bar(
            x=recent['period'],
            y=recent['delta_dev_vs_5y'],
            name='Δ vs 5yr Avg (Bcf)',
            marker_color=recent['delta_dev_vs_5y'].apply(lambda x: 'red' if x < 0 else 'green')
        ))
        fig_dev.update_layout(
            title=f"{selected_region}: Weekly Deviation vs 5-Year Avg (Bcf)",
            xaxis_title="Date",
            yaxis_title="Bcf",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_dev, use_container_width=True)

    with c2:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=recent['period'],
            y=recent['delta_zscore'],
            mode='lines+markers',
            name='Weekly Δ Z-Score'
        ))
        fig_z.add_hline(y=0, line=dict(color='black', width=1))
        fig_z.add_hline(y=1.5, line=dict(color='orange', width=1, dash='dash'))
        fig_z.add_hline(y=-1.5, line=dict(color='orange', width=1, dash='dash'))
        fig_z.update_layout(
            title=f"{selected_region}: Weekly Injection/Withdrawal Z-Score",
            xaxis_title="Date",
            yaxis_title="Z-Score",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_z, use_container_width=True)

    # --- 2D. Cumulative Deviation vs 5-Year Avg (Gas Year) ---
    # This is already limited to last ~5 gas years; keep as-is
    fig_cum = go.Figure()
    for gy, sub in storage_df.groupby('gas_year'):
        if gy >= storage_df['gas_year'].max() - 4:  # last ~5 gas years
            fig_cum.add_trace(go.Scatter(
                x=sub['period'],
                y=sub['cum_dev_vs_5y'],
                mode='lines',
                name=f"Gas Year {gy}"
            ))

    fig_cum.add_hline(y=0, line=dict(color='black', width=1))
    fig_cum.update_layout(
        title=f"{selected_region}: Cumulative Deviation vs 5-Year Avg (by Gas Year)",
        xaxis_title="Date",
        yaxis_title="Cumulative Δ vs 5yr Avg (Bcf)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_cum, use_container_width=True)

else:
    st.warning(f"⚠️ Could not load storage data for {selected_region}.")

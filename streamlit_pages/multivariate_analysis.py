# streamlit_pages/multivariate_analysis.py

import streamlit as st
import pandas as pd
import numpy as np

from backend.data_service import perform_granger_causality_test
from backend.visualization_service import (
    visualize_correlation_heatmap,
    visualize_granger_causality
)

# 페이지 제목
st.title("🔄 다변량 시계열 분석")
st.markdown("대기질 변수 간의 상관관계와 인과관계를 분석합니다.")

# 데이터 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
    st.stop()

# 상관관계 분석 섹션
st.markdown("## 변수 간 상관관계 분석")

# 분석할 변수 선택
numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
selected_variables = st.multiselect(
    "분석할 변수 선택", 
    numeric_cols,
    default=["PM10", "PM25", "O3", "NO2", "CO", "SO2"][:min(4, len(numeric_cols))]
)

if len(selected_variables) >= 2:
    # 선택된 변수들로 데이터 추출
    data = st.session_state.df[selected_variables].copy()
    
    # 결측치 제거
    data = data.dropna()
    
    # 상관관계 히트맵
    st.markdown("### 상관관계 히트맵")
    corr_fig = visualize_correlation_heatmap(data)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True, theme="streamlit")
    
    # Granger 인과성 테스트 섹션
    st.markdown("## Granger 인과성 분석")
    st.markdown("""
    Granger 인과성 테스트는 한 변수가 다른 변수를 예측하는 데 도움이 되는지 검정합니다.
    "변수 X가 변수 Y를 Granger-cause 한다"는 것은 X의 과거값이 Y의 미래값을 예측하는 데 통계적으로 유의미한 정보를 제공한다는 의미입니다.
    """)
    
    # 변수 쌍 선택
    st.markdown("### 인과관계 검정할 변수 쌍 선택")
    col1, col2 = st.columns(2)
    
    with col1:
        cause_var = st.selectbox("원인 변수 (X)", selected_variables)
    
    with col2:
        effect_var = st.selectbox("결과 변수 (Y)", 
                                 [v for v in selected_variables if v != cause_var],
                                 index=0)
    
    max_lag = st.slider("최대 시차(lag)", min_value=1, max_value=24, value=12)
    
    if st.button("Granger 인과성 검정 실행", type="primary"):
        with st.spinner("인과성 검정 중..."):
            x_series = data[cause_var]
            y_series = data[effect_var]
            
            # 정상성 확인 (비정상 시계열에는 차분 적용)
            from statsmodels.tsa.stattools import adfuller
            
            x_adf = adfuller(x_series.dropna())[1] < 0.05
            y_adf = adfuller(y_series.dropna())[1] < 0.05
            
            if not x_adf:
                st.warning(f"{cause_var}가 비정상 시계열입니다. 차분을 적용합니다.")
                x_series = x_series.diff().dropna()
            
            if not y_adf:
                st.warning(f"{effect_var}가 비정상 시계열입니다. 차분을 적용합니다.")
                y_series = y_series.diff().dropna()
            
            # Granger 인과성 검정 수행
            granger_results = perform_granger_causality_test(x_series, y_series, max_lag)
            
            # 결과 표시
            st.markdown(f"### {cause_var}에서 {effect_var}로의 Granger 인과성 결과")
            
            # 결과 테이블
            results_data = []
            for lag, result in granger_results.items():
                if 'error' in result:
                    continue
                    
                results_data.append({
                    "시차(Lag)": lag,
                    "F-통계량": f"{result['ssr_ftest']['statistic']:.4f}",
                    "p-값": f"{result['ssr_ftest']['p_value']:.4f}",
                    "인과성 여부": "있음 ✓" if result['ssr_ftest']['is_causal'] else "없음 ✗"
                })
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                st.table(results_df)
                
                # p-값 목록 추출
                lags = [row["시차(Lag)"] for row in results_data]
                p_values = [float(row["p-값"]) for row in results_data]
                
                # Plotly 시각화
                granger_fig = visualize_granger_causality(lags, p_values, cause_var, effect_var)
                st.plotly_chart(granger_fig, use_container_width=True, theme="streamlit")
                
                # 종합 결과
                significant_lags = [lag for lag, p in zip(lags, p_values) if float(p) < 0.05]
                if significant_lags:
                    st.success(f"시차 {', '.join(map(str, significant_lags))}에서 {cause_var}가 {effect_var}에 Granger 인과성이 있습니다.")
                    st.markdown(f"**해석**: {cause_var}의 변화가 {significant_lags[0]}시간 후의 {effect_var} 변화에 영향을 미칠 수 있습니다.")
                else:
                    st.info(f"{cause_var}가 {effect_var}에 Granger 인과성이 없습니다.")
            else:
                st.error("인과성 검정 중 오류가 발생했습니다.")
else:
    st.info("다변량 분석을 위해 2개 이상의 변수를 선택해주세요.")

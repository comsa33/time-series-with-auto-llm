"""
홈 페이지 - 앱 소개 및 데이터 개요
"""
import streamlit as st

from frontend.components import render_data_summary, render_station_info

# 홈페이지 내용
st.title("서울시 대기질 시계열 분석", help="서울시 IoT 데이터를 활용한 시계열 분석 앱")

# 확장 가능한 앱 소개
with st.expander("📌 앱 소개 및 사용 방법", expanded=True):
    st.markdown("""
    이 앱은 서울시 대기질 데이터를 활용하여 시계열 데이터를 분석하고 시각화하는 도구입니다.

    ### 🌟 주요 기능
    1. **데이터 탐색**: 서울시 대기질 데이터의 기본 통계 및 시각화 제공
    2. **시계열 분해**: 추세(Trend), 계절성(Seasonality), 불규칙성(Irregularity) 분석
    3. **모델 비교**: ARIMA/SARIMA, 지수평활법, Prophet, LSTM 등 다양한 예측 모델 지원
    4. **예측 성능 평가**: RMSE, MAE, R² 등 다양한 메트릭 기반 평가

    ### 🛠️ 사용 방법
    1. 사이드바에서 데이터 업로드 또는 API 수집 옵션 선택
    2. 분석할 측정소와 변수(PM10, PM25 등) 선택
    3. 각 페이지에서 원하는 분석 진행하기:
       - 시계열 시각화: 기본 시계열 데이터 탐색
       - 시계열 분해: 추세, 계절성, 불규칙성 분석
       - 정상성 & ACF/PACF: 데이터의 통계적 특성 분석
       - 모델 학습/예측: 다양한 시계열 모델 학습 및 예측
       - LLM 분석: AI 기반 데이터 해석
    """)

# 데이터가 로드된 경우 데이터 개요 표시
if st.session_state.df is not None and not st.session_state.df.empty:
    # 데이터 요약 정보 렌더링
    render_data_summary(st.session_state.df)

    # 측정소 정보 렌더링 (있는 경우)
    render_station_info(st.session_state.df)
else:
    st.info("분석을 시작하려면 사이드바에서 데이터를 업로드하거나 API에서 데이터를 가져오세요.")

"""
시계열 시각화 페이지
"""
import streamlit as st

from backend.visualization_service import visualize_timeseries

# 페이지 제목
st.title("📈 시계열 시각화")
st.markdown("서울시 대기질 데이터의 시계열 시각화를 제공합니다.")

# 데이터 및 시계열 정보 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
else:
    # 시계열 데이터 정보 표시
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric(
            label="데이터 길이",
            value=f"{len(st.session_state.series):,}",
            help="시계열 데이터 포인트 수",
            border=True
        )
    
    with stat_col2:
        st.metric(
            label="시작 날짜",
            value=f"{st.session_state.series.index.min().strftime('%Y-%m-%d %H:%M')}",
            help="첫 번째 데이터 시점",
            border=True
        )
    
    with stat_col3:
        st.metric(
            label="종료 날짜",
            value=f"{st.session_state.series.index.max().strftime('%Y-%m-%d %H:%M')}",
            help="마지막 데이터 시점",
            border=True
        )
    
    # 시계열 기본 통계
    st.markdown("### 기본 통계량")
    stats_df = st.session_state.series.describe().to_frame().T
    st.dataframe(stats_df, use_container_width=True)
    
    # 시계열 그래프 표시
    st.markdown("### 시계열 그래프")
    
    fig = visualize_timeseries()
    if fig:
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    else:
        st.error("시계열 그래프를 생성할 수 없습니다.")
    
    # 추가 설명 및 도움말
    with st.expander("📌 시계열 분석 도움말", expanded=False):
        st.markdown("""
        ### 📊 시계열 데이터 분석이란?
        
        시계열 데이터는 시간에 따라 순차적으로 기록된 데이터를 의미합니다. 이러한 데이터는 일반적인 정적 데이터와 달리 시간에 따른 
        패턴, 추세, 계절성, 주기성 등의 특성을 가집니다.
        
        ### 📈 시계열 데이터의 주요 특성
        
        1. **추세(Trend)**: 장기간에 걸친 데이터의 전반적인 상승 또는 하락 패턴
        2. **계절성(Seasonality)**: 일정한 시간 간격으로 반복되는 패턴 (예: 24시간 주기, 7일 주기, 12개월 주기 등)
        3. **주기성(Cyclicity)**: 비고정적 기간 동안의 상승과 하락 패턴 (경기 순환 등)
        4. **불규칙성(Irregularity)**: 예측할 수 없는 무작위 변동
        
        ### 🔍 다음 단계로 시도해 볼 수 있는 분석
        
        - **시계열 분해**: 다음 페이지에서 데이터를 추세, 계절성, 잔차로 분해할 수 있습니다.
        - **정상성 검정**: 데이터가 시간에 관계없이 일정한 통계적 속성을 가지는지 확인할 수 있습니다.
        - **모델 학습**: 분석한 특성을 바탕으로 예측 모델을 학습할 수 있습니다.
        """)

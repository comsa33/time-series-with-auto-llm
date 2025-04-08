"""
시계열 분해 페이지
"""
import streamlit as st

from backend.data_service import analyze_decomposition
from backend.visualization_service import visualize_decomposition

# 페이지 제목
st.title("🔄 시계열 분해")
st.markdown("시계열 데이터를 추세, 계절성, 잔차로 분해하여 분석합니다.")

# 시계열 분해 도움말
with st.expander("시계열 분해란?", expanded=True):
    st.markdown("""
    ### 🔄 시계열 분해(Time Series Decomposition)
    
    시계열 분해란, :orange[시계열 데이터를 구성하는 여러 요소(성분)]를 분리해내는 기법입니다. 이를 통해 데이터의 구조를 더 잘 이해하고, 예측력 높은 모델을 만들 수 있습니다.
    
    ##### 주요 구성 요소
    
    1. **관측값(Observed)**: 원본 시계열 데이터
    2. **추세(Trend)**: 데이터의 장기적인 상승 또는 하락 패턴
    3. **계절성(Seasonality)**: 일정한 주기로 반복되는 패턴
    4. **잔차(Residual)**: 추세와 계절성을 제거한 후 남는 불규칙한 변동
    
    ##### 분해 방식
    
    - **가법(Additive) 모델**: 원본 = 추세 + 계절성 + 잔차
    - **승법(Multiplicative) 모델**: 원본 = 추세 × 계절성 × 잔차
    
    일반적으로 시간에 따라 변동의 크기가 일정하면 가법 모델, 변동의 크기가 증가하면 승법 모델을 사용합니다.
    """)

# 데이터 및 시계열 정보 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
else:
    # 계절성 주기 선택
    min_period = 2
    max_period = min(len(st.session_state.series) // 2, 168)  # 최대 일주일(168시간) 또는 데이터 길이의 절반
    
    st.markdown("### 계절성 주기 설정")
    period = st.slider(
        "계절성 주기 (시간 단위)",
        min_value=min_period,
        max_value=max_period,
        value=st.session_state.period,
        help="계절성 주기를 선택하세요. 예: 24시간은 하루 주기를 의미합니다."
    )
    st.session_state.period = period
    
    # 계절성 주기 선택에 따른 설명
    if period == 24:
        st.info("24시간 주기는 일별 패턴을 분석합니다. 예: 매일 출퇴근 시간의 변화")
    elif period == 168:
        st.info("168시간(7일) 주기는 주간 패턴을 분석합니다. 예: 주중과 주말의 차이")
    elif period == 8760:
        st.info("8760시간(365일) 주기는 연간 패턴을 분석합니다. 예: 계절에 따른 변화")
    
    # 시계열 분해 버튼
    if st.button("시계열 분해 실행", type="primary"):
        with st.spinner("시계열 분해 중..."):
            # 시계열 분해 수행
            decomposition = analyze_decomposition(period)
            
            if decomposition:
                st.success("시계열 분해가 완료되었습니다.")
            else:
                st.error("시계열 분해 중 오류가 발생했습니다.")
    
    # 분해 결과 시각화
    if st.session_state.decomposition:
        st.markdown("### 시계열 분해 결과")
        
        # 분해 결과 요약
        trend = st.session_state.decomposition['trend'].dropna()
        seasonal = st.session_state.decomposition['seasonal'].dropna()
        resid = st.session_state.decomposition['resid'].dropna()
        
        # 메트릭 표시
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                label="추세 변동 범위",
                value=f"{trend.min():.2f} ~ {trend.max():.2f}",
                delta=f"{trend.max() - trend.min():.2f}",
                help="추세 성분의 최소값과 최대값 범위",
                border=True
            )
        
        with metric_col2:
            st.metric(
                label="계절성 변동 범위",
                value=f"{seasonal.min():.2f} ~ {seasonal.max():.2f}",
                delta=f"{seasonal.max() - seasonal.min():.2f}",
                help="계절성 성분의 최소값과 최대값 범위",
                border=True
            )
        
        with metric_col3:
            st.metric(
                label="잔차 변동 범위",
                value=f"{resid.min():.2f} ~ {resid.max():.2f}",
                delta=f"{resid.max() - resid.min():.2f}",
                help="잔차 성분의 최소값과 최대값 범위",
                border=True
            )
        
        # 분해 시각화
        decomp_fig = visualize_decomposition()
        if decomp_fig:
            st.plotly_chart(decomp_fig, use_container_width=True, theme="streamlit")
        else:
            st.error("분해 결과 시각화에 실패했습니다.")
        
        # 결과 해석 도움말
        with st.expander("분해 결과 해석 방법", expanded=False):
            st.markdown("""
            ### 📊 분해 결과 해석 방법
            
            #### 1. 추세(Trend) 성분
            - **상승 추세**: 시간이 지남에 따라 값이 증가하는 경향
            - **하락 추세**: 시간이 지남에 따라 값이 감소하는 경향
            - **안정적**: 뚜렷한 상승이나 하락 없이 일정한 범위 내에서 변동
            
            #### 2. 계절성(Seasonality) 성분
            - **강한 계절성**: 패턴이 명확하고 규칙적으로 반복됨
            - **약한 계절성**: 패턴이 존재하지만 약하거나 불규칙적
            - **계절성 주기**: 설정한 주기(이 경우 {st.session_state.period}시간)에 따른 반복 패턴
            
            #### 3. 잔차(Residual) 성분
            - **무작위적 잔차**: 특별한 패턴 없이 무작위로 분포 (이상적)
            - **패턴이 있는 잔차**: 여전히 패턴이 존재한다면 분해가 완전하지 않음을 의미
            - **큰 잔차**: 특정 시점에 큰 잔차는 이상값(outlier)이나 특별한 이벤트를 나타낼 수 있음
            
            ### 🔍 다음 단계 분석 제안
            
            1. **다양한 주기 시도**: 다른 주기 값으로 분해를 시도하여 데이터에 맞는 최적의 주기 찾기
            2. **정상성 검정**: 잔차의 정상성을 확인하여 추세와 계절성이 잘 제거되었는지 확인
            3. **모델링**: 분해 결과를 바탕으로 적절한 시계열 모델 선택 (ARIMA, 지수평활법 등)
            """)
    else:
        st.info("시계열 분해를 실행하여 결과를 확인하세요.")

"""
차분 분석 페이지
"""
import streamlit as st

from backend.data_service import (
    analyze_differencing_need,
    perform_differencing,
    prepare_differenced_train_test_data,
    analyze_stationarity,
    cached_check_stationarity
)
from backend.visualization_service import (
    visualize_timeseries,
    visualize_differencing_comparison,
)

# 페이지 제목
st.title("🔄 차분 분석 (Differencing Analysis)")
st.markdown("시계열 데이터의 정상성을 확인하고 차분을 통해 정상화합니다.")

# 데이터 및 시계열 정보 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
    st.stop()
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
    st.stop()

# 차분 분석 설명
with st.expander("차분 분석(Differencing Analysis)이란?", expanded=True):
    st.markdown("""
    ### 🔄 차분 분석 (Differencing Analysis)
    
    **차분(Differencing)**은 시계열 데이터를 정상화(stationary)하는 중요한 기법입니다. 정상성(stationarity)이란 시간에 따라 통계적 특성(평균, 분산 등)이 변하지 않는 성질을 말합니다.
    
    #### 차분의 종류:
    
    1. **일반 차분(Regular Differencing)**: 연속된 관측값 간의 차이를 계산합니다.
       - 1차 차분: y'(t) = y(t) - y(t-1)
       - 2차 차분: y''(t) = y'(t) - y'(t-1) = y(t) - 2y(t-1) + y(t-2)
    
    2. **계절 차분(Seasonal Differencing)**: 계절성 주기만큼 떨어진 관측값과의 차이를 계산합니다.
       - s기간 계절 차분: y'(t) = y(t) - y(t-s)
    
    #### 차분이 필요한 경우:
    
    - **추세(Trend)가 있는 경우**: 시간에 따라 상승하거나 하락하는 경향이 있을 때
    - **계절성(Seasonality)이 있는 경우**: 일정한 주기로 패턴이 반복될 때
    - **분산이 증가하는 경우**: 시간에 따라 변동폭이 증가할 때
    
    #### 차분의 효과:
    
    - **추세 제거**: 일반 차분은 선형 추세를 제거합니다.
    - **계절성 제거**: 계절 차분은 계절적 패턴을 제거합니다.
    - **정상성 확보**: ARIMA 등의 모델은 정상 시계열을 가정하므로, 차분을 통해 정상성을 확보해야 합니다.
    """)

# 정상성 검정 섹션
st.markdown("## 1️⃣ 정상성 검정 (Stationarity Test)")
st.markdown("차분 적용 전 시계열의 정상성을 확인합니다.")

# 원본 시계열 그래프 표시
st.markdown("### 📈 원본 시계열")
timeseries_fig = visualize_timeseries()
if timeseries_fig:
    st.plotly_chart(timeseries_fig, use_container_width=True, theme="streamlit")

# 정상성 검정 실행
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("정상성 검정 실행", type="primary", use_container_width=True):
        with st.spinner("정상성 검정 중..."):
            # 정상성 검정 수행
            stationarity_result = analyze_stationarity()
            
            if stationarity_result:
                st.success("정상성 검정이 완료되었습니다.")
            else:
                st.error("정상성 검정 중 오류가 발생했습니다.")

# 정상성 검정 결과 표시
if st.session_state.stationarity_result:
    # 정상성 결과 컨테이너
    with st.container():
        # 정상성 여부 먼저 큰 글씨로 표시
        if st.session_state.stationarity_result['is_stationary']:
            st.success("### ✅ 시계열 데이터가 정상성을 만족합니다")
            st.markdown("차분이 필요하지 않을 수 있습니다. 하지만 모델에 따라 차분이 도움이 될 수도 있습니다.")
        else:
            st.warning("### ⚠️ 시계열 데이터가 정상성을 만족하지 않습니다")
            st.markdown("차분을 통해 시계열을 정상화하는 것이 좋습니다.")
            
        # 메트릭 표시를 위한 3개 컬럼
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        # ADF 통계량 (첫 번째 메트릭)
        test_stat = st.session_state.stationarity_result['test_statistic']
        critical_1pct = st.session_state.stationarity_result['critical_values']['1%']
        # ADF 통계량과 1% 임계값의 차이
        delta_adf = test_stat - critical_1pct
        
        # 시각화: ADF 통계량이 임계값보다 작으면 좋은 것이므로 delta_color="inverse" 사용
        metric_col1.metric(
            label="ADF 통계량",
            value=f"{test_stat:.4f}",
            delta=f"{delta_adf:.4f}",
            delta_color="inverse",
            help="ADF 통계량이 임계값보다 작을수록 정상성 가능성이 높습니다",
            border=True
        )
        
        # p-값 (두 번째 메트릭)
        p_value = st.session_state.stationarity_result['p_value']
        # p-값과 0.05의 차이
        delta_p = p_value - 0.05
        
        # 시각화: p-값이 작을수록 좋은 것이므로 delta_color="inverse" 사용
        metric_col2.metric(
            label="p-값",
            value=f"{p_value:.4f}",
            delta=f"{delta_p:.4f}",
            delta_color="inverse",
            help="p-값이 0.05보다 작으면 정상성을 만족합니다",
            border=True
        )
        
        # 관측 수 (세 번째 메트릭)
        num_obs = st.session_state.stationarity_result['num_observations']
        metric_col3.metric(
            label="관측 데이터 수",
            value=f"{num_obs:,}",
            help="정상성 검정에 사용된 데이터 수",
            border=True
        )
else:
    st.info("정상성 검정을 실행하세요.")

# 차분 추천 분석 섹션
st.markdown("---")
st.markdown("## 2️⃣ 차분 추천 분석")

with col2:
    if st.button("차분 권장 분석", use_container_width=True):
        with st.spinner("차분 분석 중..."):
            # 차분 추천 분석 수행
            recommendation = analyze_differencing_need()
            
            if recommendation:
                st.success("차분 분석이 완료되었습니다.")
            else:
                st.error("차분 분석 중 오류가 발생했습니다.")

# 차분 추천 결과 표시
if hasattr(st.session_state, 'differencing_recommendation') and st.session_state.differencing_recommendation:
    # 차분 추천 결과 컨테이너
    with st.container():
        # 차분 필요 여부 표시
        if st.session_state.differencing_recommendation['needs_differencing']:
            st.warning("### ⚠️ 차분이 권장됩니다")
        else:
            st.success("### ✅ 차분이 필요하지 않을 수 있습니다")
        
        # 추천 이유 표시
        st.markdown("#### 분석 결과:")
        for reason in st.session_state.differencing_recommendation['reason']:
            st.markdown(f"- {reason}")
        
        # 차분 권장사항 표시
        st.markdown("#### 권장 차분 설정:")
        
        # 차분 설정을 위한 컬럼
        param_col1, param_col2, param_col3 = st.columns(3)
        
        # 일반 차분 차수
        diff_order = st.session_state.differencing_recommendation['diff_order']
        param_col1.metric(
            label="일반 차분 차수",
            value=f"{diff_order}차",
            help="연속된 시점 간의 차분 차수",
            border=True
        )
        
        # 계절 차분 차수
        seasonal_diff_order = st.session_state.differencing_recommendation['seasonal_diff_order']
        param_col2.metric(
            label="계절 차분 차수",
            value=f"{seasonal_diff_order}차",
            help="계절 주기 간의 차분 차수",
            border=True
        )
        
        # 계절성 주기
        seasonal_period = st.session_state.differencing_recommendation['seasonal_period']
        param_col3.metric(
            label="계절성 주기",
            value=f"{seasonal_period or '없음'}",
            help="계절성 패턴의 주기 (시간 단위)",
            border=True
        )
else:
    st.info("차분 권장 분석을 실행하세요.")

# 차분 적용 섹션
st.markdown("---")
st.markdown("## 3️⃣ 차분 적용")

# 차분 설정 UI
with st.form(key="differencing_form"):
    st.markdown("### 차분 파라미터 설정")
    
    # 권장값 사용 여부
    use_recommended = False
    if hasattr(st.session_state, 'differencing_recommendation') and st.session_state.differencing_recommendation:
        use_recommended = st.checkbox("권장 차분 설정 사용", value=True)
    
    # 차분 파라미터 설정
    diff_col1, diff_col2, diff_col3 = st.columns(3)
    
    with diff_col1:
        # 권장값이 있으면 사용, 없으면 현재 설정 또는 0
        default_diff_order = 0
        if use_recommended and hasattr(st.session_state, 'differencing_recommendation'):
            default_diff_order = st.session_state.differencing_recommendation['diff_order']
        elif hasattr(st.session_state, 'diff_order'):
            default_diff_order = st.session_state.diff_order
            
        diff_order = st.number_input(
            "일반 차분 차수",
            min_value=0,
            max_value=2,
            value=default_diff_order,
            help="연속된 시점 간의 차분 횟수 (0~2 권장)"
        )
    
    with diff_col2:
        # 권장값이 있으면 사용, 없으면 현재 설정 또는 0
        default_seasonal_diff_order = 0
        if use_recommended and hasattr(st.session_state, 'differencing_recommendation'):
            default_seasonal_diff_order = st.session_state.differencing_recommendation['seasonal_diff_order']
        elif hasattr(st.session_state, 'seasonal_diff_order'):
            default_seasonal_diff_order = st.session_state.seasonal_diff_order
            
        seasonal_diff_order = st.number_input(
            "계절 차분 차수",
            min_value=0,
            max_value=1,
            value=default_seasonal_diff_order,
            help="계절 주기 간의 차분 횟수 (0~1 권장)"
        )
    
    with diff_col3:
        # 권장값이 있으면 사용, 없으면 현재 설정 또는 기본 주기
        default_seasonal_period = st.session_state.period
        if use_recommended and hasattr(st.session_state, 'differencing_recommendation') and st.session_state.differencing_recommendation['seasonal_period']:
            default_seasonal_period = st.session_state.differencing_recommendation['seasonal_period']
            
        seasonal_period = st.selectbox(
            "계절성 주기",
            options=[24, 168, 720],  # 일별(24시간), 주별(168시간), 월별(30일) 주기
            index=0,  # 기본값: 일별 주기
            format_func=lambda x: f"{x}시간 ({x//24}일)" if x >= 24 else f"{x}시간",
            help="계절성 패턴의 주기 (시간 단위)"
        )
    
    # 모델에 차분 적용 여부
    st.markdown("### 모델 학습 설정")
    use_differencing = st.checkbox(
        "모델 학습에 차분 데이터 사용 (ARIMA, 지수평활법에만 적용)",
        value=False,
        help="차분된 데이터로 ARIMA와 지수평활법 모델을 학습하고 예측합니다. LSTM과 Prophet 모델은 항상 원본 데이터를 사용합니다. 예측 결과는 자동으로 원래 스케일로 변환됩니다."
    )
    
    # 차분 적용 버튼
    submit_button = st.form_submit_button(label="차분 적용", type="primary")
    
    if submit_button:
        # 세션 상태 업데이트
        st.session_state.diff_order = diff_order
        st.session_state.seasonal_diff_order = seasonal_diff_order
        st.session_state.use_differencing = use_differencing
        
        with st.spinner("차분 적용 중..."):
            # 차분 수행
            differenced_series = perform_differencing(diff_order, seasonal_diff_order, seasonal_period)
            
            if differenced_series is not None:
                # 차분된 데이터 분할
                prepare_differenced_train_test_data()
                st.success("차분이 성공적으로 적용되었습니다.")
            else:
                st.error("차분 적용 중 오류가 발생했습니다.")

# 차분 결과 표시
if st.session_state.differenced_series is not None:
    st.markdown("### 차분 결과")
    
    # 차분 전후 비교 시각화
    diff_fig = visualize_differencing_comparison()
    if diff_fig:
        st.plotly_chart(diff_fig, use_container_width=True, theme="streamlit")
    
    # 차분된 데이터에 대한 정상성 검정
    st.markdown("### 차분 후 정상성 검정")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("차분 후 정상성 검정", type="primary", use_container_width=True):
            with st.spinner("정상성 검정 중..."):
                # 차분된 데이터에 대한 정상성 검정 수행
                diff_stationarity = cached_check_stationarity(st.session_state.differenced_series)
                
                # 결과 표시
                if diff_stationarity['is_stationary']:
                    st.success("### ✅ 차분 후 시계열이 정상성을 만족합니다")
                else:
                    st.warning("### ⚠️ 차분 후에도 시계열이 정상성을 만족하지 않습니다")
                    st.markdown("더 높은 차수의 차분이 필요할 수 있습니다.")
                
                # 메트릭 표시
                metric_col1, metric_col2 = st.columns(2)
                
                # ADF 통계량
                test_stat = diff_stationarity['test_statistic']
                metric_col1.metric(
                    label="ADF 통계량",
                    value=f"{test_stat:.4f}",
                    help="ADF 통계량이 임계값보다 작을수록 정상성 가능성이 높습니다",
                    border=True
                )
                
                # p-값
                p_value = diff_stationarity['p_value']
                metric_col2.metric(
                    label="p-값",
                    value=f"{p_value:.4f}",
                    help="p-값이 0.05보다 작으면 정상성을 만족합니다",
                    border=True
                )
    
    with col2:
        if st.button("차분 결과 ACF/PACF", use_container_width=True):
            with st.spinner("ACF/PACF 분석 중..."):
                # 차분된 데이터에 대한 ACF, PACF 계산
                from backend.data_service import cached_get_acf_pacf
                diff_acf, diff_pacf = cached_get_acf_pacf(st.session_state.differenced_series)
                
                # ACF, PACF 시각화
                from utils.visualizer import cached_plot_acf_pacf
                acf_pacf_fig = cached_plot_acf_pacf(diff_acf, diff_pacf)
                
                if acf_pacf_fig:
                    st.plotly_chart(acf_pacf_fig, use_container_width=True, theme="streamlit")
                else:
                    st.error("ACF/PACF 그래프 생성에 실패했습니다.")
    
    # 차분 데이터 통계 정보
    st.markdown("### 차분 데이터 통계 정보")
    # 전치 없이 describe() 결과를 그대로 사용하고 열 이름만 변경
    stats_df = st.session_state.differenced_series.describe().to_frame()
    stats_df.columns = [f"차분된 {st.session_state.selected_target}"]
    st.dataframe(stats_df, use_container_width=True)

# 모델링 가이드
st.markdown("---")
st.markdown("## 다음 단계: 모델링")

if st.session_state.use_differencing:
    st.success("차분 데이터로 모델 학습이 설정되었습니다. '모델 학습 및 예측' 페이지로 이동하여 모델을 학습하세요.")
else:
    st.info("차분 데이터로 모델 학습하려면 '모델 학습에 차분 데이터 사용' 옵션을 활성화하세요.")

with st.expander("차분 데이터 모델링 가이드", expanded=False):
    st.markdown("""
    ### 차분 데이터를 이용한 모델링 가이드
    
    #### 1. ARIMA/SARIMA 모델
    - 차분을 적용했다면 ARIMA 모델의 d 파라미터를 줄일 수 있습니다 (이미 차분이 적용됨).
    - 예: 1차 차분 적용 후 ARIMA(p,1,q) 대신 ARIMA(p,0,q) 사용 가능
    
    #### 2. 지수평활법 모델
    - 차분한 데이터에 지수평활법을 적용하면 복잡한 추세와 계절성을 더 잘 포착할 수 있습니다.
    
    #### 3. LSTM 모델
    - LSTM 모델은 항상 원본 데이터를 사용합니다. 차분 설정은 적용되지 않습니다.
    - LSTM은 복잡한 패턴을 학습할 수 있는 딥러닝 모델이므로 차분 없이도 효과적입니다.
    
    #### 4. Prophet 모델
    - Prophet 모델은 항상 원본 데이터를 사용합니다. 차분 설정은 적용되지 않습니다.
    - Prophet은 내부적으로 추세와 계절성을 모델링하므로 원본 데이터 사용이 권장됩니다.
    
    > **참고**: 모델 학습 및 예측 페이지에서 차분 데이터로 모델을 학습하면, ARIMA와 지수평활법 모델에만 적용되며 예측 결과는 자동으로 원래 스케일로 변환됩니다.
    """)

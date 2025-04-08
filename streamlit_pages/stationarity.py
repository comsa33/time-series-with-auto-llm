"""
정상성 & ACF/PACF 페이지
"""
import streamlit as st

from backend.data_service import analyze_stationarity, analyze_acf_pacf
from backend.visualization_service import visualize_acf_pacf

# 페이지 제목
st.title("🔍 정상성 & ACF/PACF 분석")
st.markdown("시계열 데이터의 정상성을 검정하고 자기상관 분석을 수행합니다.")

# 데이터 및 시계열 정보 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
else:
    # 주요 섹션
    st.markdown("## 정상성 검정")
    
    # 정상성 검정 도움말
    with st.expander("정상성 검정이란?", expanded=True):
        st.markdown("""
        ### 🔍 정상성 검정(Stationarity Test)
        
        정상성 검정이란 :orange[시계열 데이터가 시간이 지나도 통계적 특성이 일정한지(=정상인지) 확인하는 검정]입니다. 즉, 평균, 분산, 자기공분산 등의 값이 시간에 따라 변하지 않는지를 확인하는 것입니다.
        
        ##### 정상성의 중요성
        
        대부분의 시계열 모델링 기법은 데이터가 정상성을 만족한다는 가정 하에 적용됩니다. 정상성을 만족하지 않는 데이터에 이러한 모델을 적용하면 예측 성능이 떨어질 수 있습니다.
        
        ##### 검정 방법
        
        여기서는 ADF(Augmented Dickey-Fuller) 검정을 사용합니다:
        - **귀무가설**: 시계열이 단위근을 가짐 (비정상)
        - **대립가설**: 시계열이 단위근을 가지지 않음 (정상)
        
        **`p-값`이 `0.05`보다 작으면** 귀무가설을 기각하고, 시계열이 정상성을 만족한다고 볼 수 있습니다.
        """)
    
    # 정상성 검정 버튼
    if st.button("정상성 검정 실행", type="primary"):
        with st.spinner("정상성 검정 중..."):
            # 정상성 검정 수행
            stationarity_result = analyze_stationarity()
            
            if stationarity_result:
                st.success("정상성 검정이 완료되었습니다.")
            else:
                st.error("정상성 검정 중 오류가 발생했습니다.")
    
    # 정상성 검정 결과 표시
    if st.session_state.stationarity_result:
        # 시각적 구분선 추가
        st.markdown("---")
        
        # 정상성 결과 컨테이너
        with st.container():
            # 정상성 여부 먼저 큰 글씨로 표시
            if st.session_state.stationarity_result['is_stationary']:
                st.success("### ✅ 시계열 데이터가 정상성을 만족합니다")
            else:
                st.warning("### ⚠️ 시계열 데이터가 정상성을 만족하지 않습니다")
                
            # 설명 추가
            with st.expander("정상성 판단 기준 설명", expanded=False):
                st.markdown("""
                - **ADF 통계량**이 임계값보다 **작을수록** 정상성 가능성이 높습니다
                - **p-값**이 0.05보다 **작으면** 정상성을 만족합니다
                - ADF 통계량이 임계값보다 작을수록, 그리고 p-값이 작을수록 정상성 가능성이 높습니다
                """)
            
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
            
            # 임계값 카드
            st.markdown("### 📊 임계값 (Critical Values)")
            
            # 임계값 표시를 위한 3개 컬럼
            crit_col1, crit_col2, crit_col3 = st.columns(3)
            
            # 각 임계값을 메트릭으로 표시
            for i, (key, value) in enumerate(st.session_state.stationarity_result['critical_values'].items()):
                # ADF 통계량과 임계값의 차이
                delta_crit = test_stat - value
                # 색상 설정: ADF 통계량이 임계값보다 작으면 좋은 것이므로 inverse 사용
                color_setting = "inverse"
                
                # 각 컬럼에 임계값 메트릭 추가
                if i == 0:  # 1% 임계값
                    crit_col1.metric(
                        label=f"임계값 ({key})",
                        value=f"{value:.4f}",
                        delta=f"{delta_crit:.4f}",
                        delta_color=color_setting,
                        help=f"ADF 통계량이 {key} 임계값보다 작으면 {key} 유의수준에서 정상성 만족",
                        border=True
                    )
                elif i == 1:  # 5% 임계값
                    crit_col2.metric(
                        label=f"임계값 ({key})",
                        value=f"{value:.4f}",
                        delta=f"{delta_crit:.4f}",
                        delta_color=color_setting,
                        help=f"ADF 통계량이 {key} 임계값보다 작으면 {key} 유의수준에서 정상성 만족",
                        border=True
                    )
                elif i == 2:  # 10% 임계값
                    crit_col3.metric(
                        label=f"임계값 ({key})",
                        value=f"{value:.4f}",
                        delta=f"{delta_crit:.4f}",
                        delta_color=color_setting,
                        help=f"ADF 통계량이 {key} 임계값보다 작으면 {key} 유의수준에서 정상성 만족",
                        border=True
                    )
    
    # ACF, PACF 분석 섹션
    st.markdown("---")
    st.markdown("## ACF/PACF 분석")
    
    # ACF/PACF 도움말
    with st.expander("ACF/PACF 분석이란?", expanded=True):
        st.markdown("""
        ### 📊 ACF/PACF 분석
        
        ACF(Autocorrelation Function)와 PACF(Partial Autocorrelation Function)는 시계열 데이터의 자기상관 특성을 분석하는 도구로, 시계열 모델의 파라미터 선택에 중요한 역할을 합니다.
        
        ##### 🔹 ACF (Autocorrelation Function, 자기상관함수)
        - 현재 시점의 값과 이전 시점들의 값들(lag) 간의 상관관계를 측정
        - 여러 시차(lag)에 걸친 전체적인 상관성을 파악함
        - AR(p) 모델에서 p값 추정에 도움
        
        ##### 🔹 PACF (Partial Autocorrelation Function, 부분 자기상관함수)
        - 중간에 끼어 있는 시점들의 영향을 제거하고, 지정한 lag와 직접적인 상관만 추정
        - 즉, lag-k와 현재 시점 사이의 순수한 직접 관계만 보는 것
        - AR(p) 모델에서 p의 결정에 매우 중요
        
        ##### 📊 해석 방법
        - **ACF가 점차 감소**: AR 모델 특성
        - **PACF가 특정 lag 이후 절단**: AR(p) 모델에서 p는 절단 시점
        - **ACF가 특정 lag 이후 절단**: MA(q) 모델에서 q는 절단 시점
        - **둘 다 점차 감소**: ARMA 모델 특성
        """)
    
    # ACF/PACF 분석 버튼
    nlags = st.slider("최대 시차(lag) 수", min_value=10, max_value=100, value=40, step=5)
    
    if st.button("ACF/PACF 분석 실행", type="primary"):
        with st.spinner("ACF/PACF 분석 중..."):
            # ACF/PACF 분석 수행
            acf_values, pacf_values = analyze_acf_pacf(nlags)
            
            if acf_values is not None and pacf_values is not None:
                st.success("ACF/PACF 분석이 완료되었습니다.")
            else:
                st.error("ACF/PACF 분석 중 오류가 발생했습니다.")
    
    # ACF/PACF 분석 결과 표시
    if st.session_state.acf_values is not None and st.session_state.pacf_values is not None:
        st.markdown("### ACF/PACF 그래프")
        
        acf_pacf_fig = visualize_acf_pacf()
        if acf_pacf_fig:
            st.plotly_chart(acf_pacf_fig, use_container_width=True, theme="streamlit")
        else:
            st.error("ACF/PACF 그래프 생성에 실패했습니다.")
        
        # 결과 해석 도움말
        with st.expander("ACF/PACF 해석 도움말", expanded=False):
            st.markdown("""
            ### 📊 ACF/PACF 그래프 해석 방법
            
            #### 모델 선택 가이드
            - **AR(p) 모델**: PACF가 lag p 이후 급격히 절단(cut off)되고, ACF가 점차 감소
            - **MA(q) 모델**: ACF가 lag q 이후 급격히 절단되고, PACF가 점차 감소
            - **ARMA(p,q) 모델**: ACF와 PACF 모두 점차 감소(tail off)
            
            #### 계절성 확인
            - 특정 간격(주기)마다 ACF 또는 PACF 값이 높게 나타나면 계절성 존재
            - 예를 들어, 24시간 간격으로 높은 상관관계가 나타나면 일별 계절성을 의미
            
            #### ARIMA 모델 파라미터 선택
            1. **차분(d)**: 정상성 검정 결과에 따라 결정 (비정상이면 d > 0)
            2. **AR 차수(p)**: PACF 그래프에서 유의하게 절단되는 시점
            3. **MA 차수(q)**: ACF 그래프에서 유의하게 절단되는 시점
            
            #### 유의성 판단
            - 점선으로 표시된 신뢰 구간을 벗어나는 막대는 통계적으로 유의미한 자기상관을 나타냄
            """)
    else:
        st.info("ACF/PACF 분석을 실행하여 결과를 확인하세요.")

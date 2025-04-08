"""
모델 학습 및 예측 페이지
"""
import streamlit as st
import pandas as pd

from frontend.session_state import reset_model_results
from frontend.components import render_model_selector
from backend.model_service import get_model_factory, train_models
from backend.data_service import prepare_train_test_data
from backend.visualization_service import (
    visualize_forecast_comparison, 
    visualize_metrics_comparison, 
    visualize_residuals
)

# 페이지 제목
st.title("🤖 모델 학습 및 예측")
st.markdown("시계열 데이터에 대한 다양한 예측 모델을 학습하고 성능을 비교합니다.")

# 데이터 및 시계열 정보 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
    st.stop()
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
    st.stop()

# 모델 학습 섹션
st.markdown("## 모델 설정 및 학습")

# 모델 팩토리 가져오기
model_factory = get_model_factory()

if model_factory is None:
    st.error("모델 팩토리 로드에 실패했습니다. pmdarima 호환성 문제일 수 있습니다.")
    st.stop()

# 모델 선택기 렌더링
selected_models, complexity = render_model_selector(model_factory)

# 모델 학습 버튼
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("모델 학습 및 예측 시작", use_container_width=True, type="primary"):
        if not selected_models:
            st.warning("최소한 하나의 모델을 선택해주세요.")
        else:
            # 훈련/테스트 데이터 준비
            if prepare_train_test_data():
                with st.spinner("모델을 학습 중입니다..."):
                    st.session_state.selected_models = selected_models
                    st.session_state.complexity = complexity
                    # 모델 학습 실행
                    train_models(selected_models, complexity)
                    st.success("모델 학습 완료!")
            else:
                st.error("훈련/테스트 데이터 준비 중 오류가 발생했습니다.")

with col2:
    if st.button("결과 초기화", use_container_width=True):
        reset_model_results()
        st.rerun()

# 모델 학습 결과 표시
if st.session_state.models_trained and st.session_state.forecasts:
    st.markdown("---")
    st.subheader("📊 모델 예측 결과")
    
    # 예측 결과 비교 시각화
    comparison_fig = visualize_forecast_comparison()
    if comparison_fig:
        st.plotly_chart(comparison_fig, use_container_width=True, theme="streamlit")
    else:
        st.error("예측 결과 시각화에 실패했습니다.")
    
    # 메트릭 비교 시각화
    st.subheader("📈 모델 성능 비교")
    metrics_fig = visualize_metrics_comparison()
    if metrics_fig:
        st.plotly_chart(metrics_fig, use_container_width=True, theme="streamlit")
    else:
        st.error("성능 메트릭 시각화에 실패했습니다.")
    
    # 메트릭 표 표시
    st.subheader("📋 모델 성능 메트릭")
    
    # 메트릭 데이터프레임 생성
    metrics_data = {}
    for model_name, metrics in st.session_state.metrics.items():
        metrics_data[model_name] = {}
        for metric_name, value in metrics.items():
            if metric_name not in ['name']:  # name은 제외
                metrics_data[model_name][metric_name] = value
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.T, use_container_width=True)  # 전치하여 모델별로 행 표시
    
    # 최적 모델 선택
    if st.session_state.best_model:
        st.success(f"### 최적 모델 (RMSE 기준): {st.session_state.best_model}")
        
        # 선택한 최적 모델 상세 분석
        if st.session_state.best_model in st.session_state.forecasts:
            with st.expander("최적 모델 상세 분석", expanded=True):
                st.subheader(f"📈 최적 모델 ({st.session_state.best_model}) 상세 분석")
                
                # 잔차 분석
                residuals_fig = visualize_residuals()
                if residuals_fig:
                    st.plotly_chart(residuals_fig, use_container_width=True, theme="streamlit")
                else:
                    st.error("잔차 분석 시각화에 실패했습니다.")
                
                # 모델 설명
                st.markdown("### 모델 해석")
                if "ARIMA" in st.session_state.best_model:
                    st.markdown("""
                    **ARIMA 모델**은 AutoRegressive Integrated Moving Average의 약자로, 시계열 데이터의 자기회귀(AR), 차분(I), 이동평균(MA) 특성을 모델링합니다.
                    - AR(p): 과거 p 시점의 값들이 현재 값에 영향을 미치는 정도
                    - I(d): 정상성을 확보하기 위해 수행한 차분의 횟수
                    - MA(q): 과거 q 시점의 오차가 현재 값에 영향을 미치는 정도
                    """)
                elif "LSTM" in st.session_state.best_model:
                    st.markdown("""
                    **LSTM(Long Short-Term Memory) 모델**은 순환 신경망(RNN)의 일종으로, 장기 의존성 문제를 해결하기 위한 특수한 구조를 가진 딥러닝 모델입니다.
                    - 복잡한 시계열 패턴 학습 가능
                    - 긴 시퀀스 처리에 효과적
                    - 비선형 관계 모델링에 강점
                    """)
                elif "Prophet" in st.session_state.best_model:
                    st.markdown("""
                    **Prophet 모델**은 Facebook에서 개발한 시계열 예측 모델로, 다양한 계절성과 휴일 효과를 고려할 수 있습니다.
                    - 추세, 계절성, 휴일 효과 등을 자동으로 분해
                    - 이상값에 강건한 특성
                    - 직관적인 파라미터 조정 가능
                    """)
                elif "지수평활법" in st.session_state.best_model or "ExpSmoothing" in st.session_state.best_model:
                    st.markdown("""
                    **지수평활법(Exponential Smoothing) 모델**은 과거 관측치에 지수적으로 감소하는 가중치를 부여하는 예측 기법입니다.
                    - 단순 지수평활법: 추세나 계절성이 없는 데이터에 적합
                    - Holt 지수평활법: 추세가 있는 데이터에 적합
                    - Holt-Winters 지수평활법: 추세와 계절성이 모두 있는 데이터에 적합
                    """)
                
                # 모델 성능 메트릭 설명
                st.markdown("### 성능 지표 해석")
                st.markdown("""
                **주요 성능 지표:**
                - **RMSE (Root Mean Squared Error)**: 예측 오차의 제곱평균의 제곱근. 낮을수록 좋음.
                - **MAE (Mean Absolute Error)**: 예측 오차의 절대값 평균. 낮을수록 좋음.
                - **MAPE (Mean Absolute Percentage Error)**: 실제값 대비 오차의 비율(%). 낮을수록 좋음.
                - **R² (Coefficient of Determination)**: 모델이 설명하는 분산의 비율. 1에 가까울수록 좋음.
                """)
    else:
        st.warning("최적 모델을 결정할 수 없습니다.")
else:
    st.info("모델 학습을 진행하여 예측 결과를 확인하세요.")

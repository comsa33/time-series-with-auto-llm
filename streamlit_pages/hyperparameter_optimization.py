# streamlit_pages/hyperparameter_optimization.py
"""
하이퍼파라미터 최적화 페이지
"""
import streamlit as st
import pandas as pd

from backend.llm_service import recommend_hyperparameters, get_model_parameters_for_recommendation
from backend.model_service import get_model_factory, train_models_with_params
from backend.visualization_service import visualize_forecast_comparison, visualize_metrics_comparison

# 페이지 제목 및 설명
st.title("🧠 하이퍼파라미터 최적화")
# streamlit_pages/hyperparameter_optimization.py (계속)
st.markdown("LLM을 활용하여 시계열 모델의 하이퍼파라미터를 최적화하고 성능을 비교합니다.")

# 데이터 및 모델 학습 상태 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
    st.stop()
elif st.session_state.train is None or st.session_state.test is None:
    st.warning("훈련/테스트 데이터가 준비되지 않았습니다. 모델 학습 페이지에서 먼저 모델을 학습해주세요.")
    st.stop()
elif not st.session_state.models_trained:
    st.warning("학습된 모델이 없습니다. 모델 학습 페이지에서 먼저 모델을 학습해주세요.")
    st.stop()

# 하이퍼파라미터 최적화 UI
st.markdown("## 모델 하이퍼파라미터 최적화")

# 모델 선택 UI
model_factory = get_model_factory()
available_models = model_factory.get_all_available_models()

# 최적화할 모델 선택
selected_model = st.selectbox(
    "최적화할 모델 선택",
    available_models,
    index=0
)

# 선택한 모델의 현재 파라미터 표시
current_params = get_model_parameters_for_recommendation(selected_model)
st.markdown("### 현재 모델 파라미터")
st.json(current_params)

# 하이퍼파라미터 추천 UI
recommend_button, clear_button = st.columns([3, 1])

with recommend_button:
    if st.button("AI 하이퍼파라미터 추천 받기", type="primary", use_container_width=True):
        with st.spinner(f"{selected_model} 모델의 최적 하이퍼파라미터 추천 중..."):
            recommendation = recommend_hyperparameters(selected_model)
            if recommendation and 'error' not in recommendation:
                st.success("하이퍼파라미터 추천이 완료되었습니다!")
            else:
                st.error("하이퍼파라미터 추천 중 오류가 발생했습니다.")

with clear_button:
    if st.button("결과 초기화", use_container_width=True):
        if 'hyperparameter_recommendations' in st.session_state:
            if selected_model in st.session_state.hyperparameter_recommendations:
                del st.session_state.hyperparameter_recommendations[selected_model]
                st.rerun()

# 추천 결과 표시 및 조정
if 'hyperparameter_recommendations' in st.session_state and selected_model in st.session_state.hyperparameter_recommendations:
    recommendation = st.session_state.hyperparameter_recommendations[selected_model]
    
    st.markdown("### AI 추천 하이퍼파라미터")
    
    # 추천 근거 표시
    with st.expander("추천 근거 확인", expanded=True):
        if 'rationale' in recommendation:
            for param, rationale in recommendation['rationale'].items():
                st.markdown(f"**{param}**: {rationale}")
        else:
            st.info("추천 근거가 제공되지 않았습니다.")
    
    # 예상 개선 효과
    if 'expected_improvement' in recommendation:
        st.markdown("### 예상 개선 효과")
        
        # 메트릭별 예상 개선 효과 표시
        for metric, improvement in recommendation['expected_improvement'].items():
            st.markdown(f"**{metric}**: {improvement}")
    
    # 추천 파라미터 표시 및 사용자 조정
    st.markdown("### 추천 파라미터 조정")
    st.markdown("AI가 추천한 파라미터를 필요에 따라 수정할 수 있습니다.")
    
    # 파라미터 조정 UI
    tuned_params = {}
    if 'recommended_parameters' in recommendation:
        for param, value in recommendation['recommended_parameters'].items():
            # 파라미터 유형에 따른 입력 위젯 선택
            if isinstance(value, bool):
                tuned_params[param] = st.checkbox(param, value=value)
            elif isinstance(value, int):
                tuned_params[param] = st.number_input(param, value=value, step=1)
            elif isinstance(value, float):
                tuned_params[param] = st.number_input(param, value=value, format="%.4f")
            elif isinstance(value, list):
                if all(isinstance(x, int) for x in value):
                    # 정수 리스트의 경우 텍스트로 입력받고 변환
                    str_value = st.text_input(param, value=str(value).replace(' ', ''))
                    try:
                        import ast
                        try:
                            tuned_params[param] = ast.literal_eval(str_value)
                        except (ValueError, SyntaxError):
                            st.warning(f"{param}: 유효한 리스트 형식이 아닙니다.")
                            tuned_params[param] = value
                    except:
                        st.warning(f"{param}: 유효한 리스트 형식이 아닙니다.")
                        tuned_params[param] = value
                else:
                    tuned_params[param] = value
            else:
                tuned_params[param] = st.text_input(param, value=str(value))
    
    # 모델 학습 버튼
    if st.button("조정된 파라미터로 모델 학습", type="primary"):
        with st.spinner("조정된 파라미터로 모델 학습 중..."):
            # 접두사 추가하여 기존 모델과 구분
            success = train_models_with_params([selected_model], tuned_params, prefix="튜닝된_")
            if success:
                st.success("모델 학습이 완료되었습니다!")
                
                # 세션 상태에 최적화 이력 저장
                if 'optimization_history' not in st.session_state:
                    st.session_state.optimization_history = {}
                
                if selected_model not in st.session_state.optimization_history:
                    st.session_state.optimization_history[selected_model] = []
                
                # 최적화 시도 정보 저장
                optimization_attempt = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "parameters": tuned_params,
                    "model_name": f"튜닝된_{selected_model}"
                }
                st.session_state.optimization_history[selected_model].append(optimization_attempt)
                
                st.rerun()
            else:
                st.error("모델 학습 중 오류가 발생했습니다.")

# 결과 비교 섹션
st.markdown("---")
st.markdown("## 최적화 결과 비교")

# 원본 모델과 최적화 모델 비교할 모델 선택
if 'optimization_history' in st.session_state and selected_model in st.session_state.optimization_history:
    # 원본 모델은 항상 포함
    comparison_models = [selected_model]
    
    # 최적화된 모델 목록 가져오기
    optimized_models = [attempt["model_name"] for attempt in st.session_state.optimization_history[selected_model]]
    
    # 비교할 최적화 모델 선택
    selected_optimized_models = st.multiselect(
        "비교할 최적화 모델 선택",
        optimized_models,
        default=optimized_models[-1:] if optimized_models else []
    )
    
    # 선택된 모든 모델
    all_selected_models = comparison_models + selected_optimized_models
    
    if len(all_selected_models) > 1:
        # 예측 결과 비교
        st.subheader("예측 결과 비교")
        
        # 선택된 모델에 대한 예측 결과만 필터링
        filtered_forecasts = {model: st.session_state.forecasts[model] 
                             for model in all_selected_models 
                             if model in st.session_state.forecasts}
        
        # 예측 결과 시각화
        comparison_fig = visualize_forecast_comparison(
            st.session_state.train,
            st.session_state.test,
            filtered_forecasts
        )
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True, theme="streamlit")
        
        # 성능 메트릭 비교
        st.subheader("성능 메트릭 비교")
        
        # 선택된 모델에 대한 메트릭만 필터링
        filtered_metrics = {model: st.session_state.metrics[model] 
                           for model in all_selected_models 
                           if model in st.session_state.metrics}
        
        # 메트릭 비교 시각화
        metrics_fig = visualize_metrics_comparison(filtered_metrics)
        if metrics_fig:
            st.plotly_chart(metrics_fig, use_container_width=True, theme="streamlit")
        
        # 상세 메트릭 테이블
        st.subheader("상세 성능 메트릭")
        
        # 메트릭 데이터프레임 생성
        metrics_data = {}
        for model_name, metrics in filtered_metrics.items():
            metrics_data[model_name] = {}
            for metric_name, value in metrics.items():
                if metric_name not in ['name']:  # name은 제외
                    metrics_data[model_name][metric_name] = value
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # 개선율 계산 (원본 대비)
        if len(all_selected_models) > 1:
            st.subheader("성능 개선율 (원본 대비)")
            
            # 원본 모델 메트릭
            baseline_metrics = {metric: value 
                               for metric, value in filtered_metrics[selected_model].items() 
                               if metric not in ['name']}
            
            # 개선율 계산
            improvement_data = {}
            for model_name, metrics in filtered_metrics.items():
                if model_name != selected_model:  # 원본 모델 제외
                    improvement_data[model_name] = {}
                    for metric_name, value in metrics.items():
                        if metric_name not in ['name']:  # name은 제외
                            baseline = baseline_metrics.get(metric_name)
                            if baseline is not None and baseline != 0:
                                # R^2는 높을수록 좋음, 나머지는 낮을수록 좋음
                                if metric_name == 'R^2':
                                    improvement = (value - baseline) / abs(baseline) * 100 if baseline != 0 else float('inf')
                                    improvement_data[model_name][metric_name] = f"{improvement:.2f}% {'↑' if improvement > 0 else '↓'}"
                                else:
                                    improvement = (baseline - value) / baseline * 100 if baseline != 0 else float('inf')
                                    improvement_data[model_name][metric_name] = f"{improvement:.2f}% {'↑' if improvement > 0 else '↓'}"
            
            # 개선율 데이터프레임 표시
            if improvement_data:
                improvement_df = pd.DataFrame(improvement_data)
                st.dataframe(improvement_df, use_container_width=True)
            else:
                st.info("개선율을 계산할 수 없습니다.")
    else:
        st.info("비교할 최적화 모델을 선택해주세요.")
else:
    st.info("아직 최적화를 수행하지 않았습니다. 상단에서 하이퍼파라미터 추천을 받고 모델을 학습해보세요.")

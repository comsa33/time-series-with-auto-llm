"""
시각화 서비스 모듈 - 시각화 관련 기능
"""
import streamlit as st

from utils.visualizer import (
    cached_plot_timeseries,
    cached_plot_decomposition,
    cached_plot_acf_pacf,
    cached_plot_forecast_comparison,
    cached_plot_metrics_comparison,
    cached_plot_residuals,
    cached_plot_differencing_comparison,
    cached_plot_stationarity_comparison,
    cached_plot_change_points,
    cached_plot_segment_means,
    cached_plot_correlation_heatmap,
    cached_plot_granger_causality,
    cached_plot_residual_acf,
    cached_plot_volatility
)

def visualize_timeseries():
    """
    시계열 데이터 시각화
    
    Returns:
        plotly.graph_objects.Figure: 시계열 그래프
    """
    if st.session_state.series is not None:
        station_text = f"{st.session_state.selected_station} " if st.session_state.selected_station else "Seoul City Overall "
        fig = cached_plot_timeseries(
            st.session_state.series,
            title=f"{station_text}{st.session_state.selected_target} 시계열 데이터",
            xlabel="날짜 (Date)",
            ylabel=st.session_state.selected_target
        )
        return fig
    return None

def visualize_decomposition():
    """
    시계열 분해 결과 시각화
    
    Returns:
        plotly.graph_objects.Figure: 분해 그래프
    """
    if st.session_state.decomposition is not None:
        decomp_fig = cached_plot_decomposition(st.session_state.decomposition)
        return decomp_fig
    return None

def visualize_acf_pacf():
    """
    ACF/PACF 시각화
    
    Returns:
        plotly.graph_objects.Figure: ACF/PACF 그래프
    """
    if st.session_state.acf_values is not None and st.session_state.pacf_values is not None:
        acf_pacf_fig = cached_plot_acf_pacf(st.session_state.acf_values, st.session_state.pacf_values)
        return acf_pacf_fig
    return None

def visualize_forecast_comparison(train_data=None, test_data=None, forecasts=None):
    """
    예측 결과 비교 시각화
    
    Args:
        train_data: 훈련 데이터 (기본값: None, 세션 상태 사용)
        test_data: 테스트 데이터 (기본값: None, 세션 상태 사용)
        forecasts: 예측 결과 딕셔너리 (기본값: None, 세션 상태 사용)
    
    Returns:
        plotly.graph_objects.Figure: 예측 비교 그래프
    """
    # 매개변수가 없으면 세션 상태 사용
    if train_data is None:
        if st.session_state.use_differencing and st.session_state.diff_train is not None:
            # 차분 모드이고 diff_train이 있는 경우 diff_train 사용
            st.info("차분 데이터 모드입니다. 원본 데이터로 시각화합니다.")
            train_data = st.session_state.train  # 원본 데이터로 시각화
        else:
            # 일반 모드
            train_data = st.session_state.train
    
    if test_data is None:
        if st.session_state.use_differencing and st.session_state.diff_test is not None:
            # 차분 모드이고 diff_test가 있는 경우 diff_test 사용
            test_data = st.session_state.test  # 원본 데이터로 시각화
        else:
            # 일반 모드
            test_data = st.session_state.test
    
    forecasts = forecasts if forecasts is not None else st.session_state.forecasts
    
    # 데이터 유효성 검사
    if train_data is None or test_data is None:
        st.error("시각화에 필요한 훈련/테스트 데이터가 없습니다.")
        
        # 디버깅 정보 표시
        st.write("### 세션 상태 확인:")
        st.write(f"train: {'존재함' if hasattr(st.session_state, 'train') and st.session_state.train is not None else '없음'}")
        st.write(f"test: {'존재함' if hasattr(st.session_state, 'test') and st.session_state.test is not None else '없음'}")
        st.write(f"diff_train: {'존재함' if hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None else '없음'}")
        st.write(f"diff_test: {'존재함' if hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None else '없음'}")
        st.write(f"use_differencing: {st.session_state.use_differencing if hasattr(st.session_state, 'use_differencing') else '설정 안됨'}")
        
        # 차분 모드이고 원본 데이터가 없는 경우, 차분 데이터 사용 시도
        if st.session_state.use_differencing:
            if train_data is None and hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None:
                st.warning("원본 train 데이터가 없어 차분 데이터를 사용합니다.")
                train_data = st.session_state.diff_train
            
            if test_data is None and hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
                st.warning("원본 test 데이터가 없어 차분 데이터를 사용합니다.")
                test_data = st.session_state.diff_test
            
            # 여전히 데이터가 없는 경우
            if train_data is None or test_data is None:
                return None
        else:
            return None
    
    if not forecasts:
        st.error("시각화할 예측 결과가 없습니다.")
        return None
    
    # 유효한 예측 결과만 필터링
    valid_forecasts = {}
    for model_name, forecast in forecasts.items():
        if forecast is not None and len(forecast) > 0:
            # 예측 결과 길이가 테스트 데이터와 다른 경우 길이 조정
            if len(forecast) != len(test_data):
                min_len = min(len(forecast), len(test_data))
                if min_len > 0:
                    st.warning(f"{model_name} 모델의 예측 길이({len(forecast)})가 테스트 데이터 길이({len(test_data)})와 다릅니다. 최소 길이({min_len})로 조정합니다.")
                    valid_forecasts[model_name] = forecast[:min_len]
                else:
                    st.warning(f"{model_name} 모델의 예측 결과를 시각화에서 제외합니다.")
                    continue
            else:
                valid_forecasts[model_name] = forecast
    
    if not valid_forecasts:
        st.error("유효한 예측 결과가 없어 시각화할 수 없습니다.")
        return None
    
    try:
        comparison_fig = cached_plot_forecast_comparison(
            train_data, 
            test_data, 
            valid_forecasts
        )
        return comparison_fig
    except Exception as e:
        st.error(f"예측 비교 시각화 중 오류 발생: {str(e)}")
        return None

def visualize_metrics_comparison(metrics=None):
    """
    성능 메트릭 비교 시각화
    
    Args:
        metrics: 메트릭 딕셔너리 (기본값: None, 세션 상태 사용)
    
    Returns:
        plotly.graph_objects.Figure: 메트릭 비교 그래프
    """
    metrics = metrics if metrics is not None else st.session_state.metrics
    
    if metrics:
        metrics_fig = cached_plot_metrics_comparison(metrics)
        return metrics_fig
    return None

def visualize_residuals(model_name=None):
    """
    잔차 분석 시각화
    
    Args:
        model_name: 모델 이름 (기본값: None, 최적 모델 사용)
        
    Returns:
        plotly.graph_objects.Figure: 잔차 분석 그래프
    """
    if model_name is None and st.session_state.best_model:
        model_name = st.session_state.best_model
        
    if (st.session_state.test is not None and 
        model_name in st.session_state.forecasts):
        best_forecast = st.session_state.forecasts[model_name]
        residuals_fig = cached_plot_residuals(st.session_state.test, best_forecast)
        return residuals_fig
    return None

def visualize_differencing_comparison():
    """
    원본 시계열과 차분된 시계열 비교 시각화
    
    Returns:
        plotly.graph_objects.Figure: 차분 비교 그래프
    """
    if st.session_state.series is not None and st.session_state.differenced_series is not None:
        # 차분 정보 텍스트 생성
        diff_info = f"일반 차분: {st.session_state.diff_order}차"
        if st.session_state.seasonal_diff_order > 0:
            diff_info += f", 계절 차분: {st.session_state.seasonal_diff_order}차 (주기: {st.session_state.period})"
        
        # 차분 시각화
        diff_fig = cached_plot_differencing_comparison(
            st.session_state.series,
            st.session_state.differenced_series,
            title=f"차분 비교 ({diff_info})"
        )
        return diff_fig
    return None

def visualize_stationarity_comparison():
    """
    ADF와 KPSS 정상성 검정 결과 비교 시각화
    
    Returns:
        plotly.graph_objects.Figure: 정상성 비교 그래프
    """
    if (hasattr(st.session_state, 'stationarity_result') and 
        hasattr(st.session_state, 'kpss_result')):
        fig = cached_plot_stationarity_comparison(
            st.session_state.stationarity_result,
            st.session_state.kpss_result,
            target_name=st.session_state.selected_target
        )
        return fig
    return None

def visualize_change_points():
    """
    구조적 변화점 시각화
    
    Returns:
        plotly.graph_objects.Figure: 변화점 그래프
    """
    if (st.session_state.series is not None and 
        hasattr(st.session_state, 'change_points_result')):
        title = f"{st.session_state.selected_target} 시계열 데이터의 구조적 변화점"
        fig = cached_plot_change_points(
            st.session_state.series,
            st.session_state.change_points_result,
            title=title
        )
        return fig
    return None

def visualize_segment_means():
    """
    세그먼트별 평균값 시각화
    
    Returns:
        plotly.graph_objects.Figure: 세그먼트 평균값 그래프
    """
    if (st.session_state.series is not None and 
        hasattr(st.session_state, 'change_points_result')):
        title = f"{st.session_state.selected_target} 세그먼트별 평균값"
        fig = cached_plot_segment_means(
            st.session_state.series,
            st.session_state.change_points_result,
            title=title
        )
        return fig
    return None

def visualize_correlation_heatmap(data):
    """
    변수 간 상관관계 히트맵 시각화
    
    Args:
        data: 상관관계를 계산할 데이터프레임
        
    Returns:
        plotly.graph_objects.Figure: 상관관계 히트맵
    """
    if data is not None and not data.empty:
        corr_matrix = data.corr()
        fig = cached_plot_correlation_heatmap(corr_matrix)
        return fig
    return None

def visualize_granger_causality(lags, p_values, cause_var, effect_var):
    """
    Granger 인과성 검정 결과 시각화
    
    Args:
        lags: 시차 목록
        p_values: 각 시차별 p값
        cause_var: 원인 변수명
        effect_var: 결과 변수명
        
    Returns:
        plotly.graph_objects.Figure: 인과성 그래프
    """
    fig = cached_plot_granger_causality(lags, p_values, cause_var, effect_var)
    return fig

def visualize_residual_acf(residuals, max_lags=20):
    """
    모델 잔차의 자기상관함수 시각화
    
    Args:
        residuals: 모델 잔차
        max_lags: 최대 시차
        
    Returns:
        plotly.graph_objects.Figure: 잔차 ACF 그래프
    """
    fig = cached_plot_residual_acf(residuals, max_lags)
    return fig

def visualize_volatility():
    """
    시계열 데이터의 변동성 시각화
    
    Returns:
        plotly.graph_objects.Figure: 변동성 그래프
    """
    if (st.session_state.series is not None and 
        hasattr(st.session_state, 'volatility_result')):
        volatility = st.session_state.volatility_result['volatility']
        title = f"{st.session_state.selected_target}의 시계열 데이터와 조건부 변동성"
        fig = cached_plot_volatility(
            st.session_state.series,
            volatility,
            title=title
        )
        return fig
    return None

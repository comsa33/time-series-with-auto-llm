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
    cached_plot_differencing_comparison
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
    train_data = train_data if train_data is not None else st.session_state.train
    test_data = test_data if test_data is not None else st.session_state.test
    forecasts = forecasts if forecasts is not None else st.session_state.forecasts
    
    if (train_data is not None and test_data is not None and forecasts):
        comparison_fig = cached_plot_forecast_comparison(
            train_data, 
            test_data, 
            forecasts
        )
        return comparison_fig
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

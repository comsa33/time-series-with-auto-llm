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
    cached_plot_residuals
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

def visualize_forecast_comparison():
    """
    예측 결과 비교 시각화
    
    Returns:
        plotly.graph_objects.Figure: 예측 비교 그래프
    """
    if (st.session_state.train is not None and 
        st.session_state.test is not None and 
        st.session_state.forecasts):
        comparison_fig = cached_plot_forecast_comparison(
            st.session_state.train, 
            st.session_state.test, 
            st.session_state.forecasts
        )
        return comparison_fig
    return None

def visualize_metrics_comparison():
    """
    성능 메트릭 비교 시각화
    
    Returns:
        plotly.graph_objects.Figure: 메트릭 비교 그래프
    """
    if st.session_state.metrics:
        metrics_fig = cached_plot_metrics_comparison(st.session_state.metrics)
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

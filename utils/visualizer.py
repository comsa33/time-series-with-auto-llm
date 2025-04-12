"""
시계열 데이터 시각화를 위한 모듈
"""
from typing import Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib import font_manager, rc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.singleton import Singleton


class TimeSeriesVisualizer(metaclass=Singleton):
    """
    시계열 데이터 시각화를 위한 클래스
    싱글턴 패턴을 적용하여 메모리 효율성 확보
    """
    
    def __init__(self):
        """
        시각화 클래스 초기화
        """
        # 기본 시각화 스타일 설정
        sns.set_style('whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 한글 폰트 설정
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 나눔고딕 폰트 경로
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_timeseries(
            self, 
            data: pd.Series, 
            title: str = "시계열 플롯 (Time Series Plot)",
            xlabel: str = "날짜 (Date)",
            ylabel: str = "값 (Value)",
            color: str = '#1f77b4',
            **kwargs
        ) -> go.Figure:
        """
        기본 시계열 플롯을 생성합니다 (Plotly 버전).
        """
        # Plotly 버전으로 구현
        fig = px.line(
            x=data.index, 
            y=data.values,
            labels={"x": xlabel, "y": ylabel},
            title=title
        )
        
        # 스타일 설정
        fig.update_layout(
            title=title,
            title_font_size=14,
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        # 날짜 형식 지정
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
        )
        
        return fig
    
    def plot_decomposition(
            self, 
            decomposition: Dict[str, pd.Series],
            **kwargs
        ) -> go.Figure:
        """
        시계열 분해 결과를 시각화합니다 (Plotly 버전).
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('관측값 (Observed)', '추세 (Trend)', '계절성 (Seasonality)', '잔차 (Residuals)'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # 관측값
        fig.add_trace(
            go.Scatter(x=decomposition['observed'].index, y=decomposition['observed'].values,
                    mode='lines', name='관측값'),
            row=1, col=1
        )
        
        # 추세
        fig.add_trace(
            go.Scatter(x=decomposition['trend'].index, y=decomposition['trend'].values,
                    mode='lines', name='추세'),
            row=2, col=1
        )
        
        # 계절성
        fig.add_trace(
            go.Scatter(x=decomposition['seasonal'].index, y=decomposition['seasonal'].values,
                    mode='lines', name='계절성'),
            row=3, col=1
        )
        
        # 잔차
        fig.add_trace(
            go.Scatter(x=decomposition['resid'].index, y=decomposition['resid'].values,
                    mode='lines', name='잔차'),
            row=4, col=1
        )
        
        # 스타일 업데이트
        fig.update_layout(
            height=800,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False
        )
        
        return fig
    
    def plot_acf_pacf(self,
                    acf_values: np.ndarray,
                    pacf_values: np.ndarray,
                    lags: int = 40,
                    **kwargs) -> go.Figure:
        """
        ACF 및 PACF 플롯을 생성합니다 (Plotly 버전).
        
        Args:
            acf_values: ACF 값
            pacf_values: PACF 값
            lags: 지연값 수
            
        Returns:
            Plotly Figure 객체
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                '자기상관 함수 (Autocorrelation Function)',
                '부분 자기상관 함수 (Partial Autocorrelation Function)'
            )
        )
        
        # x축 값 (lags)
        x = list(range(len(acf_values)))
        
        # 신뢰 구간 계산 (95%)
        confidence = 1.96 / np.sqrt(len(acf_values))
        
        # ACF 플롯 - stem 효과 (마커와 선 조합)
        for i in range(len(acf_values)):
            fig.add_trace(
                go.Scatter(
                    x=[i, i], 
                    y=[0, acf_values[i]], 
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # ACF 마커
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=acf_values, 
                mode='markers',
                marker=dict(color='blue', size=8),
                name='ACF'
            ),
            row=1, col=1
        )
        
        # 신뢰 구간 추가
        fig.add_trace(
            go.Scatter(
                x=[0, len(acf_values)-1],
                y=[confidence, confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, len(acf_values)-1],
                y=[-confidence, -confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # PACF 플롯 - stem 효과
        for i in range(len(pacf_values)):
            fig.add_trace(
                go.Scatter(
                    x=[i, i], 
                    y=[0, pacf_values[i]], 
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # PACF 마커
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=pacf_values, 
                mode='markers',
                marker=dict(color='blue', size=8),
                name='PACF'
            ),
            row=1, col=2
        )
        
        # PACF 신뢰 구간
        fig.add_trace(
            go.Scatter(
                x=[0, len(pacf_values)-1],
                y=[confidence, confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, len(pacf_values)-1],
                y=[-confidence, -confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        
        # x축 및 y축 레이블
        fig.update_xaxes(title_text='지연 (Lag)', row=1, col=1)
        fig.update_xaxes(title_text='지연 (Lag)', row=1, col=2)
        fig.update_yaxes(title_text='상관도 (Correlation)', row=1, col=1)
        fig.update_yaxes(title_text='상관도 (Correlation)', row=1, col=2)
        
        return fig

    def plot_forecast_comparison(self,
                            train: pd.Series,
                            test: pd.Series,
                            forecasts: Dict[str, np.ndarray],
                            **kwargs) -> go.Figure:
        """
        여러 모델의 예측 결과를 비교하여 시각화합니다 (Plotly 버전).
        
        Args:
            train: 훈련 데이터
            test: 테스트 데이터
            forecasts: 모델별 예측값 딕셔너리
            
        Returns:
            Plotly Figure 객체
        """
        # 그래프 생성
        fig = go.Figure()
        
        # 훈련 데이터
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train.values,
                mode='lines',
                name='Training Data',
                line=dict(color='blue', width=2)
            )
        )
        
        # 테스트 데이터
        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=test.values,
                mode='lines',
                name='Actual Test Data',
                line=dict(color='green', width=2)
            )
        )
        
        # 각 모델의 예측
        colors = ['red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            fig.add_trace(
                go.Scatter(
                    x=test.index,
                    y=forecast,
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                )
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title='예측 비교 (Forecast Comparison)',
            xaxis_title='날짜 (Date)',
            yaxis_title='값 (Value)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            height=500,
        )
        
        # 날짜 형식 지정
        fig.update_xaxes(
            tickformat="%Y-%m-%d"
        )
        
        return fig

    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        여러 모델의 성능 지표를 비교하여 시각화합니다 (Plotly 버전).
        
        Args:
            metrics: 모델별 성능 지표 딕셔너리
            
        Returns:
            Plotly Figure 객체
        """
        # 데이터 준비
        models = list(metrics.keys())
        metric_names = ['RMSE', 'MAE', 'R^2', 'MAPE']
        
        # 모든 모델에 있는 지표만 선택
        available_metrics = set.intersection(*[set(m.keys()) for m in metrics.values()])
        metric_names = [m for m in metric_names if m in available_metrics]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=len(metric_names), 
            cols=1,
            subplot_titles=[f'{metric} Comparison' for metric in metric_names]
        )
        
        # 각 메트릭별 바 차트 생성
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            
            # 특별히 R^2는 높을수록 좋음, 나머지는 낮을수록 좋음
            if metric == 'R^2':
                colors = ['green' if v > 0 else 'red' for v in values]
                # 내림차순 정렬 (높을수록 좋음)
                sorted_idx = np.argsort(values)[::-1]
            else:
                # 값을 정규화하여 색상 결정 (낮을수록 좋음)
                max_val = max(values) if values else 1
                colors = ['lightcoral' if v/max_val > 0.7 else 'lightgreen' for v in values]
                # 오름차순 정렬 (낮을수록 좋음)
                sorted_idx = np.argsort(values)
            
            # 정렬된 모델 및 값
            sorted_models = [models[i] for i in sorted_idx]
            sorted_values = [values[i] for i in sorted_idx]
            sorted_colors = [colors[i] for i in sorted_idx]
            
            # 바 차트 추가
            fig.add_trace(
                go.Bar(
                    x=sorted_models,
                    y=sorted_values,
                    text=[f'{v:.4f}' for v in sorted_values],
                    textposition='outside',
                    marker_color=sorted_colors,
                    name=metric
                ),
                row=i+1, col=1
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=300 * len(metric_names),
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # y축 타이틀 업데이트
        for i, metric in enumerate(metric_names):
            fig.update_yaxes(title_text=metric, row=i+1, col=1)
        
        return fig

    def plot_residuals(
            self,
            actual: pd.Series,
            predicted: np.ndarray,
            title: str = "Residual Analysis",
            **kwargs
        ) -> go.Figure:
        """
        잔차 분석 플롯을 생성합니다 (Plotly 버전).
        
        Args:
            actual: 실제 값
            predicted: 예측 값
            title: 그래프 제목
            
        Returns:
            Plotly Figure 객체
        """
        # 길이 맞춤
        min_len = min(len(actual), len(predicted))
        actual_values = actual.iloc[:min_len].values
        predicted_values = predicted[:min_len]
        
        # 잔차 계산
        residuals = actual_values - predicted_values
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '잔차 시계열 (Residuals Over Time)',
                '잔차 분포 (Residual Distribution)',
                '정규 Q-Q 플롯 (Normal Q-Q Plot)',
                '잔차 vs 예측값 (Residuals vs Predicted)'
            )
        )
        
        # 1. 잔차 시계열 플롯
        fig.add_trace(
            go.Scatter(
                x=actual.index[:min_len],
                y=residuals,
                mode='lines',
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # 0 라인 추가
        fig.add_trace(
            go.Scatter(
                x=[actual.index[0], actual.index[min_len-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. 잔차 히스토그램
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=20,
                marker_line_color='black',
                marker_line_width=1,
                opacity=0.7,
                name='Residual Distribution'
            ),
            row=1, col=2
        )
        
        osm, osr = stats.probplot(residuals, dist="norm", fit=False)
        
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                marker=dict(color='blue', size=6),
                name='Q-Q Plot'
            ),
            row=2, col=1
        )
        
        # 이론적인 정규분포 라인
        z = np.polyfit(osm, osr, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=p(osm),
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 잔차 vs 예측값
        fig.add_trace(
            go.Scatter(
                x=predicted_values,
                y=residuals,
                mode='markers',
                marker=dict(color='blue', size=6),
                name='Residuals vs Predicted'
            ),
            row=2, col=2
        )
        
        # 0 라인 추가
        fig.add_trace(
            go.Scatter(
                x=[min(predicted_values), max(predicted_values)],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # 축 레이블 업데이트
        fig.update_xaxes(title_text='날짜 (Date)', row=1, col=1)
        fig.update_xaxes(title_text='잔차 (Residual)', row=1, col=2)
        fig.update_xaxes(title_text='이론적 분위수 (Theoretical Quantiles)', row=2, col=1)
        fig.update_xaxes(title_text='예측값 (Predicted Values)', row=2, col=2)
        
        fig.update_yaxes(title_text='잔차 (Residual)', row=1, col=1)
        fig.update_yaxes(title_text='빈도 (Frequency)', row=1, col=2)
        fig.update_yaxes(title_text='정렬된 값 (Ordered Values)', row=2, col=1)
        fig.update_yaxes(title_text='잔차 (Residuals)', row=2, col=2)
        
        return fig

    def plot_feature_importance(self,
                            features: pd.DataFrame,
                            target: pd.Series,
                            top_n: int = 10,
                            **kwargs) -> go.Figure:
        """
        특성 중요도를 시각화합니다 (Plotly 버전).
        
        Args:
            features: 특성 데이터프레임
            target: 타겟 변수
            top_n: 표시할 상위 특성 수
            
        Returns:
            Plotly Figure 객체
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # 모든 특성이 수치형인지 확인
        numeric_features = features.select_dtypes(include=np.number).columns.tolist()
        
        # 수치형 특성만 사용
        X = features[numeric_features]
        
        # 랜덤 포레스트로 특성 중요도 계산
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, target)
        
        # 특성 중요도 정렬
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # 상위 N개 특성만 선택
        top_indices = indices[:top_n]
        top_features = [numeric_features[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        # 시각화
        fig = go.Figure()
        
        # 수평 막대 그래프
        fig.add_trace(
            go.Bar(
                y=top_features,
                x=top_importance,
                orientation='h',
                marker=dict(
                    color='rgba(50, 171, 96, 0.7)',
                    line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
                )
            )
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title='상위 특성 중요도 (Top Feature Importance)',
            xaxis_title='특성 중요도 (Feature Importance)',
            yaxis_title='특성 (Feature)',
            height=500,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # y축 역순 정렬 (중요도 높은 특성이 상단에 표시)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return fig

    def plot_differencing_comparison(
            self, 
            original_series: pd.Series, 
            differenced_series: pd.Series,
            title: str = "차분 비교 (Differencing Comparison)",
            **kwargs
        ) -> go.Figure:
        """
        원본 시계열과 차분된 시계열 비교 시각화 (Plotly 버전).
        
        Args:
            original_series: 원본 시계열 데이터
            differenced_series: 차분된 시계열 데이터
            title: 그래프 제목
            
        Returns:
            Plotly Figure 객체
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                '원본 시계열 (Original Time Series)',
                '차분된 시계열 (Differenced Time Series)'
            ),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # 원본 시계열 그래프
        fig.add_trace(
            go.Scatter(
                x=original_series.index,
                y=original_series.values,
                mode='lines',
                name='원본 데이터',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # 차분된 시계열 그래프
        fig.add_trace(
            go.Scatter(
                x=differenced_series.index,
                y=differenced_series.values,
                mode='lines',
                name='차분된 데이터',
                line=dict(color='red', width=1.5)
            ),
            row=2, col=1
        )
        
        # 0 라인 추가 (차분 그래프에만)
        fig.add_trace(
            go.Scatter(
                x=[differenced_series.index.min(), differenced_series.index.max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 스타일 설정
        fig.update_layout(
            title=title,
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # x축 및 y축 레이블
        fig.update_xaxes(title_text='날짜 (Date)', row=2, col=1)
        fig.update_yaxes(title_text='값 (Value)', row=1, col=1)
        fig.update_yaxes(title_text='차분값 (Differenced Value)', row=2, col=1)
        
        # 날짜 형식 지정
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
            row=2, col=1
        )
        
        return fig

    def plot_stationarity_comparison(
            self,
            adf_result: dict,
            kpss_result: dict,
            target_name: str = "시계열 데이터",
            **kwargs
        ) -> go.Figure:
        """
        ADF와 KPSS 정상성 검정 결과를 비교 시각화합니다.
        
        Args:
            adf_result: ADF 검정 결과 딕셔너리
            kpss_result: KPSS 검정 결과 딕셔너리
            target_name: 대상 시계열 이름
        
        Returns:
            plotly Figure 객체
        """
        # 사분면 차트 생성
        fig = go.Figure()
        
        # 정상성 상태에 따른 영역 표시
        fig.add_shape(
            type="rect", x0=0, y0=0, x1=0.5, y1=0.5, 
            fillcolor="rgba(255, 0, 0, 0.1)", line=dict(width=0)
        )
        fig.add_shape(
            type="rect", x0=0.5, y0=0, x1=1, y1=0.5, 
            fillcolor="rgba(255, 165, 0, 0.1)", line=dict(width=0)
        )
        fig.add_shape(
            type="rect", x0=0, y0=0.5, x1=0.5, y1=1, 
            fillcolor="rgba(255, 165, 0, 0.1)", line=dict(width=0)
        )
        fig.add_shape(
            type="rect", x0=0.5, y0=0.5, x1=1, y1=1, 
            fillcolor="rgba(0, 128, 0, 0.1)", line=dict(width=0)
        )
        
        # 각 영역에 텍스트 추가
        fig.add_annotation(x=0.25, y=0.25, text="비정상", showarrow=False)
        fig.add_annotation(x=0.75, y=0.25, text="수준 정상", showarrow=False)
        fig.add_annotation(x=0.25, y=0.75, text="추세 정상", showarrow=False)
        fig.add_annotation(x=0.75, y=0.75, text="정상", showarrow=False)
        
        # 현재 데이터의 위치 표시
        adf_stationary = adf_result.get('is_stationary', False)
        kpss_stationary = kpss_result.get('is_stationary', False)
        
        marker_x = 0.75 if kpss_stationary else 0.25
        marker_y = 0.75 if adf_stationary else 0.25
        
        fig.add_trace(go.Scatter(
            x=[marker_x], y=[marker_y],
            mode='markers',
            marker=dict(
                size=15,
                color='blue',
                symbol='circle'
            ),
            name=target_name
        ))
        
        # 축 레이블과 제목 설정
        fig.update_layout(
            title="정상성 검정 종합 결과",
            xaxis=dict(
                title="KPSS 검정",
                showticklabels=False,
                range=[0, 1]
            ),
            yaxis=dict(
                title="ADF 검정",
                showticklabels=False,
                range=[0, 1]
            ),
            height=400,
            showlegend=True
        )
        
        return fig

    def plot_change_points(
            self,
            series: pd.Series,
            change_points_result: dict,
            title: str = "구조적 변화점 분석",
            **kwargs
        ) -> go.Figure:
        """
        시계열 데이터의 구조적 변화점을 시각화합니다.
        
        Args:
            series: 원본 시계열 데이터
            change_points_result: 변화점 탐지 결과 딕셔너리
            title: 그래프 제목
            
        Returns:
            plotly Figure 객체
        """
        # 원본 시계열 데이터 플롯
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='원본 데이터',
            line=dict(color='blue')
        ))
        
        # 변화점에 수직선 추가
        for date in change_points_result['change_dates']:
            fig.add_shape(
                type="line",
                x0=date,
                y0=min(series),
                x1=date,
                y1=max(series),
                line=dict(color="red", width=2, dash="dash"),
            )
            # 변화점에 주석 추가
            fig.add_annotation(
                x=date,
                y=max(series),
                text=f"{date.strftime('%Y-%m-%d')}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        
        # 그래프 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title="날짜",
            yaxis_title="값",
            height=500,
            showlegend=True
        )
        
        return fig

    def plot_segment_means(
            self,
            series: pd.Series,
            change_points_result: dict,
            title: str = "세그먼트별 평균값",
            **kwargs
        ) -> go.Figure:
        """
        변화점으로 구분된 세그먼트별 평균값을 시각화합니다.
        
        Args:
            series: 원본 시계열 데이터
            change_points_result: 변화점 탐지 결과 딕셔너리
            title: 그래프 제목
            
        Returns:
            plotly Figure 객체
        """
        fig = go.Figure()
        
        # 원본 데이터 (반투명)
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='원본 데이터',
            line=dict(color='gray', width=1),
            opacity=0.5
        ))
        
        # 각 세그먼트의 평균값 수평선
        for i, segment in enumerate(change_points_result['segments']):
            start_date = pd.to_datetime(segment['start_date'])
            end_date = pd.to_datetime(segment['end_date'])
            mean_value = segment['mean']
            
            # 세그먼트 평균선
            fig.add_trace(go.Scatter(
                x=[start_date, end_date],
                y=[mean_value, mean_value],
                mode='lines',
                name=f'세그먼트 {i+1} 평균',
                line=dict(color='red', width=3)
            ))
            
            # 세그먼트 라벨
            mid_point = start_date + (end_date - start_date) / 2
            fig.add_annotation(
                x=mid_point,
                y=mean_value * 1.05,
                text=f"평균: {mean_value:.2f}",
                showarrow=False,
                font=dict(size=12)
            )
        
        # 그래프 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title="날짜",
            yaxis_title="값",
            height=500
        )
        
        return fig

    def plot_correlation_heatmap(
            self,
            correlation_matrix: pd.DataFrame,
            title: str = "변수 간 상관관계 히트맵",
            **kwargs
        ) -> go.Figure:
        """
        변수 간 상관관계 히트맵을 시각화합니다.
        
        Args:
            correlation_matrix: 상관관계 행렬
            title: 그래프 제목
            
        Returns:
            plotly Figure 객체
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.around(correlation_matrix.values, decimals=2),
            texttemplate='%{text:.2f}',
            colorbar=dict(title='상관계수')
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            width=700
        )
        
        return fig

    def plot_granger_causality(
            self,
            lags: list,
            p_values: list,
            cause_var: str,
            effect_var: str,
            significance_level: float = 0.05,
            **kwargs
        ) -> go.Figure:
        """
        Granger 인과성 검정 결과를 시각화합니다.
        
        Args:
            lags: 시차 목록
            p_values: 각 시차별 p값
            cause_var: 원인 변수명
            effect_var: 결과 변수명
            significance_level: 유의수준
            
        Returns:
            plotly Figure 객체
        """
        fig = go.Figure()
        
        # p-값 막대 그래프
        fig.add_trace(go.Bar(
            x=lags,
            y=p_values,
            marker_color=['green' if p < significance_level else 'red' for p in p_values],
            name='p-값'
        ))
        
        # 유의수준 선
        fig.add_trace(go.Scatter(
            x=[min(lags), max(lags)],
            y=[significance_level, significance_level],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name=f'유의수준 {significance_level}'
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{cause_var}에서 {effect_var}로의 Granger 인과성 p-값',
            xaxis_title='시차(Lag)',
            yaxis_title='p-값',
            height=400
        )
        
        return fig

    def plot_residual_acf(
            self,
            residuals: np.ndarray,
            max_lags: int = 20,
            title: str = "잔차의 자기상관함수 (ACF)",
            **kwargs
        ) -> go.Figure:
        """
        모델 잔차의 자기상관함수(ACF)를 시각화합니다.
        
        Args:
            residuals: 모델 잔차
            max_lags: 최대 시차
            title: 그래프 제목
            
        Returns:
            plotly Figure 객체
        """
        from statsmodels.tsa.stattools import acf
        
        # ACF 값 계산
        acf_values = acf(residuals, nlags=max_lags, fft=False)
        lags = list(range(len(acf_values)))  # range 객체를 list로 변환
        
        # 신뢰 구간 계산 (95%)
        confidence = 1.96 / np.sqrt(len(residuals))
        
        # ACF 시각화
        fig = go.Figure()
        
        # ACF 막대 그래프
        fig.add_trace(go.Bar(
            x=lags,
            y=acf_values,
            name='ACF',
            marker_color='blue'
        ))
        
        # 신뢰 구간 선
        fig.add_trace(go.Scatter(
            x=[0, max(lags)],
            y=[confidence, confidence],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='95% 신뢰구간'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, max(lags)],
            y=[-confidence, -confidence],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title='시차(Lag)',
            yaxis_title='자기상관계수',
            height=400
        )
        
        return fig

    def plot_volatility(
            self,
            series: pd.Series,
            volatility: list,
            title: str = "시계열 데이터 및 변동성",
            **kwargs
        ) -> go.Figure:
        """
        시계열 데이터와 조건부 변동성을 함께 시각화합니다.
        
        Args:
            series: 원본 시계열 데이터
            volatility: 조건부 변동성 값 리스트
            title: 그래프 제목
            
        Returns:
            plotly Figure 객체
        """
        # 변동성 시각화할 날짜 계산
        dates = series.index[-len(volatility):]
        
        fig = go.Figure()
        
        # 원본 시계열
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='원본 데이터',
            line=dict(color='blue')
        ))
        
        # 조건부 변동성
        fig.add_trace(go.Scatter(
            x=dates,
            y=volatility,
            mode='lines',
            name='조건부 변동성',
            line=dict(color='red', width=2)
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title="날짜",
            yaxis_title="값",
            height=500
        )
        
        return fig


@st.cache_data(ttl=3600)
def cached_plot_timeseries(data, title, xlabel, ylabel, color='#1f77b4'):
    """시계열 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_timeseries(data, title=title, xlabel=xlabel, ylabel=ylabel, color=color)

@st.cache_data(ttl=3600)
def cached_plot_decomposition(decomposition):
    """분해 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_decomposition(decomposition)

@st.cache_data(ttl=3600)
def cached_plot_acf_pacf(acf_values, pacf_values):
    """ACF/PACF 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_acf_pacf(acf_values, pacf_values)

@st.cache_data(ttl=3600)
def cached_plot_forecast_comparison(train, test, forecasts):
    """예측 비교 그래프 캐싱"""
    try:
        # numpy 배열을 Series로 변환 (인덱스 보장)
        for model_name, forecast in forecasts.items():
            if isinstance(forecast, np.ndarray):
                # 길이 맞춤
                min_len = min(len(test), len(forecast))
                forecasts[model_name] = pd.Series(forecast[:min_len], index=test.index[:min_len])
                
        viz = TimeSeriesVisualizer()
        return viz.plot_forecast_comparison(train, test, forecasts)
    except Exception as e:
        st.error(f"예측 비교 그래프 생성 중 오류: {str(e)}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        return None

@st.cache_data(ttl=3600)
def cached_plot_metrics_comparison(metrics):
    """메트릭 비교 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_metrics_comparison(metrics)

@st.cache_data(ttl=3600)
def cached_plot_residuals(actual, predicted, title="Residual Analysis"):
    """잔차 분석 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_residuals(actual, predicted, title=title)

@st.cache_data(ttl=3600)
def cached_plot_differencing_comparison(original_series, differenced_series, title="차분 비교 (Differencing Comparison)"):
    """차분 비교 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_differencing_comparison(original_series, differenced_series, title=title)

@st.cache_data(ttl=3600)
def cached_plot_stationarity_comparison(adf_result, kpss_result, target_name="시계열 데이터"):
    """정상성 비교 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_stationarity_comparison(adf_result, kpss_result, target_name=target_name)

@st.cache_data(ttl=3600)
def cached_plot_change_points(series, change_points_result, title="구조적 변화점 분석"):
    """구조적 변화점 분석 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_change_points(series, change_points_result, title=title)

@st.cache_data(ttl=3600)
def cached_plot_segment_means(series, change_points_result, title="세그먼트별 평균값"):
    """세그먼트별 평균값 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_segment_means(series, change_points_result, title=title)

@st.cache_data(ttl=3600)
def cached_plot_correlation_heatmap(correlation_matrix, title="변수 간 상관관계 히트맵"):
    """상관관계 히트맵 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_correlation_heatmap(correlation_matrix, title=title)

@st.cache_data(ttl=3600)
def cached_plot_granger_causality(lags, p_values, cause_var, effect_var, significance_level=0.05):
    """Granger 인과성 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_granger_causality(lags, p_values, cause_var, effect_var, significance_level=significance_level)

@st.cache_data(ttl=3600)
def cached_plot_residual_acf(residuals, max_lags=20, title="잔차의 자기상관함수 (ACF)"):
    """잔차 ACF 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_residual_acf(residuals, max_lags=max_lags, title=title)

@st.cache_data(ttl=3600)
def cached_plot_volatility(series, volatility, title="시계열 데이터 및 변동성"):
    """변동성 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_volatility(series, volatility, title=title)

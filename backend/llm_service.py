"""
LLM 서비스 모듈 - LLM 분석 관련 기능
"""
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import app_config
from prompts.hyperparameter_recommendation_prompt import HYPERPARAMETER_RECOMMENDATION_PROMPT
from backend.model_service import get_model_factory
from utils.llm_connector import LLMConnector
from prompts.time_series_analysis_prompt import TIME_SERIES_ANALYSIS_PROMPT

def check_analysis_ready():
    """
    LLM 분석을 위한 데이터와 모델이 준비되었는지 확인합니다.
    
    Returns:
        tuple: (준비 완료 여부, 메시지)
    """
    # 모델 예측 결과와 평가 지표를 우선 확인
    if not hasattr(st.session_state, 'forecasts') or not st.session_state.forecasts:
        return False, "모델 예측 결과가 없습니다. 먼저 '모델 학습 및 예측' 페이지에서 모델 학습을 완료해주세요."
        
    if not hasattr(st.session_state, 'metrics') or not st.session_state.metrics:
        return False, "모델 평가 지표가 없습니다. 먼저 '모델 학습 및 예측' 페이지에서 모델 학습을 완료해주세요."
        
    if not hasattr(st.session_state, 'best_model') or st.session_state.best_model is None:
        return False, "최적 모델 정보가 없습니다. 먼저 '모델 학습 및 예측' 페이지에서 모델 학습을 완료해주세요."
    
    # 시계열 데이터는 필수 확인
    if not hasattr(st.session_state, 'series') or st.session_state.series is None:
        return False, "시계열 데이터가 준비되지 않았습니다."
    
    # train/test 데이터는 경고만 출력하고 진행
    if not hasattr(st.session_state, 'train') or st.session_state.train is None or not hasattr(st.session_state, 'test') or st.session_state.test is None:
        st.warning("학습/테스트 데이터가 접근 불가능하지만, 모델 결과가 있으므로 분석을 진행합니다.")
    
    return True, "분석 준비 완료"

def run_llm_analysis():
    """
    LLM을 사용하여 시계열 분석 수행
    
    Returns:
        str: 분석 결과 또는 None
    """
    # 데이터 및 모델 학습 상태 확인
    is_ready, message = check_analysis_ready()
    
    if not is_ready:
        st.warning(message)
        return None
    
    with st.spinner("LLM을 통해 시계열 분석 결과를 분석 중입니다..."):
        try:
            llm_connector = LLMConnector(
                base_url=app_config.OLLAMA_SERVER,
                model=app_config.OLLAMA_MODEL
            )
            
            # 데이터 정보 수집 - 안전하게 값 추출
            data_info = {
                "target_variable": st.session_state.selected_target,
                "station": st.session_state.selected_station if hasattr(st.session_state, 'selected_station') else None,
                "seasonality_period": st.session_state.period if hasattr(st.session_state, 'period') else None,
                "data_range": {
                    "total_points": len(st.session_state.series),
                },
                "date_range": {
                    "start": str(st.session_state.series.index.min()),
                    "end": str(st.session_state.series.index.max())
                },
                "value_stats": {
                    "min": float(st.session_state.series.min()),
                    "max": float(st.session_state.series.max()),
                    "mean": float(st.session_state.series.mean()),
                    "std": float(st.session_state.series.std())
                }
            }

            # 훈련/테스트 데이터가 있는 경우만 추가
            if hasattr(st.session_state, 'train') and st.session_state.train is not None:
                data_info["data_range"]["train_points"] = len(st.session_state.train)

            if hasattr(st.session_state, 'test') and st.session_state.test is not None:
                data_info["data_range"]["test_points"] = len(st.session_state.test)
            
            # 정상성 정보 추가 (있는 경우만)
            if (hasattr(st.session_state, 'stationarity_result') 
                and st.session_state.stationarity_result is not None):
                try:
                    data_info["stationarity"] = {
                        "is_stationary": bool(st.session_state.stationarity_result["is_stationary"]),
                        "p_value": float(st.session_state.stationarity_result["p_value"]),
                        "test_statistic": float(st.session_state.stationarity_result["test_statistic"])
                    }
                except (KeyError, TypeError):
                    # 정상성 정보에 필요한 키가 없는 경우 무시
                    pass
            
            # 분해 정보 추가 (있는 경우만, 요약 정보만)
            if (hasattr(st.session_state, 'decomposition') 
                and st.session_state.decomposition is not None):
                try:
                    decomp_info = {}
                    for comp_name, comp_data in st.session_state.decomposition.items():
                        if comp_name != 'observed' and comp_data is not None:  # 원본 데이터는 제외
                            clean_data = comp_data.dropna()
                            if not clean_data.empty:
                                decomp_info[comp_name] = {
                                    "min": float(clean_data.min()),
                                    "max": float(clean_data.max()),
                                    "mean": float(clean_data.mean())
                                }
                    
                    if decomp_info:  # 비어있지 않은 경우만 추가
                        data_info["decomposition"] = decomp_info
                except Exception as e:
                    st.warning(f"분해 정보 처리 중 오류 발생 (무시됨): {e}")
            
            # 모델 결과 정보 수집
            model_results = {
                "best_model": st.session_state.best_model,
                "models": {}
            }
            
            # 각 모델의 메트릭 정보 추가
            for model_name, metrics in st.session_state.metrics.items():
                model_results["models"][model_name] = {
                    "metrics": {}
                }
                for metric_name, metric_value in metrics.items():
                    # 'name'과 같은 문자열 필드는 건너뛰기
                    if metric_name in ['name', 'model_type']:
                        continue
                        
                    # NaN 값 처리
                    if pd.isna(metric_value):
                        model_results["models"][model_name]["metrics"][metric_name] = None
                    else:
                        try:
                            # 숫자 값만 float으로 변환
                            model_results["models"][model_name]["metrics"][metric_name] = float(metric_value)
                        except (ValueError, TypeError):
                            # 숫자로 변환할 수 없는 값은 그대로 유지
                            model_results["models"][model_name]["metrics"][metric_name] = metric_value
                
                # 예측값이 있는 경우만 통계 추가
                # 모델명 매핑을 위한 사전 정의
                model_mapping = {
                    'arima': ['ARIMA', 'ARIMA/SARIMA', 'SARIMA'],
                    'exp_smoothing': ['지수평활법', 'ExpSmoothing', 'Exponential Smoothing'],
                    'prophet': ['Prophet'],
                    'lstm': ['LSTM']
                }

                # forecast 키를 찾기 위한 함수
                def find_matching_forecast_key(model_name):
                    # 정확히 일치하는 경우
                    if model_name in st.session_state.forecasts:
                        return model_name
                    
                    # 매핑을 통한 검색
                    for forecast_key in st.session_state.forecasts.keys():
                        # 양방향 부분 문자열 검색
                        if model_name in forecast_key or forecast_key in model_name:
                            return forecast_key
                        
                        # 매핑 테이블을 통한 검색
                        for key, aliases in model_mapping.items():
                            if (key == model_name or model_name in aliases) and (key == forecast_key or forecast_key in aliases):
                                return forecast_key
                    
                    return None

                # 매칭되는 forecast 키 찾기
                forecast_key = find_matching_forecast_key(model_name)

                if forecast_key:
                    forecast = st.session_state.forecasts[forecast_key]
                    if forecast is not None and len(forecast) > 0:
                        model_results["models"][model_name]["forecast_stats"] = {
                            "min": float(np.min(forecast)),
                            "max": float(np.max(forecast)),
                            "mean": float(np.mean(forecast)),
                            "std": float(np.std(forecast))
                        }
            
            # LLM 분석 요청
            analysis_result = llm_connector.analyze_time_series(
                data_info,
                model_results,
                TIME_SERIES_ANALYSIS_PROMPT
            )
            
            # 세션 상태에 분석 결과 저장
            st.session_state.llm_analysis = analysis_result
            
            return analysis_result
            
        except Exception as e:
            st.error(f"LLM 분석 중 오류 발생: {str(e)}")
            return f"## 오류 발생\n\nLLM 분석 중 오류가 발생했습니다: {str(e)}"


def recommend_hyperparameters(model_type: str) -> Dict[str, Any]:
    """
    LLM을 사용하여 하이퍼파라미터 추천 수행
    """
    # 데이터 및 모델 학습 상태 확인
    is_ready, message = check_analysis_ready()
    
    if not is_ready:
        st.warning(message)
        return None
    
    with st.spinner(f"{model_type} 모델의 하이퍼파라미터 추천 중..."):
        try:
            # 데이터 정보, 모델 정보, 성능 지표 수집
            data_info = _collect_data_info()
            model_info = get_model_parameters_for_recommendation(model_type)
            performance_metrics = _collect_performance_metrics(model_type)
            
            # LLM에 하이퍼파라미터 추천 요청
            llm_connector = LLMConnector(
                base_url=app_config.OLLAMA_SERVER,
                model=app_config.OLLAMA_MODEL
            )
            
            recommendation = llm_connector.recommend_hyperparameters(
                data_info,
                model_info,
                performance_metrics,
                HYPERPARAMETER_RECOMMENDATION_PROMPT
            )
            
            # 세션 상태에 추천 결과 저장
            if 'hyperparameter_recommendations' not in st.session_state:
                st.session_state.hyperparameter_recommendations = {}
            
            st.session_state.hyperparameter_recommendations[model_type] = recommendation
            
            return recommendation
            
        except Exception as e:
            st.error(f"하이퍼파라미터 추천 중 오류 발생: {str(e)}")
            return {"error": str(e)}


def get_model_parameters_for_recommendation(model_type: str) -> Dict[str, Any]:
    """
    LLM에 전달할 모델 파라미터 정보 수집
    """
    # 기본 파라미터 초기화
    default_params = {}
    
    # 모델 유형에 따른 기본 파라미터 설정
    if model_type == 'arima':
        default_params = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 24),
            'seasonal': True,
            'm': 24
        }
    elif model_type == 'exp_smoothing':
        default_params = {
            'model_type': 'hw',
            'trend': 'add',
            'seasonal': 'add',
            'seasonal_periods': 24,
            'damped_trend': False
        }
    elif model_type == 'prophet':
        default_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05
        }
    elif model_type == 'lstm':
        default_params = {
            'n_steps': 24,
            'lstm_units': [50],
            'dropout_rate': 0.2,
            'epochs': 50
        }
    
    # 세션 상태에서 사용된 모델 파라미터 확인
    if 'model_params' in st.session_state and model_type in st.session_state.model_params:
        return st.session_state.model_params[model_type]
    
    # 대체 검색: 유사한 이름의 모델 찾기
    if 'metrics' in st.session_state:
        for model_name, metrics in st.session_state.metrics.items():
            # 대소문자 구분 없이 모델 유형 비교
            model_type_lower = model_type.lower()
            model_name_lower = model_name.lower()
            
            # 모델 이름 포함 관계 확인
            if (model_type_lower in model_name_lower or
                model_name_lower in model_type_lower or
                (metrics.get('name') and model_type_lower in metrics.get('name', '').lower())):
                
                # 모델 팩토리에서 해당 모델 인스턴스 생성
                model_factory = get_model_factory()
                try:
                    model = model_factory.get_model(model_type)
                    params = model.get_params()
                    if params:  # 파라미터가 있으면 반환
                        return params
                except Exception as e:
                    st.warning(f"모델 파라미터 조회 중 오류 발생: {e}")
    
    # 기본 파라미터 반환
    return default_params


def _collect_data_info() -> Dict[str, Any]:
    """
    LLM에 전달할 데이터 특성 정보 수집
    """
    data_info = {}
    
    # 시계열 데이터 기본 정보
    if st.session_state.series is not None:
        data_info["target_variable"] = st.session_state.selected_target
        data_info["station"] = st.session_state.selected_station
        data_info["data_length"] = len(st.session_state.series)
        data_info["date_range"] = {
            "start": str(st.session_state.series.index.min()),
            "end": str(st.session_state.series.index.max())
        }
        data_info["statistics"] = {
            "min": float(st.session_state.series.min()),
            "max": float(st.session_state.series.max()),
            "mean": float(st.session_state.series.mean()),
            "std": float(st.session_state.series.std())
        }
    
    # 훈련/테스트 분할 정보
    if st.session_state.train is not None and st.session_state.test is not None:
        data_info["train_test_split"] = {
            "train_size": len(st.session_state.train),
            "test_size": len(st.session_state.test),
            "test_ratio": len(st.session_state.test) / (len(st.session_state.train) + len(st.session_state.test))
        }
    
    # 시계열 분해 정보 (있는 경우)
    if hasattr(st.session_state, 'decomposition') and st.session_state.decomposition is not None:
        decomp_info = {}
        for comp_name, comp_data in st.session_state.decomposition.items():
            if comp_name != 'observed' and comp_data is not None:  # 원본 데이터는 제외
                clean_data = comp_data.dropna()
                if not clean_data.empty:
                    decomp_info[comp_name] = {
                        "min": float(clean_data.min()),
                        "max": float(clean_data.max()),
                        "mean": float(clean_data.mean()),
                        "std": float(clean_data.std())
                    }
        if decomp_info:
            data_info["decomposition"] = decomp_info
            data_info["seasonality_period"] = st.session_state.period
    
    # 정상성 검정 결과 (있는 경우)
    if hasattr(st.session_state, 'stationarity_result') and st.session_state.stationarity_result is not None:
        data_info["stationarity"] = {
            "is_stationary": bool(st.session_state.stationarity_result.get('is_stationary', False)),
            "p_value": float(st.session_state.stationarity_result.get('p_value', 1.0)),
            "test_statistic": float(st.session_state.stationarity_result.get('test_statistic', 0.0))
        }
    
    # ACF/PACF 정보 (있는 경우, 요약만)
    if hasattr(st.session_state, 'acf_values') and st.session_state.acf_values is not None:
        acf_values = st.session_state.acf_values
        # 첫 10개 값과 최댓값/최솟값만 포함
        data_info["acf_summary"] = {
            "first_10": acf_values[:min(10, len(acf_values))].tolist(),
            "max_value": float(max(acf_values)),
            "min_value": float(min(acf_values))
        }
    
    if hasattr(st.session_state, 'pacf_values') and st.session_state.pacf_values is not None:
        pacf_values = st.session_state.pacf_values
        # 첫 10개 값과 최댓값/최솟값만 포함
        data_info["pacf_summary"] = {
            "first_10": pacf_values[:min(10, len(pacf_values))].tolist(),
            "max_value": float(max(pacf_values)),
            "min_value": float(min(pacf_values))
        }
    
    return data_info


def _collect_performance_metrics(model_type: str) -> Dict[str, Any]:
    """
    LLM에 전달할 모델 성능 지표 수집
    """
    performance_metrics = {}
    
    # 모델 팩토리에서 기본 모델 인스턴스 생성
    model_factory = get_model_factory()
    
    # 요청된 모델 유형과 일치하는 모델 찾기
    for model_name, metrics in st.session_state.metrics.items():
        # 모델 유형 확인 (정확한 일치 또는 포함 관계 확인)
        is_matching_model = (
            model_type == model_name or
            model_type in model_name or
            model_name in model_type or
            (metrics.get('name') and (model_type == metrics['name'] or model_type in metrics['name'] or metrics['name'] in model_type))
        )
        
        if is_matching_model:
            # 성능 지표 복사
            for metric_name, value in metrics.items():
                if metric_name != 'name':  # name은 제외
                    # NaN 값 처리
                    if pd.isna(value):
                        performance_metrics[metric_name] = None
                    else:
                        performance_metrics[metric_name] = float(value)
            
            # 예측값이 있는 경우 예측 통계 추가
            if model_name in st.session_state.forecasts:
                forecast = st.session_state.forecasts[model_name]
                if forecast is not None and len(forecast) > 0:
                    performance_metrics["forecast_stats"] = {
                        "min": float(np.min(forecast)),
                        "max": float(np.max(forecast)),
                        "mean": float(np.mean(forecast)),
                        "std": float(np.std(forecast))
                    }
            break
    
    # 모델 비교 정보: 최적 모델과의 비교
    if st.session_state.best_model and st.session_state.best_model in st.session_state.metrics:
        best_metrics = st.session_state.metrics[st.session_state.best_model]
        
        if performance_metrics:  # 이미 성능 지표가 있는 경우
            performance_metrics["comparison_with_best"] = {}
            
            for metric_name, best_value in best_metrics.items():
                if metric_name != 'name' and metric_name in performance_metrics:
                    current_value = performance_metrics[metric_name]
                    
                    # NaN 값 처리
                    if pd.isna(best_value) or pd.isna(current_value):
                        continue
                    
                    # R^2는 높을수록 좋음, 나머지는 낮을수록 좋음
                    if metric_name == 'R^2':
                        diff_percent = (current_value - best_value) / abs(best_value) * 100 if best_value != 0 else float('inf')
                    else:
                        diff_percent = (best_value - current_value) / best_value * 100 if best_value != 0 else float('inf')
                    
                    performance_metrics["comparison_with_best"][metric_name] = f"{diff_percent:.2f}%"
    
    return performance_metrics

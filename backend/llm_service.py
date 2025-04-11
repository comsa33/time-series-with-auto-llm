"""
LLM 서비스 모듈 - LLM 분석 관련 기능
"""
import streamlit as st
import pandas as pd
import numpy as np

from config.settings import app_config
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
                    # NaN 값 처리
                    if pd.isna(metric_value):
                        model_results["models"][model_name]["metrics"][metric_name] = None
                    else:
                        model_results["models"][model_name]["metrics"][metric_name] = float(metric_value)
                
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

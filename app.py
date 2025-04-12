"""
서울시 대기질 시계열 분석 메인 Streamlit 앱
"""
import os
import warnings
import streamlit as st

from config.settings import app_config
from frontend.sidebar import initialize_sidebar
from frontend.session_state import initialize_session_state

# GPU 사용 비활성화 (CPU 모드만 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TensorFlow 로그 레벨 조정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 경고 메시지 무시
warnings.filterwarnings('ignore')

def main():
    """메인 애플리케이션 진입점"""
    # 페이지 설정
    st.set_page_config(
        page_title=app_config.APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바 렌더링
    initialize_sidebar()

    # app.py
    pages = {
        "APP": [
            st.Page("streamlit_pages/introduction.py", title="📋 데이터 명세"),
        ],
        "ANALYSIS": [
            st.Page("streamlit_pages/time-series-graph.py", title="📈 시계열 시각화"),
            st.Page("streamlit_pages/decomposition.py", title="🔍 시계열 분해"),
            st.Page("streamlit_pages/stationarity.py", title="📊 정상성 & ACF/PACF"),
            st.Page("streamlit_pages/differencing.py", title="🔄 차분 분석"),
        ],
        "AI": [
            st.Page("streamlit_pages/modeling.py", title="🤖 모델 학습 및 예측"),
            st.Page("streamlit_pages/hyperparameter_optimization.py", title="🎯 하이퍼파라미터 최적화"),
            st.Page("streamlit_pages/llm_analysis.py", title="🧠 모델 결과 - LLM 분석"),
        ]
    }
    pg = st.navigation(pages)
    pg.run()
    
if __name__ == "__main__":
    main()

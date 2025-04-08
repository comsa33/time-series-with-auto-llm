"""
LLM 분석 페이지
"""
import streamlit as st

from backend.llm_service import check_analysis_ready, run_llm_analysis
from config.settings import app_config

# 페이지 제목
st.title("📊 LLM 시계열 데이터 분석")
st.markdown("인공지능 모델을 활용하여 시계열 데이터 및 예측 결과를 자동으로 분석합니다.")

# LLM 분석 설명
with st.expander("📊 LLM 분석 설정", expanded=True):
    st.info(f"Ollama 서버를 통해 {app_config.OLLAMA_MODEL} 모델로 시계열 분석 결과를 자동으로 분석합니다.")
    
    st.markdown("""
    ### 🤖 LLM 분석이란?
    
    LLM(Large Language Model) 분석은 대규모 언어 모델을 활용하여 시계열 데이터와 예측 결과를 자동으로 해석하고 통찰력을 제공하는 기능입니다.
    
    ### 📊 분석 내용
    
    LLM은 다음과 같은 내용을 분석하고 마크다운 형식으로 보고서를 생성합니다:
    
    1. **데이터 특성 요약** - 주요 패턴, 추세, 계절성 등
    2. **각 모델의 성능 비교 및 분석** (RMSE, MAE, R^2, MAPE 등 기준)
    3. **최적 모델 추천 및 그 이유**
    4. **미래 예측값에 대한 해석 및 신뢰도 평가**
    5. **데이터 특성에 따른 각 모델의 장단점 분석**
    6. **예측 성능을 더 향상시키기 위한 제안**
    
    ### ⚠️ 분석 요구사항
    
    LLM 분석을 실행하기 전에 다음 단계를 완료해야 합니다:
    
    1. 시계열 데이터 로드 및 전처리
    2. 모델 학습 및 예측 완료
    """)

# 데이터 및 모델 학습 상태 확인
is_ready, message = check_analysis_ready()

if not is_ready:
    st.warning(message)
else:
    # LLM 분석 실행 버튼
    if st.button("LLM 분석 시작", type="primary"):
        analysis_result = run_llm_analysis()
        
        if analysis_result:
            st.success("LLM 분석이 완료되었습니다!")
    
    # 이전에 분석한 결과가 있으면 표시
    if hasattr(st.session_state, 'llm_analysis') and st.session_state.llm_analysis:
        # 다운로드 버튼
        st.download_button(
            label="분석 결과 다운로드 (Markdown)",
            data=st.session_state.llm_analysis,
            file_name="time_series_analysis_report.md",
            mime="text/markdown"
        )
        
        # 결과 표시
        st.markdown(st.session_state.llm_analysis)
    elif is_ready and not st.session_state.llm_analysis:
        st.info("LLM 분석 시작 버튼을 클릭하여 분석을 실행하세요.")

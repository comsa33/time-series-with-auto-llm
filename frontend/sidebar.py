"""
사이드바 구성을 위한 모듈
"""
from datetime import datetime, timedelta
import streamlit as st

from backend.data_service import load_data, update_series
from frontend.components import show_memory_usage, render_footer

def initialize_sidebar():
    """
    사이드바 초기화 및 구성
    """
    # 데이터 로드 섹션
    render_data_load_section()
    
    # 데이터가 있을 경우 분석 옵션 섹션 표시
    if st.session_state.df is not None and not st.session_state.df.empty:
        render_analysis_options()
    
    # 메모리 사용량 표시
    show_memory_usage()

    # 푸터 렌더링
    render_footer()


def render_data_load_section():
    """
    데이터 로드 섹션 렌더링
    """
    st.sidebar.subheader("서울시 대기질 데이터 로드", help="서울시 IoT 대기질 데이터 API를 통해 데이터를 실시간으로 가져옵니다.")

    # 날짜 범위 선택
    today = datetime.now().date()
    default_end_date = today
    default_start_date = today - timedelta(days=30)

    st.sidebar.markdown("##### 📅 분석 기간 선택", help="시계열 분석을 위한 데이터 기간을 선택하세요. (최대 30일)")

    date_col1, date_col2 = st.sidebar.columns(2)

    with date_col1:
        start_date = st.date_input(
            "시작 날짜",
            default_start_date
        )
        
    with date_col2:
        # 시작일 기준으로 최대 종료일 계산 (30일 이내)
        max_end_date = start_date + timedelta(days=30)
        if today < max_end_date:
            max_end_date = today
            
        end_date = st.date_input(
            "종료 날짜",
            min(default_end_date, max_end_date),
            min_value=start_date,
            max_value=max_end_date
        )

    # 선택된 날짜 범위 일수 계산
    date_range_days = (end_date - start_date).days

    # 기간 표시 정보 및 시각화
    progress_value = min(date_range_days / 30, 1.0)
    st.sidebar.progress(progress_value)
    st.sidebar.text(f"선택된 기간: {date_range_days + 1}일 / 최대 30일")

    if date_range_days > 25:
        st.sidebar.warning("데이터 양이 많을수록 분석 시간이 길어질 수 있습니다.")

    # 데이터 가져오기 버튼
    if st.sidebar.button("데이터 가져오기"):
        with st.spinner("서울시 API에서 데이터를 가져오는 중..."):
            df = load_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            if df is not None and not df.empty:
                st.session_state.df = df
                st.rerun()  # 화면 갱신

def render_analysis_options():
    """
    분석 옵션 설정 섹션 렌더링
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 시계열 분석 옵션")
    
    # 측정소 선택
    if 'MSRSTE_NM' in st.session_state.df.columns:
        stations = ['전체 평균'] + sorted(st.session_state.df['MSRSTE_NM'].unique().tolist())
        selected_station = st.sidebar.selectbox(
            "측정소 선택", 
            stations,
            index=0 if st.session_state.selected_station is None else stations.index(st.session_state.selected_station if st.session_state.selected_station else "전체 평균")
        )
        
        if selected_station == '전체 평균':
            st.session_state.selected_station = None
        else:
            st.session_state.selected_station = selected_station
    else:
        st.session_state.selected_station = None
        st.sidebar.info("측정소 정보가 없습니다.")
    
    # 타겟 변수 선택
    import numpy as np
    numeric_columns = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    target_options = [col for col in numeric_columns if col in ['PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2']]
    
    if not target_options:
        target_options = numeric_columns
    
    if target_options:
        selected_target = st.sidebar.selectbox(
            "분석할 변수 선택", 
            target_options,
            index=5 if st.session_state.selected_target is None else target_options.index(st.session_state.selected_target)
        )
        st.session_state.selected_target = selected_target
    else:
        st.sidebar.error("분석 가능한 숫자형 변수가 없습니다.")
        return
    
    # 테스트 데이터 비율 설정
    test_size = st.sidebar.slider(
        "테스트 데이터 비율",
        min_value=0.1,
        max_value=0.5,
        value=st.session_state.test_size,
        step=0.05
    )
    st.session_state.test_size = test_size
    
    # 시리즈 데이터 업데이트
    update_series()

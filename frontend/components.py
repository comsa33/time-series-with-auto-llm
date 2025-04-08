"""
공통 UI 컴포넌트 모듈
"""
import os

import psutil
import pandas as pd
import streamlit as st


def show_memory_usage():
    """
    메모리 사용량을 사이드바에 표시
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB 단위
    
    # 사이드바 하단에 메모리 사용량 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 시스템 상태")
    st.sidebar.progress(min(memory_usage / 4000, 1.0))  # 4GB 기준
    st.sidebar.text(f"메모리 사용량: {memory_usage:.1f} MB")
    
    if memory_usage > 3500:  # 3.5GB 이상일 때 경고
        st.sidebar.warning("⚠️ 메모리 사용량이 높습니다. 불필요한 모델을 제거하거나 샘플 데이터를 사용하세요.")

def render_model_selector(model_factory):
    """
    모델 선택 UI 렌더링
    
    Args:
        model_factory: 모델 팩토리 인스턴스
    
    Returns:
        선택된 모델 목록, 모델 복잡도
    """
    with st.expander("모델 선택 및 설정", not st.session_state.models_trained):
        available_models = model_factory.get_all_available_models()
        
        selected_models = st.multiselect(
            "사용할 모델 선택",
            available_models,
            default=available_models[:] if not st.session_state.selected_models else st.session_state.selected_models
        )
        
        # 복잡도 설정 추가
        complexity = st.radio(
            "모델 복잡도 설정",
            ["간단 (빠름, 저메모리)", "중간", "복잡 (정확도 높음, 고메모리)"],
            index=0,
            horizontal=True,
            help="낮은 복잡도는 계산 속도가 빠르지만 정확도가 낮을 수 있습니다."
        )
        
        return selected_models, complexity

def render_data_summary(df):
    """
    데이터 요약 정보 표시
    
    Args:
        df: 데이터프레임
    """
    with st.expander("📋 데이터 미리보기", expanded=True):
        # 데이터 샘플 표시
        st.markdown("`ℹ️ 각 컬럼에 커서를 올리면 설명이 표시됩니다.`")
        st.data_editor(
            df.head(),
            column_config={
                "MSRDT": st.column_config.DatetimeColumn(
                    "MSRDT",
                    format="YYYY MMM D, h:mm:ss a",
                    step=60,
                    help="YYYY-MM-DD HH:mm:ss 형태의 측정 일시"
                ),
                "MSRSTE_NM": st.column_config.TextColumn(
                    "MSRSTE_NM",
                    help="서울시 측정소 이름"
                ),
                "NO2": st.column_config.ProgressColumn(
                    "NO2",
                    help="이산화질소 농도(ppm)",
                    format="%.4f",
                    min_value=0,
                    max_value=0.1,
                ),
                "O3": st.column_config.ProgressColumn(
                    "O3",
                    help="오존 농도(ppm)",
                    format="%.4f",
                    min_value=0,
                    max_value=0.15,
                ),
                "CO": st.column_config.ProgressColumn(
                    "CO",
                    help="일산화탄소 농도(ppm)",
                    format="%.4f",
                    min_value=0,
                    max_value=1,
                ),
                "SO2": st.column_config.ProgressColumn(
                    "SO2",
                    help="이산화황 농도(ppm)",
                    format="%.4f",
                    min_value=0,
                    max_value=0.01,
                ),
                "PM10": st.column_config.ProgressColumn(
                    "PM10",
                    help="미세먼지 농도 (µg/m³)",
                    format="%.1f",
                    min_value=0,
                    max_value=300,
                ),
                "PM25": st.column_config.ProgressColumn(
                    "PM25",
                    help="초미세먼지 농도 (µg/m³)",
                    format="%.1f",
                    min_value=0,
                    max_value=200,
                ),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # 정보 섹션 제목
        st.markdown("#### 📊 데이터 요약 정보")
        
        # 데이터 요약 정보를 위한 메트릭 카드 (4개 컬럼으로 배치)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        # 1. 데이터 행 수
        metric_col1.metric(
            label="📈 데이터 행 수",
            value=f"{df.shape[0]:,}",
            help="전체 데이터 레코드 수",
            border=True
        )
        
        # 2. 데이터 열 수
        metric_col2.metric(
            label="📊 데이터 열 수",
            value=f"{df.shape[1]}",
            help="데이터셋의 속성(특성) 수",
            border=True
        )
        
        # 3. 시작 날짜
        if 'MSRDT' in df.columns:
            start_date = df['MSRDT'].min()
            metric_col3.metric(
                label="📅 시작 날짜",
                value=f"{start_date.strftime('%Y-%m-%d')}",
                help="데이터의 시작 날짜",
                border=True
            )
            
            # 4. 종료 날짜
            end_date = df['MSRDT'].max()
            days_diff = (end_date - start_date).days
            metric_col4.metric(
                label="📅 종료 날짜",
                value=f"{end_date.strftime('%Y-%m-%d')}",
                delta=f"{days_diff}일",
                help="데이터의 종료 날짜 (delta는 전체 기간)",
                border=True
            )

def render_station_info(df):
    """
    측정소 정보 표시
    
    Args:
        df: 데이터프레임
    """
    with st.expander("📍 측정소 정보", expanded=True):
        if 'MSRSTE_NM' in df.columns:
            # 측정소 정보를 위한 두 개의 컬럼 (2:1 비율)
            station_col1, station_col2 = st.columns([2, 1])
            
            with station_col1:
                # expander 대신 컨테이너와 제목 사용
                st.markdown("#### 📋 측정소 목록")
                # 구분선으로 시각적 분리 효과
                st.markdown("<hr style='margin: 5px 0px 15px 0px'>", unsafe_allow_html=True)
                
                # 측정소 목록을 표 형태로 표시 (더 구조화된 형태)
                stations = sorted(df['MSRSTE_NM'].unique())
                
                # 측정소 목록을 3개 컬럼으로 정렬하여 표시 (더 읽기 쉽게)
                cols = st.columns(3)
                for i, station in enumerate(stations):
                    cols[i % 3].markdown(f"• {station}")
            
            with station_col2:
                # 측정소 수를 메트릭으로 표시
                num_stations = df['MSRSTE_NM'].nunique()
                st.metric(
                    label="🏢 측정소 수",
                    value=f"{num_stations}개",
                    help="분석 대상 측정소의 총 개수",
                    border=True
                )
                
                # 시간당 측정 빈도 계산 (대략적인 값)
                if 'MSRDT' in df.columns:
                    start_date = df['MSRDT'].min()
                    end_date = df['MSRDT'].max()
                    hours_span = (end_date - start_date).total_seconds() / 3600
                    records_per_hour = df.shape[0] / max(hours_span, 1)
                    
                    st.metric(
                        label="📊 측정 빈도",
                        value=f"{records_per_hour:.1f}회/시간",
                        help="시간당 평균 측정 빈도",
                        border=True
                    )
                    
                    # 추가 정보: 측정소별 데이터 수 분포
                    records_per_station = df.groupby('MSRSTE_NM').size().mean()
                    st.metric(
                        label="📊 측정소별 데이터",
                        value=f"{records_per_station:.1f}개",
                        help="측정소당 평균 데이터 수",
                        border=True
                    )

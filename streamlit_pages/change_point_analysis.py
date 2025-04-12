# streamlit_pages/change_point_analysis.py

import streamlit as st
import pandas as pd

from backend.data_service import detect_change_points
from backend.visualization_service import visualize_change_points, visualize_segment_means

# 페이지 제목
st.title("🔍 구조적 변화점 분석")
st.markdown("시계열 데이터의 구조적 변화점을 탐지하고 분석합니다.")

# 데이터 확인
if st.session_state.df is None:
    st.warning("데이터가 로드되지 않았습니다. 사이드바에서 데이터를 로드해주세요.")
    st.stop()
elif st.session_state.series is None:
    st.warning("시계열 데이터가 생성되지 않았습니다. 사이드바에서 분석 변수와 측정소를 선택해주세요.")
    st.stop()

# 변화점 탐지 설정
st.markdown("## 변화점 탐지 설정")

col1, col2 = st.columns(2)

with col1:
    detection_method = st.selectbox(
        "탐지 방법",
        options=['l1', 'l2', 'rbf', 'linear', 'normal', 'ar'],
        index=0,
        help="변화점 탐지 알고리즘의 비용 함수"
    )

with col2:
    min_segment_size = st.slider(
        "최소 세그먼트 크기",
        min_value=10,
        max_value=100,
        value=30,
        help="변화점 사이의 최소 데이터 포인트 수"
    )

# 변화점 탐지 실행
if st.button("변화점 탐지 실행", type="primary"):
    with st.spinner("변화점 탐지 중..."):
        change_points_result = detect_change_points(
            method=detection_method,
            min_size=min_segment_size
        )
        
        if change_points_result:
            st.success(f"변화점 탐지 완료: {change_points_result['num_changes']}개의 변화점을 발견했습니다.")
        else:
            st.error("변화점 탐지 중 오류가 발생했습니다.")

# 변화점 결과 표시 (이하 코드는 이전과 동일)
if hasattr(st.session_state, 'change_points_result'):
    result = st.session_state.change_points_result
    
    # 변화점 시각화
    st.markdown("## 변화점 시각화")
    
    change_points_fig = visualize_change_points()
    if change_points_fig:
        st.plotly_chart(change_points_fig, use_container_width=True, theme="streamlit")
    
    # 세그먼트 정보 표시
    st.markdown("## 세그먼트 정보")
    
    segments_data = []
    for i, segment in enumerate(result['segments']):
        segments_data.append({
            "세그먼트 번호": i + 1,
            "시작 일자": segment['start_date'],
            "종료 일자": segment['end_date'],
            "데이터 길이": segment['length'],
            "평균값": f"{segment['mean']:.2f}",
            "표준편차": f"{segment['std']:.2f}"
        })
    
    if segments_data:
        segments_df = pd.DataFrame(segments_data)
        st.table(segments_df)
        
        # 세그먼트 평균값 시각화
        segment_means_fig = visualize_segment_means()
        if segment_means_fig:
            st.plotly_chart(segment_means_fig, use_container_width=True, theme="streamlit")
        
        # 변화점 요약 및 해석
        st.markdown("## 변화점 분석 결과")
        
        if result['num_changes'] > 0:
            st.markdown("### 주요 변화점")
            
            for i in range(len(result['change_dates'])):
                change_date = result['change_dates'][i]
                
                # 이전 세그먼트와 다음 세그먼트 정보
                prev_segment = result['segments'][i] if i < len(result['segments']) else None
                next_segment = result['segments'][i+1] if i+1 < len(result['segments']) else None
                
                if prev_segment and next_segment:
                    # 평균값 변화 계산
                    prev_mean = float(prev_segment['mean'])
                    next_mean = float(next_segment['mean'])
                    change_pct = (next_mean - prev_mean) / prev_mean * 100 if prev_mean != 0 else float('inf')
                    
                    # 변화 방향
                    direction = "증가" if next_mean > prev_mean else "감소"
                    
                    st.markdown(f"**변화점 {i+1}: {change_date}**")
                    st.markdown(f"- {prev_segment['end_date']}까지 평균: {prev_mean:.2f}")
                    st.markdown(f"- {next_segment['start_date']}부터 평균: {next_mean:.2f}")
                    st.markdown(f"- 변화량: {abs(next_mean - prev_mean):.2f} ({direction}, {abs(change_pct):.1f}%)")
                    
                    # 변화 정도에 따른 해석
                    if abs(change_pct) > 50:
                        st.markdown(f"- **큰 폭의 {direction}**: 급격한 환경 변화 또는 정책 변화가 있었을 가능성이 있습니다.")
                    elif abs(change_pct) > 20:
                        st.markdown(f"- **중간 정도의 {direction}**: 계절적 요인이나 중요한 사건이 영향을 미쳤을 수 있습니다.")
                    else:
                        st.markdown(f"- **소폭 {direction}**: 점진적인 변화가 있었습니다.")
            
            # 종합 해석
            st.markdown("### 종합 해석")
            
            if result['num_changes'] == 1:
                st.markdown("시계열에 하나의 중요한 변화점이 존재합니다. 이 시점 전후로 데이터의 특성이 달라졌으므로, 모델링 시 이를 고려해야 합니다.")
            elif result['num_changes'] <= 3:
                st.markdown("시계열에 몇 개의 주요 변화점이 존재합니다. 각 구간별로 다른 모델을 적용하거나, 변화점을 더미 변수로 추가하는 것이 효과적일 수 있습니다.")
            else:
                st.markdown("시계열에 여러 변화점이 존재합니다. 구조적 변화가 빈번하므로, 변화에 적응할 수 있는 모델(예: Prophet, LSTM)이 적합할 수 있습니다.")
        else:
            st.info("탐지된 변화점이 없습니다. 시계열이 상대적으로 안정적이거나, 탐지 파라미터를 조정해 볼 필요가 있습니다.")
else:
    st.info("변화점 탐지를 실행하여 결과를 확인하세요.")

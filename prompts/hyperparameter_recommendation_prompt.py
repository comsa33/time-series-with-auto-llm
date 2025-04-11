# prompts/hyperparameter_recommendation_prompt.py
HYPERPARAMETER_RECOMMENDATION_PROMPT = """
# 시계열 모델 하이퍼파라미터 추천 요청

당신은 시계열 데이터 분석과 모델링 전문가입니다. 아래 제공된 데이터 정보와 모델 결과를 바탕으로 해당 모델의 성능을 향상시킬 수 있는 하이퍼파라미터를 추천해주세요.

## 데이터 정보
```json
{data_info}
```
## 현재 모델 정보
```json
{model_info}
```
## 현재 모델 성능
```json
{performance_metrics}
```
응답은 반드시 다음 JSON 형식으로 작성해주세요:
```json
{{
  "model_type": "모델 유형(arima, prophet, exp_smoothing, lstm 중 하나)",
  "recommended_parameters": {{
    "param1": value1,
    "param2": value2
  }},
  "rationale": {{
    "param1": "param1 추천 근거(한국어로)",
    "param2": "param2 추천 근거(한국어로)"
  }},
  "expected_improvement": {{
    "metric1": "예상 개선 효과 1(한국어로)",
    "metric2": "예상 개선 효과 2(한국어로)"
  }}
}}
```
정확한 파라미터 명칭과 유효 범위를 사용해주세요. 시계열 데이터의 특성(추세, 계절성, 주기 등)을 반영한 추천을 제공해주세요.
"""

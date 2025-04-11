# prompts/model_parameter_tuning_prompt.py
MODEL_PARAMETER_TUNING_PROMPT = """
# 시계열 모델 파라미터 조정 요청

당신은 시계열 데이터 분석과 모델링 전문가입니다. 아래 제공된 데이터 정보, 모델 정보, 학습 결과를 바탕으로 모델 파라미터를 조정하여 성능을 향상시켜주세요.

## 데이터 정보
```json
{data_info}
```
## 현재 모델 정보
```json
{model_info}
```
## 학습 결과
```json
{training_results}
```
응답은 반드시 다음 JSON 형식으로 작성해주세요:
```json
{
  "tuned_parameters": {
    "param1": value1,
    "param2": value2
  },
  "tuning_rationale": {
    "param1": "param1 조정 근거",
    "param2": "param2 조정 근거"
  },
  "expected_improvement": {
    "RMSE": "RMSE 예상 개선율",
    "MAE": "MAE 예상 개선율",
    "R^2": "R^2 예상 개선율"
  }
}
```
"""
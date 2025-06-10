# prompts/hyperparameter_recommendation_prompt.py
HYPERPARAMETER_RECOMMENDATION_PROMPT = """
# Time Series Model Hyperparameter Recommendation Request

You are an expert in time series data analysis and modeling. Based on the provided data information and model results, please recommend hyperparameters that can improve the performance of the current model.

## Data Information
```json
{data_info}
```
## Current Model Information
```json
{model_info}
```
## Current Model Performance
```json
{performance_metrics}
```
Please respond in the following JSON format:
```json
{{
  "model_type": "Model type (one of: arima, prophet, exp_smoothing, lstm)",
  "recommended_parameters": {{
    "param1": value1,
    "param2": value2
  }},
  "rationale": {{
    "param1": "Reason for recommending param1",
    "param2": "Reason for recommending param2"
  }},
  "expected_improvement": {{
    "metric1": "Expected improvement 1",
    "metric2": "Expected improvement 2"
  }}
}}
```
Use accurate parameter names and valid ranges. Provide recommendations that reflect the characteristics of the time series data (e.g., trend, seasonality, periodicity).
"""

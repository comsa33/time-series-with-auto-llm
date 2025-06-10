"""
시계열 분석을 위한 프롬프트 템플릿 모듈
"""

TIME_SERIES_ANALYSIS_PROMPT = """
# Time Series Analysis Request

You are a time series analysis expert. Based on the data information and model results provided below, please write a detailed analysis in Markdown format.

## Data Information
```json
{data_info}
```
## Model Analysis Results
```json
{model_results}
```
Please write a comprehensive analysis report in English, including the following:
	1.	Summary of Data Characteristics – Key patterns, trends, seasonality, etc.
	2.	Performance Comparison and Analysis of Each Model – Based on RMSE, MAE, R², MAPE, etc.
	3.	Recommendation of the Best Model and Explanation
	4.	Interpretation and Confidence Evaluation of Forecasted Values
	5.	Advantages and Disadvantages of Each Model Based on Data Characteristics
	6.	Suggestions to Further Improve Forecasting Performance

The output should be in clear and easy-to-understand Markdown format. Each section must have a heading, and use code blocks if needed to improve readability.
"""
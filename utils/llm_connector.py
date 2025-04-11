"""
Ollama LLM API 연동 모듈
시계열 분석 결과를 LLM에 전달하여 분석 결과를 마크다운으로 받아옵니다.
"""
import logging
import traceback
from typing import Dict, Any

import json
from openai import OpenAI

from utils.singleton import Singleton

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMConnector(metaclass=Singleton):
    """
    Ollama LLM API 연동 클래스
    싱글턴 패턴을 적용하여 메모리 효율성 확보
    """
    
    def __init__(
            self,
            base_url: str, 
            model: str
        ):
        """
        Ollama 연결 초기화
        
        Args:
            base_url: Ollama 서버 URL
            model: 사용할 LLM 모델명
        """
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama"  # required, but unused
        )
        
    def analyze_time_series(self, 
                           data_info: Dict[str, Any], 
                           model_results: Dict[str, Any],
                           prompt_template: str) -> str:
        """
        시계열 분석 결과를 LLM에 전달하여 마크다운 분석 결과를 받아옵니다.
        
        Args:
            data_info: 데이터 정보 딕셔너리
            model_results: 모델 학습 결과 딕셔너리
            prompt_template: 프롬프트 템플릿 문자열
            
        Returns:
            LLM이 생성한 마크다운 분석 결과
        """
        try:
            # 프롬프트 포맷팅
            formatted_prompt = prompt_template.format(
                data_info=json.dumps(data_info, ensure_ascii=False),
                model_results=json.dumps(model_results, ensure_ascii=False)
            )
            
            # LLM API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a time series analysis expert. Provide insights and analysis in markdown format."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.2  # 일관된 결과를 위해 낮은 온도 설정
            )
            
            # 응답 추출
            analysis_result = response.choices[0].message.content
            return analysis_result
            
        except Exception as e:
            logger.error(f"LLM API 호출 중 오류 발생: {e}\n{traceback.format_exc()}")
            return f"## 오류 발생\n\nLLM 분석 중 오류가 발생했습니다: {str(e)}"
    
    def analyze_time_series_stream(self, 
                                  data_info: Dict[str, Any], 
                                  model_results: Dict[str, Any],
                                  prompt_template: str) -> str:
        """
        시계열 분석 결과를 LLM에 전달하여 스트리밍 방식으로 마크다운 분석 결과를 받아옵니다.
        
        Args:
            data_info: 데이터 정보 딕셔너리
            model_results: 모델 학습 결과 딕셔너리
            prompt_template: 프롬프트 템플릿 문자열
            
        Returns:
            LLM이 생성한 마크다운 분석 결과
        """
        try:
            # 프롬프트 포맷팅
            formatted_prompt = prompt_template.format(
                data_info=json.dumps(data_info, ensure_ascii=False),
                model_results=json.dumps(model_results, ensure_ascii=False)
            )
            
            # LLM API 호출 (스트리밍)
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a time series analysis expert. Provide insights and analysis in markdown format."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.2,
                stream=True
            )
            
            # 전체 응답 수집
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    
            return full_response
            
        except Exception as e:
            logger.error(f"LLM 스트리밍 API 호출 중 오류 발생: {e}")
            return f"## 오류 발생\n\nLLM 분석 중 오류가 발생했습니다: {str(e)}"

    def recommend_hyperparameters(self, 
                            data_info: Dict[str, Any], 
                            model_info: Dict[str, Any],
                            performance_metrics: Dict[str, Any],
                            prompt_template: str) -> Dict[str, Any]:
        """
        시계열 모델의 하이퍼파라미터 추천을 LLM에 요청합니다.
        """
        try:
            # 프롬프트 포맷팅
            formatted_prompt = prompt_template.format(
                data_info=json.dumps(data_info, ensure_ascii=False),
                model_info=json.dumps(model_info, ensure_ascii=False),
                performance_metrics=json.dumps(performance_metrics, ensure_ascii=False)
            )
            
            # LLM API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a time series modeling expert. Provide hyperparameter recommendations in JSON format."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.2
            )
            
            # JSON 응답 파싱 및 반환
            recommendation_result = response.choices[0].message.content
            return self._parse_json_response(recommendation_result)
                
        except Exception as e:
            logger.error(f"하이퍼파라미터 추천 API 호출 중 오류 발생: {traceback.format_exc()}")
            return {"error": f"API 호출 오류: {str(e)}"}

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 추출
        """
        try:
            # JSON 문자열 추출 (```json과 ``` 사이의 내용)
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 일반 텍스트에서 찾기
                json_str = response_text
            
            # JSON 파싱
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.debug(f"파싱 시도한 문자열: {response_text}")
            return {"error": f"JSON 파싱 오류: {str(e)}"}
    
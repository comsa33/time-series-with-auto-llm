"""
서울시 대기질 데이터 수집 모듈
"""
import logging
from typing import Optional

import requests
import pandas as pd
from datetime import datetime, timedelta

from utils.singleton import Singleton
from config.settings import app_config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataReader(metaclass=Singleton):
    """
    데이터 수집을 담당하는 클래스
    싱글턴 패턴을 적용하여 메모리 효율성 확보
    """
    
    def __init__(self):
        """
        데이터 리더 초기화
        """
        self.api_key = app_config.SEOUL_API_KEY
        self.base_url = app_config.SEOUL_API_BASE_URL
        
    def get_seoul_air_quality(self, 
                             start_date: str, 
                             end_date: str, 
                             api_key: Optional[str] = None) -> pd.DataFrame:
        """
        서울시 대기질 정보 OpenAPI를 통해 데이터 수집
        
        Args:
            start_date: 데이터 수집 시작 날짜 (YYYY-MM-DD 형식)
            end_date: 데이터 수집 종료 날짜 (YYYY-MM-DD 형식)
            api_key: 서울시 OpenAPI 인증키 (None인 경우 기본값 사용)
            
        Returns:
            수집된 대기질 데이터 DataFrame
        """
        # API 키 설정
        if api_key is None:
            api_key = self.api_key
            
        service = app_config.SEOUL_AIR_QUALITY_SERVICE
        
        # 날짜 범위 설정
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 데이터 저장용 리스트
        all_data = []
        
        # 일별로 데이터 수집
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            
            # API 요청 (페이징 처리)
            row_count = 1
            page = 1
            while True:
                start_row = row_count
                end_row = row_count + 999  # 한번에 최대 1000개 조회 가능
                
                url = f"{self.base_url}/{api_key}/json/{service}/{start_row}/{end_row}/{date_str}"
                
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
                    data = response.json()
                    
                    # 결과가 있는지 확인
                    if service in data and 'row' in data[service]:
                        rows = data[service]['row']
                        if not rows:
                            break
                            
                        all_data.extend(rows)
                        
                        # 다음 페이지가 있는지 확인
                        if len(rows) < 1000:
                            break
                            
                        row_count += len(rows)
                    else:
                        # 결과가 없으면 종료
                        break
                except requests.RequestException as e:
                    logger.error(f"API 요청 중 오류 발생: {e}")
                    break
                except Exception as e:
                    logger.error(f"{date_str}, 페이지 {page} 처리 중 오류 발생: {e}")
                    break
                    
                page += 1
                
            # 다음 날짜로 이동
            current_dt += timedelta(days=1)
            logger.info(f"Collected data for {date_str}")
        
        # 데이터프레임으로 변환
        if all_data:
            df = pd.DataFrame(all_data)
            
            # 시간 컬럼 형식 확인 및 변환
            if not df.empty and 'MSRDT' in df.columns:
                # 디버깅을 위해 MSRDT 원본 값 샘플 출력
                logger.debug(f"MSRDT 컬럼 원본 샘플 값: {df['MSRDT'].head().tolist()}")
                
                # 문자열 전처리 후 변환
                # 끝에 '00'이 붙어 있는 경우 제거
                if df['MSRDT'].astype(str).str.endswith('00').any():
                    df['MSRDT'] = df['MSRDT'].astype(str).str.replace('00$', '', regex=True)
                    logger.debug(f"MSRDT 컬럼 전처리 후 샘플 값: {df['MSRDT'].head().tolist()}")
                
                # 날짜 형식으로 변환
                try:
                    df['MSRDT'] = pd.to_datetime(df['MSRDT'], format='%Y%m%d%H')
                except ValueError:
                    # 자동 변환 시도
                    logger.warning("MSRDT 날짜 형식 변환 실패, 자동 변환 시도")
                    df['MSRDT'] = pd.to_datetime(df['MSRDT'], errors='coerce')
                    
                    # 변환 실패한 값 확인
                    failed_conversion = df[df['MSRDT'].isna()]
                    if not failed_conversion.empty:
                        logger.warning(f"날짜 변환 실패한 값 개수: {len(failed_conversion)}")
                        logger.debug(f"변환 실패 샘플: {failed_conversion['MSRDT'].head().tolist()}")
            
            # 수치형 컬럼 변환
            numeric_columns = ['PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 결측치 처리
            present_columns = [col for col in numeric_columns if col in df.columns]
            if present_columns:
                na_counts_before = df[present_columns].isna().sum()
                logger.debug(f"결측치 처리 전 컬럼별 결측치 개수:\n{na_counts_before}")
                
                df = df.dropna(subset=present_columns)
                
                logger.info(f"최종 데이터 크기: {len(df)} 행 × {len(df.columns)} 열")
            
            return df
        else:
            logger.warning("수집된 데이터가 없습니다.")
            return pd.DataFrame()


# 모듈 사용 시 편의를 위한 래퍼 함수
def get_seoul_air_quality(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    서울시 대기질 정보 수집 래퍼 함수
    
    Args:
        api_key: 서울시 OpenAPI 인증키
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
        
    Returns:
        수집된 대기질 데이터 DataFrame
    """
    data_reader = DataReader()
    return data_reader.get_seoul_air_quality(start_date, end_date, api_key)


# 직접 실행 시 예제 코드
if __name__ == "__main__":
    API_KEY = app_config.SEOUL_API_KEY
    START_DATE = "2024-01-01"  # 시작 날짜
    END_DATE = "2024-01-31"    # 종료 날짜
    
    df = get_seoul_air_quality(API_KEY, START_DATE, END_DATE)
    
    if not df.empty:
        print(f"수집된 데이터 건수: {len(df)}")
        print(df.head())
        
        # CSV 파일로 저장
        df.to_csv(app_config.DEFAULT_DATA_FILE, index=False, encoding='utf-8-sig')
        print(f"데이터가 {app_config.DEFAULT_DATA_FILE} 파일로 저장되었습니다.")
    else:
        print("수집된 데이터가 없습니다.")

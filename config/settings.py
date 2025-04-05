"""
애플리케이션 전체에서 사용되는 설정 값을 관리하는 모듈
"""
import os
from dataclasses import dataclass

from dotenv import dotenv_values

# .env 파일에서 환경 변수 읽기
env_config = dotenv_values(".env")

@dataclass
class AppConfig:
    """
    애플리케이션 기본 설정
    """
    # 앱 기본 정보
    APP_TITLE: str = "서울시 대기질 시계열 분석"
    APP_DESCRIPTION: str = "서울시 IoT 도시데이터를 활용한 시계열 분석 앱"
    
    # 파일 경로 설정
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    DEFAULT_DATA_FILE: str = os.path.join(DATA_DIR, "seoul_air_quality_data.csv")
    
    # API 설정
    SEOUL_API_KEY: str = env_config["SEOUL_API_KEY"]
    SEOUL_API_BASE_URL: str = env_config["SEOUL_API_BASE_URL"]
    SEOUL_AIR_QUALITY_SERVICE: str = env_config["SEOUL_AIR_QUALITY_SERVICE"]
    
    # 시각화 설정
    PLOT_BACKGROUND_COLOR: str = "#ffffff"
    PLOT_GRID_COLOR: str = "#e0e0e0"
    DEFAULT_COLOR_PALETTE: str = "viridis"
    
    # 모델 학습 설정
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 32
    
    def __post_init__(self):
        """
        설정 초기화 후 필요한 디렉토리 생성
        """
        os.makedirs(self.DATA_DIR, exist_ok=True)


# 애플리케이션 전역 설정 객체
app_config = AppConfig()

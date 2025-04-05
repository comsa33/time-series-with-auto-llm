"""
LSTM 신경망 모델 구현 모듈
"""
import warnings
from typing import Dict, Any, Tuple, Optional, Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.base_model import TimeSeriesModel

# TensorFlow는 선택적으로 사용
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    
    # GPU 메모리 증가 방지
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            warnings.warn(f"GPU 메모리 설정 중 오류 발생: {e}")
    
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow 라이브러리를 사용할 수 없습니다. LSTM 모델을 사용하려면 설치가 필요합니다.")


class LSTMModel(TimeSeriesModel):
    """
    LSTM 신경망 모델 구현 클래스
    """
    
    def __init__(self, name: str = "LSTM"):
        """
        LSTM 모델 생성자
        
        Args:
            name: 모델 이름
        """
        super().__init__(name)
        self.model_params = {}
        self.history = None
        self.scaler = None
        self.n_steps = None
        self.n_features = 1
    
    def _create_sequences(self, data: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터를 입력 시퀀스와 타겟으로 변환합니다.
        
        Args:
            data: 시계열 데이터
            n_steps: 시퀀스 길이
            
        Returns:
            (입력 시퀀스, 타겟) 튜플
        """
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps])
        
        return np.array(X), np.array(y)
    
    def fit(self, 
           train_data: pd.Series, 
           n_steps: int = 24,
           lstm_units: Union[int, List[int]] = [50, 50],
           dropout_rate: float = 0.2,
           epochs: int = 100,
           batch_size: int = 32,
           validation_split: float = 0.1,
           early_stopping: bool = True,
           patience: int = 10,
           **kwargs) -> Any:
        """
        LSTM 모델을 학습합니다.
        
        Args:
            train_data: 학습 데이터
            n_steps: 시퀀스 길이
            lstm_units: LSTM 레이어의 유닛 수 (단일 값 또는 리스트)
            dropout_rate: 드롭아웃 비율
            epochs: 학습 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            early_stopping: 조기 종료 사용 여부
            patience: 조기 종료 인내심
            **kwargs: 추가 매개변수
            
        Returns:
            학습된 모델
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 라이브러리가 설치되지 않았습니다. 먼저 설치하세요.")
        
        self.train_data = train_data
        self.n_steps = n_steps
        
        try:
            # 데이터 스케일링
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1))
            
            # 시퀀스 데이터 생성
            X_train, y_train = self._create_sequences(train_scaled, n_steps)
            
            # LSTM 입력 형태로 변환
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], self.n_features)
            
            # LSTM 모델 구성
            self.model = Sequential()
            
            # LSTM 레이어 추가
            if isinstance(lstm_units, int):
                lstm_units = [lstm_units]
            
            for i, units in enumerate(lstm_units):
                return_sequences = i < len(lstm_units) - 1
                
                if i == 0:
                    # 첫 번째 레이어
                    self.model.add(LSTM(
                        units, 
                        return_sequences=return_sequences,
                        input_shape=(n_steps, self.n_features)
                    ))
                else:
                    # 나머지 레이어
                    self.model.add(LSTM(
                        units,
                        return_sequences=return_sequences
                    ))
                
                # 드롭아웃 추가
                if dropout_rate > 0:
                    self.model.add(Dropout(dropout_rate))
            
            # 출력 레이어
            self.model.add(Dense(1))
            
            # 모델 컴파일
            self.model.compile(optimizer='adam', loss='mse')
            
            # 콜백 설정
            callbacks = []
            if early_stopping:
                callbacks.append(EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                ))
            
            # 모델 학습
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # 모델 파라미터 저장
            self.model_params = {
                'n_steps': n_steps,
                'lstm_units': lstm_units,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split
            }
            
            self.is_fitted = True
            return self.model
            
        except Exception as e:
            warnings.warn(f"LSTM 모델 학습 중 오류 발생: {e}")
            self.is_fitted = False
            return None
    
    def predict(self, steps: int = 1, test_data: Optional[pd.Series] = None, **kwargs) -> np.ndarray:
        """
        미래 값을 예측합니다.
        
        Args:
            steps: 예측할 기간 수
            test_data: 테스트 데이터 (있는 경우)
            **kwargs: 추가 매개변수
            
        Returns:
            예측값 배열
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 라이브러리가 설치되지 않았습니다. 먼저 설치하세요.")
        
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        if test_data is None:
            # 예측 횟수만큼 반복 예측
            forecast = self._multi_step_forecast(steps)
        else:
            # 테스트 기간 동안 예측
            forecast = self._predict_test_period(test_data)
        
        return forecast
    
    def _multi_step_forecast(self, steps: int) -> np.ndarray:
        """
        다중 스텝 예측을 수행합니다.
        
        Args:
            steps: 예측할 기간 수
            
        Returns:
            예측값 배열
        """
        # 마지막 n_steps 데이터 가져오기
        last_sequence = self.scaler.transform(
            self.train_data.values[-self.n_steps:].reshape(-1, 1)
        ).reshape(1, self.n_steps, self.n_features)
        
        # 예측값 저장 배열
        forecast = np.zeros(steps)
        
        # 순차적으로 예측
        current_sequence = last_sequence.copy()
        
        for i in range(steps):
            # 현재 시퀀스로 다음 값 예측
            next_pred = self.model.predict(current_sequence, verbose=0)[0][0]
            forecast[i] = next_pred
            
            # 시퀀스 업데이트
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        # 예측값 역스케일링
        forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        
        return forecast
    
    def _predict_test_period(self, test_data: pd.Series) -> np.ndarray:
        """
        테스트 기간에 대한 예측을 수행합니다.
        
        Args:
            test_data: 테스트 데이터
            
        Returns:
            예측값 배열
        """
        # 전체 시계열 데이터 (훈련 + 테스트)
        full_data = pd.concat([self.train_data, test_data])
        
        # 예측값 저장 배열
        forecast = np.zeros(len(test_data))
        
        # 테스트 데이터 각 시점에 대해 예측
        for i in range(len(test_data)):
            # 현재 시점까지의 데이터
            current_data = full_data.iloc[:(len(self.train_data) + i)]
            
            # 마지막 n_steps 데이터 가져오기
            input_sequence = self.scaler.transform(
                current_data.values[-self.n_steps:].reshape(-1, 1)
            ).reshape(1, self.n_steps, self.n_features)
            
            # 다음 값 예측
            next_pred = self.model.predict(input_sequence, verbose=0)[0][0]
            
            # 예측값 저장
            forecast[i] = next_pred
        
        # 예측값 역스케일링
        forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        
        return forecast
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        학습 이력을 반환합니다.
        
        Returns:
            학습 이력 딕셔너리
        """
        if self.history is None:
            return {}
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', [])
        }
    
    def get_params(self) -> Dict[str, Any]:
        """
        모델 파라미터를 반환합니다.
        
        Returns:
            모델 파라미터 딕셔너리
        """
        return self.model_params

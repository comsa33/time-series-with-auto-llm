"""
트랜스포머 모델 구현 모듈
"""
import warnings
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.base_model import TimeSeriesModel

# TensorFlow는 선택적으로 사용
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
    from tensorflow.keras.optimizers import Adam
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
    warnings.warn("TensorFlow 라이브러리를 사용할 수 없습니다. 트랜스포머 모델을 사용하려면 설치가 필요합니다.")


class TransformerEncoder(tf.keras.layers.Layer):
    """
    트랜스포머 인코더 레이어 구현
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        트랜스포머 인코더 초기화
        
        Args:
            embed_dim: 임베딩 차원
            num_heads: 멀티헤드 어텐션의 헤드 수
            ff_dim: 피드포워드 네트워크의 차원
            dropout: 드롭아웃 비율
        """
        super(TransformerEncoder, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        """
        순전파 연산 수행
        
        Args:
            inputs: 입력 텐서
            training: 학습 모드 여부
            
        Returns:
            변환된 출력 텐서
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(TimeSeriesModel):
    """
    트랜스포머 기반 시계열 예측 모델 구현 클래스
    """
    
    def __init__(self, name: str = "Transformer"):
        """
        트랜스포머 모델 생성자
        
        Args:
            name: 모델 이름
        """
        super().__init__(name)
        self.model_params = {}
        self.history = None
        self.scaler = None
        self.window_size = None
        
        if not TF_AVAILABLE:
            warnings.warn("TensorFlow 라이브러리가 설치되지 않았습니다. 모델을 사용하기 전에 설치하세요.")
    
    def create_sliding_window_data(self, data: np.ndarray, window_size: int, output_size: int = 1):
        """
        슬라이딩 윈도우 방식으로 데이터 생성
        
        Args:
            data: 원본 시계열 데이터
            window_size: 입력 윈도우 크기
            output_size: 출력 윈도우 크기
            
        Returns:
            입력 시퀀스와 타겟값 튜플
        """
        X, y = [], []
        for i in range(len(data) - window_size - output_size + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + output_size])
        return np.array(X), np.array(y)
    
    def build_transformer_model(self, 
                               input_shape, 
                               output_size, 
                               embed_dim, 
                               num_heads, 
                               ff_dim, 
                               num_layers, 
                               dropout):
        """
        트랜스포머 모델 구축
        
        Args:
            input_shape: 입력 데이터 형태
            output_size: 출력 차원
            embed_dim: 임베딩 차원
            num_heads: 멀티헤드 어텐션의 헤드 수
            ff_dim: 피드포워드 네트워크의 차원
            num_layers: 트랜스포머 인코더 레이어 수
            dropout: 드롭아웃 비율
            
        Returns:
            구축된 모델
        """
        inputs = Input(shape=input_shape)
        x = Dense(embed_dim)(inputs)

        for _ in range(num_layers):
            x = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout)(x)

        # 마지막 타임스텝의 출력만 사용
        x = Dense(output_size)(x[:, -1, :])
        model = Model(inputs, x)
        return model
    
    def fit(self, 
           train_data: pd.Series, 
           window_size: int = 24,
           embed_dim: int = 64,
           num_heads: int = 4,
           ff_dim: int = 128,
           num_layers: int = 2,
           dropout_rate: float = 0.1,
           learning_rate: float = 0.001,
           epochs: int = 50,
           batch_size: int = 32,
           validation_split: float = 0.1,
           early_stopping: bool = True,
           patience: int = 10,
           **kwargs) -> Any:
        """
        트랜스포머 모델을 학습합니다.
        
        Args:
            train_data: 학습 데이터
            window_size: 시퀀스 길이
            embed_dim: 임베딩 차원
            num_heads: 멀티헤드 어텐션의 헤드 수
            ff_dim: 피드포워드 네트워크의 차원
            num_layers: 트랜스포머 인코더 레이어 수
            dropout_rate: 드롭아웃 비율
            learning_rate: 학습률
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
        self.window_size = window_size
        
        try:
            # 데이터 스케일링
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = self.scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
            
            # 시퀀스 데이터 생성
            X_train, y_train = self.create_sliding_window_data(train_scaled, window_size)
            
            # 모델 입력 형태 변환
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            # 트랜스포머 모델 구성
            self.model = self.build_transformer_model(
                input_shape=(window_size, 1),
                output_size=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
                dropout=dropout_rate
            )
            
            # 모델 컴파일
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate), 
                loss="mse"
            )
            
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
                'window_size': window_size,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'ff_dim': ff_dim,
                'num_layers': num_layers,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'batch_size': batch_size
            }
            
            self.is_fitted = True
            return self.model
            
        except Exception as e:
            warnings.warn(f"트랜스포머 모델 학습 중 오류 발생: {e}")
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
        # 마지막 window_size 데이터 가져오기
        last_sequence = self.scaler.transform(
            self.train_data.values[-self.window_size:].reshape(-1, 1)
        ).reshape(1, self.window_size, 1)
        
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
            
            # 마지막 window_size 데이터 가져오기
            input_sequence = self.scaler.transform(
                current_data.values[-self.window_size:].reshape(-1, 1)
            ).reshape(1, self.window_size, 1)
            
            # 다음 값 예측
            next_pred = self.model.predict(input_sequence, verbose=0)[0][0]
            
            # 예측값 역스케일링 후 저장
            forecast[i] = self.scaler.inverse_transform(np.array([[next_pred]]))[0, 0]
        
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

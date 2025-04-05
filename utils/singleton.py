"""
싱글턴 패턴 구현 모듈
메모리 효율성을 위해 여러 곳에서 재사용되는 객체를 위한 싱글턴 패턴을 제공합니다.
"""
from typing import Dict, Any, Type, TypeVar

T = TypeVar('T')


class Singleton(type):
    """
    싱글턴 메타클래스
    이 메타클래스를 상속받은 클래스는 자동으로 싱글턴으로 동작합니다.
    """
    _instances: Dict[Any, Any] = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    @classmethod
    def clear_instance(cls, target_cls: Type[T]) -> None:
        """
        지정된 클래스의 싱글턴 인스턴스를 제거합니다.
        주로 테스트나 리소스 해제에 사용됩니다.
        
        Args:
            target_cls: 제거할 싱글턴 인스턴스의 클래스
        """
        if target_cls in cls._instances:
            del cls._instances[target_cls]

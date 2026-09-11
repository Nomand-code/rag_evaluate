from pydantic import BaseModel, Field, field_validator
from typing import Literal


# ==========================================
# REQUEST SCHEMAS (Входные данные)
# ==========================================

from pydantic import BaseModel, Field, field_validator, ValidationInfo

class EvaluationRequest(BaseModel):
    """Схема для запроса на оценку качества RAG-ответа"""
    question: str = Field(..., min_length=3, description="Вопрос пользователя")
    context: str = Field(..., min_length=10, description="Контекст, полученный из Vector DB")
    answer: str = Field(..., min_length=3, description="Ответ, сгенерированный LLM")

    @field_validator('context', 'answer', 'question')
    @classmethod
    def context_cannot_be_placeholder(cls, v: str, info: ValidationInfo) -> str:
        """Защита от мусорных данных, которые часто прилетают из RAG"""
        bad_phrases = [
            "no context found", 
            "нет информации", 
            "контекст не найден",
            "null", 
            "none",
            "не удалось найти"
        ]
        
        if v.strip().lower() in bad_phrases:
            # Динамически подставляем имя поля в текст ошибки
            field_name = info.field_name
            raise ValueError(f"Поле '{field_name}' не может быть плейсхолдером отсутствия данных")
            
        return v



# ==========================================
# RESPONSE SCHEMAS (Выходные данные)
# ==========================================

class MetricScore(BaseModel):
    """Базовая модель для одной метрики оценки"""
    score: float = Field(..., ge=0.0, le=1.0, description="Оценка от 0.0 до 1.0")
    reasoning: str = Field(..., min_length=10, description="Обоснование оценки от LLM-Judge")


class EvaluationResponse(BaseModel):
    """Схема для ответа с результатами оценки"""
    faithfulness: MetricScore = Field(..., description="Насколько ответ соответствует контексту (нет галлюцинаций)")
    answer_relevance: MetricScore = Field(..., description="Насколько ответ релевантен исходному вопросу")
    
    # Вычисляемое поле: общий статус на основе пороговых значений
    status: Literal["pass", "fail"] = Field(..., description="Общий статус оценки")
    
    @field_validator('status', mode='after')
    @classmethod
    def determine_status(cls, v: str, info) -> str:
        """Автоматически определяем pass/fail на основе скоров"""
        # Если оба скора выше 0.7, считаем оценку успешной
        if info.data.get('faithfulness').score >= 0.7 and info.data.get('answer_relevance').score >= 0.7:
            return "pass"
        return "fail"
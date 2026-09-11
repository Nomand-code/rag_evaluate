import asyncio
from fastapi import HTTPException, status
from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.schemas.evaluation import EvaluationRequest, EvaluationResponse
from app.core.config import settings

class JudgeService:
    def __init__(self):
        # Инициализируем клиент
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = "gemini-2.5-flash"  
        
        self._system_prompt = """Ты — строгий и объективный LLM-Judge, оценивающий качество ответов RAG-системы.
        Твоя задача — оценить два аспекта:
        1. Faithfulness (Достоверность): Насколько ответ строго основан на предоставленном контексте? Есть ли галлюцинации или выдуманные факты?
        2. Answer Relevance (Релевантность): Насколько полно и точно ответ решает исходный вопрос пользователя?

        ПРАВИЛА ОЦЕНКИ:
        - Сначала напиши подробное обоснование (reasoning) для каждой метрики.
        - Только после рассуждений выставь оценку (score) от 0.0 до 1.0.
        - Оценка 1.0 означает идеальное соответствие. Оценка 0.0 означает полное несоответствие или галлюцинацию.
        - Не будь слишком мягким. Если в ответе есть информация, которой нет в контексте, снижай Faithfulness."""

    def _build_prompt(self, request: EvaluationRequest) -> str:
        return f"""<question>
            {request.question}
            </question>

            <context>
            {request.context}
            </context>

            <answer_to_evaluate>
            {request.answer}
            </answer_to_evaluate>

            Оцени этот ответ, строго следуя правилам."""

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        prompt = self._build_prompt(request)
        
        try:
            # ИСПОЛЬЗУЕМ .aio.models ДЛЯ АСИНХРОННОГО ВЫЗОВА
            # asyncio.wait_for отлично работает с асинхронными методами
            response = await asyncio.wait_for( 
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        # 1. ОБЯЗАТЕЛЬНО ПЕРЕДАЕМ СИСТЕМНЫЙ ПРОМПТ
                        system_instruction=self._system_prompt,
                        
                        response_mime_type="application/json",
                        # 2. ПЕРЕДАЕМ САМУ PYDANTIC МОДЕЛЬ (новый SDK это поддерживает из коробки)
                        response_schema=EvaluationResponse, 
                    ),
                ), 
                timeout=30
            )
            
            # response.parsed уже будет объектом EvaluationResponse (валидация под капотом SDK)
            return response.parsed

        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gemini API Request Timeout"
            )
        except APIError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini API Error: {str(e)}"
            )

# Singleton
judge_service = JudgeService()
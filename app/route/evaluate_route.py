from fastapi import APIRouter, HTTPException
from app.schemas.evaluation import EvaluationRequest, EvaluationResponse
from app.services.judge import judge_service

# Создаем роутер. prefix добавится ко всем путям, tags — для красивой Swagger-документации
router = APIRouter(prefix="/evaluator", tags=["RAG Evaluation"])


@router.get("/", summary="Информация о сервисе")
async def info():
    return {"message": "RAG Evaluator Service is running", "status": "healthy"}


@router.post(
    "/evaluate", 
    response_model=EvaluationResponse,
    summary="Оценить качество RAG-ответа",
    description="Принимает вопрос, контекст и ответ LLM. Возвращает оценки Faithfulness, Answer Relevance и общий статус."
)
async def evaluate_rag_response(evaluation_request: EvaluationRequest):
    
    try:
        # Вызываем сервис и ждем результат (асинхронно, не блокируя event loop)
        result = await judge_service.evaluate(request=evaluation_request)
        return result
    except Exception as e:
        # Ловим любые ошибки LLM и возвращаем понятный 500 статус
        raise HTTPException(status_code=500, detail=f"Ошибка при оценке LLM: {str(e)}")

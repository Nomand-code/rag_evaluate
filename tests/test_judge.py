"""
Тесты для сервиса оценки RAG.
Запуск: pytest tests/test_judge.py -v
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.schemas.evaluation import EvaluationRequest, EvaluationResponse
from app.services.judge import JudgeService
from app.main import app


# ===========================================================================
# ГРУППА 1: Тесты Pydantic-схем (валидация входных данных)
# ===========================================================================
# Здесь мы НЕ мокаем ничего. Мы просто проверяем, что Pydantic правильно
# отклоняет плохие данные ещё ДО того, как они попадут в бизнес-логику.
# ===========================================================================

class TestEvaluationRequest:
    """Тесты для валидации входного запроса"""

    def test_valid_request(self):
        """Нормальные данные должны проходить валидацию без ошибок"""
        request = EvaluationRequest(
            question="Кто основал Москву?",
            context="Москва была основана в 1147 году Юрием Долгоруким",
            answer="Москву основал Юрий Долгорукий в 1147 году"
        )
        # Если дошли сюда без исключения — тест пройден
        assert request.question == "Кто основал Москву?"
        assert "1147" in request.context

    def test_empty_context_fails(self):
        """Пустой контекст должен отклоняться (min_length=10)"""
        # pytest.raises — это блок, который ЖДЁТ исключение.
        # Если исключение НЕ возникнет — тест упадёт.
        with pytest.raises(ValidationError):
            EvaluationRequest(
                question="Вопрос нормальный",
                context="",  # слишком короткий
                answer="Ответ нормальный"
            )

    def test_placeholder_context_fails(self):
        """Мусорные плейсхолдеры из RAG должны отклоняться твоим валидатором"""
        with pytest.raises(ValidationError) as error_info:
            EvaluationRequest(
                question="Вопрос",
                context="no context found",  # мусор из RAG
                answer="Ответ"
            )
        # Проверяем что ошибка действительно про плейсхолдер
        assert "плейсхолдером" in str(error_info.value)

    def test_placeholder_answer_fails(self):
        """Валидатор должен ловить мусор и в поле answer тоже"""
        with pytest.raises(ValidationError):
            EvaluationRequest(
                question="Вопрос нормальный",
                context="Контекст нормальный длинный",
                answer="none"  # мусор
            )


# ===========================================================================
# ГРУППА 2: Тесты JudgeService (бизнес-логика + моки)
# ===========================================================================
# Здесь мы тестируем логику оценки. Мы НЕ хотим реально дергать Google API:
#   - Это стоит денег (пусть даже flash бесплатный — это медленно)
#   - Тесты должны быть БЫСТРЫМИ и стабильными
# Поэтому мы подменяем (мокаем) Google-клиент на заглушку.
# ===========================================================================

class TestJudgeService:
    """Тесты сервиса с моками Google API"""

    @pytest.mark.asyncio  # Помечаем тест как асинхронный (нужен pytest-asyncio)
    @patch('app.services.judge.genai.Client')  # Подменяем genai.Client на мок
    async def test_evaluate_success(self, mock_client_class):
        """Сценарий: Google API отвечает корректно -> получаем валидный EvaluationResponse"""
        
        # === ШАГ 1: Настраиваем заглушку Google API ===
        fake_response = Mock()
        fake_response.parsed = {
            "faithfulness": {
                "score": 0.9,
                "reasoning": "Ответ строго основан на контексте, галлюцинаций нет"
            },
            "answer_relevance": {
                "score": 0.85,
                "reasoning": "Ответ полностью отвечает на вопрос пользователя"
            },
            "status": "pass"
        }
        
        # Создаём мок-инстанс клиента с методом generate_content
        mock_client_instance = Mock()
        # AsyncMock нужен, потому что в judge.py используется await
        mock_client_instance.models.generate_content = AsyncMock(return_value=fake_response)
        
        # Когда код вызовет genai.Client(...), он получит наш мок
        mock_client_class.return_value = mock_client_instance
        
        # === ШАГ 2: Создаём сервис (теперь он использует мок вместо реального API) ===
        service = JudgeService()
        
        # === ШАГ 3: Вызываем evaluate ===
        request = EvaluationRequest(
            question="Кто основал Москву?",
            context="Москва была основана в 1147 году Юрием Долгоруким",
            answer="Москву основал Юрий Долгорукий"
        )
        
        result = await service.evaluate(request)
        
        # === ШАГ 4: Проверяем результат ===
        assert isinstance(result, EvaluationResponse)
        assert result.faithfulness.score == 0.9
        assert result.answer_relevance.score == 0.85
        # Статус пересчитан твоим валидатором (0.9 и 0.85 оба >= 0.7)
        assert result.status == "pass"
        assert "галлюцинаций нет" in result.faithfulness.reasoning

    @pytest.mark.asyncio
    @patch('app.services.judge.genai.Client')
    async def test_evaluate_returns_fail_status(self, mock_client_class):
        """Сценарий: низкие скори -> статус автоматически становится 'fail'"""
        fake_response = Mock()
        fake_response.parsed = {
            "faithfulness": {
                "score": 0.3,  # Низкий скор
                "reasoning": "Много галлюцинаций в ответе"
            },
            "answer_relevance": {
                "score": 0.8,
                "reasoning": "Ответ частично релевантен"
            },
            "status": "pass"  # LLM прислала 'pass', но наш валидатор перезапишет!
        }
        
        mock_client_instance = Mock()
        mock_client_instance.models.generate_content = AsyncMock(return_value=fake_response)
        mock_client_class.return_value = mock_client_instance
        
        service = JudgeService()
        request = EvaluationRequest(
            question="Вопрос",
            context="Контекст достаточно длинный для теста",
            answer="Ответ на проверку"
        )
        
        result = await service.evaluate(request)
        
        # Ключевая проверка: наш @field_validator перезаписал 'pass' на 'fail'
        # потому что faithfulness (0.3) < 0.7
        assert result.status == "fail"

    @pytest.mark.asyncio
    @patch('app.services.judge.genai.Client')
    async def test_evaluate_timeout(self, mock_client_class):
        """Сценарий: Google API не отвечает за 30 сек -> HTTPException 504"""
        from fastapi import HTTPException
        
        mock_client_instance = Mock()
        # Мок возвращает TimeoutError как будто API завис
        mock_client_instance.models.generate_content = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        mock_client_class.return_value = mock_client_instance
        
        service = JudgeService()
        request = EvaluationRequest(
            question="Вопрос",
            context="Контекст достаточно длинный для теста",
            answer="Ответ"
        )
        
        # Проверяем что сервис выбросит HTTPException с кодом 504
        with pytest.raises(HTTPException) as exc_info:
            await service.evaluate(request)
        
        assert exc_info.value.status_code == 504


# ===========================================================================
# ГРУППА 3: Интеграционный тест через FastAPI TestClient
# ===========================================================================
# TestClient — это инструмент FastAPI, который запускает приложение
# в памяти и позволяет делать HTTP-запросы без реального сервера.
# ===========================================================================

class TestRouter:
    """Тесты роутера (эндпоинтов)"""

    @patch('app.services.judge.genai.Client')
    def test_evaluate_endpoint(self, mock_client_class):
        """POST /evaluator/evaluate должен возвращать JSON с оценками"""
        
        # Настраиваем мок
        fake_response = Mock()
        fake_response.parsed = {
            "faithfulness": {"score": 0.9, "reasoning": "Всё отлично"},
            "answer_relevance": {"score": 0.9, "reasoning": "Отличный ответ"},
            "status": "pass"
        }
        mock_instance = Mock()
        mock_instance.models.generate_content = AsyncMock(return_value=fake_response)
        mock_client_class.return_value = mock_instance
        
        # Создаём тестовый клиент (не нужно запускать uvicorn!)
        client = TestClient(app)
        
        # Делаем HTTP POST запрос
        response = client.post(
            "/evaluator/evaluate",
            json={
                "question": "Кто основал Москву?",
                "context": "Москва была основана в 1147 году Юрием Долгоруким",
                "answer": "Юрий Долгорукий"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "faithfulness" in data
        assert "status" in data
        assert data["status"] in ["pass", "fail"]

    def test_evaluate_endpoint_validation_error(self):
        """Если отправить плохие данные, FastAPI вернёт 422 Unprocessable Entity"""
        client = TestClient(app)
        
        response = client.post(
            "/evaluator/evaluate",
            json={
                "question": "Вопрос",
                "context": "",  # Пустой -> Pydantic отклонит
                "answer": "Ответ"
            }
        )
        
        # 422 — это стандартный код FastAPI для ошибок валидации
        assert response.status_code == 422

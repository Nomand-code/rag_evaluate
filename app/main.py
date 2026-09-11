# main.py
from fastapi import FastAPI
import uvicorn
from app.route.evaluate_route import router as evaluate_router


app = FastAPI(title="FastAPI RAG Evaluator")

# Подключаем роутер оценки
app.include_router(evaluate_router)


@app.get("/", summary="Health check")
async def root():
    return {"message": "Good", "status": "running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8976, reload=True)

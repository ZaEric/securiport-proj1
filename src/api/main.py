from fastapi import FastAPI

from src.api.routes.sources import router as sources_router
from src.api.routes.processing import router as processing_router
from src.api.routes.nlp import router as nlp_router
from src.api.routes.report import router as report_router


app = FastAPI(
    title="Securiport Sentiment Analysis API",
    version="0.1.0",
    description="API for running and debugging individual stages of the pipeline.",
)

app.include_router(sources_router)
app.include_router(processing_router)
app.include_router(nlp_router)
app.include_router(report_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Securiport Sentiment Analysis API is running.",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.predict import load_artifacts
from api.routes import router, set_artifacts
from src.utils.logger import get_logger

logger = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement des artefacts au démarrage
    logger.info("Chargement des artefacts ML...")
    arts = load_artifacts()
    set_artifacts(arts)
    logger.info("✅ Modèle chargé — API prête")
    yield
    logger.info("Arrêt de l'API")


app = FastAPI(
    title="Readmission Hospitalière — API ML",
    description=(
        "Prédiction du risque de réadmission hospitalière dans les 30 jours "
        "pour patients diabétiques. Modèle : LightGBM (NaN natif, AUC-PR optimisé)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
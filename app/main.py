
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import camera, alerts, dashboard
from app.config import BASE_DIR

app = FastAPI(title="VIGILANT - AI Surveillance System")

app.mount("/static", StaticFiles(directory=f"{BASE_DIR}/app/static"), name="static")

app.include_router(camera.router, prefix="/camera", tags=["camera"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(dashboard.router, tags=["dashboard"])


from fastapi import APIRouter
from app.database.database import JSONDatabase

router = APIRouter()
db = JSONDatabase()

@router.get("/")
async def get_alerts():
    return {"alerts": db.get_all_alerts()}

@router.get("/recent")
async def get_recent_alerts(limit: int = 10):
    return {"alerts": db.get_recent_alerts(limit)}

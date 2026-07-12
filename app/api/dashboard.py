
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app.database.database import JSONDatabase
from app.config import BASE_DIR

router = APIRouter()
templates = Jinja2Templates(directory=f"{BASE_DIR}/app/templates")
db = JSONDatabase()

@router.get("/")
async def dashboard(request: Request):
    recent_alerts = db.get_recent_alerts(10)
    total_alerts = len(db.get_all_alerts())
    high_severity = len([a for a in db.get_all_alerts() if a.get("severity") == "HIGH"])
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "recent_alerts": recent_alerts,
        "total_alerts": total_alerts,
        "high_severity": high_severity
    })

@router.get("/alerts")
async def alerts_page(request: Request):
    all_alerts = db.get_all_alerts()[::-1]
    return templates.TemplateResponse("alerts.html", {
        "request": request,
        "alerts": all_alerts
    })

"""Main API router that aggregates all sub-routers."""

from fastapi import APIRouter
from web_dashboard.api.auth import router as auth_router

from web_dashboard.api.control import router as control_router
from web_dashboard.api.dashboard import router as dashboard_router
from web_dashboard.api.models import router as models_router
from web_dashboard.api.secure_settings_api import router as secure_settings_router
from web_dashboard.api.settings_api import router as settings_router
from web_dashboard.api.symbols import router as symbols_router
from web_dashboard.api.system_health import router as system_health_router
from web_dashboard.api.trades import router as trades_router

api_router = APIRouter()

api_router.include_router(auth_router, tags=["auth"])
api_router.include_router(dashboard_router, tags=["dashboard"])
api_router.include_router(models_router, tags=["models"])
api_router.include_router(trades_router, tags=["trades"])
api_router.include_router(control_router, tags=["control"])
api_router.include_router(symbols_router, tags=["symbols"])
api_router.include_router(secure_settings_router, tags=["secure-settings"])
api_router.include_router(settings_router, tags=["settings"])
api_router.include_router(system_health_router, tags=["system-health"])

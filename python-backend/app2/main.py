# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from app2.eda import do_bivariate_eda
from app2.session_manager import session



app = FastAPI(title="Automated ML Model Builder - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.post("/eda/bivariate")
def run_bivariate_eda():
    """
    Run bivariate EDA and save generated plots to /static/plots folder.
    """
    result = do_bivariate_eda(session)
    return result
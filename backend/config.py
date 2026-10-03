import os

class Settings:
    PROJECT_NAME: str = "Stock Market Direction Prediction System"
    VERSION: str = "1.0.0"
    DEFAULT_TICKER: str = "TCS"
    DEFAULT_MODEL: str = "xgboost"
    
    # Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_RAW_DIR: str = os.path.join(BASE_DIR, "data", "raw")
    DATA_PROCESSED_DIR: str = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]

settings = Settings()

import uvicorn

from frontend.app import app as shiny_app
from src.routes import app as fastapi_app

fastapi_app.mount("/", shiny_app)

if __name__ == "__main__":
    print("🚀 Starting KTC System...")
    print("📝 API Docs: http://127.0.0.1:8000/docs")
    print("📊 AI Assistant Dashboard: http://127.0.0.1:8000")

    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

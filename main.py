import uvicorn

from frontend.app import app as shiny_app
from src.routes import app as fastapi_app
from src.tools.document_tools import get_vector_db

fastapi_app.mount("/", shiny_app)

if __name__ == "__main__":
    print("🚀 Starting KTC System...")

    print("⏳ Loading Chroma Vector Database into memory... please wait.")
    try:
        get_vector_db()
        print("✅ Vector Database loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load Vector DB at startup. It will load on first use. Error: {e}")

    print("📝 API Docs: http://127.0.0.1:8000/docs")
    print("📊 AI Assistant Dashboard: http://127.0.0.1:8000")

    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

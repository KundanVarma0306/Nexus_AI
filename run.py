import uvicorn
import os
from config.settings import settings

if __name__ == "__main__":
    # Ensure directories exist
    settings.ensure_directories()
    
    print("🚀 Starting AI Research Assistant...")
    print(f"Backend & Frontend available at: http://localhost:{settings.api_port}")
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

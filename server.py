import uvicorn
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("Job Fraud Detection FastAPI Server")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    print("\nStarting FastAPI server...")
    print("Server will be available at: http://localhost:5000")
    print("API endpoint: http://localhost:5000/predict")
    print("Health check: http://localhost:5000/health")
    print("Documentation: http://localhost:5000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        from app import app
        
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=5000,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
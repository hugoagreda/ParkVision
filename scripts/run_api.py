from backend.core.config import settings

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)

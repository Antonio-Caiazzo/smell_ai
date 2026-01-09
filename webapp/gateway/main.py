import os

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Service URLs (Configurable via Env Vars)
# Default to localhost for local development/testing
AI_ANALYSIS_SERVICE = os.getenv(
    "AI_ANALYSIS_SERVICE_URL", os.getenv("AI_ANALYSIS_SERVICE", "http://localhost:8001")
)
STATIC_ANALYSIS_SERVICE = os.getenv(
    "STATIC_ANALYSIS_SERVICE_URL",
    os.getenv("STATIC_ANALYSIS_SERVICE", "http://localhost:8002"),
)
REPORT_SERVICE = os.getenv(
    "REPORT_SERVICE_URL", os.getenv("REPORT_SERVICE", "http://localhost:8003")
)


@app.get("/")
def read_root():
    return {"message": "Welcome to CodeSmile API Gateway"}


# Proxy requests to AI Analysis Service
@app.post("/api/detect_smell_ai")
async def detect_smell_ai(request: dict):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AI_ANALYSIS_SERVICE}/detect_smell_ai",
                json=request,
                timeout=500.0,  # Set a timeout (in seconds)
            )
        return response.json()
    except httpx.RequestError as exc:
        return {
            "success": False,
            "error": f"Request to AI Analysis Service failed: {str(exc)}",
        }
    except httpx.TimeoutException:
        return {"success": False, "error": "Request to AI Analysis Service timed out"}


# Proxy requests to Static Analysis Service
# Proxy requests to Static Analysis Service
@app.post("/api/detect_smell_static")
async def detect_smell_static(request: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{STATIC_ANALYSIS_SERVICE}/detect_smell_static", json=request
        )
    return response.json()


@app.post("/api/detect_call_graph")
async def detect_call_graph(
    request: Request,
    file: UploadFile = File(None),
    code_snippet: str = Form(None),
    include_call_graph: bool = Form(True),
):
    async with httpx.AsyncClient() as client:
        if file:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(
                f"{STATIC_ANALYSIS_SERVICE}/detect_call_graph",
                files=files,
                data={"include_call_graph": str(include_call_graph)},
            )
        else:
            response = await client.post(
                f"{STATIC_ANALYSIS_SERVICE}/detect_call_graph",
                data={
                    "code_snippet": code_snippet,
                    "include_call_graph": str(include_call_graph),
                },
            )
    return response.json()


# Proxy requests to Report Service
@app.post("/api/generate_report")
async def generate_report(request: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{REPORT_SERVICE}/generate_report", json=request)
    return response.json()

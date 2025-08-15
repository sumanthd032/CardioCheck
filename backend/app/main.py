from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

#Create a FastAPI app instance
app = FastAPI(title="CardioCheck API")

# Mount the 'frontend' directory to serve static files like index.html
# This allows the FastAPI server to also deliver our webpage.
app.mount("/static", StaticFiles(directory="../../frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("../../frontend/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
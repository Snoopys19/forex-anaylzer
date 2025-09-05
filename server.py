# This is your original server.py from the backend.
# Replace this file with your actual server.py from forex-backend-export zip.
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True, "message": "Replace this server.py with your original backend code."}

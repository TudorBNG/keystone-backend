from fastapi import FastAPI
from mangum import Mangum
from fastapi.middleware.cors import CORSMiddleware

from router.api import router

app = FastAPI(root_path="/")

@app.get("/")
def read_root():
    return {"Welcome": "Welcome to the FastAPI on Lambda"}

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

handler = Mangum(app)


# if __name__ == "__main__":
#     uvicorn.run(app, port=8000)
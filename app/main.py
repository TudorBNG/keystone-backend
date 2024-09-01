from fastapi import FastAPI
from mangum import Mangum
from starlette.middleware.cors import CORSMiddleware

from router.api import router

app = FastAPI(root_path="/")

@app.get("/")
def read_root():
    return {"Welcome": "Welcome to the FastAPI on Lambda"}

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = Mangum(app=app)


# if __name__ == "__main__":
#     uvicorn.run(app, port=8000)
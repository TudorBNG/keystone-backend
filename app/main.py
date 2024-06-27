from fastapi import FastAPI
from mangum import Mangum

from router.api import router

app = FastAPI(root_path="/")


app.include_router(router, prefix="/api")

handler = Mangum(app)


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
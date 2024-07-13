Install dependencies

`cd keystone-backend`

`python -m venv kstn_venv`

`source kstn_venv/bin/activate`

`pip install -r requirements.txt`

Run server

`cd app && uvicorn main:app --reload`

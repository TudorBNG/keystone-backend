from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, FileResponse
import fitz
import boto3
from botocore.client import Config
import json
import numpy as np
from typing import Dict
from dotenv import load_dotenv
import os

from services.helper import find_sections_in_page, sections_parser

from services.spec_handler import extract_keys_from_spec

load_dotenv()

router = APIRouter(prefix='/api')

# set the name of your bucket
BUCKET = os.environ.get('BUCKET') or "keystone-bucket"

s3 = boto3.client('s3')

@router.get("/")
async def index_root():
    return {"message": "Hello!"}

@router.post("/presigned-url")
async def create_presigned_url(user: str, filename: str, method: str) -> Dict[str, str]:
    path = f"{user}/specs/{filename}"

    config = {
        "Bucket": BUCKET,
        "Key": path
    }

    if method == "put_object":
        config["ContentType"] = "application/pdf"
        config["ACL"] = "public-read"

    presigned_url = s3.generate_presigned_url(method, Params=config, ExpiresIn=60000)

    return {"url": presigned_url, "key": path}


### Specs

@router.get("/specs")
async def get_user_library(user: str):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{user}/specs/")

    return [obj['Key'].split("/specs/")[-1] for obj in response.get('Contents', [])]


### Keys 

@router.get("/keys")
async def get_keys(user: str, filename: str):
    try:
        file_data = s3.get_object(Bucket=BUCKET, Key=f"{user}/keys/{filename}.json")
        
        json_data = json.loads(file_data["Body"].read())
        
        return json_data

    except s3.exceptions.NoSuchKey:
        return {"message": f"Spec not found in S3"}
    except Exception as error:
        return {"message": f"Error accessing S3: {str(error)}"}

@router.post("/keys")
async def save_keys(user: str, filename: str, keys: str = Form(...)):
    jsons = json.dumps(keys)
    path = f"{user}/keys/{filename}.json"
    s3.put_object(Bucket=BUCKET, Key=path, Body=jsons)
    return JSONResponse(content={"message":"Keys have been saved"}, status_code=200)

@router.post("/extract-keys")
async def extract_keys(user: str, filename: str):
    try:
        path = f"{user}/specs/{filename}"
        
        file_data = s3.get_object(Bucket=BUCKET, Key=path)

        file_stream = file_data["Body"].read()
        spec = fitz.open(stream=file_stream, filetype="pdf")

        keys = extract_keys_from_spec(spec)

        spec.close()

        keys['label'] = np.random.choice(["Additional Cost", "Deliverable", "None"], size=len(keys))

        return keys.to_json()

    except Exception as exception:
        return JSONResponse(content={"error": f"Error on extracting the keys: {str(exception)}"}, status_code=500)


### Sections

@router.get("/sections")
async def get_sections(user: str, filename: str):
    try:
        path = f"{user}/specs/{filename}"

        result_file = s3.get_object(Bucket=BUCKET, Key=path)
        
        file_stream = result_file["Body"].read()

        spec = fitz.open(stream=file_stream, filetype="pdf")

        sections_list = {}

        aggregated_sections = []

        for page_number, page in enumerate(spec):
            page_text = page.get_text()
            page_sections = find_sections_in_page(page_text)
            
            parsed_page_sections = list(filter(lambda x : x not in aggregated_sections, sections_parser(page_sections)))

            if len(parsed_page_sections) < 5 and parsed_page_sections:
                for parsed_section in parsed_page_sections:
                    sections_list[parsed_section] = page_number
                aggregated_sections += parsed_page_sections

        return sections_list

    except s3.exceptions.NoSuchKey:
        return {"message": f"Spec not found in S3: {filename}"}
    except Exception as error:
        return {"message": f"Error accessing S3: {error}"}
         

    

### DEPRECATED
# async def get_sections(user: str, filename: str):
#     try:
#         path = f"{user}/specs/{filename}"

#         result_file = s3.get_object(Bucket=BUCKET, Key=path)
        
#         file_stream = result_file["Body"].read()

#         spec = fitz.open(stream=file_stream, filetype="pdf")

#         sections_number = collections.defaultdict(int)

#         current_section_area = collections.defaultdict(int)

#         page_section_map = {}

#         print('Starts processing the spec')

#         for i, page in enumerate(spec):

#             text = page.get_text("blocks", sort=True)
#             page_section_map = map_sections_to_page(text, i, page_section_map)

#             if len(page_section_map) and len(page_section_map[i]):
#                 section = page_section_map[i][0]

#                 # sections_number[section] += 1 

#                 if section not in current_section_area:
#                     current_section_area[section] = i+1

#         # return {key: value for key, value in current_section_area.items() if sections_number(key) > 5}
#         return {key: value for key, value in current_section_area.items()}

#     except s3.exceptions.NoSuchKey:
#         return {"message": f"Spec not found in S3: {filename}"}
#     except Exception as error:
#         return {"message": f"Error accessing S3: {error}"}
         
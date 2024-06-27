from fastapi import APIRouter

router = APIRouter()

# set the name of your bucket
BUCKET = "keystone-specs3"

s3 = boto3.client('s3')


@router.get("/specs")
async def get_user_library(user: str):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{user}/specs/")

    return [obj['Key'].split("/specs/")[-1] for obj in response.get('Contents', [])]


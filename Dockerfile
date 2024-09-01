FROM public.ecr.aws/lambda/python:3.11

COPY ./app ${LAMBDA_TASK_ROOT}


ENV LD_PRELOAD='/var/lang/lib/python3.11/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install --default-timeout=1000 -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.handler" ]

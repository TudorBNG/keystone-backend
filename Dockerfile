FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

# Copy function code
ADD app ${LAMBDA_TASK_ROOT}

# ENV LD_PRELOAD='/var/lang/lib/python3.11/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.handler" ]
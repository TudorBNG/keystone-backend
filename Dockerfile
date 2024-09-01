FROM public.ecr.aws/lambda/python:3.10
# Copy function code
COPY ./app ${LAMBDA_TASK_ROOT}
# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt .
RUN pip install --default-timeout=1000 -r requirements.txt
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.handler" ]
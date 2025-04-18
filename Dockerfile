# pull python base image
FROM python:3.10-slim

# copy application files
ADD /heartfailure_model_api /heartfailure_model_api/

# specify working directory
WORKDIR /heartfailure_model_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements/requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /heartfailure_model_api


# Copy the requirements file from outside the API folder
COPY ./requirements/requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all application code
COPY ./heartfailure_model_api/ .

# Expose the FastAPI port
EXPOSE 8001

# Run FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
# Run the Abot backend server in production-ready mode (gunicorn)

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Define default environment variables
ENV PORT=8080
ENV APP_MODULE=langcorn_app:app
ENV LOG_LEVEL=info
ENV WEB_CONCURRENCY=2

# Copy requirements file and install
COPY ./requirements.txt ./requirements.txt
COPY ./requirements.deploy.txt ./requirements.deploy.txt
RUN pip install -q -r requirements.txt -r requirements.deploy.txt

# Copy all app files to /app folder
COPY . /app

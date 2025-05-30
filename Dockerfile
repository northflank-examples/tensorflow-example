# NOTE: This is the 
#       In case you want to run a different version of Tensorflow 
#       see https://hub.docker.com/r/tensorflow/tensorflow/tags
FROM tensorflow/tensorflow:2.18.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


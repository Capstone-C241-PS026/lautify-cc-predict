FROM python:3.11-slim

WORKDIR /app

RUN pip install h5py --only-binary h5py

RUN apt-get update && apt-get install -y python3-opencv gcc python3-dev

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
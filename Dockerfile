FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y python3-opencv gcc python3-dev && \
    apt-get install -y unzip && \
    apt-get clean && \ 
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/* 
    
COPY requirements.txt .

RUN pip install --no-cache-dir h5py --only-binary h5py

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/temp && \
    curl -o /app/temp/best.pt https://storage.googleapis.com/lautify.appspot.com/models/BestModel.pt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]

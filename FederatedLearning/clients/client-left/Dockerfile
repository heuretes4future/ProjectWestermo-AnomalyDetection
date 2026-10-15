FROM python:3.11-slim

WORKDIR /app

COPY client.py .
COPY *.csv .

RUN pip install flwr==1.4.0 scikit-learn pandas

CMD ["python", "client.py"]

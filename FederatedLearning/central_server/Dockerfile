FROM python:3.11-slim

WORKDIR /app

COPY server.py . 

RUN pip install flwr==1.4.0 scikit-learn pandas

RUN pip install flwr scikit-learn pandas numpy matplotlib

CMD ["python", "server.py"]

FROM python:3.8-slim

WORKDIR /app

# copy requirements first
COPY backend/app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# copy backend code
COPY backend /app

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
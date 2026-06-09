FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "180", "--bind", "0.0.0.0:7860", "app:app"]
FROM python:3.12-slim

WORKDIR /app

# Salin model MLflow hasil training ke dalam image
COPY serving_model/ /app/model/

# Install dependency sesuai kebutuhan model (versi terkunci dari requirements model)
RUN pip install --no-cache-dir -r /app/model/requirements.txt

EXPOSE 8080

# Jalankan REST API serving model saat container dijalankan
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080", "--env-manager", "local"]

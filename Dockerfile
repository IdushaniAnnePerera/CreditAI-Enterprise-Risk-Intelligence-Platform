FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire backend
COPY backend/ ./backend/

# Copy frontend dist so FastAPI can serve it
COPY frontend/dist/ ./frontend/dist/

WORKDIR /app/backend

# Retrain models at build time (generates fresh .pkl files for this environment)
RUN python retrain.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

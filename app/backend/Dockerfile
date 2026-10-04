FROM python:3.13-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow numpy onnxruntime

COPY src /app/src
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

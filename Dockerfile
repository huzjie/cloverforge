# cloverforge — diffusion language model reference implementation
FROM python:3.11-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir .

EXPOSE 8788

ENV PYTHONUNBUFFERED=1
CMD ["cloverforge", "serve", "--host", "0.0.0.0", "--port", "8788"]

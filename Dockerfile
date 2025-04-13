FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Set environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install pip, cron, and update package list
RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .

COPY setup.sh .

RUN chmod +x setup.sh && \
    ./setup.sh

COPY . .

# EXPOSE THE API PORT
EXPOSE 8000

CMD ["uvicorn", "--reload", "--port", "8000", "main:app"]
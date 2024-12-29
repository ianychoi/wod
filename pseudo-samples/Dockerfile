# Base Image for the target container module
FROM python:3.9-slim

# Install essential System Dependencies
RUN dnf install -y \
    chromium \
    chromium-driver \
    fonts-noto-cjk

# Python Dependencies
RUN pip install selenium==4.25.0 \
    && pip install -r requirements.txt

# Application Setup
COPY app.py .env ./
COPY posts.txt ./

CMD ["python", "app.py"]

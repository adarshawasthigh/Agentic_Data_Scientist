# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Create a non-root user with ID 1000 (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set the working directory to the user's home folder
WORKDIR /home/user/app

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the non-root user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy requirements file to the working directory
COPY --chown=user:user requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . .

# Expose the port HF Spaces expects
EXPOSE 7860

# This prevents the "AxiosError 403" when uploading files
CMD ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]

# 1. Use an official, lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install necessary system utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy all application files into the container
COPY . .

# 5. Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Expose the port used by Hugging Face
EXPOSE 7860

# 7. Run Streamlit while disabling XSRF and CORS blocking flags
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
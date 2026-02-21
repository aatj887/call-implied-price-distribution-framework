# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for some OpenBB extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- THE FIX ---
# Run the OpenBB build during the Docker image creation
# This creates the .build.lock file and static assets in a writable environment
RUN python -c "import openbb; openbb.build()"

# Copy the rest of your app code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 7860

# Command to run the app
CMD ["streamlit", "run", "gui.py", "--server.port=7860", "--server.address=0.0.0.0"]
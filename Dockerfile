# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install (all dependencies ship as wheels, no compiler needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 7860

# Command to run the app
CMD ["streamlit", "run", "gui.py", "--server.port=7860", "--server.address=0.0.0.0"]

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY backend-chatbot/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY backend-chatbot/backend/ .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run run_server.py when the container launches
CMD ["python", "run_server.py"]
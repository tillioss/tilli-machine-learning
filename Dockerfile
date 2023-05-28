# Use the official Ubuntu base image
FROM ubuntu:latest

# Update the package lists and install Python 3
RUN apt-get update && apt-get install -y python3 python3-pip

# Install required Python packages
RUN pip3 install joblib numpy pandas flask scikit-learn scipy matplotlib

# Create a working directory
WORKDIR /app

# Copy the server directory into the container
COPY server /app/server
COPY model /app/model

# Set the working directory to the server directory
WORKDIR /app/server


# Start the server
EXPOSE 8080
CMD ["python3", "server.py"]
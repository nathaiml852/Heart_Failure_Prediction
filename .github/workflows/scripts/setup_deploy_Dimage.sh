#!/bin/bash

# -------------------------------
# Configuration (edit these!)
# -------------------------------
DOCKER_IMAGE=“nathaiml/heartfailure-fastapi:latest”
CONTAINER_NAME=“heartfailure_fastapi_container"
HOST_PORT=8001
CONTAINER_PORT=8001

# -------------------------------
# Step 1: Update & Install Docker
# -------------------------------
echo "Updating system and installing prerequisites..."
sudo apt-get update -y
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

echo "Adding Docker's official GPG key..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "Setting up Docker stable repository..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "Installing Docker Engine..."
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Starting and enabling Docker..."
sudo systemctl start docker
sudo systemctl enable docker

echo "Adding user to docker group..."
sudo usermod -aG docker $USER

# -------------------------------
# Step 2: Pull Docker Image
# -------------------------------
echo "Pulling Docker image: $DOCKER_IMAGE"
docker pull $DOCKER_IMAGE

# -------------------------------
# Step 3: Stop & Remove Existing Container
# -------------------------------
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm $CONTAINER_NAME
fi

# -------------------------------
# Step 4: Run New Container
# -------------------------------
echo "Running new container: $CONTAINER_NAME"
docker run -d \
  --name $CONTAINER_NAME \
  -p $HOST_PORT:$CONTAINER_PORT \
  $DOCKER_IMAGE

# -------------------------------
# Step 5: Confirm Deployment
# -------------------------------
echo "Deployment complete. Running containers:"
docker ps

echo ""
echo "🚀 App should now be available at http://<EC2-PUBLIC-IP>:${HOST_PORT}"
echo "👉 NOTE: You may need to log out and log back in for docker group permissions to take effect."
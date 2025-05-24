#!/bin/bash
# Run Docker container with proper configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_TAG="latest"
CONTAINER_NAME="lightning-hydra-dev"
GPU_FLAG=""
MOUNT_DATA=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_FLAG="--gpus all"
            shift
            ;;
        --cpu)
            IMAGE_TAG="dev"
            shift
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --no-mount)
            MOUNT_DATA=false
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--gpu] [--cpu] [--name container_name] [--no-mount]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Starting Lightning-Hydra-Template container...${NC}"

# Build docker run command
DOCKER_CMD="docker run -it --rm"
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
DOCKER_CMD="$DOCKER_CMD $GPU_FLAG"
DOCKER_CMD="$DOCKER_CMD --ipc=host"
DOCKER_CMD="$DOCKER_CMD --ulimit memlock=-1"
DOCKER_CMD="$DOCKER_CMD --ulimit stack=67108864"

# Volume mounts
DOCKER_CMD="$DOCKER_CMD -v $(pwd):/workspace"
DOCKER_CMD="$DOCKER_CMD -v $HOME/.cache:/home/user/.cache"
DOCKER_CMD="$DOCKER_CMD -v $HOME/.ssh:/home/user/.ssh:ro"

if [ "$MOUNT_DATA" = true ] && [ -d "$HOME/datasets" ]; then
    echo -e "${YELLOW}Mounting ~/datasets to /datasets${NC}"
    DOCKER_CMD="$DOCKER_CMD -v $HOME/datasets:/datasets:ro"
fi

# Environment variables
if [ -f .env ]; then
    DOCKER_CMD="$DOCKER_CMD --env-file .env"
fi

# Ports
DOCKER_CMD="$DOCKER_CMD -p 8888:8888"  # Jupyter
DOCKER_CMD="$DOCKER_CMD -p 6006:6006"  # TensorBoard

# Image
DOCKER_CMD="$DOCKER_CMD lightning-hydra-template:$IMAGE_TAG"

# Execute
echo -e "${YELLOW}Running: $DOCKER_CMD${NC}"
eval $DOCKER_CMD

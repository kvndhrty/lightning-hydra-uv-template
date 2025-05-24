#!/bin/bash
# Build Docker images for the project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Lightning-Hydra-Template Docker images...${NC}"

# Parse arguments
BUILD_TYPE=${1:-gpu}
NO_CACHE=${2:-false}

CACHE_FLAG=""
if [ "$NO_CACHE" = "true" ]; then
    CACHE_FLAG="--no-cache"
fi

if [ "$BUILD_TYPE" = "gpu" ]; then
    echo -e "${YELLOW}Building GPU-enabled image...${NC}"
    docker build $CACHE_FLAG -t lightning-hydra-template:latest -f Dockerfile .
elif [ "$BUILD_TYPE" = "dev" ]; then
    echo -e "${YELLOW}Building lightweight development image...${NC}"
    docker build $CACHE_FLAG -t lightning-hydra-template:dev -f Dockerfile.dev .
elif [ "$BUILD_TYPE" = "all" ]; then
    echo -e "${YELLOW}Building all images...${NC}"
    docker build $CACHE_FLAG -t lightning-hydra-template:latest -f Dockerfile .
    docker build $CACHE_FLAG -t lightning-hydra-template:dev -f Dockerfile.dev .
else
    echo -e "${RED}Unknown build type: $BUILD_TYPE${NC}"
    echo "Usage: $0 [gpu|dev|all] [no-cache]"
    exit 1
fi

echo -e "${GREEN}Build complete!${NC}"

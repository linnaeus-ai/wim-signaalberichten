FROM python:3.12-slim-bookworm

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

COPY . .

# Set up for use in VS Code dev containers and install additional tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    apt-transport-https \
    lsb-release \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Install Bicep CLI (architecture-aware)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        BICEP_ARCH="linux-x64"; \
    elif [ "$ARCH" = "arm64" ]; then \
        BICEP_ARCH="linux-arm64"; \
    else \
        echo "Unsupported architecture: $ARCH"; exit 1; \
    fi && \
    curl -Lo /usr/local/bin/bicep "https://github.com/Azure/bicep/releases/latest/download/bicep-$BICEP_ARCH" && \
    chmod +x /usr/local/bin/bicep

# Verify installations (optional, but good for build logs)
RUN echo "--- Verification ---" && \
    echo "Architecture: $(dpkg --print-architecture)" && \
    az --version && \
    bicep --version

# Install dependencies
RUN pip install -r requirements.txt
#!/usr/bin/env bash
# GCP VM startup script — installs NVIDIA drivers, Docker, and nvidia-container-toolkit.
# Run this once after SSH-ing into the VM, OR attach as --metadata-from-file startup-script=.
# Tested on Ubuntu 22.04 LTS with T4 GPU.
set -euo pipefail

echo "=== 1. System update ==="
apt-get update -y && apt-get upgrade -y

echo "=== 2. Install NVIDIA driver 535 ==="
apt-get install -y linux-headers-"$(uname -r)"
apt-get install -y nvidia-driver-535
# Verify driver after reboot; skip reboot here since startup scripts continue
modprobe nvidia || true

echo "=== 3. Install Docker ==="
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== 4. Install nvidia-container-toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  git clone <your-repo> && cd rag-demo"
echo "  cp .env.example .env && nano .env"
echo "  docker compose up -d"
echo "  bash scripts/pull_models.sh --docker"
echo "  docker compose exec app python scripts/index_documents.py --input data/raw/"
echo ""
echo "  App:      http://\$(curl -s ifconfig.me):8000"
echo "  Langfuse: http://\$(curl -s ifconfig.me):3000"

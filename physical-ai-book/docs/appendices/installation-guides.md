---
sidebar_position: 1
---

# Appendix A: Installation Guides

## Table of Contents
1. [System Requirements](#system-requirements)
2. [ROS 2 Humble Installation](#ros-2-humble-installation)
3. [Docker Setup](#docker-setup)
4. [Gazebo Installation](#gazebo-installation)
5. [Unity Setup for Robotics](#unity-setup-for-robotics)
6. [NVIDIA Isaac Setup](#nvidia-isaac-setup)
7. [Development Tools](#development-tools)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04 LTS or 22.04 LTS (recommended)
- **CPU**: Multi-core processor (Intel i5 or equivalent)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 50 GB free space minimum
- **GPU**: NVIDIA GPU with compute capability 6.0+ (GTX 1060 or better)
- **VRAM**: 8 GB minimum for complex simulations

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores)
- **RAM**: 32 GB
- **Storage**: 1 TB SSD
- **GPU**: NVIDIA RTX 3070 or better
- **VRAM**: 12+ GB

## ROS 2 Humble Installation

### Method 1: Debian Packages (Recommended)

```bash
# Set locale
locale  # check for UTF-8
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y locales
sudo apt install software-properties-common
sudo add-apt-repository universe

# Add the ROS 2 GPG key
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update and install ROS 2
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
sudo apt install ros-humble-ros-base
```

### Install ROS 2 Development Tools

```bash
# Essential development tools
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Install additional packages
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-isaac-ros-*  # For Isaac integration
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

### Set Up Environment

```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Create ROS workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Docker Setup

### Install Docker Engine

```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Set up Docker's apt repository
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Install NVIDIA Container Toolkit

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

## Gazebo Installation

### Install Gazebo Garden (Recommended)

```bash
# Add Gazebo's Ubuntu package repository
echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list

# Setup keys
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Update packages
sudo apt update

# Install Gazebo Garden
sudo apt install gz-garden

# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-ros2-control
sudo apt install ros-humble-gazebo-msgs
```

### Verify Installation

```bash
# Test Gazebo
gz sim

# Test ROS 2 integration
ros2 launch gazebo_ros gazebo.launch.py
```

## Unity Setup for Robotics

### Install Unity Hub and Unity Editor

```bash
# Download Unity Hub AppImage
wget https://public-cdn.cloud.unity3d.com/hub/prod/UnityHub.AppImage
chmod +x UnityHub.AppImage

# Run Unity Hub
./UnityHub.AppImage
```

### Manual Installation (Alternative)

```bash
# Install dependencies
sudo apt update
sudo apt install libgtk-3-0 libnss3 libcups2 libasound2 libxtst6 xdg-utils

# Download Unity installer from Unity website and run
# Follow the installation wizard
```

### ROS# Unity Package Setup

```bash
# Clone ROS# package
git clone https://github.com/siemens/ros-sharp.git
cd ros-sharp/Unity3D/Assets

# Or import via Unity Package Manager
# In Unity: Window -> Package Manager -> Add package from git URL
# Use: https://github.com/siemens/ros-sharp.git
```

## NVIDIA Isaac Setup

### Install Isaac ROS

```bash
# Install Isaac ROS dependencies
sudo apt update
sudo apt install ros-humble-isaac-ros-point-cloud-map-builder
sudo apt install ros-humble-isaac-ros-visual-slac
sudo apt install ros-humble-isaac-ros-segmentation
sudo apt install ros-humble-isaac-ros-augmentations
sudo apt install ros-humble-isaac-ros-bit-mapper
sudo apt install ros-humble-isaac-ros-cbir-mode-classifier
sudo apt install ros-humble-isaac-ros-cbir-mode-indexer
sudo apt install ros-humble-isaac-ros-cbir-mode-searcher
sudo apt install ros-humble-isaac-ros-cbir-mode-trainer
sudo apt install ros-humble-isaac-ros-cortex
sudo apt install ros-humble-isaac-ros-deep-learner
sudo apt install ros-humble-isaac-ros-detect-net
sudo apt install ros-humble-isaac-ros-ego-tracing
sudo apt install ros-humble-isaac-ros-gems
sudo apt install ros-humble-isaac-ros-image-encoder
sudo apt install ros-humble-isaac-ros-isaac-sim-camera
sudo apt install ros-humble-isaac-ros-isaac-sim-ground-truth
sudo apt install ros-humble-isaac-ros-isaac-sim-lidar
sudo apt install ros-humble-isaac-ros-isaac-sim-occupancy-grid
sudo apt install ros-humble-isaac-ros-isaac-sim-point-cloud
sudo apt install ros-humble-isaac-ros-isaac-sim-segmentation
sudo apt install ros-humble-isaac-ros-isaac-sim-stereo-depth
sudo apt install ros-humble-isaac-ros-lama-occupancy-grid-localization
sudo apt install ros-humble-isaac-ros-manipulation
sudo apt install ros-humble-isaac-ros-message-bridge
sudo apt install ros-humble-isaac-ros-nitros-camera-info-type
sudo apt install ros-humble-isaac-ros-nitros-disparity-image-type
sudo apt install ros-humble-isaac-ros-nitros-image-type
sudo apt install ros-humble-isaac-ros-nitros-occupancy-grid-type
sudo apt install ros-humble-isaac-ros-nitros-point-cloud-type
sudo apt install ros-humble-isaac-ros-nitros-pose-cov-stamped-type
sudo apt install ros-humble-isaac-ros-nitros-pose-stamped-type
sudo apt install ros-humble-isaac-ros-nitros-range-type
sudo apt install ros-humble-isaac-ros-nitros-tensor-list-type
sudo apt install ros-humble-isaac-ros-nitros-type-adapter
sudo apt install ros-humble-isaac-ros-people-segmentation
sudo apt install ros-humble-isaac-ros-perception
sudo apt install ros-humble-isaac-ros-point-cloud-transport
sudo apt install ros-humble-isaac-ros-realsense-camera
sudo apt install ros-humble-isaac-ros-segment-any-anything
sudo apt install ros-humble-isaac-ros-segmentation-encoder
sudo apt install ros-humble-isaac-ros-stereo-image-publisher
sudo apt install ros-humble-isaac-ros-stereo-image-subscriber
sudo apt install ros-humble-isaac-ros-stitch-fixture
sudo apt install ros-humble-isaac-ros-stereo-depth
sudo apt install ros-humble-isaac-ros-visual-inspector
sudo apt install ros-humble-isaac-ros-visual-logger
sudo apt install ros-humble-isaac-ros-visual-odometry
sudo apt install ros-humble-isaac-ros-visual-slac
```

### Install Isaac Sim via Omniverse

```bash
# Download Omniverse Launcher
wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage

# Launch Omniverse Launcher and install Isaac Sim
# Search for "Isaac Sim" in the apps section and install
```

## Development Tools

### Python Development Environment

```bash
# Install Python 3.8 or higher
sudo apt install python3.8 python3.8-dev python3.8-venv

# Install pip packages
pip3 install --upgrade pip
pip3 install numpy scipy matplotlib jupyter pandas
pip3 install opencv-python open3d
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers openai
pip3 install speechrecognition pyttsx3 vosk
pip3 install scikit-learn scikit-image
```

### Development IDE Setup

#### VS Code Installation

```bash
# Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code

# Install ROS 2 extensions
code --install-extension ms-iot.vscode-ros
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
```

#### Useful VS Code Extensions for ROS 2

```json
{
  "recommendations": [
    "ms-iot.vscode-ros",
    "ms-python.python",
    "ms-vscode.cpptools",
    "ms-vscode.cmake-tools",
    "twxs.cmake",
    "redhat.vscode-yaml",
    "ms-azuretools.vscode-docker",
    "GitHub.copilot"
  ]
}
```

### Git and Version Control

```bash
# Install Git
sudo apt install git git-lfs

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "code --wait"

# Install Git LFS for large files
git lfs install
```

## Troubleshooting

### Common Issues and Solutions

#### 1. ROS 2 Installation Issues

**Problem**: GPG key errors during ROS 2 installation
```bash
# Solution: Update key manually
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key
sudo apt-key add /tmp/ros.key
```

**Problem**: Colcon build fails with missing dependencies
```bash
# Solution: Install missing packages
rosdep install --from-paths src --ignore-src -r -y
```

#### 2. GPU/CUDA Issues

**Problem**: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install CUDA if missing
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
```

#### 3. Docker Permission Issues

**Problem**: Cannot connect to Docker daemon
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in, or run:
newgrp docker
```

#### 4. Network Issues with ROS 2

**Problem**: Nodes cannot communicate across machines
```bash
# Solution: Set up RMW and network settings
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0  # Set same domain ID on all machines
```

#### 5. Gazebo Startup Issues

**Problem**: Gazebo crashes on startup
```bash
# Solution: Check OpenGL support
glxinfo | grep "OpenGL version"
# Install Mesa drivers if needed
sudo apt install mesa-utils
```

### System Health Checks

#### Check ROS 2 Installation

```bash
# Verify ROS 2 installation
printenv | grep ROS
ros2 --version

# Test ROS 2 communication
ros2 topic list
ros2 node list

# Test basic functionality
ros2 run turtlesim turtlesim_node &
ros2 run turtlesim turtle_teleop_key
```

#### Check GPU/CUDA Support

```bash
# Check GPU status
nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Check Isaac ROS packages
ros2 pkg list | grep isaac
```

#### Check System Resources

```bash
# Monitor system resources
htop
nvidia-smi  # GPU usage
df -h       # Disk usage
free -h     # Memory usage
```

### Performance Optimization

#### GPU Memory Management

```bash
# Check GPU memory usage
nvidia-smi -q -d MEMORY

# Set GPU memory fraction in Python (for TensorFlow/PyTorch)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### Docker Resource Limits

```bash
# Run containers with resource limits
docker run --gpus all --memory=8g --cpus=4 <image_name>

# Or create a compose file with limits
```

### Backup and Recovery

#### ROS 2 Workspace Backup

```bash
# Create backup of workspace
tar -czf ros2_workspace_backup.tar.gz ~/ros2_ws

# Restore workspace
tar -xzf ros2_workspace_backup.tar.gz -C ~/
```

#### System Configuration Backup

```bash
# Backup ROS 2 environment setup
cp ~/.bashrc ~/.bashrc.backup

# Backup custom configurations
cp -r ~/.ros ~/.ros.backup
```

## Quick Setup Script

For convenience, here's a script to install the core requirements:

```bash
#!/bin/bash
# physical_ai_setup.sh - Quick setup script for Physical AI development

set -e  # Exit on error

echo "Starting Physical AI development environment setup..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y build-essential cmake git wget curl htop

# Install Python and pip
sudo apt install -y python3 python3-pip python3-dev python3-venv

# Install ROS 2 Humble dependencies
sudo apt install -y locales software-properties-common

# Set up locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS 2 repository
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool

# Initialize rosdep
sudo rosdep init || echo "rosdep already initialized"
rosdep update

# Install additional ROS packages
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-gazebo-ros-pkgs

# Set up ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy scipy matplotlib jupyter opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Setup complete! Please run 'source ~/.bashrc' or restart your terminal."
echo "To verify installation, run: 'ros2 --version'"
```

Save this script as `physical_ai_setup.sh`, make it executable with `chmod +x physical_ai_setup.sh`, and run it with `./physical_ai_setup.sh`.

## Post-Installation Verification

After completing the installation, verify everything works:

```bash
# Source the environment
source ~/.bashrc

# Check ROS 2
ros2 --version

# Check Python packages
python3 -c "import torch; import cv2; import numpy; print('Python packages OK')"

# Check GPU (if available)
nvidia-smi || echo "No NVIDIA GPU detected"

# Test basic ROS 2 functionality
ros2 run demo_nodes_py talker &
sleep 2
ros2 run demo_nodes_py listener &
sleep 5
pkill -f talker
pkill -f listener
```

## Getting Started

Once installation is complete, begin with:

1. **Basic ROS 2 tutorials**: Follow the official ROS 2 tutorials
2. **Gazebo simulation**: Start with simple robot models
3. **Isaac ROS**: Explore the perception packages if you have NVIDIA GPU
4. **Development**: Create your first ROS 2 package

Congratulations! You now have a complete Physical AI development environment set up.
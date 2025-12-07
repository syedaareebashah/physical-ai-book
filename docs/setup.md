---
sidebar_position: 3
---

# Setup Guide

This guide will help you set up your development environment for Physical AI & Humanoid Robotics.

## Development Environment Options

### Option 1: Native Ubuntu Setup (Recommended)
For the best performance and compatibility, we recommend using Ubuntu 20.04 or 22.04 LTS.

### Option 2: Windows with WSL2
Windows users can use Windows Subsystem for Linux (WSL2) to run Ubuntu.

### Option 3: macOS
macOS is supported but may require additional configuration for hardware access.

## Ubuntu Setup

### Update System Packages
```bash
sudo apt update && sudo apt upgrade -y
```

### Install Essential Dependencies
```bash
sudo apt install curl wget git python3-pip build-essential cmake -y
```

### Install Python Dependencies
```bash
pip3 install numpy matplotlib scipy jupyter
```

### Install ROS 2 (Humble Hawksbill)
Add the ROS 2 repository and install:

```bash
# Add the ROS 2 apt repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe

# Add the ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update and install ROS 2
sudo apt update
sudo apt install ros-humble-desktop ros-humble-ros-base -y
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Install NVIDIA Isaac Dependencies
```bash
# Install CUDA (if you have an NVIDIA GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda

# Install Isaac Sim prerequisites
sudo apt install python3.8-venv python3.8-dev -y
```

## Windows with WSL2 Setup

### Install WSL2
Open PowerShell as Administrator and run:
```powershell
wsl --install Ubuntu-22.04
```

### Configure WSL2 for GUI Applications (Optional)
```bash
# Install VcXsrv or X410 for GUI applications
# Add to your .bashrc:
echo 'export DISPLAY=:0' >> ~/.bashrc
echo 'export LIBGL_ALWAYS_INDIRECT=1' >> ~/.bashrc
```

### Continue with Ubuntu Setup
After WSL2 installation, continue with the Ubuntu setup instructions above.

## macOS Setup

### Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Dependencies
```bash
brew install python3 git cmake wget curl
pip3 install numpy matplotlib scipy jupyter
```

### Install ROS 2 via Binary
```bash
# Install ROS 2 via Homebrew (community supported)
brew install ros/humble/ros-humble
source /opt/ros/humble/setup.bash
```

## Development Tools

### Install VS Code
```bash
# Ubuntu
sudo snap install --classic code

# Windows/WSL2 - Download from https://code.visualstudio.com/
# macOS
brew install --cask visual-studio-code
```

### Recommended VS Code Extensions
- Python (Microsoft)
- ROS (Microsoft)
- C/C++ (Microsoft)
- GitLens (Git extension)
- Docker (Microsoft)

### Install Docker
```bash
# Ubuntu
sudo apt install docker.io -y
sudo usermod -aG docker $USER

# macOS/Windows
# Download Docker Desktop from https://www.docker.com/products/docker-desktop/
```

## Verification Steps

### Verify Python Environment
```bash
python3 --version
pip3 list | grep -E "(numpy|matplotlib|scipy)"
```

### Verify ROS 2 Installation
```bash
source /opt/ros/humble/setup.bash
ros2 --version
ros2 topic list
```

### Test Python ROS 2 Interface
```python
import rclpy
from std_msgs.msg import String
print("ROS 2 Python interface working correctly!")
```

## Troubleshooting

### Common Issues and Solutions

#### ROS 2 Installation Issues
- Ensure your locale is set to UTF-8: `locale` should show UTF-8
- If you get GPG errors, try: `sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys [KEY]`

#### Python Path Issues
- Make sure you're using Python 3.8 or higher
- Check your PYTHONPATH: `echo $PYTHONPATH`
- Ensure ROS 2 Python packages are accessible

#### Permission Issues
- For serial port access: `sudo usermod -a -G dialout $USER`
- For Docker: `sudo usermod -aG docker $USER`
- Log out and back in for group changes to take effect

## Next Steps

Once your environment is set up, proceed to Module 1 to learn about the Robotic Nervous System (ROS 2).
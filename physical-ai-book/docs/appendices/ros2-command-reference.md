---
sidebar_position: 2
---

# Appendix B: ROS 2 Command Reference

## Table of Contents
1. [Core ROS 2 Commands](#core-ros-2-commands)
2. [Package Management](#package-management)
3. [Node Operations](#node-operations)
4. [Topic and Service Commands](#topic-and-service-commands)
5. [Parameter Management](#parameter-management)
6. [Launch System](#launch-system)
7. [Debugging and Monitoring](#debugging-and-monitoring)
8. [Navigation Commands](#navigation-commands)
9. [Simulation Commands](#simulation-commands)
10. [Isaac ROS Commands](#isaac-ros-commands)
11. [Common Workflows](#common-workflows)
12. [Troubleshooting Commands](#troubleshooting-commands)

## Core ROS 2 Commands

### Basic Information Commands

```bash
# Check ROS 2 version
ros2 --version

# List available commands
ros2 --help

# Check ROS 2 environment
printenv | grep ROS

# Set ROS domain ID (for network isolation)
export ROS_DOMAIN_ID=0

# Set RMW implementation
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp

# Check ROS 2 network configuration
ros2 doctor
```

### Workspace and Build Commands

```bash
# Create a new package
ros2 pkg create --build-type ament_python my_robot_package
ros2 pkg create --build-type ament_cmake my_robot_cpp_package

# Build workspace
cd ~/ros2_ws
colcon build
colcon build --packages-select my_package_name  # Build specific package
colcon build --symlink-install  # Symlink instead of copying
colcon build --event-handlers console_direct+  # Verbose output

# Source the workspace
source install/setup.bash
source install/local_setup.bash  # Source only current workspace

# Clean build artifacts
rm -rf build/ install/ log/
```

## Package Management

### Package Discovery and Information

```bash
# List all packages
ros2 pkg list

# Find a specific package
ros2 pkg list | grep <package_name>

# Get package information
ros2 pkg info <package_name>

# Find package path
ros2 pkg prefix <package_name>

# Get package executables
ros2 pkg executables <package_name>

# Find package share directory
find /opt/ros/humble -name "<package_name>" -type d
```

### Package Creation and Modification

```bash
# Create package with dependencies
ros2 pkg create --build-type ament_python my_robot_pkg --dependencies rclpy std_msgs geometry_msgs sensor_msgs

# Add dependency to existing package
# Edit package.xml and add:
# <depend>rclpy</depend>
# <depend>std_msgs</depend>

# Create launch directory
mkdir -p launch
touch launch/__init__.py

# Create config directory
mkdir -p config
```

## Node Operations

### Node Management

```bash
# List active nodes
ros2 node list

# Get information about a specific node
ros2 node info /node_name

# Get node graph
ros2 run rqt_graph rqt_graph

# Kill a specific node
ros2 lifecycle set /node_name shutdown
# Or use system tools
pkill -f "node_name"
killall -9 node_executable_name
```

### Node Execution

```bash
# Run a node
ros2 run <package_name> <executable_name>

# Run with arguments
ros2 run <package_name> <executable_name> --arg1 value1 --arg2 value2

# Run with remappings
ros2 run <package_name> <executable_name> --ros-args -r topic1:=new_topic1 -r topic2:=new_topic2

# Run with parameters
ros2 run <package_name> <executable_name> --ros-args -p param1:=value1 -p param2:=value2

# Run with namespace
ros2 run <package_name> <executable_name> --ros-args --ros-namespace my_namespace

# Run in background
ros2 run <package_name> <executable_name> &
```

## Topic and Service Commands

### Topic Operations

```bash
# List all topics
ros2 topic list

# List topics with types
ros2 topic list -t

# Get topic information
ros2 topic info /topic_name

# Echo topic messages
ros2 topic echo /topic_name
ros2 topic echo /topic_name --field data  # Echo specific field
ros2 topic echo /topic_name --csv  # Output as CSV
ros2 topic echo /topic_name --field-rate 1.0  # Echo at 1 Hz

# Publish to a topic
ros2 topic pub /topic_name std_msgs/msg/String "data: 'Hello World'"
ros2 topic pub /topic_name geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}' -1  # Publish once

# Get topic statistics
ros2 topic hz /topic_name  # Check message rate
ros2 topic bw /topic_name  # Check bandwidth (if publisher provides it)
```

### Service Operations

```bash
# List all services
ros2 service list

# List services with types
ros2 service list -t

# Get service information
ros2 service info /service_name

# Call a service
ros2 service call /service_name service_package/srv/ServiceType '{request_field: value}'
ros2 service call /trigger_reset std_srvs/srv/Trigger

# Get service type
ros2 service type /service_name
```

### Action Operations

```bash
# List all actions
ros2 action list

# List actions with types
ros2 action list -t

# Get action information
ros2 action info /action_name

# Send action goal
ros2 action send_goal /action_name action_package/action/ActionType '{goal_field: value}'

# Get action type
ros2 action type /action_name
```

## Parameter Management

### Parameter Operations

```bash
# List parameters of a node
ros2 param list /node_name

# Get parameter value
ros2 param get /node_name param_name

# Set parameter value
ros2 param set /node_name param_name value

# List parameter descriptions
ros2 param describe /node_name param_name

# Dump all parameters to file
ros2 param dump /node_name

# Load parameters from file
ros2 param load /node_name /path/to/param_file.yaml

# Declare parameters in launch files
# In Python launch file:
# declare_param = DeclareLaunchArgument('param_name', default_value='default_value')
```

### Parameter Files

```bash
# Create parameter file
cat > config/my_params.yaml << EOF
my_node:
  ros__parameters:
    param1: 10
    param2: "string_value"
    param3: true
    array_param: [1.0, 2.0, 3.0]
EOF

# Load parameters
ros2 param load /my_node config/my_params.yaml

# Use parameters in launch files
# In launch file:
# Node(
#     package='my_package',
#     executable='my_node',
#     parameters=['config/my_params.yaml']
# )
```

## Launch System

### Launch File Operations

```bash
# Run a launch file
ros2 launch <package_name> <launch_file>.py

# Run with arguments
ros2 launch <package_name> <launch_file>.py arg_name:=arg_value

# List launch arguments
ros2 launch <package_name> <launch_file>.py --show-args

# Run with multiple arguments
ros2 launch <package_name> <launch_file>.py arg1:=val1 arg2:=val2

# Dry run (show what would be launched)
ros2 launch <package_name> <launch_file>.py --dry-run
```

### Common Launch Commands

```bash
# Launch Gazebo
ros2 launch gazebo_ros gazebo.launch.py

# Launch Navigation
ros2 launch nav2_bringup navigation_launch.py
ros2 launch nav2_bringup bringup_launch.py

# Launch Isaac Sim
# Typically done through Omniverse Launcher

# Launch RViz
ros2 launch rviz2 rviz2.launch.py

# Launch multiple launch files
ros2 launch package1 launch1.py &
ros2 launch package2 launch2.py
```

## Debugging and Monitoring

### System Monitoring

```bash
# Monitor all topics
ros2 topic list | xargs -I {} ros2 topic echo {} --field-rate 1.0

# Monitor specific topics continuously
watch -n 1 'ros2 topic list'

# Check system status
ros2 lifecycle list /node_name  # For lifecycle nodes

# Monitor TF tree
ros2 run tf2_tools view_frames
# Then check the generated frames.pdf

# Echo TF transforms
ros2 run tf2_ros tf2_echo map base_link
```

### Performance Monitoring

```bash
# Check topic rates
ros2 topic hz /topic_name

# Monitor CPU and memory
htop
# Or for ROS-specific monitoring:
ros2 run topicos topicos

# Check network usage
nethogs
iftop

# Monitor ROS communications
ros2 run rqt_plot rqt_plot
ros2 run rqt_console rqt_console
```

### Debugging Tools

```bash
# ROS 2 Doctor
ros2 doctor --report

# Check for unreachable nodes
ros2 lifecycle list --all

# Monitor logs
ros2 run rqt_logger_level rqt_logger_level
# Or check log files:
ls ~/.ros/log/latest/
tail -f ~/.ros/log/latest/*.log

# Debug with GDB
gdb --args ros2 run <package> <executable>
```

## Navigation Commands

### Navigation 2 Commands

```bash
# Launch Navigation
ros2 launch nav2_bringup navigation_launch.py
ros2 launch nav2_bringup bringup_launch.py use_sim_time:=true

# Send navigation goal programmatically
# In Python:
# from geometry_msgs.msg import PoseStamped
# goal_publisher.publish(pose_stamped)

# Check navigation status
ros2 lifecycle list /bt_navigator

# Reset navigation
ros2 lifecycle set /bt_navigator configure
ros2 lifecycle set /bt_navigator activate

# Check costmaps
ros2 run rqt_reconfigure rqt_reconfigure
# Or echo costmap topics:
ros2 topic echo /local_costmap/costmap
ros2 topic echo /global_costmap/costmap
```

### Map Server Operations

```bash
# Run map server
ros2 run nav2_map_server map_server

# Load map
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate

# Save map
ros2 run nav2_map_server map_saver_cli -f /path/to/map_name

# Check loaded map
ros2 topic echo /map
```

## Simulation Commands

### Gazebo Commands

```bash
# Launch Gazebo
gz sim -r  # Run in headless mode
gz sim     # Run with GUI

# Launch through ROS 2
ros2 launch gazebo_ros gazebo.launch.py

# Spawn robot in Gazebo
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/model.sdf
ros2 run gazebo_ros spawn_entity.py -entity my_robot -topic robot_description -x 1 -y 2 -z 0

# Check Gazebo topics
gz topic -l  # List all topics
gz topic -i -t /world/default/state  # Get topic info

# Echo Gazebo topics
gz topic -e -t /world/default/model/my_robot/odometry
```

### RViz Commands

```bash
# Launch RViz
ros2 run rviz2 rviz2

# Launch with config
ros2 run rviz2 rviz2 -d /path/to/config.rviz

# Create new RViz config
# Use the GUI to configure, then save as .rviz file
```

## Isaac ROS Commands

### Isaac ROS Package Management

```bash
# List Isaac ROS packages
ros2 pkg list | grep isaac

# Check Isaac ROS installation
ros2 pkg info isaac_ros_visual_slam
ros2 pkg info isaac_ros_detectnet
ros2 pkg info isaac_ros_segmentation

# Launch Isaac ROS components
ros2 launch isaac_ros_visual_slam visual_slam_node.launch.py
ros2 launch isaac_ros_detectnet detectnet.launch.py
ros2 launch isaac_ros_segmentation segmentation.launch.py
```

### Isaac ROS Monitoring

```bash
# Check Isaac ROS topics
ros2 topic list | grep isaac

# Monitor Isaac ROS performance
ros2 run rqt_plot rqt_plot __name:=isaac_monitor

# Check Isaac ROS parameters
ros2 param list /visual_slam_node
ros2 param list /detectnet_node
```

## Common Workflows

### Development Workflow

```bash
# 1. Create and build package
ros2 pkg create --build-type ament_python my_robot_pkg
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source install/setup.bash

# 2. Test the package
ros2 run my_robot_pkg my_node

# 3. Launch with configuration
ros2 launch my_robot_pkg my_launch.py

# 4. Monitor during development
ros2 topic list
ros2 topic echo /my_topic
```

### Debugging Workflow

```bash
# 1. Check system state
ros2 node list
ros2 topic list
ros2 service list

# 2. Monitor specific components
ros2 topic echo /problematic_topic
ros2 param list /problematic_node

# 3. Check logs
ros2 run rqt_console rqt_console

# 4. Restart problematic components
ros2 lifecycle set /node_name shutdown
# Then restart the node
```

### Simulation Testing Workflow

```bash
# 1. Launch simulation
ros2 launch gazebo_ros gazebo.launch.py
ros2 run gazebo_ros spawn_entity.py -entity robot -file model.sdf

# 2. Launch robot control
ros2 launch my_robot_pkg robot_control.launch.py

# 3. Monitor in RViz
ros2 run rviz2 rviz2 -d config/my_robot.rviz

# 4. Send test commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}' -1
```

## Troubleshooting Commands

### Common Issues and Solutions

```bash
# Issue: Nodes cannot communicate
# Solution: Check network and domain
printenv | grep ROS
ros2 doctor

# Issue: Cannot find packages
# Solution: Source the workspace
source ~/ros2_ws/install/setup.bash

# Issue: Topic/service not found
# Solution: Check if nodes are running
ros2 node list
ros2 topic list
ros2 service list

# Issue: Permission denied for device
# Solution: Check user groups
groups $USER
# Add to dialout group for serial devices:
sudo usermod -a -G dialout $USER

# Issue: No GPU detected
# Solution: Check CUDA installation
nvidia-smi
nvidia-ml-py3 check
```

### Network Troubleshooting

```bash
# Check multicast support
ros2 doctor --products

# Test network connectivity
ping <other_machine_ip>

# Check firewall settings
sudo ufw status

# Test ROS 2 communication between machines
# On machine 1:
export ROS_DOMAIN_ID=0
ros2 run demo_nodes_cpp talker
# On machine 2:
export ROS_DOMAIN_ID=0
ros2 run demo_nodes_cpp listener
```

### Process Management

```bash
# Kill all ROS 2 processes
pkill -f ros
killall -9 ros2
killall -9 gz
killall -9 rviz2

# Check for zombie processes
ps aux | grep -E "(ros|gz|rviz)" | grep -v grep

# Clean up shared memory
sudo rm -rf /dev/shm/rt_*
sudo rm -rf /tmp/ros_*
```

### Log Management

```bash
# Find ROS logs
find ~ -name "*.log" -path "*/ros/*" -newer /tmp/start_time 2>/dev/null

# Check recent logs
ls -latr ~/.ros/log/latest/

# Monitor logs in real-time
tail -f ~/.ros/log/latest/*.log

# Clean old logs
find ~/.ros/log -name "*.log" -mtime +7 -delete
```

## Performance Optimization Commands

### Resource Monitoring

```bash
# Monitor CPU usage per process
top -p $(pgrep -d',' -f ros)

# Monitor memory usage
free -h
cat /proc/meminfo | grep MemAvailable

# Monitor disk I/O
iotop -p $(pgrep -d',' -f ros)

# Monitor network
iftop -i $(route | grep '^default' | awk '{print $8}')
```

### Process Priority Adjustment

```bash
# Set real-time priority for critical processes
sudo chrt -f 99 ros2 run critical_pkg critical_node

# Set nice value for background processes
nice -n 10 ros2 run background_pkg background_node

# Monitor process priorities
ps -eo pid,nice,priority,comm | grep ros
```

## Quick Reference Commands

### Most Frequently Used Commands

```bash
# Development
ros2 run <pkg> <exec>
ros2 launch <pkg> <launch_file>.py
colcon build --packages-select <pkg>
source install/setup.bash

# Monitoring
ros2 node list
ros2 topic list
ros2 topic echo /topic
ros2 param list /node

# Troubleshooting
ros2 doctor
ros2 run rqt_console rqt_console
ros2 run rqt_graph rqt_graph
```

### Emergency Commands

```bash
# Stop all ROS processes
pkill -f ros2
pkill -f gz
pkill -f rviz

# Reset ROS environment
unset ROS_DOMAIN_ID
unset RMW_IMPLEMENTATION
source /opt/ros/humble/setup.bash

# Clean workspace
cd ~/ros2_ws
rm -rf build/ install/ log/
colcon build
```

## Aliases and Shortcuts

### Useful Bash Aliases

Add these to your `~/.bashrc`:

```bash
# ROS 2 aliases
alias sb='source /opt/ros/humble/setup.bash'
alias sws='source ~/ros2_ws/install/setup.bash'
alias cb='cd ~/ros2_ws && colcon build'
alias cl='cd ~/ros2_ws && colcon build --symlink-install'
alias rl='ros2 launch'
alias rr='ros2 run'
alias rt='ros2 topic'
alias rs='ros2 service'
alias ra='ros2 action'
alias rp='ros2 param'
alias rn='ros2 node'

# Quick monitoring
alias rmon='ros2 node list && ros2 topic list'
alias rtop='ros2 run rqt_top rqt_top'
alias rgraph='ros2 run rqt_graph rqt_graph'
alias rconsole='ros2 run rqt_console rqt_console'
```

This reference provides comprehensive command-line access to all major ROS 2 operations needed for Physical AI development. Bookmark this page for quick reference during development.
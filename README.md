# FFI-VTR
## A visual navigation system based on feature point optical flow
This is a visual navigation system based on feature point optical flow. The framework we adopt is VINS-fusion, and our main work is in the loop_fusion package. Please configure everything else according to VINS-fusion [link](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion).
## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04.
ROS Kinetic or Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver** Suggest using 1.14.0
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3. **VINS-Fusion**
Please refer to this [link](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion).

## 2. Build FFI-VTR
Clone the repository and catkin_make:
```
cd  ~/catkin_ws/src/VINS-Fusion/loop_fusion
find . -type f -delete
git clone https://github.com/wangjks/FFI-VTR.git
cd ~/catkin_ws
catkin_make
```
Note: The onnxrruntime-linux-arch64-1.10.0 folder needs to be selected based on the device model, downloaded with the corresponding file, and modified to the correct directory in the code.[link](https://github.com/microsoft/onnxruntime/releases)

## 3. Use FFI-VTR
Please refer to VINS-fusion


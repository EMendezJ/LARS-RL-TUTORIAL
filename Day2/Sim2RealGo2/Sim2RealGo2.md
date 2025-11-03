# Sim 2 Real Go2

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)

## Overview

This section provides a practical exploration of Unitree’s framework designed for Sim2Real policy transfer, enabling direct deployment of trained control policies from simulation onto real robotic platforms.

<div align="center">

| <div align="center"><b>End Result</b></div> |
|---|
| <img src="https://github.com/user-attachments/assets/a5545a67-d834-4cd4-831a-2c3dd71d9614" width="240px" height="240px" style="border-radius:10px;"> |

</div>


## Unitree RL Lab Installation

- Install the Unitree RL IsaacLab standalone environments.
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.

- Download unitree robot description files

  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l 
    ```
  - Running a task:

    ```bash
    python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity
    ```


## Unitree SDK Installation

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
cd unitree_sdk2
mkdir build && cd build
cmake .. 
sudo make install
```
## Compile the robot_controller
```bash
cd unitree_rl_lab/deploy/robots/go2
mkdir build && cd build
cmake .. && make
```

## Unitree MuJoCo Installation

Download the mujoco [release](https://github.com/google-deepmind/mujoco/releases), and extract it to the `~/.mujoco` directory;

```
cd unitree_mujoco/simulate/
ln -s ~/.mujoco/mujoco-3.3.6 mujoco
```


### 2. Compile unitree_mujoco
```bash
cd unitree_mujoco/simulate
mkdir build && cd build
cmake ..
make -j4
```
### 3. Test:
Run:
```bash
./unitree_mujoco -r go2 -s scene_terrain.xml
```
You should see the mujoco simulator with the Go2 robot loaded.

### 4. Edit config.yaml for Go2 Interaction
```bash
cd unitree_mujoco/simulate
vim config.yaml

#FINAL CONFIG FILE MUST LOOK LIKE THIS

robot: "go2"  # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
robot_scene: "scene.xml" # Robot scene, /unitree_robots/[robot]/scene.xml 

domain_id: 0  # Domain id
interface: "wlp131s0f0" # Interface CHANGE TO WHICHEVER NETWORK CARD CORRESPOND TO YOUR MULTICAST (USE ip a)

use_joystick: 1 # Simulate Unitree WirelessController using a gamepad
joystick_type: "xbox" # support "xbox" and "switch" gamepad layout
joystick_device: "/dev/input/js0" # Device path
joystick_bits: 16 # Some game controllers may only have 8-bit accuracy

print_scene_information: 1 # Print link, joint and sensors information of robot

enable_elastic_band: 0 # Virtual spring band, used for lifting h1

```


## SIM 2 SIM 


```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

```bash
cd unitree_rl_lab/deploy/robots/go2/build
./go2_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```
![8c8f4d5d-85c6-4d8f-bf9e-b2bc7399f1de](https://github.com/user-attachments/assets/abfd544d-0505-4e86-b159-f7ae171727b5)

## SIM 2 REAL

First connect via ethernet to the robot, then set up the network

<img width="961" height="806" alt="Screenshot from 2025-11-02 23-13-33" src="https://github.com/user-attachments/assets/f5c79639-d1b1-4036-9516-7602bbdc61c6" />

An unconventional method to stop on-board control program has been used throughout the tutorial, which is the following:

```bash
cd ~/unitree_sdk2/build/bin
sudo ./go2_stand_example enxf8e43b808e06  #Change to the network card name corresponding to the IP address 192.168.123.222
```

After stopping onboard controller, run the policy controller (Program will automatically load the most recent policy)
```bash
cd unitree_rl_lab/deploy/robots/go2/build
./go2_ctrl --network eth0 # eth0 is the network interface name.
```


## LOADING PRE TRAINED POLICY

[A pre trained policy can be found here](https://github.com/EMendezJ/LARS-RL-TUTORIAL/tree/main/Day2/Sim2RealGo2/TrainedGo2LocomotionPolicy), move the full folder to unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/

Then run

```bash
cd unitree_rl_lab/

python scripts/rsl_rl/play.py --task=Unitree-Go2-Velocity --checkpoint=/home/{USER}/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/TrainedGo2LocomotionPolicy/model_13800.pt
```

After this pre trained policy can be deployed in both sim and real. 



## Acknowledgements

[This repository is built upon the support and contributions of Unitree Robotics open-source projects.](https://github.com/unitreerobotics)

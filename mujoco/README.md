Source: https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_arm_mujoco.html

## Clone Repo
```shell
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git ~/trossen_arm_mujoco
```

## Mamba/Conda
```shell
conda create --name trossen_mujoco_env python=3.10
conda activate trossen_mujoco_env

```
```shell
mamba create --name trossen_mujoco_env python=3.10
mamba activate trossen_mujoco_env
```

## Install packages
```shell
cd ~/trossen_arm_mujoco
pip install -e .
```

## Verify Installation
```shell
python3 trossen_arm_mujoco/scripts/wxai_pick_place.py
```

## Assets

All robot models are located in trossen_arm_mujoco/assets/. The assets folder contains all robot models organized by robot type, along with mesh files and scene configurations.
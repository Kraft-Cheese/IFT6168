import os
os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import numpy as np
import gymnasium as gym
from pathlib import Path

SCENE_DIR = Path(__file__).parent

TRAIN_SCENES = [
    SCENE_DIR / "scene_train_1.xml",
    SCENE_DIR / "scene_train_2.xml",
    SCENE_DIR / "scene_train_3.xml",
    SCENE_DIR / "scene_train_4.xml",
    SCENE_DIR / "scene_train_5.xml",
    SCENE_DIR / "scene_train_6.xml",
    SCENE_DIR / "scene_train_7.xml",
    SCENE_DIR / "scene_train_8.xml",
]

TEST_SCENES = [
    SCENE_DIR / "scene_test_1.xml",
    SCENE_DIR / "scene_test_2.xml",
    SCENE_DIR / "scene_test_3.xml",
    SCENE_DIR / "scene_test_4.xml",
]

# Bounds for the RX150 end-effector (in metres)
TARGET_X_RANGE = (0.10, 0.35)
TARGET_Y_RANGE = (-0.20, 0.20)
TARGET_Z_RANGE = (0.10, 0.35)

# Episode length in simulation steps
MAX_STEPS = 500


class RX150ReachEnv(gym.Env):
    """
    Gymnasium environment for the RX150 reach task (TBD for pick-and-place)

    Each instance corresponds to one scene XML or one visual environment/domain
    The task is always the same: move the end-effector to a randomly placed red sphere target
    (Later we can change the sphere color/shape/etc... as well)

    Observations: RGB camera image from the sim (480 x 640 x 3)
    Action: target joint positions for 7 actuators
    [waist, shoulder, elbow, wrist_angle, wrist_rotate, left_finger, right_finger]
    Reward: distance(ee, target)
    env_id: to discern between different environments/domains for V-REx/EQRM training loop and per-env risk calculation
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, scene_path: str | Path, env_id: int):
        super().__init__()

        self.scene_path = str(scene_path)
        self.env_id = env_id  # used by V-REx/EQRM training loop

        # Load model
        os.chdir(SCENE_DIR)
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.ee_site_id = self.model.site("ee_site").id
        self.target_body_id = self.model.body("target").id

        # Bounds are ctrlrange
        ctrl_low  = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        # Action space is joint positions for actuators
        self.action_space = gym.spaces.Box(
            low=ctrl_low.astype(np.float32),
            high=ctrl_high.astype(np.float32),
            dtype=np.float32,
        )

        # Observation is raw RGB image
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )

        self._step_count = 0
        self._target_pos = np.zeros(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize target position within reachable workspace
        self._target_pos = np.array([
            self.np_random.uniform(*TARGET_X_RANGE),
            self.np_random.uniform(*TARGET_Y_RANGE),
            self.np_random.uniform(*TARGET_Z_RANGE),
        ])
        self.model.body_pos[self.target_body_id] = self._target_pos

        # Reset arm to init (all joints = 0, fingers open)
        self.data.qpos[:] = 0.0
        self.data.qpos[5] = 0.015   # left_finger open
        self.data.qpos[6] = -0.015  # right_finger open
        self.data.ctrl[:] = self.data.qpos[:7]

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # apply action as target joint positions
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_success()
        truncated = self._step_count >= MAX_STEPS

        info = {
            "env_id": self.env_id, # needed for checking per env risk
            "ee_pos": self._get_ee_pos(),
            "target_pos": self._target_pos.copy(),
            "distance": np.linalg.norm(self._get_ee_pos() - self._target_pos),
            "success": terminated,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._get_obs()

    def close(self):
        self.renderer.close()

    def _get_obs(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="front_cam")
        return self.renderer.render()

    def _get_ee_pos(self) -> np.ndarray:
        # site_xpos is updated by mj_forward/mj_step
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_reward(self) -> float:
        dist = np.linalg.norm(self._get_ee_pos() - self._target_pos)
        return -dist

    def _is_success(self) -> bool:
        dist = np.linalg.norm(self._get_ee_pos() - self._target_pos)
        return dist < 0.02  # 2cm threshold


# Lists of train and test envs
def make_train_envs() -> list[RX150ReachEnv]:
    return [RX150ReachEnv(scene, env_id=i) for i, scene in enumerate(TRAIN_SCENES)]


def make_test_envs() -> list[RX150ReachEnv]:
    return [RX150ReachEnv(scene, env_id=i) for i, scene in enumerate(TEST_SCENES)]


# Test

if __name__ == "__main__":
    env = RX150ReachEnv(TRAIN_SCENES[0], env_id=0)
    obs, _ = env.reset()
    print(f"Observation (shape should be (480, 640, 3)): {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Target pos: {env._target_pos}")

    # take random actions, print reward and distance to target
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"Reward: {reward}")
    print(f"Distance: {info['distance']}m")
    print(f"env_id: {info['env_id']}")
    print("Test passed")
    env.close()

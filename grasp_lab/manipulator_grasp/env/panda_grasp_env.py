import os.path
import sys
import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

sys.path.append('../../grasp_lab')

from manipulator_grasp.arm.robot import Robot
from manipulator_grasp.arm.robot import Panda
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj


class PandaGraspEnv:

    def __init__(self, headless=True):
        self.sim_hz = 500
        self.headless = headless

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(7)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

    def reset(self):
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.robot = Panda()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "link0").t)  

        self.robot_q = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
        self.robot.set_joint(self.robot_q)

        self.joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6", "joint7"
        ]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 不再附加末端执行器
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.1) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)

        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        # 创建离屏渲染器（在headless下也可以使用），仅在非headless时创建viewer
        try:
            self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
            self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
            self.mj_renderer.update_scene(self.mj_data, 0)
            self.mj_depth_renderer.update_scene(self.mj_data, 0)
            self.mj_depth_renderer.enable_depth_rendering()
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data) if not self.headless else None
        except Exception as e:
            print(f"Warning: Failed to create renderers: {e}")
            print("Falling back to no-render mode for video recording")
            self.mj_renderer = None
            self.mj_depth_renderer = None
            self.mj_viewer = None

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        return None

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        if not self.headless and self.mj_viewer is not None:
            self.mj_viewer.sync()

    def render(self):
        # 使用离屏渲染器渲染（headless下也可用）。如渲染器不可用则回退为全零帧
        if self.mj_renderer is None or self.mj_depth_renderer is None:
            return {
                'img': np.zeros((self.height, self.width, 3), dtype=np.uint8),
                'depth': np.zeros((self.height, self.width), dtype=np.float32)
            }

        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }


if __name__ == '__main__':
    env = PandaGraspEnv()
    env.reset()
    for _ in range(10000):
        env.step()
    imgs = env.render()
    env.close()

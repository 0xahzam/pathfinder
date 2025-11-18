import mujoco
import numpy as np
from loguru import logger
from planner import Action, BehaviorPlanner
import glfw
import time


class MuJoCoSimulator:
    def __init__(self, xml_path: str = "data/scene.xml", target_class: str = "cup"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.planner = BehaviorPlanner(target_class=target_class)

        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_camera"
        )

        if not glfw.init():
            raise Exception("glfw init failed")

        self.window = glfw.create_window(1200, 900, "Pathfinder MuJoCo", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )

        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = 90
        self.cam.elevation = -20
        self.cam.distance = 8
        self.cam.lookat = np.array([0, 0, 0.5])

        self.opt = mujoco.MjvOption()

        logger.success("mujoco simulator initialized")

    def detect_objects_in_view(self):
        detections = []

        robot_pos = self.data.qpos[:2]
        robot_angle = self.data.qpos[2]

        cup_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_cup")
        cup_pos = self.data.xpos[cup_id][:2]

        rel_pos = cup_pos - robot_pos
        dist = np.linalg.norm(rel_pos)

        angle_to_cup = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = angle_to_cup - robot_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        if abs(angle_diff) < np.radians(30) and dist < 5:
            screen_x = 320 + int(np.degrees(angle_diff) * 10)
            size = int(2000 / max(dist, 0.5))

            detections.append(
                {
                    "class": "cup",
                    "confidence": max(0.6, 1.0 - dist / 5),
                    "bbox": [
                        float(screen_x - size // 2),
                        float(240 - size),
                        float(screen_x + size // 2),
                        float(240),
                    ],
                }
            )

        chair_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_chair"
        )
        chair_pos = self.data.xpos[chair_id][:2]

        rel_pos = chair_pos - robot_pos
        dist = np.linalg.norm(rel_pos)

        angle_to_chair = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = angle_to_chair - robot_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        if abs(angle_diff) < np.radians(30) and dist < 5:
            screen_x = 320 + int(np.degrees(angle_diff) * 10)
            size = int(3000 / max(dist, 0.5))

            detections.append(
                {
                    "class": "chair",
                    "confidence": max(0.6, 1.0 - dist / 5),
                    "bbox": [
                        float(screen_x - size // 2),
                        float(240 - size),
                        float(screen_x + size // 2),
                        float(240),
                    ],
                }
            )

        return detections

    def apply_action(self, action: Action):
        ctrl = np.zeros(3)

        if action == Action.APPROACH_TARGET or action == Action.MOVE_FORWARD:
            angle = self.data.qpos[2]
            ctrl[0] = np.cos(angle) * 0.5
            ctrl[1] = np.sin(angle) * 0.5
        elif action == Action.TURN_LEFT:
            ctrl[2] = 0.3
        elif action == Action.TURN_RIGHT:
            ctrl[2] = -0.3
        elif action == Action.SEARCH:
            ctrl[2] = 0.15

        self.data.ctrl[:] = ctrl

    def render(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def run(self, steps: int = 2000):
        logger.info(f"starting simulation for {steps} steps")

        step = 0
        start_time = time.time()

        while not glfw.window_should_close(self.window) and step < steps:
            detections = self.detect_objects_in_view()

            action, context = self.planner.decide(detections)

            if step % 100 == 0:
                logger.info(
                    f"step {step}: action={action.value}, detections={len(detections)}"
                )

            self.apply_action(action)

            mujoco.mj_step(self.model, self.data)

            self.render()

            step += 1

            time.sleep(0.01)

        glfw.terminate()
        logger.success(f"simulation complete in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "cup"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    sim = MuJoCoSimulator(target_class=target)
    sim.run(steps=steps)

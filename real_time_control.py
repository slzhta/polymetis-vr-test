import copy
import time
import random
import torch
import threading
import numpy as np
from polymetis import RobotInterface, GripperInterface
from collections import deque
from torchcontrol.policies import velocity

import line_profiler
import atexit

profile = line_profiler.LineProfiler()
profile.print_stats(output_unit=1)
atexit.register(profile.print_stats)


class VRControlTest(object):
    def __init__(self, robot_mode="fixed", policy_id="none"):
        self.robot = RobotInterface(ip_address="localhost")
        self.gripper = gripper = GripperInterface(ip_address="localhost")
        self.robot.go_home()

        current_xyz, current_quat = self.robot.get_ee_pose()
        # self.robot.move_to_ee_pose(current_xyz, torch.tensor([0.717, 0, 0.717, 0], dtype=torch.float32), time_to_go=2)

        self.stop = False
        self.save_freq = 3
        self.vr_freq = 100
        self.command = None
        self.next_desired_gripper_width = None
        self.last_gripper_state = None
        self.robot_mode = robot_mode
        self.policy_id = policy_id

        self.vr_listener_thread = threading.Thread(
            target=self._vr_listener, daemon=True
        )
        self.control_thread = threading.Thread(target=self._robot_control, daemon=True)

    def run(self):
        self.vr_listener_thread.start()
        self.control_thread.start()

        if self.policy_id == "none":
            time.sleep(30)
            self.stop_signal()
        else:
            while not self.stop:
                pass
        
        time.sleep(5) # Wait other thread stop

    def regen_target_velocity(self):
        self.target_velocity = torch.randn_like(self.target_velocity) * 0.05

    def clip_target_pos(self):
        if self.target_pos[0] < 0.3:
            self.target_pos[0] = 0.3
        if self.target_pos[0] > 0.6:
            self.target_pos[0] = 0.6
        if self.target_pos[1] < -0.2:
            self.target_pos[1] = -0.2
        if self.target_pos[1] > 0.2:
            self.target_pos[1] = 0.2
        if self.target_pos[2] < 0.2:
            self.target_pos[2] = 0.2
        if self.target_pos[2] > 0.5:
            self.target_pos[2] = 0.5

    def stop_signal(self):
        self.stop = True

    def load_policy(self):
        if self.policy_id == "none":
            return None
        elif self.policy_id == "pick":
            current_xyz, current_quat = self.robot.get_ee_pose()

            target_xyz = torch.tensor([0.5, 0, 0.13])

            policy = []

            for step in range(500):
                now_xyz = (500 - step) / 500 * current_xyz + step / 500 * target_xyz
                now_quat = current_quat
                policy.append({
                    "xyz": now_xyz,
                    "quat": now_quat,
                    "trigger": 0
                })
            
            for step in range(200):
                policy.append({
                    "xyz": target_xyz,
                    "quat": current_quat,
                    "trigger": 1
                })
            
            for step in range(500):
                now_xyz = (500 - step) / 500 * target_xyz + step / 500 * current_xyz
                now_quat = current_quat
                policy.append({
                    "xyz": now_xyz,
                    "quat": now_quat,
                    "trigger": 1
                })
            
            return policy

    def _vr_listener(self):
        """
        Randomly generate a sequence of pose
        The xyz of pose is in ([0.4, 0.6], [-0.1, 0.1], [0.3, 0.5])
        The orientation now fix (1, 0, 0, 0), will add some noise when confirm is safe

        In a periord of time, the velocity will be similar
        """
        self.command = {}

        current_xyz, current_quat = self.robot.get_ee_pose()

        self.target_pos = current_xyz
        self.target_velocity = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.last_check_time = self.last_update_time = time.time()
        
        self.policy = self.load_policy()
        self.policy_frame = 0

        while not self.stop:
            t0 = time.time()

            if self.policy_id == "none":
                if self.last_check_time + 0.5 < t0:
                    self.last_check_time = t0
                    if random.random() < 0.2:
                        self.regen_target_velocity()
            
                current_velocity = self.target_velocity + torch.randn_like(self.target_velocity) * 0.01
                self.target_pos += current_velocity * (t0 - self.last_update_time)
                self.last_update_time = t0

                self.clip_target_pos()
                trigger_value = 0
            else:
                self.target_pos = self.policy[self.policy_frame]["xyz"]
                current_quat = self.policy[self.policy_frame]["quat"]
                trigger_value = self.policy[self.policy_frame]["trigger"]

                self.policy_frame += 1
                if self.policy_frame >= len(self.policy):
                    self.stop_signal()

            self.command["vr_pos"] = self.target_pos 
            self.command["vr_quat"] = current_quat

            self.command["trigger"] = trigger_value

            duration = time.time() - t0
            if duration < 1 / self.vr_freq:
                time.sleep(1 / self.vr_freq - duration)

    @profile
    def _robot_control(self):
        elapsed_time = deque(maxlen=100)
        self.robot.start_cartesian_impedance()
        while not self.stop:
            if (
                self.command is not None
                and "trigger" in self.command
                and "vr_pos" in self.command
            ):
                t0 = time.time()
                command_copy = copy.deepcopy(self.command) # No lock, a bug?
                
                self.next_desired_gripper_width = desired_gripper_width = 0.85 * (
                    1 - command_copy["trigger"]
                )

                if self.last_gripper_state != command_copy["trigger"]:
                    self.gripper.goto(desired_gripper_width, 0.05, 1)
                    self.last_gripper_state = command_copy["trigger"]

                # current_xyz, current_quat = self.robot.get_ee_pose()
                target_ee_xyz, target_ee_quat = self.command["vr_pos"], self.command["vr_quat"]

                if self.robot_mode == "position":
                    ret = self.robot.update_desired_ee_pose(target_ee_xyz, target_ee_quat)
                elif self.robot_mode == "fixed":
                    # print("Current ee pose:", current_xyz, current_quat)
                    print("Desired ee pose:", target_ee_xyz, target_ee_quat)
                
                duration = time.time() - t0
                elapsed_time.append(duration)
                if duration < 1 / self.vr_freq:
                    time.sleep(1 / self.vr_freq - duration)
                if np.mean(elapsed_time) > 1.0 / self.save_freq * 1.5:
                    print("[Warning]: robot runs too slow",
                          "robot freq",
                          1 / np.mean(elapsed_time))
    
        print("Robot freq:", 1 / np.mean(elapsed_time))


if __name__=="__main__":
    
    controller = VRControlTest(robot_mode="position", policy_id="pick")
    controller.run()

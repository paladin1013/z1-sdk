import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import json
from typing import List, Dict
from dataclasses import dataclass, asdict
from spacemouse.spacemouse_shared_memory import Spacemouse
from multiprocessing.managers import SharedMemoryManager


@dataclass
class Frame:
    time_tag: float
    joint_q: List[float]
    """6 elements, from joint 1 to joint 6, unit: rad"""
    joint_dq: List[float]
    """6 elements, from joint 1 to joint 6, unit: rad/s"""
    joint_tau: List[float]
    """6 elements, from joint 1 to joint 6, unit: N*m"""
    ee_posture: List[float]
    """6 elements, end effector posture, [row, pitch, yaw, x, y, z], unit: meter"""
    gripper_q: float
    """Range from [0, 1]"""


class PoseTracker:

    def __init__(
            self,
            arm: sdk.ArmInterface,
            teleop_dt: float,
            track_dt: float
        ):
        self.arm = arm
        self.teleop_dt = teleop_dt
        self.track_dt = track_dt
        self.tracked_frames:List[Frame] = []
        self.start_time = time.monotonic()
        self.arm_ctrl_dt = arm._ctrlComp.dt
        assert self.track_dt % self.arm_ctrl_dt == 0, f"track_dt should be a multiple of arm._ctrlComp.dt {arm._ctrlComp.dt}"
        assert self.teleop_dt % self.track_dt == 0, f"teleop_dt should be a multiple of track_dt {track_dt}"

    def reset(self):
        self.tracked_frames = []
        self.start_time = time.monotonic()

    def track_frame(self):
        new_frame = Frame(
            time_tag = time.monotonic() - self.start_time,
            joint_q = self.arm.lowstate.q[:6],
            joint_dq = self.arm.lowstate.dq[:6],
            joint_tau = self.arm.lowstate.tau[:6],
            ee_posture = self.arm.lowstate.endPosture.tolist(),
            gripper_q=self.arm.lowstate.q[6],
        )
        self.tracked_frames.append(new_frame)

    def start_teleop_tracking(self, duration: float, back_to_start=True):
        self.arm.loopOn()

        if back_to_start:
            self.arm.backToStart()

        self.arm.startTrack(sdk.ArmFSMState.CARTESIAN)
        
        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=0.3, max_value=500) as sm:
                print("Teleop tracking ready. Waiting for spacemouse movement to start.")

                while True:
                    state = sm.get_motion_state_transformed()
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    if state.any() or button_left or button_right:
                        print(f"Start tracking! Duration: {duration}s")
                        break

                self.reset() # will reset self.tracked_frames and self.start_time

                while time.monotonic() - self.start_time < duration:
                    print(f"Time elapsed: {time.monotonic() - self.start_time:.03f}s", end="\r")
                    state = sm.get_motion_state_transformed()
                    # Spacemouse state is in the format of (x y z roll pitch yaw)
                    directions = np.zeros(7, dtype=np.float64)
                    directions[:3] = state[3:]
                    directions[3:6] = state[:3]
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    if button_left and not button_right:
                        directions[6] = 1
                    elif button_right and not button_left:
                        directions[6] = -1
                    else:
                        directions[6] = 0
                    # Unitree arm direction is in the format of (roll pitch yaw x y z gripper)

                    new_start_time = time.monotonic()
                    for i in range(int(self.teleop_dt/self.track_dt)):
                        self.track_frame()
                        for j in range(int(self.track_dt/self.arm_ctrl_dt)):
                            self.arm.cartesianCtrlCmd(directions, 0.3, 0.3)

                            # Sleep `remaining_time` to match with reference time
                            reference_time = new_start_time + i*self.track_dt + (j+1) * self.arm_ctrl_dt
                            remaining_time = max(0, reference_time - time.monotonic())
                            time.sleep(remaining_time)
                            
        if back_to_start:
            self.arm.backToStart()

        self.arm.loopOff()

    def start_passive_tracking(self, duration: float):

        self.arm.loopOn()
        self.arm.setFsm(sdk.ArmFSMState.PASSIVE)

        print("Passive tracking ready. Waiting for arm passive movements.")
        init_pose = np.array(self.arm.lowstate.q)
        while True:
            if np.linalg.norm(np.array(self.arm.lowstate.q) - init_pose) > 0.001:
                print(f"Start tracking! Duration: {duration}s")
                break
            time.sleep(0.01)

        self.reset()
        self.start_time = time.monotonic()

        while time.monotonic() - self.start_time < duration:
            print(f"Time elapsed: {time.monotonic() - self.start_time:.03f}s", end="\r")
            self.track_frame()
            time.sleep(self.track_dt)
        
        self.arm.loopOff()

    
    def save_frames(self, file_name:str):
        with open(file_name, "w") as f:
            json.dump([asdict(frame) for frame in self.tracked_frames], f)



if __name__ == "__main__":
    arm = sdk.ArmInterface(hasGripper=True)

    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=0.02)

    pt.start_teleop_tracking(duration=10)

    pt.save_frames("logs/trajectories/test_traj_teleop.json")
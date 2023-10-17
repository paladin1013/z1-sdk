import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import json
from typing import List, Dict
from dataclasses import dataclass
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


def track_passive_movement(arm: sdk.ArmInterface, duration: float, freq: float) -> List[Frame]:
    
    arm.setFsm(sdk.ArmFSMState.PASSIVE)
    trajectory:List[Frame] = []
    t_start = time.monotonic()
    
    while time.monotonic() - t_start < duration:
        new_frame = Frame(
            time_tag = time.monotonic() - t_start,
            joint_q = arm.lowstate.q[:6],
            joint_dq = arm.lowstate.dq[:6],
            joint_tau = arm.lowstate.tau[:6],
            ee_posture = arm.lowstate.endPosture.tolist(),
            gripper_q=arm.lowstate.q[6],
        )
        trajectory.append(new_frame)
        time.sleep(1/freq)

    return trajectory
    
    

def teleop_test():

    arm = sdk.ArmInterface(hasGripper=True)

    arm.loopOn()
    arm.backToStart()
    arm.startTrack(sdk.ArmFSMState.CARTESIAN)
    dt = arm._ctrlComp.dt
    freq = 100 # Hz
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=0.3, max_value=500) as sm:
            try:
                while True:
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
                    # print(state)

                    for k in range(int(1/freq/dt)):
                        arm.cartesianCtrlCmd(directions, 0.3, 0.3)
                        time.sleep(dt)

            except KeyboardInterrupt:
                print("Finished! Arm will go back to the start position.")
    arm.backToStart()
    arm.loopOff()


if __name__ == "__main__":
    teleop_test()
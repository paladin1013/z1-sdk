import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import matplotlib.pyplot as plt
import json
from typing import List, Dict
from dataclasses import dataclass

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
    
    



arm = sdk.ArmInterface(hasGripper=True)

arm.loopOn()
arm.setFsm(sdk.ArmFSMState.PASSIVE)

prev_q = arm.lowstate.q

while True:
    t = time.time()
    while True:
        new_q = arm.lowstate.q
        if prev_q != new_q:
            break
        time.sleep(0.0001)
    update_time = time.time()-t
    prev_q = new_q
    print(f"Update time: {update_time:.4f} "+", ".join(f"{q:+.03f}({dq:+.03f})" for (q, dq) in zip(arm.lowstate.q,arm.lowstate.dq)))
    
arm.loopOff()


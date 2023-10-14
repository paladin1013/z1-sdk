import sys
import unitree_arm_interface
import time
import numpy as np

print("Press ctrl+\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState
arm.loopOn()

arm.startTrack(armState.JOINTCTRL)

def moveSingleJoint(arm: unitree_arm_interface.ArmInterface, joint_id: int, dq: float, duration: float, rest_time: float=0.2):
    """Please refer to https://dev-z1.unitree.com/brief/parameter.html for the joint number and their range. 
    7 indicates the gripper."""
    assert joint_id in range(1, 8), "Joint id should be integers between 1 and 7, Please refer to https://dev-z1.unitree.com/brief/parameter.html for the joint id"
    dt = arm._ctrlComp.dt
    speed = np.array([0,0,0,0,0,0,0], dtype=np.float64)
    speed[joint_id-1] = dq
    for i in range(0, int(duration/dt)):
        arm.jointCtrlCmd(speed, 1)
        time.sleep(dt)
    time.sleep(rest_time)

# Duration of each movement. Speed will be adjust accordingly (the angle distance should be kept the same)
duration = 10 # (s)
# To keep safe, please make sure speed*duration = 0.5 
speed = 0.5/duration

moveSingleJoint(arm=arm, joint_id=1, dq=speed, duration=duration)
moveSingleJoint(arm=arm, joint_id=1, dq=-speed, duration=duration)

moveSingleJoint(arm=arm, joint_id=3, dq=-speed, duration=2.5*duration)
moveSingleJoint(arm=arm, joint_id=2, dq=speed, duration=2.5*duration)

moveSingleJoint(arm=arm, joint_id=4, dq=speed, duration=duration)
moveSingleJoint(arm=arm, joint_id=4, dq=-speed, duration=2*duration)
moveSingleJoint(arm=arm, joint_id=4, dq=speed, duration=duration)

moveSingleJoint(arm=arm, joint_id=5, dq=speed, duration=duration)
moveSingleJoint(arm=arm, joint_id=5, dq=-speed, duration=2*duration)
moveSingleJoint(arm=arm, joint_id=5, dq=speed, duration=duration)

moveSingleJoint(arm=arm, joint_id=6, dq=speed, duration=duration)
moveSingleJoint(arm=arm, joint_id=6, dq=-speed, duration=2*duration)
moveSingleJoint(arm=arm, joint_id=6, dq=speed, duration=duration)

moveSingleJoint(arm=arm, joint_id=7, dq=-speed, duration=duration)
moveSingleJoint(arm=arm, joint_id=7, dq=speed, duration=duration)

moveSingleJoint(arm=arm, joint_id=2, dq=-speed, duration=2.5*duration)
moveSingleJoint(arm=arm, joint_id=3, dq=speed, duration=2*duration)

arm.backToStart()
arm.loopOff()


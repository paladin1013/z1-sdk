import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)
arm =  sdk.ArmInterface(hasGripper=True)

arm.loopOn()
arm.startTrack(sdk.ArmFSMState.JOINTCTRL)

steps = 10000


for i in range(steps):
    arm.setArmCmd(
        np.zeros(6, dtype=np.float64), 
        np.zeros(6, dtype=np.float64), 
        np.zeros(6, dtype=np.float64)
    )
    arm.setGripperCmd(0, 0, 0)
    arm.sendRecv()
    time.sleep(arm._ctrlComp.dt)
    if i % 50 == 0:
        print(", ".join(f"{q:.03f}" for q in arm.lowstate.q))


arm.setFsm(sdk.ArmFSMState.PASSIVE)
arm.calibration()

arm.backToStart()
arm.setFsm(sdk.ArmFSMState.PASSIVE)
while True:
    print(", ".join(f"{q:.03f}" for q in arm.lowstate.q+arm.q.tolist()))
    time.sleep(0.1)
arm.loopOff()
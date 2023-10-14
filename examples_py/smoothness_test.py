import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import matplotlib.pyplot as plt

def moveToJointQ(arm: sdk.ArmInterface, targetQ: npt.NDArray[np.float64], expected_time: float):

    dt = arm._ctrlComp.dt
    num_steps = int(expected_time/dt)
    # print(num_steps)
    currentQ = np.append(arm.q, arm.gripperQ)
    speed = (targetQ-currentQ)/expected_time

    error = np.zeros(num_steps,dtype=np.float64)
    referenceQ = np.arange(0,1,1/num_steps)[:,None] * (targetQ-currentQ)[None,:] + currentQ[None,:]

    for i in range(0, num_steps):
        currentQ = np.append(arm.q, arm.gripperQ)
        curr_err = referenceQ[i] - currentQ
        error[i] = np.linalg.norm(curr_err)
        arm.jointCtrlCmd(curr_err/dt, 1)
        time.sleep(dt)

    # currentQ = np.append(arm.q, arm.gripperQ)
    # print(f"Error: {currentQ-targetQ}")
    plt.plot(error[1:])
    plt.show()
    print(f"error: {error.mean()}, +/- {error.std()}")



np.set_printoptions(precision=3, suppress=True)
arm =  sdk.ArmInterface(hasGripper=True)
armState = sdk.ArmFSMState
arm.loopOn()
arm.setFsmLowcmd()

kp = arm._ctrlComp.lowcmd.kp
kp[1] /= 3 # Decrease the stiffness of joint 2
print(f"Decrease kp of joint 2 from {arm._ctrlComp.lowcmd.kp[1]} to {kp[1]}")
kd = arm._ctrlComp.lowcmd.kd
arm.loopOff()

arm.loopOn()
arm.setFsmLowcmd()
kp = arm._ctrlComp.lowcmd.kp
print(f"Kp of joint 2 is set to {kp[1]}")
arm.loopOff()


arm.loopOn()
arm._ctrlComp.lowcmd.setControlGain(kp, kd)
arm.sendRecv()

arm.backToStart()
arm.startTrack(armState.JOINTCTRL)

moveToJointQ(arm, np.array([0.0, 0, -1.0, -0.54, 0.0, 0.0, 0.0]), expected_time=2)
moveToJointQ(arm, np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0, 0.0]), expected_time=10)

arm.backToStart()
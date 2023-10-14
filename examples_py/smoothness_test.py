import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import matplotlib.pyplot as plt

def moveToJointQ(arm: sdk.ArmInterface, targetQ: npt.NDArray[np.float64], expected_time: float):

    dt = arm._ctrlComp.dt
    num_steps = int(expected_time/dt)
    # print(num_steps)
    currentQ = np.array(arm.lowstate.q)
    speed = (targetQ-currentQ)/expected_time

    error = np.zeros(num_steps,dtype=np.float64)
    referenceQ = np.arange(0,1,1/num_steps)[:,None] * (targetQ-currentQ)[None,:] + currentQ[None,:]

    for i in range(0, num_steps):
        currentQ = np.array(arm.lowstate.q)
        curr_err = referenceQ[i] - currentQ
        speed = np.clip(curr_err, 1, -1)
        error[i] = np.linalg.norm(curr_err)
        arm.jointCtrlCmd(speed, 1)
        time.sleep(dt)

    # currentQ = np.append(arm.q, arm.gripperQ)
    # print(f"Error: {currentQ-targetQ}")
    plt.plot(error[1:])
    plt.show()
    print(f"error: {error.mean()}, +/- {error.std()}")



np.set_printoptions(precision=3, suppress=True)
arm =  sdk.ArmInterface(hasGripper=True)

arm.loopOn()

arm.backToStart()
arm.startTrack(sdk.ArmFSMState.JOINTCTRL)

moveToJointQ(arm, np.array([0.0, 0, -1.0, -0.54, 0.0, 0.0, 0.0]), expected_time=2)
moveToJointQ(arm, np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0, 0.0]), expected_time=10)

arm.backToStart()
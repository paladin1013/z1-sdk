import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)
arm =  sdk.ArmInterface(hasGripper=True)

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
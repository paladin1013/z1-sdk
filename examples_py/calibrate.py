import unitree_arm_interface as sdk
import time
arm =  sdk.ArmInterface(hasGripper=True)

arm.loopOn()
arm.setFsm(sdk.ArmFSMState.PASSIVE)

print("Please adjust arm position and press Ctrl-C to confirm")

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Set as home position")

arm.calibration()
print("Calibration completed")
arm.loopOff()
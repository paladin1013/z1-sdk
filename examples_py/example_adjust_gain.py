import unitree_arm_interface as sdk
import time
import numpy as np

np.set_printoptions(precision=3, suppress=True)
arm = sdk.ArmInterface(hasGripper=True)
armModel = arm._ctrlComp.armModel
arm.sendRecv()
print(f"state:{arm.getCurrentState()}, kp: {arm.lowcmd.kp}")
arm.setFsmLowcmd()
print(f"state:{arm.getCurrentState()}, kp: {arm.lowcmd.kp}")
arm.lowcmd.setControlGain([kp/2 for kp in arm.lowcmd.kp], arm.lowcmd.kd)
arm.loopOn()
print(f"state:{arm.getCurrentState()}, kp: {arm.lowcmd.kp}")
arm.setFsm(sdk.ArmFSMState.PASSIVE)
print(f"state:{arm.getCurrentState()}, kp: {arm.lowcmd.kp}")
arm.setFsm(sdk.ArmFSMState.JOINTCTRL)
print(f"state:{arm.getCurrentState()}, kp: {arm.lowcmd.kp}")
arm.loopOff()
print(f"state:{arm.getCurrentState()}, kp: {arm.lowcmd.kp}")
arm.setFsmLowcmd()
print(f"state:{arm.getCurrentState()}, kp: {arm.lowcmd.kp}")

# lastPos = arm.lowstate.getQ()
# targetPos = np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0]) #forward
# duration = 1000

# current_kp = arm._ctrlComp.lowcmd.kp
# current_kd = arm._ctrlComp.lowcmd.kd

# print(f"current_kp: {current_kp}")
# new_kp = [kp/4 for kp in current_kp]
# print(f"new_kp: {new_kp}")
# arm._ctrlComp.lowcmd.setControlGain(new_kp, current_kd)

# for i in range(0, duration):
#     arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
#     arm.qd = (targetPos-lastPos)/(duration*0.002) # set velocity
#     arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6)) # set torque
#     arm.gripperQ = -1*(i/duration)
#     current_kp = arm._ctrlComp.lowcmd.kp
#     current_kd = arm._ctrlComp.lowcmd.kd
#     print(f"tau: {arm.tau} current_kp: {current_kp}")
#     arm.setArmCmd(arm.q, arm.qd, arm.tau)
#     arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
#     arm.sendRecv()# udp connection
#     # print(arm.lowstate.getQ())
#     time.sleep(arm._ctrlComp.dt)

# arm.loopOn()
# arm.backToStart()
# arm.loopOff()
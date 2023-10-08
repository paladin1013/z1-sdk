import unitree_arm_interface
import time
import numpy as np

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armModel = arm._ctrlComp.armModel
arm.setFsmLowcmd()

lastPos = arm.lowstate.getQ()
targetPos = np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0]) #forward
duration = 1000

current_kp = arm._ctrlComp.lowcmd.kp
current_kd = arm._ctrlComp.lowcmd.kd

print(f"current_kp: {current_kp}")
new_kp = [kp/4 for kp in current_kp]
print(f"new_kp: {new_kp}")
arm._ctrlComp.lowcmd.setControlGain(new_kp, current_kd)

for i in range(0, duration):
    arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
    arm.qd = (targetPos-lastPos)/(duration*0.002) # set velocity
    arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6)) # set torque
    arm.gripperQ = -1*(i/duration)
    current_kp = arm._ctrlComp.lowcmd.kp
    current_kd = arm._ctrlComp.lowcmd.kd
    print(f"tau: {arm.tau} current_kp: {current_kp}")
    arm.setArmCmd(arm.q, arm.qd, arm.tau)
    arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
    arm.sendRecv()# udp connection
    # print(arm.lowstate.getQ())
    time.sleep(arm._ctrlComp.dt)

arm.loopOn()
arm.backToStart()
arm.loopOff()
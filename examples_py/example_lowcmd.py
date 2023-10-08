import sys
sys.path.append("../lib")
import unitree_arm_interface
import time
import numpy as np

print("Press ctrl+\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armModel = arm._ctrlComp.armModel
arm.setFsmLowcmd()

duration = 5000
lastPos = arm.lowstate.getQ()
targetPos = np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0]) #forward

t_start = time.monotonic()
dt = arm._ctrlComp.dt
# dt = 1/200
for i in range(0, duration):
    t0 = time.monotonic()
    arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
    arm.qd = (targetPos-lastPos)/(duration*dt) # set velocity
    arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6)) # set torque
    arm.gripperQ = -1*(i/duration)
    t1 = time.monotonic()
    arm.setArmCmd(arm.q, arm.qd, arm.tau)
    t2 = time.monotonic()
    arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
    t3 = time.monotonic()
    # arm.sendRecv()# udp connection
    arm.sendRecv()
    t4 = time.monotonic()
    # print(arm.lowstate.getQ())
    
    t_end_desired = i * dt + t_start
    t_now = time.monotonic()
    if t_now < t_end_desired:
        time.sleep(t_end_desired - t_now)
    
    t5 = time.monotonic()
    print(f"id {t1-t0:.2e}, arm cmd {t2-t1:.2e}, gripper cmd {t3-t2:.2e}, sendrecv {t4-t3:.2e}, slept {t5-t4:.2e}, overall {t5-t0:.2e}")

arm.loopOn()
arm.backToStart()
arm.loopOff()

import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
from typing import Optional

class Controller:
    """Base PID controller"""
    def __init__(self, kp: float, ki: float, kd: float, input_size: int, output_limit: float):
        """
        output_limit: rad/s
        update_freq: Hz
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.input_size = input_size
        self.prev_error = np.zeros(input_size, dtype=np.float64)
        self.integral = np.zeros(input_size, dtype=np.float64)
        self.output_limit = output_limit

    def reset(self):
        self.prev_error = np.zeros(self.input_size, dtype=np.float64)
        self.integral = np.zeros(self.input_size, dtype=np.float64)

    def update(self, setpoint: npt.NDArray[np.float64], measured_value: npt.NDArray[np.float64], debug_joint:Optional[int]=None):
        """Output velocity will be in rad/s
        Note that the joint number starts from 1 and is different from the numpy array index. 
        See https://dev-z1.unitree.com/brief/parameter.html for detailed information."""
        error = setpoint - measured_value
        self.integral += error 
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        output = output.clip(-self.output_limit, self.output_limit)
        if debug_joint:
            print(f"Joint{debug_joint}: Target {setpoint[debug_joint-1]:+.03f} \
Current {measured_value[debug_joint-1]:+.03f} \
Error {error[debug_joint-1]:+.03f} \
P {self.kp * error[debug_joint-1]:+.03f} \
I {self.ki * self.integral[debug_joint-1]:+.03f} \
D {self.kd * derivative[debug_joint-1]:+.03f} \
Output {output[debug_joint-1]:+.03f} \
")
        return output

def pidAdjust(arm: sdk.ArmInterface, target_joint: npt.NDArray[np.float64], max_time: float):

    dt = arm._ctrlComp.dt*10
    steps = max_time/dt
    arm.startTrack(sdk.ArmFSMState.JOINTCTRL)
    output_limit = 0.5
    controller = Controller(3, 0.0005, 1, 7, output_limit)
    output_velocity = controller.update(target_joint, np.array(arm.lowstate.q))

    for i in range(int(steps)):
        output_velocity = controller.update(target_joint, np.array(arm.lowstate.q), debug_joint=1)
        assert np.all(np.abs(output_velocity) <= output_limit)
        # if i % 50 == 0:
            # print(", ".join(f"{q:+.03f} ({v:+.03f})" for (q, v) in zip(arm.lowstate.q, output_velocity)))
        for j in range(int(dt/arm._ctrlComp.dt)):
            arm.jointCtrlCmd(output_velocity, 1)
            time.sleep(arm._ctrlComp.dt)



np.set_printoptions(precision=3, suppress=True)
arm =  sdk.ArmInterface(hasGripper=True)


targetPos = np.zeros(7, dtype=np.float64)
targetPos[0] = 0.5
targetPos[2] = -0.3
arm.loopOn()

arm.backToStart()

pidAdjust(arm, targetPos, max_time=10)

time.sleep(1)
arm.backToStart()
arm.loopOff()
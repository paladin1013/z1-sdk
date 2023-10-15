import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt

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

    def update(self, setpoint: npt.NDArray[np.float64], measured_value: npt.NDArray[np.float64]):
        """Output velocity will be in rad/s"""
        error = setpoint - measured_value
        self.integral += error 
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        output = output.clip(-self.output_limit, self.output_limit)
        return output

def pidAdjust(arm: sdk.ArmInterface, target_joint: npt.NDArray[np.float64], max_time: float):

    dt = arm._ctrlComp.dt
    steps = max_time/dt
    arm.startTrack(sdk.ArmFSMState.JOINTCTRL)
    output_limit = 0.5
    controller = Controller(1, 0.01, 0.01, 7, output_limit)
    output_velocity = controller.update(target_joint, np.array(arm.lowstate.q))

    for i in range(int(steps)):
        output_velocity = controller.update(target_joint, np.array(arm.lowstate.q))
        assert np.all(np.abs(output_velocity) <= output_limit)
        arm.jointCtrlCmd(output_velocity, 1)
        if i % 50 == 0:
            print(", ".join(f"{q:+.03f} ({v:+.03f})" for (q, v) in zip(arm.lowstate.q, output_velocity)))
        time.sleep(dt)



np.set_printoptions(precision=3, suppress=True)
arm =  sdk.ArmInterface(hasGripper=True)


targetPos = np.zeros(7, dtype=np.float64)
targetPos[0] = 0.5
arm.loopOn()
pidAdjust(arm, targetPos, 10)
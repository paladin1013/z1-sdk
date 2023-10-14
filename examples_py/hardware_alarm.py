import time

import serial
import unitree_arm_interface


class HardwareAlarm:
    def __init__(
        self,
        usb_port: str,
        baud_rate: int = 9600,
        turn_on: bytes = b"\xA0\x01\x00\xA1",
        turn_off: bytes = b"\xA0\x01\x01\xA2",
    ):
        self.serial = serial.Serial(usb_port, baud_rate)
        self.turn_off_str = turn_on
        self.turn_on_str = turn_off

    def alert(self, time_s: float = 2, freq_Hz: float = 20):
        for i in range(int(time_s * freq_Hz)):
            self.serial.write(self.turn_on_str)
            time.sleep(0.5 / freq_Hz)
            self.serial.write(self.turn_off_str)
            time.sleep(0.5 / freq_Hz)

    def __del__(self):
        self.serial.close()

    def inspect_arm(
        self, arm: unitree_arm_interface.ArmInterface, inspect_freq_Hz: float = 10
    ):
        
        state = arm.getCurrentState()
        if state != unitree_arm_interface.ArmFSMState.LOWCMD:
            arm.setFsmLowcmd()
        arm.sendRecv()
        print(arm.lowstate.getQTau())


if __name__ == "__main__":
    alarm = HardwareAlarm("/dev/ttyUSB0")
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    while True:
        alarm.inspect_arm(arm)
        time.sleep(0.1)

import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import json
from typing import List, Dict
from dataclasses import dataclass, asdict, field
from spacemouse.spacemouse_shared_memory import Spacemouse
from multiprocessing.managers import SharedMemoryManager


@dataclass
class Frame:

    time_tag: float
    joint_q: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, from joint 1 to joint 6, unit: rad"""
    joint_dq: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, from joint 1 to joint 6, unit: rad/s"""
    joint_tau: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, from joint 1 to joint 6, unit: N*m"""
    ee_posture: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, end effector posture, [row, pitch, yaw, x, y, z], unit: meter or rad"""
    gripper_q: float = 0
    """Range from [0, 1]"""


class PoseTracker:
    def __init__(self, arm: sdk.ArmInterface, teleop_dt: float, track_dt: float):

        self.arm = arm
        self.teleop_dt = teleop_dt
        self.track_dt = track_dt
        self.tracked_frames: List[Frame] = []
        self.start_time = time.monotonic()
        self.arm_ctrl_dt = arm._ctrlComp.dt
        assert (
            self.track_dt % self.arm_ctrl_dt == 0
        ), f"track_dt should be a multiple of arm._ctrlComp.dt {arm._ctrlComp.dt}"
        assert (
            self.teleop_dt % self.track_dt == 0
        ), f"teleop_dt should be a multiple of track_dt {track_dt}"

    def reset(self):

        self.tracked_frames = []
        self.start_time = time.monotonic()

    def track_frame(self):

        new_frame = Frame(
            time_tag=time.monotonic() - self.start_time,
            joint_q=self.arm.lowstate.q[:6],
            joint_dq=self.arm.lowstate.dq[:6],
            joint_tau=self.arm.lowstate.tau[:6],
            ee_posture=self.arm.lowstate.endPosture.tolist(),
            gripper_q=self.arm.lowstate.q[6],
        )
        self.tracked_frames.append(new_frame)

    def start_teleop_tracking(self, duration: float, back_to_start=True):

        self.arm.loopOn()
        if back_to_start:
            self.arm.backToStart()

        self.arm.startTrack(sdk.ArmFSMState.CARTESIAN)

        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=0.3, max_value=500) as sm:
                print(
                    "Teleop tracking ready. Waiting for spacemouse movement to start."
                )

                while True:
                    state = sm.get_motion_state_transformed()
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    if state.any() or button_left or button_right:
                        print(f"Start tracking! Duration: {duration}s")
                        break

                self.reset()  # will reset self.tracked_frames and self.start_time

                while time.monotonic() - self.start_time <= duration:
                    print(
                        f"Time elapsed: {time.monotonic() - self.start_time:.03f}s",
                        end="\r",
                    )
                    state = sm.get_motion_state_transformed()
                    # Spacemouse state is in the format of (x y z roll pitch yaw)
                    directions = np.zeros(7, dtype=np.float64)
                    directions[:3] = state[3:]
                    directions[3:6] = state[:3]
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    if button_left and not button_right:
                        directions[6] = 1
                    elif button_right and not button_left:
                        directions[6] = -1
                    else:
                        directions[6] = 0
                    # Unitree arm direction is in the format of (roll pitch yaw x y z gripper)

                    new_start_time = time.monotonic()
                    for i in range(int(self.teleop_dt / self.track_dt)):
                        self.track_frame()
                        for j in range(int(self.track_dt / self.arm_ctrl_dt)):
                            oriSpeed = 0.6
                            posSpeed = 0.3
                            self.arm.cartesianCtrlCmd(directions, oriSpeed, posSpeed)

                            # Sleep `remaining_time` to match with reference time
                            reference_time = (
                                new_start_time
                                + i * self.track_dt
                                + (j + 1) * self.arm_ctrl_dt
                            )
                            remaining_time = max(0, reference_time - time.monotonic())
                            time.sleep(remaining_time)

        print("Teleoperation completed!")

        if back_to_start:
            self.arm.backToStart()

        self.arm.loopOff()

    def start_passive_tracking(self, duration: float):

        self.arm.loopOn()
        self.arm.setFsm(sdk.ArmFSMState.PASSIVE)

        print("Passive tracking ready. Waiting for arm passive movements.")
        init_pose = np.array(self.arm.lowstate.q)
        while True:
            if np.linalg.norm(np.array(self.arm.lowstate.q) - init_pose) > 0.001:
                print(f"Start tracking! Duration: {duration}s")
                break
            time.sleep(0.01)

        self.reset()
        self.start_time = time.monotonic()

        while time.monotonic() - self.start_time < duration:
            print(f"Time elapsed: {time.monotonic() - self.start_time:.03f}s", end="\r")
            self.track_frame()
            time.sleep(self.track_dt)

        self.arm.loopOff()

    def save_frames(self, file_name: str):

        with open(file_name, "w") as f:
            json.dump([asdict(frame) for frame in self.tracked_frames], f)

    def load_frames(self, file_name: str, override_tracked_frames=False) -> List[Frame]:

        with open(file_name, "r") as f:
            json_data = json.load(f)
        frames = [Frame(**frame) for frame in json_data]
        if override_tracked_frames:
            self.tracked_frames = frames
        return frames

    def replay_trajectory(
        self,
        trajectory: List[Frame],
        ctrl_method: sdk.ArmFSMState,
        init_timeout: float = 5,
        back_to_start=True,
    ) -> bool:
        """Replay trajectory and recording new frames at the same time. Will return True if succeed"""

        assert ctrl_method in [
            sdk.ArmFSMState.JOINTCTRL,
            sdk.ArmFSMState.CARTESIAN,
        ], "ctrl_method should be either sdk.ArmFSMState.JOINTCTRL or sdk.ArmFSMState.CARTESIAN"

        print(f"Start replaying trajectory!")
        self.arm.loopOn()

        if ctrl_method == sdk.ArmFSMState.JOINTCTRL:
            assert any(
                [any(frame.joint_q) for frame in trajectory]
            ), "frame.joint_q in trajectory is not initialized"

            self.arm.backToStart()
            self.arm.startTrack(sdk.ArmFSMState.JOINTCTRL)
            init_start_time = time.monotonic()

            while True:
                self.arm.setArmCmd(
                    np.array(trajectory[0].joint_q),
                    np.array(trajectory[0].joint_dq),
                    np.zeros(6, dtype=np.float64),
                )
                time.sleep(self.arm_ctrl_dt)
                if (
                    np.linalg.norm(
                        np.array(trajectory[0].ee_posture)
                        - np.array(self.arm.lowstate.endPosture)
                    )
                    < 0.05
                ):
                    print(f"Initialization complete! Start replaying trajectory.")
                    break
                if time.monotonic() - init_start_time > init_timeout:
                    print(
                        f"Failed initialization in {init_timeout}. Please reset trajectory or timeout."
                    )
                    return False

            self.reset()
            frame_num = 0
            while True:
                elapsed_time = time.monotonic() - self.start_time
                print(f"Elapsed time: {elapsed_time:3.f}s", end="\r")
                # Find the frame right after elapsed time
                for k in range(frame_num, len(trajectory)):
                    if trajectory[k].time_tag > elapsed_time:
                        frame_num = k
                        break
                else:
                    print(f"Finish replaying trajectory!")
                    if back_to_start:
                        self.arm.backToStart()
                    self.arm.loopOff()
                    return True

                self.arm.setArmCmd(
                    np.array(trajectory[frame_num].joint_q),
                    np.array(trajectory[frame_num].joint_dq),
                    np.zeros(6, dtype=np.float64),
                )
                if elapsed_time / self.track_dt >= len(self.tracked_frames):
                    self.track_frame()
                time.sleep(self.arm_ctrl_dt)

        else:
            home_joint_q = np.zeros(6, dtype=np.float64)
            home_transformation = self.arm._ctrlComp.armModel.forwardKinematics(
                home_joint_q, 6
            )
            home_posture = sdk.homoToPosture(home_transformation)
            assert (
                np.linalg.norm(
                    np.array(trajectory[0].ee_posture) - np.array(home_posture)
                )
                < 0.1
            ), f"Trajectory starting point should be close enough to home position {home_posture}"
            return False


if __name__ == "__main__":

    arm = sdk.ArmInterface(hasGripper=True)

    duration = 10
    track_dt = 0.002

    file_name = f"logs/trajectories/teleop_duration{duration}_dt{track_dt}.json"
    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt)
    # pt.start_teleop_tracking(duration)
    # pt.save_frames()

    trajectory = pt.load_frames(file_name)
    pt.replay_trajectory(trajectory, sdk.ArmFSMState.JOINTCTRL)
    pt.save_frames(file_name.replace("teleop", "replay"))

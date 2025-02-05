from curses import window
import numpy as np
import unitree_arm_interface as sdk
import time
from .spacemouse import Spacemouse
from multiprocessing.managers import SharedMemoryManager
from .trajectory import Trajectory, Frame
from typing import Optional, cast, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy.typing as npt
from queue import Queue


def precise_sleep_until(target: float, slack_time: float = 0.001):
    remaining_time = target - time.monotonic()
    if remaining_time < 0:
        return
    if (remaining_time - slack_time) > 0:
        time.sleep(remaining_time - slack_time)
    while target > time.monotonic():
        # spin lock
        pass
    return


class PoseTracker:
    def __init__(
        self,
        arm: sdk.ArmInterface,
        input_dt: float,
        record_dt: float,
        kp: Optional[List[float]] = None,
        kd: Optional[List[float]] = None,
    ):
        self.arm = arm
        self.input_dt = input_dt
        """Time interval between two consecutive input commands (e.g. teleop or policy)."""
        self.record_dt = record_dt
        """Time interval between two consecutive recorded frames."""
        self.tracked_frames: List[Frame] = []
        self.start_time = time.monotonic()
        self.arm_ctrl_dt = arm._ctrlComp.dt
        if kp is None:
            kp = [20.0, 30.0, 30.0, 20.0, 15.0, 10.0, 20.0]
        if kd is None:
            kd = [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
        self.kp = kp
        self.kd = kd
        assert (
            self.record_dt % self.arm_ctrl_dt == 0
        ), f"record_dt should be a multiple of arm._ctrlComp.dt {arm._ctrlComp.dt}"
        assert (
            self.input_dt % self.record_dt == 0
        ), f"input_dt should be a multiple of record_dt {record_dt}"
        self.spacemouse_queue: Queue[npt.NDArray[np.float64]] = Queue()

    def reset(self):
        self.tracked_frames: List[Frame] = []
        self.start_time = time.monotonic()

    def track_frame(self):
        new_frame = Frame(
            timestamp=time.monotonic() - self.start_time,
            joint_q=self.arm.lowstate.q[:6],
            joint_dq=self.arm.lowstate.dq[:6],
            joint_tau=self.arm.lowstate.tau[:6],
            ee_posture=self.arm.lowstate.endPosture.tolist(),
            gripper_q=[self.arm.lowstate.q[6]],
        )
        self.tracked_frames.append(new_frame)

    def reset_spacemouse_queue(self, window_size: int = 10):
        self.spacemouse_queue: Queue[npt.NDArray[np.float64]] = Queue(window_size)

    def get_smoothed_spacemouse_output(self, sm: Spacemouse):
        assert (
            self.spacemouse_queue.maxsize > 0
        ), "Please call reset_spacemouse_queue() to initialize the queue"
        state = sm.get_motion_state_transformed()
        if (
            self.spacemouse_queue.maxsize > 0
            and self.spacemouse_queue._qsize() == self.spacemouse_queue.maxsize
        ):
            self.spacemouse_queue._get()
        self.spacemouse_queue.put_nowait(state)

        return np.mean(np.array(list(self.spacemouse_queue.queue)), axis=0)

    def start_teleop_tracking(
        self,
        duration: float,
        back_to_start=True,
        oriSpeed: float = 0.6,
        posSpeed: float = 0.3,
    ):
        self.arm.loopOn()
        self.arm.setFsm(sdk.ArmFSMState.PASSIVE)

        # Not sure whether this kd adjustment work in cartesian space control
        if back_to_start:
            print(
                "Setting the arm back to start. Pass `back_to_start=False` to disable this initialization"
            )
            self.arm.backToStart()
            self.arm.setArmCmd(
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )

        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=0.3, max_value=500) as sm:
                print(
                    "Teleop tracking ready. Waiting for spacemouse movement to start."
                )
                self.reset_spacemouse_queue(window_size=10)
                while True:
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    state = self.get_smoothed_spacemouse_output(sm)
                    if state.any() or button_left or button_right:
                        print(f"Start tracking! Duration: {duration}s")
                        break

                self.reset()  # will reset self.tracked_frames and se lf.start_time

                directions = np.zeros(7, dtype=np.float64)
                spacemouse_input_queue = Queue()
                while time.monotonic() - self.start_time <= duration:
                    print(
                        f"Time elapsed: {time.monotonic() - self.start_time:.03f}s",
                        end="\r",
                    )
                    # Spacemouse state is in the format of (x y z roll pitch yaw)
                    prev_directions = directions
                    directions = np.zeros(7, dtype=np.float64)
                    state = self.get_smoothed_spacemouse_output(sm)
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
                    ctrl_frame_num = int(self.input_dt / self.arm_ctrl_dt)
                    for k in range(ctrl_frame_num):
                        interp_directions = (
                            directions * k + prev_directions * (ctrl_frame_num - k)
                        ) / ctrl_frame_num
                        assert all(abs(interp_directions)) <= 1
                        self.arm.cartesianCtrlCmd(interp_directions, oriSpeed, posSpeed)

                        if k % int(self.record_dt / self.arm_ctrl_dt) == 0:
                            self.track_frame()

                        # Sleep `remaining_time` to match with reference time
                        reference_time = new_start_time + (k + 1) * self.arm_ctrl_dt
                        remaining_time = max(0, reference_time - time.monotonic())
                        time.sleep(remaining_time)

        print("Teleoperation completed!")

        if back_to_start:
            self.arm.backToStart()
            self.arm.setArmCmd(
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )

        self.arm.loopOff()
        return Trajectory(frames=self.tracked_frames)

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
            time.sleep(self.record_dt)

        self.arm.loopOff()

    def init_arm(
        self,
        init_frame: Frame,
        ctrl_method: sdk.ArmFSMState,
        init_timeout: float = 5,
        start_from_home: bool = True,
    ):
        assert ctrl_method in [
            sdk.ArmFSMState.JOINTCTRL,
            sdk.ArmFSMState.CARTESIAN,
            sdk.ArmFSMState.LOWCMD,
        ], "ctrl_method should be either sdk.ArmFSMState.JOINTCTRL, sdk.ArmFSMState.CARTESIAN or sdk.ArmFSMState.LOWCMD"
        self.arm.setFsm(sdk.ArmFSMState.PASSIVE)
        if start_from_home:
            self.arm.backToStart()
            self.arm.setArmCmd(
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )

        if ctrl_method == sdk.ArmFSMState.JOINTCTRL:
            self.arm.startTrack(sdk.ArmFSMState.JOINTCTRL)
            self.arm.setArmCmd(
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )
        elif ctrl_method == sdk.ArmFSMState.LOWCMD:
            self.arm.setFsmLowcmd()
            self.arm.lowcmd.setControlGain(self.kp, self.kd)
        init_start_time = time.monotonic()

        while True:
            # Initialize arm position
            self.arm.setArmCmd(
                np.array(init_frame.joint_q),
                # np.array(init_frame.joint_dq),
                np.zeros(6, dtype=np.float64),  # Initialize at zero velocity
                np.zeros(6, dtype=np.float64),
            )
            self.arm.setGripperCmd(
                init_frame.gripper_q[0],
                init_frame.gripper_dq[0],
                # 0.0,
                0.0,
            )
            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                self.arm.sendRecv()
            time.sleep(self.arm_ctrl_dt)
            if (
                np.linalg.norm(
                    np.array(init_frame.joint_q) - np.array(self.arm.lowstate.q[0:6])
                )
                < 0.05
            ):
                print(f"Initialization complete! Start replaying trajectory.")
                return True
            if time.monotonic() - init_start_time > init_timeout:
                print(
                    f"Failed initialization in {init_timeout}. Please reset trajectory or timeout."
                )
                return False

    # @profile
    def replay_traj(
        self,
        trajectory: Trajectory,
        ctrl_method: sdk.ArmFSMState,
        back_to_start=True,
        start_from_home=True,
    ) -> Trajectory:
        """Replay trajectory and recording new frames at the same time. Will return True if succeed"""

        assert ctrl_method in [
            sdk.ArmFSMState.JOINTCTRL,
            sdk.ArmFSMState.CARTESIAN,
            sdk.ArmFSMState.LOWCMD,
        ], "ctrl_method should be either sdk.ArmFSMState.JOINTCTRL, sdk.ArmFSMState.CARTESIAN or sdk.ArmFSMState.LOWCMD"

        self.arm.loopOn()

        if ctrl_method == sdk.ArmFSMState.JOINTCTRL:
            assert trajectory.is_initialized(
                "joint_q"
            ), "frame.joint_q in trajectory is not initialized"
            assert self.init_arm(
                trajectory[0],
                ctrl_method,
                start_from_home=start_from_home,
            ), "Initialization failed"
        elif ctrl_method == sdk.ArmFSMState.LOWCMD:
            assert trajectory.is_initialized(
                "joint_q"
            ), "frame.joint_q in trajectory is not initialized"
            assert self.init_arm(
                trajectory[0],
                ctrl_method,
                start_from_home=start_from_home,
            ), "Initialization failed"
            self.arm.loopOff()
        else:
            raise NotImplementedError("Cartesian control not implemented yet")
            home_joint_q = np.zeros(6, dtype=np.float64)
            home_transformation = self.arm._ctrlComp.armModel.forwardKinematics(
                home_joint_q, 6
            )
            home_posture = sdk.homoToPosture(home_transformation)
            assert (
                np.linalg.norm(
                    np.array(trajectory.np_arrays["ee_posture"][0])
                    - np.array(home_posture)
                )
                < 0.1
            ), f"Trajectory starting point should be close enough to home position {home_posture}"
            return False

        self.reset()
        traj_frame_num = 0  # Frame number to be set as target of the new trajectory
        ctrl_frame_num = (
            0  # Frame number of the arm control communciation (period=self.arm_ctrl_dt)
        )
        while True:
            elapsed_time = time.monotonic() - self.start_time
            ctrl_frame_num += 1
            target_timestamp = ctrl_frame_num * self.arm_ctrl_dt

            # Find the frame right after elapsed time
            for k in range(traj_frame_num, trajectory.timestamps.shape[0]):
                if trajectory.timestamps[k] > target_timestamp:
                    traj_frame_num = k
                    break
            else:
                # Return the recorded trajectory
                print(f"\nFinish replaying trajectory!")
                time.sleep(0.5)
                if back_to_start:
                    self.arm.loopOn()
                    self.arm.backToStart()
                    self.arm.setArmCmd(
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64),
                    )
                    self.arm.loopOff()
                return Trajectory(frames=self.tracked_frames)

            loop_start_time = time.monotonic()
            target_frame = trajectory.interp_single_frame(
                # elapsed_time, traj_frame_num, method="scipy", interp_attrs=["joint_q", "gripper_q"]
                target_timestamp,
                traj_frame_num,
                method="linear",
            )
            interp_end_time = time.monotonic()

            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                joint_tau = np.array(target_frame.joint_tau)
                self.arm.setArmCmd(
                    np.array(target_frame.joint_q),
                    np.array(target_frame.joint_dq),
                    joint_tau,
                )
                self.arm.setGripperCmd(target_frame.gripper_q[0], 0.0, 0.0)

            elif ctrl_method == sdk.ArmFSMState.JOINTCTRL:
                # Using joint_dq for speed normalization
                joint_direction = np.array(target_frame.joint_dq) / np.linalg.norm(
                    target_frame.joint_dq
                )
                gripper_direction = np.array(target_frame.gripper_dq) / np.linalg.norm(
                    target_frame.joint_dq
                )
                direction = np.append(joint_direction, gripper_direction)
                self.arm.jointCtrlCmd(direction, np.linalg.norm(target_frame.joint_dq))

            # Maintain a control frequency of self.arm_ctrl_dt and reduce accumulated error

            precise_sleep_until(self.start_time + target_timestamp)
            sleep_end_time = time.monotonic()

            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                self.arm.sendRecv()
            sendrecv_end_time = time.monotonic()

            if target_timestamp / self.record_dt >= len(self.tracked_frames):
                self.track_frame()

            print(
                f"Elapsed: {sleep_end_time - self.start_time:>5.3f}, interp: {interp_end_time - loop_start_time:.5f}, \
sleep: {sleep_end_time - interp_end_time:.5f}, \
sendrecv: {sendrecv_end_time - sleep_end_time:.5f}",
                # end="\r",
            )

    # def move_to_target(self, target_frame: Frame, method: sdk.ArmFSMState, duration: float = 5.0):

    #     self.reset() # Reset to record new trajectory

    #     if method == sdk.ArmFSMState.LOWCMD:

    #         while True:
    #             self.arm.setArmCmd(
    #                 np.array(target_frame.joint_q),
    #                 np.array(target_frame.joint_dq),
    #                 np.zeros(6, dtype=np.float64),
    #             )

    #     else:
    #         raise NotImplementedError(f"Control method {method} not implemented yet")

    def generate_joint_traj(self, start_frame: Frame, end_frame: Frame):
        """Generate a joint trajectory from start_frame to end_frame with a given duration"""

        current_joint_q = np.array(start_frame.joint_q)
        current_gripper_q = np.array(start_frame.gripper_q)
        target_joint_q = np.array(end_frame.joint_q)
        target_gripper_q = np.array(end_frame.gripper_q)
        duration = end_frame.timestamp - start_frame.timestamp
        timestamps = start_frame.timestamp + np.linspace(
            0, duration, int(duration / self.input_dt)
        )
        traj = Trajectory(timestamps=timestamps)
        traj.np_arrays["joint_q"] = np.linspace(
            current_joint_q, target_joint_q, timestamps.shape[0]
        )
        traj.np_arrays["gripper_q"] = np.linspace(
            current_gripper_q, target_gripper_q, timestamps.shape[0]
        )
        traj.update_joint_dq()
        traj.smoothen_joint_dq_start_end()
        traj.update_joint_tau()

        return traj

    def go_to_joint_pos(
        self,
        joint_q: List[float],
        gripper_q: List[float],
        duration: float,
        ctrl_method=sdk.ArmFSMState.LOWCMD,
    ):
        assert len(joint_q) == 6, "joint_q should be a list of 6 floats"
        assert len(gripper_q) == 1, "gripper_q should be a list of 1 float"

        self.arm.loopOn()
        # Keep sufficient time to sync with the robot
        time.sleep(0.1)
        self.arm.loopOff()
        current_frame = Frame(
            timestamp=0,
            joint_q=self.arm.lowstate.q[:6],
            joint_dq=self.arm.lowstate.dq[:6],
            joint_tau=self.arm.lowstate.tau[:6],
            gripper_q=self.arm.lowstate.q[6:],
            gripper_dq=self.arm.lowstate.dq[6:],
            gripper_tau=self.arm.lowstate.tau[6:],
        )

        target_frame = Frame(
            timestamp=duration,
            joint_q=joint_q,
            gripper_q=gripper_q,
        )

        reference_traj = self.generate_joint_traj(current_frame, target_frame)
        tracked_traj = self.replay_traj(
            reference_traj, ctrl_method, back_to_start=False, start_from_home=False
        )

        return reference_traj, tracked_traj

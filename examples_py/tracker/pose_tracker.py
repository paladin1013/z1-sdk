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


class PoseTracker:
    def __init__(
        self,
        arm: sdk.ArmInterface,
        teleop_dt: float,
        track_dt: float,
        stiffness: Optional[float] = None,
    ):
        self.arm = arm
        self.teleop_dt = teleop_dt
        self.track_dt = track_dt
        self.tracked_traj: Trajectory = Trajectory()
        self.start_time = time.monotonic()
        self.arm_ctrl_dt = arm._ctrlComp.dt
        self.stiffness = stiffness
        assert (
            self.track_dt % self.arm_ctrl_dt == 0
        ), f"track_dt should be a multiple of arm._ctrlComp.dt {arm._ctrlComp.dt}"
        assert (
            self.teleop_dt % self.track_dt == 0
        ), f"teleop_dt should be a multiple of track_dt {track_dt}"

    def reset(self):
        self.tracked_traj = Trajectory()
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
        self.tracked_traj.frames.append(new_frame)

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

                while True:
                    state = sm.get_motion_state_transformed()
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    if state.any() or button_left or button_right:
                        print(f"Start tracking! Duration: {duration}s")
                        break

                self.reset()  # will reset self.tracked_traj and self.start_time

                directions = np.zeros(7, dtype=np.float64)

                while time.monotonic() - self.start_time <= duration:
                    print(
                        f"Time elapsed: {time.monotonic() - self.start_time:.03f}s",
                        end="\r",
                    )
                    state = sm.get_motion_state_transformed()
                    # Spacemouse state is in the format of (x y z roll pitch yaw)
                    prev_directions = directions
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
                    ctrl_frame_num = int(self.teleop_dt / self.arm_ctrl_dt)
                    for k in range(ctrl_frame_num):
                        interp_directions = (
                            directions * k + prev_directions * (ctrl_frame_num - k)
                        ) / ctrl_frame_num
                        assert all(abs(interp_directions)) <= 1
                        self.arm.cartesianCtrlCmd(interp_directions, oriSpeed, posSpeed)

                        if k % int(self.track_dt / self.arm_ctrl_dt) == 0:
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

    def init_arm(
        self, init_frame: Frame, ctrl_method: sdk.ArmFSMState, init_timeout: float = 5
    ):
        assert ctrl_method in [
            sdk.ArmFSMState.JOINTCTRL,
            sdk.ArmFSMState.CARTESIAN,
            sdk.ArmFSMState.LOWCMD,
        ], "ctrl_method should be either sdk.ArmFSMState.JOINTCTRL, sdk.ArmFSMState.CARTESIAN or sdk.ArmFSMState.LOWCMD"
        self.arm.setFsm(sdk.ArmFSMState.PASSIVE)
        self.arm.backToStart()
        self.arm.setArmCmd(
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )
        if ctrl_method == sdk.ArmFSMState.JOINTCTRL:
            self.arm.startTrack(sdk.ArmFSMState.JOINTCTRL)
        elif ctrl_method == sdk.ArmFSMState.LOWCMD:
            self.arm.setFsmLowcmd()
            default_kp = np.array([20.0, 30.0, 30.0, 20.0, 15.0, 10.0, 20.0])
            kd = self.arm.lowcmd.kd
            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                assert (
                    self.stiffness is not None and 0 < self.stiffness <= 1
                ), "stiffness should be initialized in (0, 1]"
            self.arm.lowcmd.setControlGain(self.stiffness * default_kp, kd)
        init_start_time = time.monotonic()

        while True:
            # Initialize arm position
            self.arm.setArmCmd(
                np.array(init_frame.joint_q),
                np.array(init_frame.joint_dq),
                np.zeros(6, dtype=np.float64),
            )
            self.arm.setGripperCmd(
                init_frame.gripper_q[0],
                init_frame.gripper_dq[0],
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

    def replay_traj(
        self,
        trajectory: Trajectory,
        ctrl_method: sdk.ArmFSMState,
        back_to_start=True,
    ) -> bool:
        """Replay trajectory and recording new frames at the same time. Will return True if succeed"""

        assert ctrl_method in [
            sdk.ArmFSMState.JOINTCTRL,
            sdk.ArmFSMState.CARTESIAN,
            sdk.ArmFSMState.LOWCMD,
        ], "ctrl_method should be either sdk.ArmFSMState.JOINTCTRL, sdk.ArmFSMState.CARTESIAN or sdk.ArmFSMState.LOWCMD"

        print(f"Start replaying trajectory!")
        self.arm.loopOn()

        if ctrl_method == sdk.ArmFSMState.JOINTCTRL:
            assert trajectory.is_initialized(
                "joint_q"
            ), "frame.joint_q in trajectory is not initialized"
            assert self.init_arm(
                trajectory.frames[0], ctrl_method
            ), "Initialization failed"
        elif ctrl_method == sdk.ArmFSMState.LOWCMD:
            assert trajectory.is_initialized(
                "joint_q"
            ), "frame.joint_q in trajectory is not initialized"
            assert self.init_arm(
                trajectory.frames[0], ctrl_method
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
                    np.array(trajectory.frames[0].ee_posture) - np.array(home_posture)
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
            # print(f"Elapsed time: {elapsed_time:.3f}s", end="\r")
            # Find the frame right after elapsed time
            for k in range(traj_frame_num, len(trajectory.frames)):
                if trajectory.frames[k].timestamp > elapsed_time:
                    traj_frame_num = k
                    break
            else:
                print(f"\nFinish replaying trajectory!")
                if back_to_start:
                    self.arm.loopOn()
                    self.arm.backToStart()
                    self.arm.setArmCmd(
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64),
                    )
                self.arm.loopOff()
                self.tracked_traj.update_np_arrays()
                return True
            loop_start_time = time.monotonic()
            target_frame = trajectory.interp_single_frame(
                # elapsed_time, traj_frame_num, method="scipy", interp_attrs=["joint_q", "gripper_q"]
                elapsed_time,
                traj_frame_num,
                method="linear",
            )
            interp_end_time = time.monotonic()

            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                joint_tau = np.array(target_frame.joint_tau)
            else:
                joint_tau = np.zeros(6, dtype=np.float64)

            self.arm.setArmCmd(
                np.array(target_frame.joint_q),
                np.array(target_frame.joint_dq) * 0.1,
                joint_tau,
            )
            self.arm.setGripperCmd(target_frame.gripper_q[0], 0.0, 0.0)
            set_gripper_cmd_end_time = time.monotonic()

            if elapsed_time / self.track_dt >= len(self.tracked_traj.frames):
                self.track_frame()

            ctrl_frame_num += 1

            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                self.arm.sendRecv()
            sendrecv_end_time = time.monotonic()

            # Maintain a control frequency of self.arm_ctrl_dt and reduce accumulated error
            reference_time = self.start_time + (ctrl_frame_num + 1) * self.arm_ctrl_dt
            remaining_time = max(0, reference_time - time.monotonic())

            time.sleep(remaining_time)

            sleep_end_time = time.monotonic()

            print(
                f"Elapsed: {sleep_end_time - self.start_time:>5.3f}, interp: {interp_end_time - loop_start_time:.5f}, \
sendrecv: {sendrecv_end_time - set_gripper_cmd_end_time:.5f}, \
sleep: {sleep_end_time - sendrecv_end_time:.5f}",
                end="\r",
            )

    def compare_traj(
        self,
        reference_traj: Trajectory,
        tracked_traj: Trajectory,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        fig: Optional[Figure] = None,
    ):
        """
        Draw 3*3 subplots.
        Three lines in each subplot: reference, tracked, difference.
        Three columns in each subplot: joint_q, joint_dq, joint_tau.
        """
        reference_traj = reference_traj.interp_traj(
            [frame.timestamp for frame in tracked_traj.frames]
        )
        diff_traj = reference_traj.calc_difference(tracked_traj)
        if fig is None:
            fig = plt.figure()
        axes = fig.subplots(3, 3)
        axes = cast(List[List[Axes]], axes)
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = min(
                reference_traj.frames[-1].timestamp,
                tracked_traj.frames[-1].timestamp,
            )
        attr_names = ["joint_q", "joint_dq", "joint_tau"]
        trajs = [reference_traj, tracked_traj, diff_traj]
        traj_names = ["reference", "tracked", "difference"]
        for i in range(3):
            for j in range(3):
                trajs[i].plot_attr(
                    attr_names[j],
                    ax=axes[i][j],
                    title=f"{traj_names[i]} {attr_names[j]}",
                )
                axes[i][j].set_xlim(start_time, end_time)
        return diff_traj, fig

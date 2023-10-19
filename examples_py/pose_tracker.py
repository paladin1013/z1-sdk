import numpy as np
import unitree_arm_interface as sdk
import time
import numpy.typing as npt
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from spacemouse.spacemouse_shared_memory import Spacemouse
from multiprocessing.managers import SharedMemoryManager
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


@dataclass
class Frame:
    timestamp: float
    joint_q: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, from joint 1 to joint 6, unit: rad"""
    joint_dq: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, from joint 1 to joint 6, unit: rad/s"""
    joint_tau: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, from joint 1 to joint 6, unit: N*m"""
    ee_posture: List[float] = field(default_factory=lambda: [0 for k in range(6)])
    """6 elements, end effector posture, [row, pitch, yaw, x, y, z], unit: meter or rad"""
    gripper_q: List[float] = field(default_factory=lambda: [0 for k in range(1)])
    """1 element (make it a list so that it is compatible with other attributes). Range from [0, 1]"""
    gripper_dq: List[float] = field(default_factory=lambda: [0 for k in range(1)])
    """1 element (make it a list so that it is compatible with other attributes)."""
    gripper_tau: List[float] = field(default_factory=lambda: [0 for k in range(1)])
    """1 element (make it a list so that it is compatible with other attributes)."""

    LIST_ATTRS = [            
        "joint_q",
        "joint_dq",
        "joint_tau",
        "ee_posture",
        "gripper_q",
        "gripper_dq",
        "gripper_tau",
    ]
    """All list-like attributes"""


    def __mul__(self, ratio: float):
        new_frame_dict = {}
        for attr_name in Frame.LIST_ATTRS:
            new_frame_dict[attr_name] = [
                val * ratio for val in getattr(self, attr_name)
            ]
        return Frame(self.timestamp * ratio, **new_frame_dict)

    def __add__(self, frame: "Frame"):
        new_frame_dict = {}
        for attr_name in self.LIST_ATTRS:
            new_frame_dict[attr_name] = [
                val1 + val2
                for val1, val2 in zip(
                    getattr(self, attr_name), getattr(frame, attr_name)
                )
            ]
        return Frame(self.timestamp + frame.timestamp, **new_frame_dict)
    


class Trajectory:
    def __init__(
        self, frames: Optional[List[Frame]] = None, file_name: Optional[str] = None
    ):
        if frames is None:
            if file_name is None:
                self.frames: List[Frame] = []
            else:
                with open(file_name, "r") as f:
                    json_data = json.load(f)
                    self.frames = [Frame(**frame) for frame in json_data]
        else:
            self.frames = frames
        
        self.interp_functions: Dict[str, List[interp1d]] = {}


    def save_frames(self, file_name: str):
        with open(file_name, "w") as f:
            json.dump([asdict(frame) for frame in self.frames], f)

    def is_initialized(self, attr_name):
        assert attr_name in Frame.LIST_ATTRS

        return any([any(getattr(frame, attr_name)) for frame in self.frames])

    def interp_traj(self, new_timestamps: List[float]):
        """This method will interpolate a new trajectory, using the postures and the joint states from
        the reference trajectory, and query at the input timestamps."""

        ref_timestamps = [frame.timestamp for frame in self.frames]

        def interpnd(
            x: List[float], y: List[List[float]], x_new: List[float]
        ) -> List[List[float]]:
            assert all(
                [len(_) == len(y[0]) for _ in y]
            ), "The input data of y should have the same dimension."
            raw_results: List[List[float]] = []
            dim = len(y[0])
            for k in range(dim):
                y_k = [_[k] for _ in y]
                f_k = interp1d(x, y_k, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
                raw_results.append(f_k(x_new))

            results = [
                [raw_results[i][j] for i in range(dim)] for j in range(len(x_new))
            ]
            return results

        new_traj = Trajectory([Frame(timestamp) for timestamp in new_timestamps])

        # Interpolate all attributes
        for attr_name in Frame.LIST_ATTRS:
            ref_val = [getattr(frame, attr_name) for frame in self.frames]

            interp_val = interpnd(ref_timestamps, ref_val, new_timestamps)
            for k in range(len(new_timestamps)):
                setattr(new_traj.frames[k], attr_name, interp_val[k])

        return new_traj

    def init_interp_function(self):
        assert self.frames, "self.frames is empty! Please initialize the trajectory."
        timestamps = [frame.timestamp for frame in self.frames]
        for attr_name in Frame.LIST_ATTRS:
            dim = len(getattr(self.frames[0], attr_name))
            self.interp_functions[attr_name] = []
            for k in range(dim):
                vals = [getattr(frame, attr_name)[k] for frame in self.frames]
                f_k = interp1d(timestamps, vals, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
                self.interp_functions[attr_name].append(f_k)

        

    def interp_single_frame(
        self,
        new_timestamp: float,
        next_frame_idx: Optional[int] = None,
        method: str = "linear",
        interp_attrs: List[str] = Frame.LIST_ATTRS
    ):
        """next_frame_idx is the frame number that is the first to have a larger timestamp than new_timestamp.
        Will return the self.frames[0] if next_frame_idx = 0
        Will return the self.frames[-1] if next_frame_idx >= len(self.frames)"""

        if next_frame_idx is None:
            for k, frame in enumerate(self.frames):
                if frame.timestamp > new_timestamp:
                    next_frame_idx = k
                    break
            else:
                next_frame_idx = len(self.frames)

        if next_frame_idx == 0:
            return self.frames[0]
        elif next_frame_idx >= len(self.frames):
            return self.frames[-1]
        else:
            assert (
                self.frames[next_frame_idx - 1].timestamp
                <= new_timestamp
                <= self.frames[next_frame_idx].timestamp
            ), "Wrong next_frame_idx"

            if method == "linear":
                prev_ratio = (
                    new_timestamp - self.frames[next_frame_idx - 1].timestamp
                ) / (
                    self.frames[next_frame_idx].timestamp
                    - self.frames[next_frame_idx - 1].timestamp
                )
                new_frame = self.frames[next_frame_idx - 1] * prev_ratio + \
                    self.frames[next_frame_idx] * (1 - prev_ratio)
            
            elif method == "scipy":
                raise NotImplementedError("Computation speed is too slow. Need to be optimized")
                start_time = time.monotonic()
                if not self.interp_functions:
                    self.init_interp_function()

                new_frame = Frame(new_timestamp)
                for attr_name in interp_attrs:
                    dim = len(getattr(new_frame, attr_name))
                    new_val:List[float] = []
                    for k in range(dim):
                        new_val.append(self.interp_functions[attr_name][k](new_timestamp))
                    setattr(new_frame, attr_name, new_val)
                
                print(f"Interpolate time {time.monotonic() - start_time:.5f}s")


            else:
                raise NotImplementedError(
                    f"Interpolation method {method} not implemented."
                )

            return new_frame
    
    def calc_difference(self, new_traj: "Trajectory"):
        """
        Calculate the difference of all attributes between two trajectories.
        When applying this method, please make sure new_traj has the same timestamps as the current one.
        """
        assert len(self.frames) == len(new_traj.frames), "The two trajectories should have the same length"
        diff_traj = Trajectory()
        for k in range(len(self.frames)):
            assert self.frames[k].timestamp == new_traj.frames[k].timestamp, "The two trajectories should have the same timestamps"
            diff_traj.frames.append(self.frames[k]*(-1) + new_traj.frames[k])
            diff_traj.frames[-1].timestamp = self.frames[k].timestamp
        return diff_traj

    def calc_difference_norm(
        self, new_traj: "Trajectory", attr_name: str, specify_axis: Optional[int] = None
    ):
        """
        Calculate the difference of `attr_name` between two trajectories.

        `attr_name` should be one of "joint_q", "joint_dq", "joint_tau", "ee_posture", "gripper_q", "gripper_dq", "gripper_tau".
        If `specify_axis` is not set, this method will take the norm of the difference of all axes.
        Otherwise it will only return the specified single axis (`new_traj - self`).

        When applying this method, please make sure new_traj has the same timestamps as the current one.
        If not, please first use interp_traj to get a interpolated trajectory.
        """
        assert attr_name in Frame.LIST_ATTRS

        if specify_axis is not None:
            assert (
                type(specify_axis) == int
            ), f"`specify_axis` should be an integer from 0 to {len(getattr(self.frames[0], attr_name))}"
            results: List[float] = [
                getattr(new_traj.frames[k], attr_name)[specify_axis]
                - getattr(frame, attr_name)[specify_axis]
                for k, frame in enumerate(self.frames)
            ]
        else:
            results: List[float] = []
            for k, frame in enumerate(self.frames):
                new_val = np.array(getattr(new_traj.frames[k], attr_name))
                self_val = np.array(getattr(frame, attr_name))
                results.append(float(np.linalg.norm(new_val - self_val)))

        return results
    
    def measure_noise(self, padding: int = 5):
        """Calculate noise of the trajectory through calculate average variance in a neighborhood around each element."""
        avg_var = Frame(0)
        for attr_name in Frame.LIST_ATTRS:
            val_matrix = np.array([getattr(frame, attr_name) for frame in self.frames])
            var_matrix = np.zeros_like(val_matrix, dtype=np.float64)
            for k in range(var_matrix.shape[0]):
                # Calculate the variance between [k-padding, k+padding] in each column 
                var_matrix[k] = np.std(val_matrix[max(0, k-padding):min(k+padding, var_matrix.shape[0]-1)], axis=0)
            setattr(avg_var, attr_name, np.mean(var_matrix, axis=0).tolist())
        
        return avg_var
    
    def get_moving_average(self, padding: int = 5):
        """Apply np.convolve to smoothen the trajectory"""
        new_frames = [Frame(frame.timestamp) for frame in self.frames]
        for attr_name in Frame.LIST_ATTRS:
            val_matrix = np.array([getattr(frame, attr_name) for frame in self.frames])
            # Apply padding
            padding_up = np.ones((padding, 1))@val_matrix[0].reshape((1, -1))
            padding_down = np.ones((padding, 1))@val_matrix[-1].reshape((1, -1))
            val_matrix_with_padding = np.concatenate([padding_up, val_matrix, padding_down], axis=0)
            filter_result = convolve2d(val_matrix_with_padding, np.ones((padding*2+1, 1)), mode="valid")/(2*padding+1)
            for k, new_frame in enumerate(new_frames):
                setattr(new_frame, attr_name, filter_result[k].tolist())
        return Trajectory(new_frames)

    def plot_attr(self, attr_name):
        assert attr_name in Frame.LIST_ATTRS
        timestamps = [frame.timestamp for frame in self.frames]
        vals = [getattr(frame, attr_name) for frame in self.frames]
        dim = len(vals[0])
        plt.figure()
        for k in range(dim):
            plt.plot(timestamps, [val[k] for val in vals])
        plt.legend([f"{attr_name}_{k}" for k in range(dim)])
        plt.xlabel("Time (s)")

class PoseTracker:
    def __init__(self, arm: sdk.ArmInterface, teleop_dt: float, track_dt: float):
        self.arm = arm
        self.teleop_dt = teleop_dt
        self.track_dt = track_dt
        self.tracked_traj: Trajectory = Trajectory()
        self.start_time = time.monotonic()
        self.arm_ctrl_dt = arm._ctrlComp.dt
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
        posSpeed: float = 0.3
    ):
        assert 0 < stiffness <= 1, "stiffness should be in (0, 1]"
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

        
    def init_arm(self, init_frame: Frame, ctrl_method: sdk.ArmFSMState, init_timeout: float = 5):
        assert ctrl_method in [
            sdk.ArmFSMState.JOINTCTRL,
            sdk.ArmFSMState.CARTESIAN,
            sdk.ArmFSMState.LOWCMD,
        ], "ctrl_method should be either sdk.ArmFSMState.JOINTCTRL, sdk.ArmFSMState.CARTESIAN or sdk.ArmFSMState.LOWCMD"

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
            time.sleep(self.arm_ctrl_dt)
            if (
                np.linalg.norm(
                    np.array(init_frame.joint_q)
                    - np.array(self.arm.lowstate.q[0:6])
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
            assert self.init_arm(trajectory.frames[0], ctrl_method), "Initialization failed"
        elif ctrl_method == sdk.ArmFSMState.LOWCMD:
            assert trajectory.is_initialized(
                "joint_q"
            ), "frame.joint_q in trajectory is not initialized"
            assert self.init_arm(trajectory.frames[0], ctrl_method), "Initialization failed"
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
        ctrl_frame_num = 0  # Frame number of the arm control communciation (period=self.arm_ctrl_dt)
        while True:
            elapsed_time = time.monotonic() - self.start_time
            print(f"Elapsed time: {elapsed_time:.3f}s", end="\r")
            # Find the frame right after elapsed time
            for k in range(0, len(trajectory.frames)):
                if trajectory.frames[k].timestamp > elapsed_time:
                    traj_frame_num = k
                    break
            else:
                print(f"Finish replaying trajectory!")
                if back_to_start:
                    self.arm.backToStart()
                    self.arm.setArmCmd(
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64),
                    )
                self.arm.loopOff()
                return True

            target_frame = trajectory.interp_single_frame(
                # elapsed_time, traj_frame_num, method="scipy", interp_attrs=["joint_q", "gripper_q"]
                elapsed_time, traj_frame_num, method="linear"
            )

            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                    # np.zeros(6, dtype=np.float64),
                joint_tau = self.arm._ctrlComp.armModel.inverseDynamics(
                    np.array(target_frame.joint_q),
                    np.array(target_frame.joint_dq),
                    np.zeros(6, dtype=np.float64),
                    np.zeros(6, dtype=np.float64),
                )
            else:
                joint_tau = np.zeros(6, dtype=np.float64)

            self.arm.setArmCmd(
                np.array(target_frame.joint_q),
                np.array(target_frame.joint_dq),
                # np.zeros(6, dtype=np.float64),
                joint_tau,
            )
            self.arm.setGripperCmd(
                target_frame.gripper_q[0], 0.0, 0.0
            )

            if elapsed_time / self.track_dt >= len(self.tracked_traj.frames):
                self.track_frame()

            # Maintain a control frequency of self.arm_ctrl_dt and reduce accumulated error
            reference_time = (
                self.start_time + (ctrl_frame_num + 1) * self.arm_ctrl_dt
            )
            remaining_time = max(0, time.monotonic() - reference_time)
            ctrl_frame_num += 1

            if ctrl_method == sdk.ArmFSMState.LOWCMD:
                self.arm.sendRecv()

            time.sleep(remaining_time)

            


if __name__ == "__main__":
    arm = sdk.ArmInterface(hasGripper=True)
    arm.setArmCmd(
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
    )

    duration = 15
    track_dt = 0.01
    stiffness = 0.5

    teleop_file_name = f"logs/trajectories/teleop_duration{duration}_dt{track_dt}.json"
    replay_file_name = f"logs/trajectories/replay_duration{duration}_dt{track_dt}.json"
    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt)
    
    # pt.start_teleop_tracking(duration)
    # pt.tracked_traj.save_frames(teleop_file_name)

    ref_traj = Trajectory(file_name=teleop_file_name)

    default_kp = [20.0, 30.0, 30.0, 20.0, 15.0, 10.0, 20.0]
    kd = arm.lowcmd.kd
    arm.setFsmLowcmd()
    arm.lowcmd.setControlGain([kp*stiffness for kp in default_kp], kd)

    pt.replay_traj(ref_traj, ctrl_method=sdk.ArmFSMState.JOINTCTRL)
    pt.tracked_traj.save_frames(replay_file_name)

    replay_traj = Trajectory(file_name=replay_file_name)
    replay_timestamps = [frame.timestamp for frame in replay_traj.frames]

    interp_traj = ref_traj.interp_traj(replay_timestamps)
    diff_traj = interp_traj.calc_difference(replay_traj)
    ref_traj.plot_attr("joint_q")
    replay_traj.plot_attr("joint_q")
    diff_traj.plot_attr("joint_q")
    plt.show()
    
    # teleop_traj = Trajectory(file_name=teleop_file_name)
    # replay_traj = Trajectory(file_name=replay_file_name)

    # print(teleop_traj.measure_noise().joint_tau)
    # # print(replay_traj.measure_noise())

    # filtered_traj = teleop_traj.get_moving_average()

    # print(filtered_traj.measure_noise().joint_tau)

    # teleop_traj.plot_attr("joint_tau")
    # filtered_traj.plot_attr("joint_tau")
    # plt.show()

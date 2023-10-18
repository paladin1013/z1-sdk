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

class Trajectory:
    def __init__(self, frames: Optional[List[Frame]]=None, file_name:Optional[str]=None):
        if frames is None:
            if file_name is None:
                self.frames:List[Frame] = []
            else:
                with open(file_name, "r") as f:
                    json_data = json.load(f)
                    self.frames = [Frame(**frame) for frame in json_data]
        else:
            self.frames = frames
    

    def save_frames(self, file_name: str):
        with open(file_name, "w") as f:
            json.dump([asdict(frame) for frame in self.frames], f)

    def is_initialized(self, attr_name):
        assert attr_name in ["joint_q", "joint_dq", "joint_tau", "ee_posture", "gripper_q"]

        return any([any(getattr(frame, attr_name)) for frame in self.frames])

    def interp_traj(self, new_timestamps: List[float]):
        """This method will interpolate a new trajectory, using the postures and the joint states from 
        the reference trajectory, and query at the input timestamps."""
        
        ref_timestamps = [frame.timestamp for frame in self.frames]
    

        def interpnd(x:List[float], y:List[List[float]], x_new:List[float]) -> List[List[float]]:
            assert all([len(_)==len(y[0]) for _ in y]), "The input data of y should have the same dimension."
            raw_results:List[List[float]] = []
            dim = len(y[0])
            for k in range(dim):
                y_k = [_[k] for _ in y]
                f_k = interp1d(x, y_k, bounds_error=False, fill_value="extrapolate")
                raw_results.append(f_k(x_new))

            results = [[raw_results[i][j] for i in range(dim)] for j in range(len(x_new))]
            return results
        
        new_traj = Trajectory([Frame(timestamp) for timestamp in new_timestamps])

        # Interpolate all attributes
        for attr_name in ["joint_q", "joint_dq", "joint_tau", "ee_posture", "gripper_q"]:

            ref_val = [getattr(frame, attr_name) for frame in self.frames]

            interp_val = interpnd(ref_timestamps, ref_val, new_timestamps)
            for k in range(len(new_timestamps)):
                setattr(new_traj.frames[k], attr_name, interp_val[k])

        return new_traj
    
    def calc_difference(self, new_traj: "Trajectory", attr_name: str, specify_axis: Optional[int]=None):
        """
        Calculate the difference of `attr_name` between two trajectories. 

        `attr_name` should be one of "joint_q", "joint_dq", "joint_tau", "ee_posture", "gripper_q".
        If `specify_axis` is not set, this method will take the norm of the difference of all axes.
        Otherwise it will only return the specified single axis (`new_traj - self`).

        When applying this method, please make sure new_traj has the same timestamps as the current one.
        If not, please first use interp_traj to get a interpolated trajectory.
        """
        assert attr_name in ["joint_q", "joint_dq", "joint_tau", "ee_posture", "gripper_q"]
        
        if specify_axis is not None:
            assert type(specify_axis) == int, f"`specify_axis` should be an integer from 0 to {len(getattr(self.frames[0], attr_name))}"
            results:List[float] = [
                    getattr(new_traj.frames[k], attr_name)[specify_axis] - getattr(frame, attr_name)[specify_axis]
                    for k, frame in enumerate(self.frames)
                    ]
        else:
            results:List[float] = []
            for k, frame in enumerate(self.frames):
                new_val = np.array(getattr(new_traj.frames[k], attr_name))
                self_val = np.array(getattr(frame, attr_name))
                results.append(float(np.linalg.norm(new_val - self_val)))

        return results
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

                self.reset()  # will reset self.tracked_traj and self.start_time

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



    def replay_traj(
        self,
        trajectory: Trajectory,
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
            assert trajectory.is_initialized("joint_q"), "frame.joint_q in trajectory is not initialized"

            self.arm.backToStart()
            self.arm.startTrack(sdk.ArmFSMState.JOINTCTRL)
            init_start_time = time.monotonic()

            while True:
                self.arm.setArmCmd(
                    np.array(trajectory.frames[0].joint_q),
                    np.array(trajectory.frames[0].joint_dq),
                    np.zeros(6, dtype=np.float64),
                )
                time.sleep(self.arm_ctrl_dt)
                if (
                    np.linalg.norm(
                        np.array(trajectory.frames[0].ee_posture)
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
                print(f"Elapsed time: {elapsed_time:.3f}s", end="\r")
                # Find the frame right after elapsed time
                for k in range(frame_num, len(trajectory.frames)):
                    if trajectory.frames[k].timestamp > elapsed_time:
                        frame_num = k
                        break
                else:
                    print(f"Finish replaying trajectory!")
                    if back_to_start:
                        self.arm.backToStart()
                    self.arm.loopOff()
                    return True

                self.arm.setArmCmd(
                    np.array(trajectory.frames[frame_num].joint_q),
                    np.array(trajectory.frames[frame_num].joint_dq),
                    np.zeros(6, dtype=np.float64),
                )
                if elapsed_time / self.track_dt >= len(self.tracked_traj.frames):
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
                    np.array(trajectory.frames[0].ee_posture) - np.array(home_posture)
                )
                < 0.1
            ), f"Trajectory starting point should be close enough to home position {home_posture}"
            return False
    
if __name__ == "__main__":

    arm = sdk.ArmInterface(hasGripper=True)

    duration = 10
    track_dt = 0.002

    teleop_file_name = f"logs/trajectories/teleop_duration{duration}_dt{track_dt}.json"
    replay_file_name = f"logs/trajectories/replay_duration{duration}_dt{track_dt}.json"
    # pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt)
    # pt.start_teleop_tracking(duration)
    # pt.tracked_traj.save_frames(teleop_file_name)

    ref_traj = Trajectory(file_name=teleop_file_name)
    replay_traj = Trajectory(file_name=replay_file_name)
    replay_timestamps = [frame.timestamp for frame in replay_traj.frames]
    
    interp_traj = ref_traj.interp_traj(replay_timestamps)

    diff = interp_traj.calc_difference(replay_traj, "joint_q")

    plt.plot(replay_timestamps, diff)
    plt.show()

    # pt.replay_traj(trajectory, sdk.ArmFSMState.JOINTCTRL)
    # pt.tracked_traj.save_frames(replay_file_name)

    


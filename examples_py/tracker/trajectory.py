import numpy as np
import numpy.typing as npt
import json
from typing import List, Dict, Optional, cast
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
import unitree_arm_interface as sdk
from tqdm import tqdm
import time
@dataclass
class Frame:
    # Using np.ndarray might be better
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

        self.np_arrays: Dict[str, npt.NDArray[np.float64]] = {}
        """For faster computation as a whole trajectory. Values should be kept the same as self.frames. 
        Keys includes `timestamps` and all attributes in `Frame.LIST_ATTRS`"""

        self.update_np_arrays()

    def update_np_arrays(self):    
        """Synchronize the numpy arrays with the current frames"""
        self.np_arrays["timestamps"] = np.array([frame.timestamp for frame in self.frames])
        for attr_name in Frame.LIST_ATTRS:
            self.np_arrays[attr_name] = np.array([getattr(frame, attr_name) for frame in self.frames])

    def update_frames(self):
        """Synchronize the frames with the current numpy arrays"""
        # Match the length of frames and timestamps:
        if len(self.frames) > len(self.np_arrays["timestamps"]):
            self.frames = self.frames[:len(self.np_arrays["timestamps"])]
        elif len(self.frames) < len(self.np_arrays["timestamps"]):
            for k in range(len(self.np_arrays["timestamps"]) - len(self.frames)):
                self.frames.append(Frame(0))


        for k, frame in enumerate(self.frames):
            frame.timestamp = self.np_arrays["timestamps"][k]
            for attr_name in Frame.LIST_ATTRS:
                if attr_name in self.np_arrays and self.np_arrays[attr_name].shape[0] > k:
                    array:npt.NDArray = self.np_arrays[attr_name][k]
                    setattr(frame, attr_name, array.tolist())

    def save_frames(self, file_name: str):
        with open(file_name, "w") as f:
            json.dump([asdict(frame) for frame in self.frames], f)

    def is_initialized(self, attr_name):
        assert attr_name in Frame.LIST_ATTRS

        return any([any(getattr(frame, attr_name)) for frame in self.frames])

    def interp_traj(self, new_timestamps: List[float], attr_names: List[str] = Frame.LIST_ATTRS, update_frames: bool = True):
        """This method will interpolate a new trajectory, using the postures and the joint states from
        the reference trajectory, and query at the input timestamps.
        To accelarate, set update_frames to False"""

        if not self.interp_functions:
            self.init_interp_function()
            print("Interpolation functions initialized")

        # Create a new trajectory
        new_traj = Trajectory()
        new_traj.np_arrays["timestamps"] = np.array(new_timestamps)
        # Interpolate all attributes
        for attr_name in attr_names:
            # ref_val = [getattr(frame, attr_name) for frame in self.frames]
            ref_val = self.np_arrays[attr_name]
            dim = ref_val.shape[1]
            interp_val = np.zeros((len(new_timestamps), dim), dtype=np.float64)
            for k in range(dim):
                interp_val[:, k] = self.interp_functions[attr_name][k](new_timestamps)
            new_traj.np_arrays[attr_name] = interp_val.copy()
        if update_frames:
            new_traj.update_frames()
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
        self, new_traj: "Trajectory", attr_name: str
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the difference of `attr_name` between two trajectories.

        `attr_name` should be one of "joint_q", "joint_dq", "joint_tau", "ee_posture", "gripper_q", "gripper_dq", "gripper_tau".
        Otherwise it will only return the specified single axis (`new_traj - self`).

        When applying this method, please make sure new_traj has the same timestamps as the current one.
        If not, please first use interp_traj to get a interpolated trajectory.
        """
        assert attr_name in Frame.LIST_ATTRS

        assert all(self.np_arrays['timestamps'] == new_traj.np_arrays['timestamps']), \
            "The two trajectories should have the same timestamps"
        results = np.linalg.norm(new_traj.np_arrays[attr_name] - self.np_arrays[attr_name], axis=1)

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

    def plot_attr(self, attr_name, ax:Optional[Axes] = None, title: Optional[str] = None):
        assert attr_name in Frame.LIST_ATTRS
        timestamps = [frame.timestamp for frame in self.frames]
        vals = [getattr(frame, attr_name) for frame in self.frames]
        dim = len(vals[0])
        if ax is None:
            fig = plt.figure()
            ax = plt.axes()
            ax = cast(Axes, ax)
            fig.add_axes(ax)
        for k in range(dim):
            ax.plot(timestamps, [val[k] for val in vals])
        ax.legend([f"{attr_name}_{k}" for k in range(dim)])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{title}")

    def update_joint_dq(self, padding: int=5):
        """Calculate average speed in [k-padding, k+padding] for each joint and update in self.frames[k].joint_dq"""
        for k, frame in enumerate(self.frames):
            if k < padding or k >= len(self.frames) - padding:
                continue
            prev_frame = self.frames[k-padding]
            next_frame = self.frames[k+padding]
            joint_dq_np = (np.array(next_frame.joint_q) - np.array(prev_frame.joint_q))/(next_frame.timestamp - prev_frame.timestamp)
            frame.joint_dq = joint_dq_np.tolist()

    def update_joint_tau(self, padding: int=1):
        """Calculate average torque in [k-padding, k+padding] for each joint and update in self.frames[k].joint_tau"""
        for k, frame in enumerate(self.frames):
            if k < padding or k >= len(self.frames) - padding:
                continue
            prev_frame = self.frames[k-padding]
            next_frame = self.frames[k+padding]
            joint_ddq = (np.array(next_frame.joint_dq) - np.array(prev_frame.joint_dq))/(next_frame.timestamp - prev_frame.timestamp)

            # For z1 arm inverse dynamics
            arm = sdk.ArmInterface(hasGripper=True)
            frame.joint_tau = arm._ctrlComp.armModel.inverseDynamics(
                np.array(frame.joint_q),
                np.array(frame.joint_dq),
                joint_ddq,
                np.zeros(6, dtype=np.float64)
            ).tolist()

    def rescale_speed(self, new_scale: float, update_joint_dq: bool=True, update_joint_tau: bool=True):
        """Rescale the speed of the trajectory by a factor of `new_scale`"""
        for frame in self.frames:
            frame.timestamp = frame.timestamp / new_scale
        if update_joint_dq:
            self.update_joint_dq()
        if update_joint_tau:
            assert update_joint_dq, "update_joint_tau requires update_joint_dq to be True"
            self.update_joint_tau()
        self.update_np_arrays()

    def time_shift(self, time_offset: float, inplace: bool=True, update_frames = True):
        """Shift the time of the trajectory by `time_offset`. To speedup, set update_frames to False"""
        if inplace:
            self.np_arrays["timestamps"] += time_offset
            if update_frames:
                self.update_frames()
            return self
        else:
            new_traj = Trajectory()
            new_traj.np_arrays = self.np_arrays.copy()
            new_traj.np_arrays["timestamps"] = self.np_arrays["timestamps"] + time_offset
            if update_frames:
                new_traj.update_frames()
            return new_traj

    def truncate(self, end_time: float, inplace: bool=True):
        """Truncate the current trajectory to the input end_time"""
        if inplace:
            for k, frame in enumerate(self.frames):
                if frame.timestamp > end_time:
                    self.frames = self.frames[:k]
                    self.update_np_arrays()
                    return self
        else:
            new_traj = Trajectory()
            for frame in self.frames:
                if frame.timestamp > end_time:
                    break
                new_traj.frames.append(frame)
            new_traj.update_np_arrays()
            return new_traj
        
    def calc_delay(
            self, 
            new_traj: "Trajectory", 
            time_precision: float = 0.002, 
            delay_min: float = -0.1,
            delay_max: float = 0.1,
            attr_name:str="joint_q"
        ):
        """Calculate the delay off the new_traj with respect to the current trajectory. 
        Will find the time offset that minimizes the difference of `attr_name` of the two trajectories."""
        self.init_interp_function()
        time_offsets = np.arange(delay_min, delay_max, time_precision)
        diffs = np.zeros_like(time_offsets)
        for k, time_offset in enumerate(time_offsets):
            # New trajectory should be shifted to the left if offset > 0
            new_traj_with_offset = new_traj.time_shift(-time_offset, inplace=False, update_frames=False)
            new_timestamps = new_traj_with_offset.np_arrays["timestamps"].tolist()
            interp_traj = self.interp_traj(new_timestamps, attr_names=[attr_name], update_frames=False)
            diff_sum = np.sum(interp_traj.calc_difference_norm(new_traj_with_offset, attr_name))
            diffs[k] = diff_sum
            # print(f"Time offset {time_offset:.5f}s, diff {diff_sum:.5f}")

        return time_offsets[np.argmin(diffs)]
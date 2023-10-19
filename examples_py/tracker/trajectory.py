import numpy as np
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
import unitree_arm_interface as sdk

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

    def update_joint_dq(self, padding: int=5):
        """Calculate average speed in [k-padding, k+padding] for each joint and update in self.frames[k].joint_dq"""
        for k, frame in enumerate(self.frames):
            if k < padding or k >= len(self.frames) - padding:
                continue
            prev_frame = self.frames[k-padding]
            next_frame = self.frames[k+padding]
            frame.joint_dq = (np.array(next_frame.joint_q) - np.array(prev_frame.joint_q))/(next_frame.timestamp - prev_frame.timestamp)

    def update_joint_tau(self, arm: sdk.ArmInterface, padding: int=1):
        """Calculate average torque in [k-padding, k+padding] for each joint and update in self.frames[k].joint_tau"""
        for k, frame in enumerate(self.frames):
            if k < padding or k >= len(self.frames) - padding:
                continue
            prev_frame = self.frames[k-padding]
            next_frame = self.frames[k+padding]
            joint_ddq = (np.array(next_frame.joint_dq) - np.array(prev_frame.joint_dq))/(next_frame.timestamp - prev_frame.timestamp)
            frame.joint_tau = arm._ctrlComp.armModel.inverseDynamics(
                np.array(frame.joint_q),
                np.array(frame.joint_dq),
                joint_ddq,
                np.zeros(6, dtype=np.float64)
            ).tolist()

import numpy as np
import numpy.typing as npt
import json
from typing import List, Dict, Optional, Tuple, cast
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
import unitree_arm_interface as sdk
from tqdm import tqdm
import time
from sklearn.linear_model import LinearRegression
import warnings


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

    LIST_ATTRS = {
        "joint_q": 6,
        "joint_dq": 6,
        "joint_tau": 6,
        "ee_posture": 6,
        "gripper_q": 1,
        "gripper_dq": 1,
        "gripper_tau": 1,
    }
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
        self,
        frames: Optional[List[Frame]] = None,
        file_name: Optional[str] = None,
        timestamps: Optional[npt.NDArray[np.float64]] = None,
    ):
        self.np_arrays: Dict[str, npt.NDArray[np.float64]] = {}

        if frames is not None:
            frame_num = len(frames)
            self.np_arrays["timestamps"] = np.zeros(frame_num, dtype=np.float64)
            for attr_name in Frame.LIST_ATTRS:
                self.np_arrays[attr_name] = np.zeros(
                    (frame_num, Frame.LIST_ATTRS[attr_name]), dtype=np.float64
                )
            for k, frame in enumerate(frames):
                self.np_arrays["timestamps"][k] = frame.timestamp
                for attr_name in Frame.LIST_ATTRS:
                    self.np_arrays[attr_name][k] = np.array(getattr(frame, attr_name))

        elif file_name is not None:
            with open(file_name, "r") as f:
                json_data = json.load(f)
                frame_num = len(json_data)
                self.np_arrays["timestamps"] = np.zeros(frame_num, dtype=np.float64)
                for attr_name in Frame.LIST_ATTRS:
                    self.np_arrays[attr_name] = np.zeros(
                        (frame_num, Frame.LIST_ATTRS[attr_name]), dtype=np.float64
                    )
                for k, frame in enumerate(json_data):
                    self.np_arrays["timestamps"][k] = frame["timestamp"]
                    for attr_name in Frame.LIST_ATTRS:
                        self.np_arrays[attr_name][k] = np.array(frame[attr_name])

        elif timestamps is not None:
            self.np_arrays["timestamps"] = timestamps
            for attr_name in Frame.LIST_ATTRS:
                self.np_arrays[attr_name] = np.zeros(
                    (timestamps.shape[0], Frame.LIST_ATTRS[attr_name]), dtype=np.float64
                )

        self.interp_functions: Dict[str, List[interp1d]] = {}

        """For faster computation as a whole trajectory. Values should be kept the same as self.frames. 
        Keys includes `timestamps` and all attributes in `Frame.LIST_ATTRS`"""

    def copy(self):
        """Return a copy of the current trajectory"""
        new_traj = Trajectory()
        for key, val in self.np_arrays.items():
            new_traj.np_arrays[key] = val.copy()
        for key, val in self.interp_functions.items():
            new_traj.interp_functions[key] = val.copy()
        return new_traj

    def __getitem__(self, idx: int):
        if idx == -1:
            idx = self.np_arrays["timestamps"].shape[0] - 1
        if idx < 0 or idx >= self.np_arrays["timestamps"].shape[0]:
            raise ValueError(
                f"{idx} out of range. maximum: {self.np_arrays['timestamps'].shape[0]-1}"
            )
        attr_dict = {}
        for attr_name in Frame.LIST_ATTRS:
            attr_dict[attr_name] = self.np_arrays[attr_name][idx]
        frame = Frame(self.np_arrays["timestamps"][idx], **attr_dict)
        return frame

    def save_frames(self, file_name: str):
        with open(file_name, "w") as f:
            frame_dicts = []
            for k in range(self.np_arrays["timestamps"].shape[0]):
                frame_dict = {}
                frame_dict["timestamp"] = self.np_arrays["timestamps"][k]
                for attr_name in Frame.LIST_ATTRS:
                    frame_dict[attr_name] = self.np_arrays[attr_name][k].tolist()
                frame_dicts.append(frame_dict)
            json.dump(frame_dicts, f)

    def is_initialized(self, attr_name):
        assert attr_name in Frame.LIST_ATTRS
        return self.np_arrays[attr_name].any()

    def interp_traj(
        self,
        new_timestamps: npt.NDArray[np.float64],
        attr_names: List[str] = list(Frame.LIST_ATTRS.keys()),
    ):
        """This method will interpolate a new trajectory, using the postures and the joint states from
        the reference trajectory, and query at the input timestamps."""

        if not self.interp_functions:
            self.init_interp_function()

        # Create a new trajectory
        new_traj = Trajectory()
        new_traj.np_arrays["timestamps"] = new_timestamps.copy()
        # Interpolate all attributes
        for attr_name in attr_names:
            ref_val = self.np_arrays[attr_name]
            dim = ref_val.shape[1]
            interp_val = np.zeros((new_timestamps.shape[0], dim), dtype=np.float64)
            for k in range(dim):
                interp_val[:, k] = self.interp_functions[attr_name][k](new_timestamps)
            new_traj.np_arrays[attr_name] = interp_val.copy()
        return new_traj

    def init_interp_function(self):
        assert (
            "timestamps" in self.np_arrays and self.np_arrays["timestamps"].shape
        ), "self.np_arrays['timestamps'] is empty! Please initialize the trajectory."
        timestamps = self.np_arrays["timestamps"].copy()
        for attr_name in Frame.LIST_ATTRS:
            dim = self.np_arrays[attr_name].shape[1]
            self.interp_functions[attr_name] = []
            for k in range(dim):
                vals = self.np_arrays[attr_name][:, k].squeeze()
                fill_value = "extrapolate"
                fill_value = cast(float, fill_value)  # To make mypy happy
                f_k = interp1d(
                    timestamps,
                    vals,
                    bounds_error=False,
                    fill_value=fill_value,
                    assume_sorted=True,
                )
                self.interp_functions[attr_name].append(f_k)

    def interp_single_frame(
        self,
        new_timestamp: float,
        next_frame_idx: Optional[int] = None,
        method: str = "linear",
        interp_attrs: List[str] = list(Frame.LIST_ATTRS.keys()),
    ):
        """next_frame_idx is the frame number that is the first to have a larger timestamp than new_timestamp.
        Will return the self.np_arrays["timestamps"][0] if next_frame_idx = 0
        Will return the self.np_arrays["timestamps"][-1] if next_frame_idx >= len(self.np_arrays["timestamps"])
        """

        if next_frame_idx is None:
            next_frame_idx = np.searchsorted(
                self.np_arrays["timestamps"], new_timestamp
            ).item()

        if next_frame_idx == 0:
            return self[0]
        elif next_frame_idx >= len(self.np_arrays["timestamps"]):
            return self[-1]
        else:
            assert (
                self.np_arrays["timestamps"][next_frame_idx - 1]
                <= new_timestamp
                <= self.np_arrays["timestamps"][next_frame_idx]
            ), "Wrong next_frame_idx"

            if method == "linear":
                prev_ratio = (
                    new_timestamp - self.np_arrays["timestamps"][next_frame_idx - 1]
                ) / (
                    self.np_arrays["timestamps"][next_frame_idx]
                    - self.np_arrays["timestamps"][next_frame_idx - 1]
                )
                new_frame_dict: Dict[str, List[float]] = {}
                for attr_name in interp_attrs:
                    prev_val = self.np_arrays[attr_name][next_frame_idx - 1]
                    next_val = self.np_arrays[attr_name][next_frame_idx]
                    new_frame_dict[attr_name] = (
                        prev_val * prev_ratio + next_val * (1 - prev_ratio)
                    ).tolist()
                new_frame = Frame(new_timestamp, **new_frame_dict)

            elif method == "scipy":
                raise NotImplementedError(
                    "Computation speed is too slow. Need to be optimized"
                )
                start_time = time.monotonic()
                if not self.interp_functions:
                    self.init_interp_function()

                new_frame = Frame(new_timestamp)
                for attr_name in interp_attrs:
                    dim = len(getattr(new_frame, attr_name))
                    new_val: List[float] = []
                    for k in range(dim):
                        new_val.append(
                            self.interp_functions[attr_name][k](new_timestamp)
                        )
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
        assert len(self.np_arrays["timestamps"]) == len(
            new_traj.np_arrays["timestamps"]
        ), "The two trajectories should have the same length"
        diff_traj = Trajectory()
        assert np.all(
            self.np_arrays["timestamps"] == new_traj.np_arrays["timestamps"]
        ), "The two trajectories should have the same timestamps"
        diff_traj.np_arrays["timestamps"] = self.np_arrays["timestamps"].copy()
        for attr_name in Frame.LIST_ATTRS:
            diff_traj.np_arrays[attr_name] = (
                new_traj.np_arrays[attr_name] - self.np_arrays[attr_name]
            )
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

        assert all(
            self.np_arrays["timestamps"] == new_traj.np_arrays["timestamps"]
        ), "The two trajectories should have the same timestamps"
        results = np.linalg.norm(
            new_traj.np_arrays[attr_name] - self.np_arrays[attr_name], axis=1
        )

        return results

    def calc_local_variance(self, padding: int = 5):
        """Calculate noise of the trajectory through calculate average variance in a neighborhood around each element."""
        avg_var = Frame(0)
        for attr_name in Frame.LIST_ATTRS:
            val_matrix = self.np_arrays[attr_name]
            var_matrix = np.zeros_like(val_matrix, dtype=np.float64)
            for k in range(var_matrix.shape[0]):
                # Calculate the variance between [k-padding, k+padding] in each column
                var_matrix[k] = np.std(
                    val_matrix[
                        max(0, k - padding) : min(k + padding, var_matrix.shape[0] - 1)
                    ],
                    axis=0,
                )
            setattr(avg_var, attr_name, np.mean(var_matrix, axis=0).tolist())

        return avg_var

    def plot_attr(
        self, attr_name, ax: Optional[Axes] = None, title: Optional[str] = None
    ):
        assert attr_name in Frame.LIST_ATTRS
        timestamps = self.np_arrays["timestamps"].copy()
        vals = self.np_arrays[attr_name]
        dim = vals.shape[1]
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

    def update_joint_dq(self, window_width: float = 0.1, method: str = "linfit"):
        """Calculate average speed in [timestamp-window_width/2, timestamp+window_width/2] for each joint and update in self.np_arrays["joint_dq"]"""

        assert method in ["linfit", "diff"]

        for k in range(self.np_arrays["timestamps"].shape[0]):
            frame_nums = np.argwhere(
                np.logical_and(
                    self.np_arrays["timestamps"]
                    >= self.np_arrays["timestamps"][k] - window_width / 2,
                    self.np_arrays["timestamps"]
                    <= self.np_arrays["timestamps"][k] + window_width / 2,
                )
            ).squeeze()
            assert frame_nums.shape[0] > 0, "No frame found in the neighborhood"
            start_frame = min(frame_nums)
            end_frame = max(frame_nums)

            if method == "diff":
                self.np_arrays["joint_dq"][k] = (
                    self.np_arrays["joint_q"][start_frame]
                    - self.np_arrays["joint_q"][end_frame]
                ) / (
                    self.np_arrays["timestamps"][start_frame]
                    - self.np_arrays["timestamps"][end_frame]
                )
            elif method == "linfit":
                # Use linear fit between [start_frame, end_frame] to calculate the speed
                reg = LinearRegression().fit(
                    self.np_arrays["timestamps"][
                        start_frame : end_frame + 1, np.newaxis
                    ],
                    self.np_arrays["joint_q"][start_frame : end_frame + 1],
                )
                self.np_arrays["joint_dq"][k] = reg.coef_.squeeze()

    def apply_moving_average(self, attr_names: List[str], window_width: float = 0.2):
        """Apply np.convolve to smoothen the attribute in place"""
        for attr_name in attr_names:
            assert attr_name in Frame.LIST_ATTRS

            val_matrix = self.np_arrays[attr_name].copy()
            # Apply padding
            avg_period = (
                self.np_arrays["timestamps"][-1] - self.np_arrays["timestamps"][0]
            ) / self.np_arrays["timestamps"].shape[0]
            padding_num = int(window_width / avg_period)
            padding_up = np.ones((padding_num, 1)) @ val_matrix[0].reshape((1, -1))
            padding_down = np.ones((padding_num, 1)) @ val_matrix[-1].reshape((1, -1))
            val_matrix_with_padding = np.concatenate(
                [padding_up, val_matrix, padding_down], axis=0
            )
            self.np_arrays[attr_name] = convolve2d(
                val_matrix_with_padding, np.ones((padding_num * 2 + 1, 1)), mode="valid"
            ) / (2 * padding_num + 1)
            assert (
                self.np_arrays[attr_name].shape == val_matrix.shape
            ), "Shape mismatch after moving average"

    def smoothen_joint_dq_start_end(
        self, window_width: float = 0.5, method: str = "linear"
    ):
        """Smoothen the joint_dq both at the start and the end of the trajectory (first window_width seconds)
        by applying a filter function directly to the first few frames.,
        so that the speed is continuous at the start"""

        # Start

        frame_nums = np.searchsorted(self.np_arrays["timestamps"], window_width)
        assert frame_nums > 0, "No frame found in the first window_width seconds"
        if method == "linear":
            coeff = self.np_arrays["timestamps"][:frame_nums] / window_width
            self.np_arrays["joint_dq"][:frame_nums] = (
                self.np_arrays["joint_dq"][:frame_nums] * coeff[:, np.newaxis]
            )
        elif method == "quadratic":
            coeff = (self.np_arrays["timestamps"][:frame_nums] / window_width) ** 2
            self.np_arrays["joint_dq"][:frame_nums] = (
                self.np_arrays["joint_dq"][:frame_nums] * coeff[:, np.newaxis]
            )

        # End
        final_timestamp = self.np_arrays["timestamps"][-1]
        frame_num_end = np.searchsorted(
            self.np_arrays["timestamps"], final_timestamp - window_width, side="right"
        )
        assert (
            self.np_arrays["timestamps"].shape[0] - frame_num_end > 0
        ), "No frame found in the last window_width seconds"

        if method == "linear":
            coeff = (
                final_timestamp - self.np_arrays["timestamps"][frame_num_end:]
            ) / window_width
            self.np_arrays["joint_dq"][frame_num_end:] = (
                self.np_arrays["joint_dq"][frame_num_end:] * coeff[:, np.newaxis]
            )
        elif method == "quadratic":
            coeff = (
                (final_timestamp - self.np_arrays["timestamps"][frame_num_end:])
                / window_width
            ) ** 2
            self.np_arrays["joint_dq"][frame_num_end:] = (
                self.np_arrays["joint_dq"][frame_num_end:] * coeff[:, np.newaxis]
            )

    def update_joint_tau(self, window_width: float = 0.1):
        """Calculate average torque in [timestamp-window_width/2, timestamp+window_width/2]
        for each joint and update in self.np_arrays["joint_tau"]"""

        arm = sdk.ArmInterface(hasGripper=True)

        for k in range(self.np_arrays["timestamps"].shape[0]):
            frame_nums = np.argwhere(
                np.logical_and(
                    self.np_arrays["timestamps"]
                    >= self.np_arrays["timestamps"][k] - window_width / 2,
                    self.np_arrays["timestamps"]
                    <= self.np_arrays["timestamps"][k] + window_width / 2,
                )
            ).squeeze()
            assert frame_nums.shape[0] > 0, "No frame found in the neighborhood"
            start_frame = min(frame_nums)
            end_frame = max(frame_nums)

            joint_ddq = (
                self.np_arrays["joint_dq"][start_frame]
                - self.np_arrays["joint_dq"][end_frame]
            ) / (
                self.np_arrays["timestamps"][start_frame]
                - self.np_arrays["timestamps"][end_frame]
            )

            # z1 arm inverse dynamics
            self.np_arrays["joint_tau"][k] = arm._ctrlComp.armModel.inverseDynamics(
                self.np_arrays["joint_q"][k],
                self.np_arrays["joint_dq"][k],
                joint_ddq,
                np.zeros(6, dtype=np.float64),
            )

    def rescale_speed(
        self,
        new_scale: float,
        update_joint_dq: bool = True,
        update_joint_tau: bool = True,
    ):
        """Rescale the speed of the trajectory by a factor of `new_scale`"""

        self.np_arrays["timestamps"] = self.np_arrays["timestamps"] / new_scale
        if update_joint_dq:
            self.update_joint_dq()
            self.smoothen_joint_dq_start_end()
        if update_joint_tau:
            assert (
                update_joint_dq
            ), "update_joint_tau requires update_joint_dq to be True"
            self.update_joint_tau()

    def time_shift(self, time_offset: float, inplace: bool = True):
        """Shift the time of the trajectory by `time_offset`."""
        if inplace:
            self.np_arrays["timestamps"] += time_offset
            return self
        else:
            new_traj = Trajectory()
            new_traj.np_arrays = self.np_arrays.copy()
            new_traj.np_arrays["timestamps"] = (
                self.np_arrays["timestamps"] + time_offset
            )
            return new_traj

    def truncate(self, end_time: float, inplace: bool = True):
        """Truncate the current trajectory to the input end_time"""
        if inplace:
            next_frame_id = np.argmax(self.np_arrays["timestamps"] > end_time)
            if next_frame_id == 0:
                # All timestamps <= end_time, No need to truncate
                return self
            else:
                self.np_arrays["timestamps"] = self.np_arrays["timestamps"][
                    :next_frame_id
                ]
                for attr_name in Frame.LIST_ATTRS:
                    self.np_arrays[attr_name] = self.np_arrays[attr_name][
                        :next_frame_id
                    ]
                return self
        else:
            new_traj = Trajectory()
            next_frame_id = np.argmax(self.np_arrays["timestamps"] > end_time)
            if next_frame_id == 0:
                new_traj.np_arrays = self.np_arrays.copy()
            else:
                new_traj.np_arrays["timestamps"] = self.np_arrays["timestamps"][
                    :next_frame_id
                ]
                for attr_name in Frame.LIST_ATTRS:
                    new_traj.np_arrays[attr_name] = self.np_arrays[attr_name][
                        :next_frame_id
                    ]
            return new_traj

    def calc_delay(
        self,
        new_traj: "Trajectory",
        time_precision: float = 0.001,
        delay_min: float = 0,
        delay_max: float = 0.5,
        attr_name: str = "joint_q",
    ) -> Tuple[np.float64, npt.NDArray[np.float64]]:
        """Calculate the delay off the new_traj with respect to the current trajectory.
        Will find the time offset that minimizes the difference of `attr_name` of the two trajectories.
        Return the time offset for minimized absolute difference and the joint-wise time offsets.
        """
        self.init_interp_function()
        time_offsets = np.arange(delay_min, delay_max, time_precision)
        avg_diffs = np.zeros_like(time_offsets, dtype=np.float64)
        joint_diffs = np.zeros(
            (time_offsets.shape[0], self.np_arrays[attr_name].shape[1]),
            dtype=np.float64,
        )
        for k, time_offset in enumerate(time_offsets):
            # New trajectory should be shifted to the left if offset > 0
            new_traj_with_offset = new_traj.time_shift(-time_offset, inplace=False)
            new_timestamps = new_traj_with_offset.np_arrays["timestamps"].copy()
            interp_traj = self.interp_traj(new_timestamps, attr_names=[attr_name])

            avg_diffs[k] = np.sum(
                interp_traj.calc_difference_norm(new_traj_with_offset, attr_name)
            )
            shifted_vals = new_traj_with_offset.np_arrays[attr_name]
            interp_vals = interp_traj.np_arrays[attr_name]
            joint_diffs[k] = np.sum(np.abs(shifted_vals - interp_vals), axis=0)

        avg_minimum_id = np.argmin(avg_diffs).item()
        joint_minimum_id = np.argmin(joint_diffs, axis=0)
        if time_offsets[avg_minimum_id] + time_precision >= delay_max:
            warnings.warn(
                f"trajectory.calc_delay: delay exceeds {delay_max}. Please increase delay_max."
            )
        return time_offsets[avg_minimum_id], time_offsets[joint_minimum_id]

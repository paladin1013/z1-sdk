from matplotlib.axes import Axes
from matplotlib.figure import Figure
import unitree_arm_interface as sdk
from typing import Generator, List, Optional, Tuple, cast
import numpy as np
import numpy.typing as npt
import os
from .pose_tracker import PoseTracker
from .trajectory import Trajectory
import matplotlib.pyplot as plt


class ExperimentRunner:
    """A wrapper to run multiple experiments and analyze results to optimize parameters"""

    def __init__(
        self,
        teleop_dt: float,
        track_dt: float,
        demo_duration: float,
        replay_duration: float,
        trial_id: int,
        replay_speeds: List[float],
        stiffnesses: List[float],
    ):
        self.teleop_dt = teleop_dt
        self.track_dt = track_dt
        self.trial_id = trial_id
        self.demo_duration = demo_duration
        self.replay_duration = replay_duration
        self.data_dir = (
            f"logs/trajectories/duration{demo_duration}_dt{track_dt}_trial{trial_id}"
        )
        self.replay_speeds = replay_speeds
        self.stiffnesses = stiffnesses

    def record_teleop_demo(self):
        arm = sdk.ArmInterface(hasGripper=True)
        pt = PoseTracker(arm, teleop_dt=self.teleop_dt, track_dt=self.track_dt)
        pt.start_teleop_tracking(self.demo_duration)
        os.makedirs(self.data_dir)
        pt.tracked_traj.save_frames(f"{self.data_dir}/teleop.json")

    def sweep_params(self):
        """Will sweep parameters in self.replay_speeds and self.stiffnesses and record trajectories"""

        teleop_traj = Trajectory(file_name=f"{self.data_dir}/teleop.json")
        for replay_speed in self.replay_speeds:
            reference_file = f"{self.data_dir}/duration{self.replay_duration}_speed{replay_speed:.1f}_reference.json"
            reference_traj = teleop_traj.copy()
            reference_traj.rescale_speed(replay_speed)
            reference_traj.apply_moving_average(["joint_dq"])
            reference_traj.truncate(self.replay_duration)
            reference_traj.save_frames(reference_file)

            for stiffness in self.stiffnesses:
                replay_file = f"{self.data_dir}/duration{self.replay_duration}_speed{replay_speed:.1f}_stiffness{stiffness:.1f}_replay.json"
                arm = sdk.ArmInterface(hasGripper=True)
                pt = PoseTracker(
                    arm,
                    teleop_dt=self.teleop_dt,
                    track_dt=self.track_dt,
                    stiffness=stiffness,
                )

                print(
                    f"Start trajectory replaying: speed {replay_speed:.1f}, stiffness {stiffness:.1f}"
                )
                pt.replay_traj(reference_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
                pt.tracked_traj.save_frames(replay_file)

    def load_experiment_results(self):
        """Generate a list of tuple including (replay_speed, stiffness, reference_traj, replay_traj)"""
        for replay_speed in self.replay_speeds:
            reference_file = f"{self.data_dir}/duration{self.replay_duration}_speed{replay_speed:.1f}_reference.json"
            reference_traj = Trajectory(file_name=reference_file)

            for stiffness in self.stiffnesses:
                replay_file = f"{self.data_dir}/duration{self.replay_duration}_speed{replay_speed:.1f}_stiffness{stiffness:.1f}_replay.json"
                replay_traj = Trajectory(file_name=replay_file)
                yield replay_speed, stiffness, reference_traj, replay_traj

    def analyze_delay(self):
        """After sweeping parameters, analyze the delay of each trajectory. Including average delay and delay of each joints"""

        for (
            replay_speed,
            stiffness,
            reference_traj,
            replay_traj,
        ) in self.load_experiment_results():
            # Compare reference traj with replay traj
            avg_delay, joint_wise_delay = reference_traj.calc_delay(replay_traj)
            joint_delay_str = ", ".join([f"{delay:.3f}" for delay in joint_wise_delay])
            print(
                f"Speed: {replay_speed:.1f} Stiffness: {stiffness: .1f} Average delay: {avg_delay:.3f}, Joint-wise delay: {joint_delay_str}"
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
            tracked_traj.np_arrays["timestamps"]
        )
        diff_traj = reference_traj.calc_difference(tracked_traj)
        if fig is None:
            fig = plt.figure(figsize=(24, 16))
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

    def analyze_precision(self):
        """After sweeping parameters, analyze the precision of each trajectory."""

        for (
            replay_speed,
            stiffness,
            reference_traj,
            replay_traj,
        ) in self.load_experiment_results():
            # Compare reference traj with replay traj
            avg_delay, joint_wise_delay = reference_traj.calc_delay(replay_traj)
            replay_traj.time_shift(float(-avg_delay))
            diff_traj, fig = self.compare_traj(reference_traj, replay_traj)

            joint_wise_diff = np.sum(
                np.abs(diff_traj.np_arrays["joint_q"]), axis=0
            ) / len(diff_traj.np_arrays["joint_q"])
            avg_diff = np.mean(joint_wise_diff)
            joint_diff_str = ", ".join([f"{diff:.4f}" for diff in joint_wise_diff])

            var_frame = replay_traj.calc_local_variance()

            avg_dq_noise = np.mean(var_frame.joint_dq)
            avg_tau_noise = np.mean(var_frame.joint_tau)

            print(
                f"Speed: {replay_speed:.1f} Stiffness: {stiffness: .1f} Delay: {avg_delay:.3f} Average difference: {avg_diff:.4f} \
dq var: {avg_dq_noise:.4f} tau var: {avg_tau_noise:.4f}"
            )

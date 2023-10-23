import unitree_arm_interface as sdk
from typing import List
import numpy as np
import numpy.typing as npt
from .pose_tracker import PoseTracker
from .trajectory import Trajectory


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
        self.data_dir = f"logs/trajectories/duration{demo_duration}_dt{track_dt}_trial{trial_id}"
        self.replay_speeds = replay_speeds
        self.stiffnesses = stiffnesses

    
    def record_teleop_demo(self):
        arm = sdk.ArmInterface(hasGripper=True)
        pt = PoseTracker(arm, teleop_dt=self.teleop_dt, track_dt=self.track_dt)
        pt.start_teleop_tracking(self.demo_duration)
        pt.tracked_traj.save_frames(f"{self.data_dir}/teleop.json")


    def param_sweep(self):
        """Will sweep parameters in self.replay_speeds and self.stiffnesses and record trajectories"""

        teleop_traj = Trajectory(file_name=f"{self.data_dir}/teleop.json")
        for replay_speed in self.replay_speeds:
            reference_file = (
                f"{self.data_dir}/duration{self.replay_duration}_speed{replay_speed:.1f}_reference.json"
            )
            reference_traj = teleop_traj.copy()
            reference_traj.rescale_speed(replay_speed)
            reference_traj.truncate(self.replay_duration)
            reference_traj.save_frames(reference_file)

            for stiffness in self.stiffnesses:
                replay_file = (
                    f"{self.data_dir}/duration{self.replay_duration}_speed{replay_speed:.1f}_stiffness{stiffness:.1f}_replay.json"
                )
                arm = sdk.ArmInterface(hasGripper=True)
                pt = PoseTracker(arm, teleop_dt=self.teleop_dt, track_dt=self.track_dt, stiffness=stiffness)
                
                print(f"Start trajectory replaying: speed {replay_speed:.1f}, stiffness {stiffness:.1f}")
                pt.replay_traj(reference_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
                pt.tracked_traj.save_frames(replay_file)
            
        
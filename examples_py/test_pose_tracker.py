import unitree_arm_interface as sdk
import numpy as np
from tracker import PoseTracker, Trajectory, Frame
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    arm = sdk.ArmInterface(hasGripper=True)
    arm.setArmCmd(
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
    )

    duration = 10
    track_dt = 0.002
    trial = 2
    replay_speed = 0.2
    truncate_time = 10

    data_dir = f"logs/trajectories/duration{duration}_dt{track_dt}_trial{trial}"

    os.makedirs(data_dir, exist_ok=True)
    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt, stiffness=1)

    # pt.start_teleop_tracking(duration)
    # pt.tracked_traj.save_frames(f"{data_dir}/tracked.json")

    ## Update reference trajectory
    teleop_traj = Trajectory(
        file_name=f"{data_dir}/tracked.json"
    )  # Load tracked trajectory as reference
    teleop_traj.plot_attr("joint_dq", title="teleop trajectory")

    for replay_speed in [0.1, 0.2, 0.3, 0.5, 0.7, 1]:
        reference_file = (
            f"{data_dir}/speed{replay_speed}_truncate{truncate_time}_reference.json"
        )
        replay_file = (
            f"{data_dir}/speed{replay_speed}_truncate{truncate_time}_replay_lowcmd.json"
        )
        reference_traj = teleop_traj.copy()
        reference_traj.rescale_speed(replay_speed)
        reference_traj.truncate(truncate_time)
        reference_traj.save_frames(reference_file)
        pt.replay_traj(reference_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
        pt.tracked_traj.save_frames(replay_file)

    # ## Start replaying
    # pt.track_dt = 0.002
    # pt.stiffness = 1

    # replay_traj = Trajectory(file_name=replay_file)
    # replay_traj.plot_attr("joint_dq", title="replayed trajectory")
    # print(reference_traj.calc_delay(replay_traj, delay_min=0, delay_max=0.1, time_precision=0.001))
    # pt.compare_traj(reference_traj, replay_traj)
    # plt.show()

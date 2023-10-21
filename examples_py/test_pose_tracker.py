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
    trial = 1
    replay_speed = 0.2
    truncate_time = 10

    data_dir = f"logs/trajectories/duration{duration}_dt{track_dt}_trial{trial}"

    reference_file = f"{data_dir}/speed{replay_speed}_truncate{truncate_time}_reference.json"
    replay_file = f"{data_dir}/speed{replay_speed}_truncate{truncate_time}_replay_lowcmd.json"

    os.makedirs(data_dir, exist_ok=True)
    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt)

    # pt.start_teleop_tracking(duration)
    # pt.tracked_traj.save_frames(f"{data_dir}/tracked.json")
    
    ## Update reference trajectory
    teleop_traj = Trajectory(file_name=f"{data_dir}/tracked.json") # Load tracked trajectory as reference
    teleop_traj.plot_attr("joint_dq", title="teleop trajectory")
    reference_traj = teleop_traj
    reference_traj.rescale_speed(replay_speed)
    reference_traj.truncate(truncate_time)
    reference_traj.save_frames(reference_file)


    ## Start replaying
    pt.track_dt = 0.002
    pt.stiffness = 1
    
    # pt.replay_traj(reference_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
    # pt.tracked_traj.save_frames(replay_file)


    replay_traj = Trajectory(file_name=replay_file)
    # replay_traj.plot_attr("joint_dq", title="replayed trajectory")
    print(reference_traj.calc_delay(replay_traj, delay_min=0, delay_max=0.1, time_precision=0.001))
    # pt.compare_traj(reference_traj, replay_traj)
    # plt.show()

    # for stiffness in np.arange(0.2, 0.5, 0.1):
    #     replay_file_name = f"logs/trajectories/replay_duration{duration}_dt{track_dt}_stiffness{stiffness}.json"
    #     pt.stiffness = stiffness
    #     pt.replay_traj(ref_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
    #     pt.tracked_traj.save_frames(replay_file_name)


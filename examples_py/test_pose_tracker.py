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

    duration = 5
    track_dt = 0.002
    
    data_dir = f"logs/trajectories/duration{duration}_dt{track_dt}"
    os.makedirs(data_dir, exist_ok=True)
    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt)

    # pt.start_teleop_tracking(duration)
    # pt.tracked_traj.save_frames(f"{data_dir}/tracked.json")
    
    teleop_traj = Trajectory(file_name=f"{data_dir}/tracked.json") # Load tracked trajectory as reference

    teleop_traj.plot_attr("joint_dq", title="teleop trajectory")
    reference_traj = teleop_traj
    reference_traj.rescale_speed(0.1)
    reference_traj.truncate(10)
    # reference_traj.save_frames(f"{data_dir}/reference.json")

    # reference_traj.plot_attr("joint_dq", title="reference trajectory")

    pt.track_dt = 0.002
    # pt.replay_traj(reference_traj, ctrl_method=sdk.ArmFSMState.JOINTCTRL)
    # pt.tracked_traj.save_frames(f"{data_dir}/replay_joint_ctrl.json")
    # replay_traj = Trajectory(file_name=f"{data_dir}/replay_joint_ctrl.json")
    pt.stiffness = 1
    # pt.replay_traj(reference_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
    # pt.tracked_traj.save_frames(f"{data_dir}/replay_lowcmd.json")
    replay_traj = Trajectory(file_name=f"{data_dir}/replay_lowcmd.json")
    # replay_traj.plot_attr("joint_dq", title="replayed trajectory")
    print(reference_traj.calc_delay(replay_traj, offset_min=-0.15, offset_max=-0.05, time_precision=0.001))
    # pt.compare_traj(reference_traj, replay_traj)
    # plt.show()

    # for stiffness in np.arange(0.2, 0.5, 0.1):
    #     replay_file_name = f"logs/trajectories/replay_duration{duration}_dt{track_dt}_stiffness{stiffness}.json"
    #     pt.stiffness = stiffness
    #     pt.replay_traj(ref_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
    #     pt.tracked_traj.save_frames(replay_file_name)





    # replay_traj = Trajectory(file_name=replay_file_name)
    # replay_timestamps = [frame.timestamp for frame in replay_traj.frames]

    # interp_traj = ref_traj.interp_traj(replay_timestamps)
    # diff_traj = interp_traj.calc_difference(replay_traj)
    # ref_traj.plot_attr("joint_q")
    # replay_traj.plot_attr("joint_q")
    # diff_traj.plot_attr("joint_q")
    # plt.show()
    
    # teleop_traj = Trajectory(file_name=teleop_file_name)
    # replay_traj = Trajectory(file_name=replay_file_name)

    # print(teleop_traj.measure_noise().joint_dq)
    # # print(replay_traj.measure_noise())

    # teleop_traj.update_joint_dq()

    # print(teleop_traj.measure_noise().joint_dq)

    # teleop_traj.plot_attr("joint_tau")
    # filtered_traj.plot_attr("joint_tau")
    # plt.show()

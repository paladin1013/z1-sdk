import unitree_arm_interface as sdk
import numpy as np
from tracker import PoseTracker, Trajectory, Frame


if __name__ == "__main__":
    arm = sdk.ArmInterface(hasGripper=True)
    arm.setArmCmd(
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
    )

    duration = 10
    track_dt = 0.02
    stiffness = 0.3

    teleop_file_name = f"logs/trajectories/teleop_duration{duration}_dt{track_dt}.json"
    replay_file_name = f"logs/trajectories/replay_duration{duration}_dt{track_dt}_stiffness{stiffness}.json"
    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt, stiffness=stiffness)
    
    # pt.start_teleop_tracking(duration)
    # pt.tracked_traj.save_frames(teleop_file_name)

    ref_traj = Trajectory(file_name=teleop_file_name)

    ref_traj.update_joint_dq(padding=2)
    ref_traj.update_joint_tau(arm)
    pt.replay_traj(ref_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
    pt.tracked_traj.save_frames(replay_file_name)

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

import unitree_arm_interface as sdk
import numpy as np
from tracker import PoseTracker, Trajectory, Frame
import os
import matplotlib.pyplot as plt





if __name__ == "__main__":
    duration = 10
    track_dt = 0.002
    trial = 7
    replay_speed = 0.2
    truncate_time = 10

    data_dir = f"logs/trajectories/duration{duration}_dt{track_dt}_trial{trial}"

    os.makedirs(data_dir, exist_ok=True)

    arm = sdk.ArmInterface(hasGripper=True)

    pt = PoseTracker(arm, teleop_dt=0.02, track_dt=track_dt, stiffness=1)

    pt.start_teleop_tracking(duration)
    pt.tracked_traj.save_frames(f"{data_dir}/tracked.json")

    ## Update reference trajectory
    teleop_traj = Trajectory(
        file_name=f"{data_dir}/tracked.json"
    )  # Load tracked trajectory as reference
    teleop_traj.plot_attr("joint_dq", title="teleop trajectory")

    replay_speed = 0.5
    reference_file = (
        f"{data_dir}/speed{replay_speed:.1f}_truncate{truncate_time}_reference.json"
    )
    reference_traj = teleop_traj.copy()
    reference_traj.rescale_speed(replay_speed)
    print("finish rescaling speed")
    reference_traj.truncate(truncate_time)
    reference_traj.plot_attr("joint_q", title=f"replay speed {replay_speed}")
    reference_traj.plot_attr("joint_dq", title=f"replay speed {replay_speed}")
    reference_traj.save_frames(reference_file)

    reference_traj = Trajectory(file_name=reference_file)

    # for stiffness in np.arange(0.2, 1, 0.1):
    #     arm = sdk.ArmInterface(hasGripper=True)
    #     pt.arm = arm
    #     pt.stiffness = stiffness
    #     # Warning!!! For some reasons, the final state in the trajectory of low level command is keeped in the sdk object
    #     # If one have used the low level command and then call arm.backToStart(), even if the arm is reset to the home position,
    #     # the last state in the low level command trajectory will be kept and the arm will move to that state immediately
    #     # once arm.setFsm(sdk.ArmFSMState.LOWCMD) or arm.setFsmLowCmd() is called.
    #     # To avoid this, please instantiate a new sdk interface object every time before running in low level command.
    #     # TODO: find out some other ways to fix this problem
    #     replay_file = f"{data_dir}/speed{replay_speed:.1f}_truncate{truncate_time}_replay_lowcmd_stiffness{stiffness:.1f}.json"


    #     print(f"finish saving reference trajectory for speed{replay_speed:.1f} stiffness{stiffness:.1f}")
    #     pt.replay_traj(reference_traj, ctrl_method=sdk.ArmFSMState.LOWCMD)
    #     pt.tracked_traj.save_frames(replay_file)

    for stiffness in np.arange(0.2, 1, 0.1):
        reference_file = (
            f"{data_dir}/speed{replay_speed:.1f}_truncate{truncate_time}_reference.json"
        )
        replay_file = f"{data_dir}/speed{replay_speed:.1f}_truncate{truncate_time}_replay_lowcmd_stiffness{stiffness:.1f}.json"
        reference_traj = Trajectory(file_name=reference_file)
        replay_traj = Trajectory(file_name=replay_file)

        # interp_traj = replay_traj.interp_traj(reference_traj.np_arrays["timestamps"].tolist())
        # diff_traj = reference_traj.calc_difference(interp_traj)

        avg_delay = reference_traj.calc_delay(replay_traj)
        joint_delays = [reference_traj.calc_delay(replay_traj, specify_joint=k) for k in range(6)]
        joint_results = ' '.join([f"{delay:.3f}" for delay in joint_delays])
        print(
            f"Stiffness: {stiffness:.1f} Avg delay: {avg_delay:.3f} Joint {joint_results}"
        )

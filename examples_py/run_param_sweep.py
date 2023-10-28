from tracker import ExperimentRunner
import matplotlib.pyplot as plt
import unitree_arm_interface as sdk

if __name__ == "__main__":
    runner = ExperimentRunner(
        teleop_dt=0.02,
        track_dt=0.002,
        demo_duration=10,
        replay_duration=10,
        trial_id=5,
        # replay_speeds=[0.2, 0.5, 1.0],
        replay_speeds=[1],
        stiffnesses=[1]
        # kp_scales=[1],
        # kd_scales=[0.5, 0.75, 1.0, 1.5]
    )
    # runner.record_teleop_demo()
    # runner.sweep_params()

    # runner.analyze_precision()

    steps = [
        # ([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0),
        # ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0),
        ([0.0, 0.0, -1.25, 0.0, 0.0, 0.0], 5.0),
        ([0.0, 1.25, -1.25, 0.0, 0.0, 0.0], 5.0),
        ([0.0, 0.0, -1.25, 0.0, 0.0, 0.0], 5.0),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0),
    ]
    runner.joint_movement_test(steps, ctrl_method=sdk.ArmFSMState.JOINTCTRL)
    runner.joint_movement_analysis(steps, ctrl_method=sdk.ArmFSMState.JOINTCTRL)
    runner.joint_movement_test(steps, ctrl_method=sdk.ArmFSMState.LOWCMD)
    runner.joint_movement_analysis(steps, ctrl_method=sdk.ArmFSMState.LOWCMD)

    plt.show()

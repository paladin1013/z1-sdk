from tracker import ExperimentRunner
import matplotlib.pyplot as plt
if __name__ == "__main__":
    runner = ExperimentRunner(
        teleop_dt=0.02,
        track_dt=0.002,
        demo_duration=10,
        replay_duration=20,
        trial_id=2,
        # replay_speeds=[0.2, 0.5, 1.0],
        replay_speeds=[0.5],
        stiffnesses=[0.5, 1, 1.5]
        # kp_scales=[1],
        # kd_scales=[0.5, 0.75, 1.0, 1.5]
    )
    # runner.record_teleop_demo()
    # runner.sweep_params()
    
    runner.analyze_precision()
    plt.show()
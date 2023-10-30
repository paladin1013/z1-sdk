from typing import List, Tuple
from tracker import ExperimentRunner, PoseTracker
import matplotlib.pyplot as plt
import unitree_arm_interface as sdk
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="teleop_replay", version_base="1.2")
def run_teleop_replay(conf: OmegaConf):
    runner: ExperimentRunner = hydra.utils.instantiate(conf.runner) # type: ignore
    runner.record_teleop_demo(conf.teleop_duration) # type: ignore
    runner.replay_teleop_demo(conf.replay_duration, conf.replay_speed) # type: ignore
    runner.analyze_replay_precision()
    plt.show()

@hydra.main(config_path="config", config_name="joint_movement", version_base="1.2")
def run_joint_movement(conf: OmegaConf):
    runner: ExperimentRunner = hydra.utils.instantiate(conf.runner) # type: ignore
    steps:List[Tuple[List[float], float]] = []
    for step in conf.steps: # type: ignore
        steps.append((list(step.joint_pos), step.duration)) # type: ignore
    if conf.ctrl_method == "jointctrl": # type: ignore
        ctrl_method = sdk.ArmFSMState.JOINTCTRL
    elif conf.ctrl_method == "lowcmd": # type: ignore
        ctrl_method = sdk.ArmFSMState.LOWCMD
    runner.joint_movement_test(steps, ctrl_method) # type: ignore
    runner.joint_movement_analysis(steps, ctrl_method) # type: ignore
    plt.show()


if __name__ == "__main__":
    # run_joint_movement()
    run_teleop_replay()
    

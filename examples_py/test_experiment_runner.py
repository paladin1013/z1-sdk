from tracker import ExperimentRunner, PoseTracker
import matplotlib.pyplot as plt
import unitree_arm_interface as sdk
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="teleop_replay", version_base="1.2")
def main(conf: OmegaConf):
    runner: ExperimentRunner = hydra.utils.instantiate(conf)
    # runner.record_teleop_demo()
    runner.replay_teleop_demo()
    runner.analyze_replay_precision()
    plt.show()


if __name__ == "__main__":
    main()
    # runner.record_teleop_demo()
    # runner.sweep_params()
    # runner.analyze_precision()
    
    # steps = [
    #     # ([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0),
    #     # ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0),
    #     ([0.0, 0.0, -1.25, 0.0, 0.0, 0.0], 5.0),
    #     ([0.0, 1.25, -1.25, 0.0, 0.0, 0.0], 5.0),
    #     ([0.0, 0.0, -1.25, 0.0, 0.0, 0.0], 5.0),
    #     ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0),
    # ]
    # runner.joint_movement_test(steps, ctrl_method=sdk.ArmFSMState.JOINTCTRL)
    # runner.joint_movement_analysis(steps, ctrl_method=sdk.ArmFSMState.JOINTCTRL)
    # runner.joint_movement_test(steps, ctrl_method=sdk.ArmFSMState.LOWCMD)
    # runner.joint_movement_analysis(steps, ctrl_method=sdk.ArmFSMState.LOWCMD)
    # runner.joint_movement_replay(steps, file_prefix="jointctrl")
    # runner.joint_movement_analysis(steps, ctrl_method=sdk.ArmFSMState.LOWCMD, file_prefix="replay")
    # runner.joint_movement_replay(steps, file_prefix="lowcmd_reference")
    # runner.joint_movement_analysis(steps, ctrl_method=sdk.ArmFSMState.LOWCMD, file_prefix="replay")
    # runner.joint_movement_id_analysis(steps, ctrl_method=sdk.ArmFSMState.LOWCMD)   
    

    # plt.show()

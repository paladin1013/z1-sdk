from typing import Any, List, Tuple, overload
from enum import Enum
import numpy as np
import numpy.typing as npt

def homoToPosture(homoMat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """convert homogeneous matrix (4*4) to posture vector (6*1)"""
    ...

def postureToHomo(homoMat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """convert posture vector matrix (6*1) to homogeneous (4*4)"""
    ...

class ArmFSMState(Enum):
    INVALID: ...
    PASSIVE: ...
    JOINTCTRL: ...
    CARTESIAN: ...
    MOVEJ: ...
    MOVEL: ...
    MOVEC: ...
    TRAJECTORY: ...
    TOSTATE: ...
    SAVESTATE: ...
    TEACH: ...
    TEACHREPEAT: ...
    CALIBRATION: ...
    SETTRAJ: ...
    BACKTOSTART: ...
    NEXT: ...
    LOWCMD: ...

class LowlevelState:
    endPosture: npt.NDArray[np.float64]
    q: List[float]
    """7 elements, including gripper q as the last element"""
    dq: List[float]
    ddq: List[float]
    tau: List[float]
    temperature: List[int]
    errorstate: List[int]
    """
    0x01 : phase current is too large
    0x02 : phase leakage
    0x04 : motor winding overheat or temperature is too large
    0x20 : parameters jump
    0x40 : Ignore
    """
    isMotorConnected: List[int]
    """
    0: OK
    1: communication between lower computer and motor disconnect once
    2: communication between lower computer and motor has CRC erro once
    """
    def getQ(self) -> npt.NDArray[np.float64]: ...
    def getQd(self) -> npt.NDArray[np.float64]: ...
    def getQdd(self) -> npt.NDArray[np.float64]: ...
    def getQTau(self) -> npt.NDArray[np.float64]: ...
    def getGripperQ(self) -> float: ...
    def getGripperQd(self) -> float: ...
    def getGripperTau(self) -> float: ...

class LowlevelCmd:

    # Constructors
    def __init__(self): ...
    def __del__(self): ...

    # Public members
    twsit: npt.NDArray[np.float64]
    q: List[float]
    dq: List[float]
    tau: List[float]
    kp: List[float]
    kd: List[float]
    posture: npt.NDArray[np.float64]

    # Methods
    def setZeroDq(self) -> None:
        """dq = {0}"""
        ...

    def setZeroTau(self) -> None:
        """tau = {0}"""
        ...

    def setZeroKp(self) -> None:
        """kp = {0}"""
        ...

    def setZeroKd(self) -> None:
        """kd = {0}"""
        ...

    @overload
    def setControlGain(self) -> None:
        """
        set default arm kp & kd (Only used in State_LOWCMD)
        kp = [20, 30, 30, 20, 15, 10]
        kd = [2000, 2000, 2000, 2000, 2000, 2000]
        """
        ...

    @overload
    def setControlGain(self, KP: List[float], KW: List[float]) -> None:
        """kp = KP, kd = KW"""
        ...

    @overload
    def setQ(self, qInput: List[float]) -> None:
        """q = qInput"""
        ...

    @overload
    def setQ(self, qInput: npt.NDArray[np.float64]) -> None:
        """q = qInput"""
        ...

    def setQd(self, qDInput: npt.NDArray[np.float64]) -> None:
        """dq = qDInput"""
        ...

    def setTau(self, tauInput: npt.NDArray[np.float64]) -> None:
        """tau = tauInput"""
        ...

    def setPassive(self) -> None:
        """
        setZeroDq()
        setZeroTau()
        setZeroKp()
        kd = [10, 10, 10, 10, 10, 10]
        """
        ...

    @overload
    def setGripperGain(self) -> None:
        """
        set default gripper kp & kd (Only used in State_LOWCMD)
        kp.at(kp.size()-1) = 20
        kd.at(kd.size()-1) = 2000
        """
        ...

    @overload
    def setGripperGain(self, KP: float, KW: float) -> None:
        """
        kp.at(kp.size()-1) = KP
        kd.at(kd.size()-1) = KW
        """
        ...

    def setGripperZeroGain(self) -> None:
        """set gripper kp&kd to zero"""
        ...

    def setGripperQ(self, qInput: float) -> None:
        """q.at(q.size()-1) = qInput;"""
        ...

    def getGripperQ(self) -> float:
        """return q.at(q.size()-1);"""
        ...

    def setGripperQd(self, qdInput: float) -> None:
        """dq.at(dq.size()-1) = qdInput;"""
        ...

    def getGripperQd(self) -> float:
        """return dq.at(dq.size()-1);"""
        ...

    def setGripperTau(self, tauInput: float) -> None:
        """tau.at(tau.size()-1) = tauInput;"""
        ...

    def getGripperTau(self) -> float:
        """return tau.at(tau.size()-1);"""
        ...

    def getQ(self) -> npt.NDArray[np.float64]:
        """return Vec6 from std::vector<double> q, without gripper"""
        ...

    def getQd(self) -> npt.NDArray[np.float64]:
        """return Vec6 from std::vector<double> dq, without gripper"""
        ...

class CtrlComponents:
    armModel: Z1Model
    dt: float
    """Read only; default: 0.002"""
    lowcmd: LowlevelCmd
    lowstate: LowlevelState

# This class is inherited from ArmModel, but ArmModel is not exposed in the python wrapper
class Z1Model:
    def __init__(
        self,
        endPosLocal: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64),
        endEffectorMass: float = 0.0,
        endEffectorCom: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64),
        endEffectorInertia: npt.NDArray[np.float64] = np.zeros(
            (3, 3), dtype=np.float64
        ),
    ) -> None: ...
    def checkInSingularity(self, q: npt.NDArray[np.float64]) -> bool:
        """
        Function: Check whether joint1 and joint5 is coaxial
                x5^2 + y5^2 < 0.1^2
        Inputs: q: current joint variables
        Returns: bool
        """
        ...
    def jointProtect(
        self, q: npt.NDArray[np.float64], qd: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Function: Limit q & qd inputs to valid values
        Inputs: q: set in range[_jointQMin, _jointQMax]
                qd: set in range[-_jointSpeedMax, _jointSpeedMax]
        Returns: (q, qd)
        """
        ...
    def getJointQMax(self) -> List[float]: ...
    def getJointQMin(self) -> List[float]: ...
    def getJointSpeedMax(self) -> List[float]: ...
    def inverseKinematics(
        self,
        TDes: npt.NDArray[np.float64],
        qPast: npt.NDArray[np.float64],
        checkInWorkSpace: bool = False,
    ) -> Tuple[bool, npt.NDArray[np.float64]]:
        """
        Function: Computes inverse kinematics in the space frame with the analytical approach
        Inputs: TDes: The desired end-effector configuration
                qPast: An initial guess and result output of joint angles that are close to
                    satisfying TDes
                checkInWorkSpace: whether q_result should be around qPast
                        example: there is a position defined by q_temp which is within the C-space
                                if qPast == np.zeros(6),
                                the function will return false while checkInWorkSpace is false,
                                and return true while checkInWorkSpace is true.
                                Normally, you can use qPast = np.zeros(6) and checkInWorkSpace = true
                                to check whether q_temp has inverse kinematics solutions
        Returns: success: A logical value where TRUE means that the function found
                        a solution and FALSE means that it ran through the set
                        number of maximum iterations without finding a solution
                q_result: Joint angles that achieve T within the specified tolerances,
        """
        ...
    def forwardKinematics(self, q: npt.NDArray[np.float64], index: int = 6) -> npt.NDArray[np.float64]:
        """
        Function: compute end effector frame (used for current spatial position calculation)
        Inputs: q: current joint angles
                index: it can set as 0,1,...,6
                if index ==  6, then compute end efftor frame,
                else compute joint_i frame
        Returns: Transfomation matrix representing the end-effector frame when the joints are
                        at the specified coordinates
        """
        ...
    def inverseDynamics(
        self,
        q: npt.NDArray[np.float64],
        qd: npt.NDArray[np.float64],
        qdd: npt.NDArray[np.float64],
        Ftip: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Function: This function uses forward-backward Newton-Euler iterations to caculate inverse dynamics
        Inputs: q: joint angles
                qd: joint velocities
                qdd: joint accelerations
                Ftip: Spatial force applied by the end-effector
        Returns: required joint forces/torques
        """
        ...
    def CalcJacobian(self, q: npt.NDArray[np.float64]):
        """
        Function: Gives the space Jacobian
        Inputs: q: current joint angles
        Returns: 6x6 Spatial Jacobian
        """
        ...
    def solveQP(
        self,
        TDes: npt.NDArray[np.float64],
        qPast: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """
        Function: The function use quadprog++ to slove equation: qd = J.inverse() * twist, even if J has no inverse
        Inputs: twist: spatial velocity [R_dot, p_dot]
                qPast: current joint angles
                dt : compute period
        Returns: qd_result: joint velocity that are corresponding to twist
        """
        ...

class ArmInterface:
    def __init__(self, hasGripper: bool) -> None: ...

    q: npt.NDArray[np.float64]
    qd: npt.NDArray[np.float64]
    tau: npt.NDArray[np.float64]
    gripperQ: float
    gripperQd: float
    gripperTau: float
    lowstate: LowlevelState
    """same as _ctrlComp->lowstate"""
    lowcmd: LowlevelCmd
    """same as _ctrlComp->lowcmd"""
    _ctrlComp: CtrlComponents

    def setFsmLowcmd(self) -> None:
        """Set arm to low command mode"""
        ...
    def getCurrentState(self) -> ArmFSMState: ...
    def loopOn(self): ...
    def loopOff(self): ...
    def setFsm(self, fsm: ArmFSMState) -> bool:
        """
        Function: Change z1_ctrl state to fsm, wait until change complete
        Input:    ArmFSMState
        Output:   Whether swtich to fsm correctly
        Note:     eaxmple: Only State_Passive could switch to State_LowCmd
        """
        ...
    def backToStart(self):
        """
        Move arm to home position
        wait until arrival home position, and then switch to State_JointCtrl
        """
        ...
    def labelRun(self, label: str):
        """
        Move arm to label position
        wait until arrival label position, and then switch to State_JointCtrl
        label
        which should exist in z1_controller/config/savedArmStates.csv.
        The number of characters in label cannot be greater than 10.(char name[10])
        """
        ...
    def labelSave(self, label: str):
        """
        Function: Save current position as a label to saveArmStates.csv
                  Switch to State_JointCtrl when done
        Input:    label
                  name to save, which shouldn't exist in z1_controller/config/saveArmStates.csv.
                  The number of characters in label cannot be greater than 10.(char name[10])
        Output:   None
        """
        ...
    def teach(self, label: str):
        """
        Function: Save current position as a label to saveArmStates.csv
                  Switch to State_JointCtrl when done
        Input:    label
                  name to save, which shouldn't exist in z1_controller/config/saveArmStates.csv.
                  The number of characters in label cannot be greater than 10.(char name[10])
        Output:   None
        """
        ...
    def teachRepeat(self, label: str) -> None:
        """
        Function: Switch to State_Teach
        Input:    label
                  Teach trajectory will be saved as Traj_label.csv in directory z1_controller/config/
                  The number of characters in label cannot be greater than 10.(char name[10])
        Output:   None
        """
        ...
    def calibration(self) -> None:
        """
        Function: Calibrate the motor, make current position as home position
        Input:    None
        Output:   None
        """
        ...
    @overload
    def MoveJ(self, posture: npt.NDArray[np.float64], maxSpeed: float):
        """
        Function: Move the robot in a joint path
        Input:    posture: target position, (roll pitch yaw x y z), unit: meter
                  maxSpeed: the maximum joint speed when robot is moving, unit: radian/s
                  range:[0, pi]
        Output:   None
        """
        ...
    @overload
    def MoveJ(
        self, posture: npt.NDArray[np.float64], gripperPos: float, maxSpeed: float
    ) -> bool:
        """
        Function: Move the robot in a joint path, and control the gripper at the same time
        Input:    posture: target position, (roll pitch yaw x y z), unit: meter
                  gripperPos: target angular
                    unit: radian
                    range:[-pi/2, 0]
                  maxSpeed: the maximum joint speed when robot is moving
                    unit: radian/s
                    range:[0, pi]
        Output:   whether posture has inverse kinematics
        """
        ...
    @overload
    def MoveL(self, posture: npt.NDArray[np.float64], maxSpeed: float) -> bool:
        """
        Function: Move the robot in a linear path
        Input:    posture: target position, (roll pitch yaw x y z), unit: meter
                  maxSpeed: the maximum joint speed when robot is moving, unit: m/s
        Output:   whether posture has inverse kinematics
        """
        ...
    @overload
    def MoveL(
        self, posture: npt.NDArray[np.float64], gripperPos: float, maxSpeed: float
    ) -> bool:
        """
        Function: Move the robot in a linear path, and control the gripper at the same time
        Input:    posture: target position, (roll pitch yaw x y z), unit: meter
                  gripperPos: target angular, unit: radian
                    range:[-pi/2, 0]
                  maxSpeed: the maximum joint speed when robot is moving, unit: m/s
        Output:   whether posture has inverse kinematics
        """
        ...
    @overload
    def MoveC(
        self,
        middlePosutre: npt.NDArray[np.float64],
        endPosture: npt.NDArray[np.float64],
        maxSpeed: float,
    ) -> bool:
        """
        Function: Move the robot in a circular path
        Input:    middle posture: determine the shape of the circular path
                  endPosture: target position, (roll pitch yaw x y z), unit: meter
                  maxSpeed: the maximum joint speed when robot is moving, unit: m/s
        Output:   whether posture has inverse kinematics
        """
        ...
    @overload
    def MoveC(
        self,
        middlePosutre: npt.NDArray[np.float64],
        endPosture: npt.NDArray[np.float64],
        gripperPos: float,
        maxSpeed: float,
    ) -> bool:
        """
        Function: Move the robot in a circular path, and control the gripper at the same time
        Input:    middle posture: determine the shape of the circular path
                  endPosture: target position, (roll pitch yaw x y z), unit: meter
                  gripperPos: target angular, unit: radian
                    range:[-pi/2, 0]
                  maxSpeed: the maximum joint speed when robot is moving, unit: m/s
        Output:   whether posture has inverse kinematics
        """
        ...
    def startTrack(self, fsm: "ArmFSMState") -> bool:
        """
        Function: Control robot with q&qd command in joint space or posture command in cartesian space
        Input:    fsm: ArmFSMState::JOINTCTRL or ArmFSMState::CARTESIAN
        Output:   whether posture has inverse kinematics
        Description: 
                1. ArmFSMState::JOINTCTRL, 
                if you run function startTrack(ArmFSMState::JOINTCTRL), 
                firstly, the following parameters will be set at the first time:
                    q : <---- lowstate->getQ()
                    qd: <---- lowstate->getQd()
                    gripperQ:  <----   lowstate->getGripperQ()
                    gripperQd: <----  lowstate->getGripperQd()
                then you can change these parameters to control robot
                2. ArmFSMState::CARTESIAN, 
                if you run function startTrack(ArmFSMState::CARTESIAN), 
                firstly, the following parameters will be set at the first time:
                    twist.setZero()
                then you can change it to control robot, [Based on the object coordinate system]
        """
        ...
    def sendRecv(self) -> None:
        """
        Function: send udp message to z1_ctrl and receive udp message from it
        Input:    None
        Output:   None\n
        Description: 
            sendRecvThread will run sendRecv() at a frequency of 500Hz
            ctrlcomp.sendRecv() is called in unitreeArm.sendRecv(),
            and set command parameters in unitreeArm to lowcmd automatically
            If you want to control robot under JOINTCTRL, CARTESIAN or LOWCMD,
            instead of MovecJ, MoveL, MoveC, and so on
            it is recommended to create your own thread to process command parameters
            (see stratTrack() description)
            and execute sendRecv() at the end of thread
        """
        ...
    def setWait(self, Y_N: bool) -> None:
        """
        Function: whether to wait for the command to finish
        Input:    true or false
        Output:   None\n
        Description: 
            For example, MoveJ will send a trajectory command to z1_controller and then
            run usleep() to wait for the trajectory execution to complete.
            If set [wait] to false, MoveJ  will send command only and user should judge 
            for youself whether the command is complete.
                Method 1: if(_ctrlComp->recvState.state != fsm)
                    After trajectory complete, the FSM will switch to ArmFSMState::JOINTCTRL
                    automatically
                Method 2: if((lowState->endPosture - endPostureGoal).norm() < error)
                    Check whether current posture reaches the target
            Related functions:
                MoveJ(), MoveL(), MoveC(), backToStart(), labelRun(), teachRepeat()
        """
        ...
    def jointCtrlCmd(
        self, directions: npt.NDArray[np.float64], jointSpeed: float
    ) -> None:
        """
        Function: set q & qd command automatically by input parameters
        Input:    directions: movement directions [include gripper], range:[-1,1]
                   J1, J2, J3, J4, J5, J6, gripper
                  jointSpeed: range: [0, pi]
        Output:   None\n
        Description: 
            The function is typically used to control the robot by keyboard or joystick
            When a key is pressed, the directions[i] sets to 1 or -1, and the function will 
            automatically execute the following command:
                    qd = directions * jointSpeed
                    q += qd * _ctrlComp->dt
            if directions == 0, the robot stop moving
        """
        ...
    def cartesianCtrlCmd(
        self, directions: npt.NDArray[np.float64], oriSpeed: float, posSpeed: float
    ) -> None:
        """
        Function: set spatial velocity command automatically by input parameters
        Input:    directions: movement directions [include gripper], range:[-1,1]
                   roll, pitch, yaw, x, y, z, gripper
                  oriSpeed: range: [0, 0.6]
                  posSpeed: range: [0, 0.3]
                  gripper joint speed is set to 1.0
        Output:   None\n
        Description: 
            The function is typically used to control the robot by keyboard or joystick
            When a key is pressed, the directions[i] is set to 1 or -1, and the function will 
            automatically execute the following command:
                    postureDelta = directions * speed
                    postureGoal  = postureDelta + posturePast
                    Tgoal = postureToHomo(postureGoal)
                    Tpast = postureToHomo(posturePast)
                    omega  = so3ToVec(MatrixLog3( Tpast.Rot.Transpose * Tgoal.Rot ))
                    v      = Tdelta.t
                    twist  = (omega, v)
            if directions == 0, the robot stop moving
        """
        ...
    def setArmCmd(
        self,
        q: npt.NDArray[np.float64],
        qd: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
    ) -> None:
        """
        Function: Set six joints commands to class lowcmd
        Input:    q:  joint angle
                  qd: joint velocity
                  tau: joint (Only used in State_LOWCMD)
        Output:   None
        """
        ...
    def setGripperCmd(
        self, gripperPos: float, gripperSpeed: float, gripperTorque: float = 0.0
    ) -> None:
        """
        Function: Set gripper command to class lowcmd
        Input:    gripperPos: target angular
                    range:[-pi/2, 0]
                  gripperSpeed: target speed, range:[0, 1.0]
                  gripperTorque: target torque, default:0, range:[0, 1.0]
        Output:   None
        """
        ...

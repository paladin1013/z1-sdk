from typing import List, Tuple
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
    def getQ(self) -> npt.NDArray[np.float64]: ...
    def getQd(self) -> npt.NDArray[np.float64]: ...
    def getQdd(self) -> npt.NDArray[np.float64]: ...
    def getQTau(self) -> npt.NDArray[np.float64]: ...

class CtrlComponents:
    armModel: Z1Model
    dt: float
    """Read only; default: 0.002"""


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
    def forwardKinematics(self, q: npt.NDArray[np.float64], index: int = 6):
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

from typing import List
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr

from ../geometry import Geometry3D, Capsule
from ../utils import MathUtils
from .robot import Robot, get_transformation_mdh, wrap


class Panda(Robot):
    def __init__(self) -> None:
        super().__init__()

        self._dof = 7
        self.q0 = [0.0 for _ in range(self._dof)]

        # DH Parameters (MDH)
        self.alpha_array = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
        self.a_array = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
        self.d_array = [0.333, 0, 0.316, 0, 0.384, 0, 0.107]
        self.theta_array = [0 for _ in range(self._dof)]
        self.sigma_array = [0 for _ in range(self._dof)]

        # Inertial Parameters (mass, CoM, inertia, motor inertia)
        # These values are simplified/estimated from official data
        masses = [4.0, 4.0, 3.0, 2.7, 2.7, 1.7, 1.6]
        centers = [
            [0, 0, 0.1], [0, 0.05, 0], [0, 0, 0.15],
            [0.05, 0, 0], [0, 0, 0.1], [0, 0, 0.05], [0.02, 0, 0]
        ]
        inertias = [np.diag([0.01, 0.01, 0.01]) for _ in range(self._dof)]
        Jms = [0.1 for _ in range(self._dof)]

        # Create DH links
        links = []
        for i in range(self._dof):
            links.append(
                rtb.DHLink(
                    d=self.d_array[i],
                    alpha=self.alpha_array[i],
                    a=self.a_array[i],
                    offset=self.theta_array[i],
                    mdh=True,
                    m=masses[i],
                    r=centers[i],
                    I=inertias[i],
                    Jm=Jms[i],
                    G=1.0
                )
            )
        self.robot = rtb.DHRobot(links)

        # Build transformation chain
        T = SE3()
        for i in range(self._dof):
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)
            self._Ms.append(Ti.A)
            T = T * Ti
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))

            Gm = np.zeros((6, 6))
            Gm[:3, :3] = inertias[i]
            Gm[3:, 3:] = masses[i] * np.eye(3)
            AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -np.array(centers[i])))
            self._Gs.append(AdT.T @ Gm @ AdT)
            self._Jms.append(Jms[i])

        self._Ms.append(np.eye(4))

    def ikine(self, Tep: SE3) -> np.ndarray:
        sol = self.robot.ikine_LM(Tep, q0=self.q0)
        if sol.success:
            return sol.q
        else:
            return np.array([])

    def get_geometries(self) -> List[Geometry3D]:
        Ts = []
        T = SE3()
        for i in range(self.dof):
            T = T * get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                           self.sigma_array[i], self.q0[i])
            Ts.append(T)

        geometries = []
        radii = [0.06, 0.05, 0.05, 0.045, 0.045, 0.04, 0.04]
        lengths = [0.15, 0.25, 0.2, 0.2, 0.2, 0.15, 0.1]
        for i in range(self.dof):
            geom_T = Ts[i] * SE3.Trans(0, 0, lengths[i] / 2)
            geometries.append(Capsule(geom_T, radii[i], lengths[i]))

        return geometries


if __name__ == "__main__":
    robot = Panda()
    T = robot.fkine(robot.q0)
    print("End-effector pose:", T)
    robot.move_cartesian(T)
    print("Inverse kinematics result:", robot.get_joint())

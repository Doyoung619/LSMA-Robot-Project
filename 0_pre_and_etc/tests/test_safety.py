import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from rita_robot.motion import RobotModel, create_plan, signed, SPEED, ACCELERATION
from rita_robot.robot import Controller
from kortex_api.autogen.messages import Base_pb2

ROOT=Path(__file__).resolve().parents[1]


class Event:
    def __init__(self): self.flag=False
    def is_set(self): return self.flag
    def set(self): self.flag=True


class FakeRobot:
    def __init__(self,q,state=Base_pb2.ARMSTATE_SERVOING_READY):
        self.sent=[]
        self.f=SimpleNamespace(
            base=SimpleNamespace(fault_bank_a=0,fault_bank_b=0,tool_pose_x=0,tool_pose_y=0,
                                 tool_pose_z=.6,active_state=state,active_state_connection_identifier=8),
            actuators=[SimpleNamespace(position=x%360,velocity=0,fault_bank_a=0,fault_bank_b=0) for x in q])
    def feedback(self): return self.f
    def speeds(self,v): self.sent.append(v)


class DanceSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=RobotModel(ROOT/'data/robot/GEN3-6DOF_VISION_URDF_ARM_V01.urdf')
        cls.q=np.array([-24.,-49.,-88.,58.,-120.32,118.])
        cls.info={'period_seconds':60/117.,'first_beat_seconds':.33}

    def test_boundary_pose_is_recentered_before_dance(self):
        plan=create_plan(self.model,self.q,self.info,40)
        self.assertGreater(plan.center[4],self.q[4]+15)
        self.assertTrue(np.all(plan.q>=self.model.lower+8))
        self.assertTrue(np.all(plan.q<=self.model.upper-8))

    def test_many_seeds_and_tempos_obey_dynamic_limits(self):
        for bpm in [70,117,180]:
            for seed in range(8):
                info={'period_seconds':60/bpm,'first_beat_seconds':.2}
                plan=create_plan(self.model,self.q,info,32,seed=seed)
                self.assertTrue(np.all(np.max(np.abs(plan.v),axis=0)<=SPEED+.001))
                self.assertTrue(np.all(np.max(np.abs(plan.a),axis=0)<=ACCELERATION+.001))
                np.testing.assert_allclose(plan.q[[0,-1]],np.tile(plan.center,(2,1)),atol=1e-9)
                self.assertLess(np.max(np.abs(plan.v[[0,-1]])),.01)
                # Multiple joints together must not stop at phrase boundaries.
                inner=np.linalg.norm(plan.v[300:-300],axis=1)
                self.assertGreater(np.min(inner),.05)

    def test_dof_mask_locks_unselected_oscillations(self):
        for dof in range(1,7):
            plan=create_plan(self.model,self.q,self.info,12,dof=dof)
            np.testing.assert_allclose(plan.q[:,dof:],np.tile(plan.center[dof:],(len(plan.q),1)))

    def test_signed_wrap(self):
        np.testing.assert_allclose(signed([359,1,239]),[-1,1,-121])

    def controller(self,robot):
        plan=create_plan(self.model,self.q,self.info,12)
        snap={'q':self.q.tolist(),'speed_limits':SPEED,'accel_limits':ACCELERATION}
        return Controller(robot,self.model,snap,plan,SimpleNamespace(value=0),Event(),Event(),io.StringIO())

    def test_fault_never_sends_motion(self):
        robot=FakeRobot(self.q);robot.f.actuators[2].fault_bank_a=1
        with self.assertRaisesRegex(RuntimeError,'fault'):
            self.controller(robot).tick(self.q,np.zeros(6),'prepare')
        self.assertEqual(robot.sent,[])

    def test_other_controller_never_receives_motion(self):
        robot=FakeRobot(self.q,Base_pb2.ARMSTATE_SERVOING_MANUALLY_CONTROLLED)
        with self.assertRaisesRegex(RuntimeError,'not ready'):
            self.controller(robot).tick(self.q,np.zeros(6),'prepare')
        self.assertEqual(robot.sent,[])

    def test_operator_stop_cannot_restart(self):
        robot=FakeRobot(self.q);control=self.controller(robot)
        control.first_command_time=control.last_tick-1
        with self.assertRaisesRegex(RuntimeError,'will not restart'):
            control.tick(self.q,np.zeros(6),'prepare')
        self.assertEqual(robot.sent,[])

    def test_recovery_only_moves_inward(self):
        robot=FakeRobot(self.q);control=self.controller(robot)
        control.last_tick-=.025
        with self.assertRaisesRegex(RuntimeError,'Outward recovery'):
            control.tick(self.q,np.array([0,0,0,0,-1,0]),'prepare')
        self.assertEqual(robot.sent,[])

    def test_control_hang_does_not_send_more_commands(self):
        robot=FakeRobot(self.q);control=self.controller(robot)
        control.armed.set();control.last_tick-=.3
        with self.assertRaisesRegex(RuntimeError,'deadline'):
            control.tick(self.q,np.zeros(6),'prepare')
        self.assertEqual(robot.sent,[])


if __name__=='__main__':unittest.main()

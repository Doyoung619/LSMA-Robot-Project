"""Contract tests use injected transports only; no robot is contacted or moved."""
import builtins
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rita_core.contracts import DanceSegment
from rita_core.profiles import default_profile, load_profile
from rita_core.adapters import make_adapter, AdapterError, CapabilityError, DispatchDeadlineMiss
from rita_core.adapters.misty import MistyAdapter, JOINTS, SENSORS
from rita_core.adapters.kinova import KinovaAdapter


def segment(profile, start=0.):
    a = np.array(profile.neutral)
    b = a+np.minimum(.05, (profile.upper-a)/2)
    controls = np.array([[a, a, a, b, b, b], [b, b, b, a, a, a]])
    return DanceSegment(start=start, duration=2., knots=np.array([start, start+1, start+2]),
                        controls=controls, validation={"valid": True}, metadata={"joint_names": list(profile.joint_names), "units": "rad"})


class AdapterContracts(unittest.TestCase):
    def test_profiles_require_explicit_units_and_order(self):
        data = default_profile().to_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mock.json"
            data["units"] = "degrees"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "rad"):
                load_profile(path)
            data["units"] = "rad"
            data["joint_names"][1] = data["joint_names"][0]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_profile(path)

    def test_mock_has_no_hardware_import_or_network(self):
        original = builtins.__import__
        def checked(name, *args, **kwargs):
            if name.startswith(("rclpy", "kortex_api", "websocket")):
                raise AssertionError("hardware import in mock")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", checked), patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            profile = default_profile()
            adapter = make_adapter("mock", profile)
            adapter.advance(0)
            adapter.connect()
            adapter.execute_segment(segment(profile), scheduled_monotonic=0)
            adapter.advance(.5)
            self.assertTrue(adapter.read_state().fresh)
            adapter.stop()
            self.assertEqual(adapter.commands, [])
            self.assertTrue(adapter.stopped)
            adapter.close()

    def test_mock_lag_missing_feedback_deadline_and_disconnect(self):
        p = default_profile()
        adapter = make_adapter("mock", p, command_delay=.1, tracking_delay=.2)
        adapter.advance(0)
        adapter.connect()
        s = segment(p)
        adapter.execute_segment(s, scheduled_monotonic=0)
        adapter.advance(.5)
        np.testing.assert_allclose(adapter.read_state().q, s.at(.2)[0], atol=1e-10)
        adapter.feedback_dropout = 1
        self.assertFalse(adapter.read_state().fresh)
        adapter.stop()
        with self.assertRaises(DispatchDeadlineMiss):
            adapter.execute_segment(s, scheduled_monotonic=0)
        adapter.disconnect_at = .6
        adapter.advance(.6)
        with self.assertRaises(ConnectionError):
            adapter.read_state()

    def test_misty_degree_conversion_schema_and_rate_compilation(self):
        p = default_profile("misty")
        now = [1.]
        adapter = MistyAdapter(p, clock=lambda: now[0], enable_feedback=False)
        adapter.connected = True
        for sensor in SENSORS:
            self.assertTrue(adapter.ingest_feedback({"message": {"SensorName": sensor, "Value": 90}}))
        np.testing.assert_allclose(adapter.read_state().q, np.pi/2)
        # Current firmware may omit sensorName: bound per-sensor eventName is used.
        self.assertTrue(adapter.ingest_feedback({"eventName": "Rita_head_yaw", "message": {"value": -45}}))
        self.assertAlmostEqual(adapter.read_state().q[2], -np.pi/4)
        now[0] = 2
        self.assertFalse(adapter.read_state().fresh)
        commands = adapter.compile_segment(segment(p))
        self.assertGreaterEqual(min(np.diff([c["offset"] for c in commands])), p.command_period-.000001)
        for c in commands:
            self.assertIn("Duration", c["head"])
            self.assertNotIn("Velocity", c["head"])
            self.assertNotIn("Duration", c["arms"])
            self.assertEqual(c["arms"]["LeftArmVelocity"], 15)
            self.assertEqual(c["head"]["Units"], "degrees")

    def test_misty_halt_path_and_readonly_does_not_send_motion(self):
        p = default_profile("misty")
        calls = []
        def http(method, path, payload):
            calls.append((method, path, payload))
            return {"status": "Success", "result": {}}
        adapter = MistyAdapter(p, http=http, enable_feedback=False)
        adapter.connect()
        adapter.close()
        self.assertTrue(all(method == "GET" for method, _, _ in calls))
        adapter = MistyAdapter(p, execute=True, http=http, enable_feedback=False)
        adapter.connected = True
        adapter.stop()
        self.assertEqual(calls[-1][1], "/api/halt")
        self.assertNotIn("/api/drive/stop", [c[1] for c in calls])
        with self.assertRaises(PermissionError):
            adapter.execute_segment(segment(p))

    def test_kinova_rejects_wrong_model_and_order(self):
        p = default_profile("kinova")
        adapter = KinovaAdapter(p)
        adapter.connected = True
        bad = SimpleNamespace(name=list(reversed(p.joint_names)), position=[0]*6, velocity=[0]*6)
        self.assertFalse(adapter.ingest_feedback(bad))
        with self.assertRaisesRegex(AdapterError, "order"):
            adapter.read_state()
        p.model = "jaco2"
        with self.assertRaises(CapabilityError):
            KinovaAdapter(p)

    def test_kinova_reference_compile_preserves_q_dq_ddq_and_cancel(self):
        p = default_profile("kinova")
        adapter = KinovaAdapter(p, execute=True)
        s = segment(p)
        compiled = adapter.compile_segment(s)
        self.assertEqual(compiled["joint_names"], list(p.joint_names))
        for point, t in zip(compiled["points"], s.knots):
            for values, expected in zip((point["positions"], point["velocities"], point["accelerations"]), s.at(t)):
                np.testing.assert_allclose(values, expected)
        calls = []
        class Future:
            def done(self): return True
            def result(self): return SimpleNamespace(goals_canceling=["id"])
        class Goal:
            def cancel_goal_async(self): calls.append("cancel_goal_async"); return Future()
        adapter._goal_handle = Goal()
        self.assertTrue(adapter.stop()["cancel_accepted"])
        self.assertEqual(calls, ["cancel_goal_async"])

    def test_hardware_rejects_nonzero_chunk_boundary(self):
        p = default_profile("kinova")
        s = segment(p)
        s.controls[0, 1, 0] += .001
        with self.assertRaises(CapabilityError):
            KinovaAdapter(p).compile_segment(s)

    def test_hardware_compiler_trims_only_exact_constant_terminal_hold(self):
        p = default_profile("kinova")
        s = segment(p)
        hold = np.repeat(s.controls[-1, -1][None, :], 6, axis=0)
        s = DanceSegment(start=0, duration=2.3, knots=np.array([0, 1, 2, 2.3]),
                         controls=np.concatenate([s.controls, hold[None]], axis=0), validation={"valid": True})
        compiled = KinovaAdapter(p).compile_segment(s)
        self.assertAlmostEqual(compiled["execution_duration"], 2.)
        self.assertAlmostEqual(compiled["terminal_hold_seconds"], .3)
        self.assertAlmostEqual(compiled["points"][-1]["time_from_start"], 2.)
        s.controls[-1, 2, 0] += .000001
        compiled = KinovaAdapter(p).compile_segment(s)
        self.assertAlmostEqual(compiled["execution_duration"], 2.3)

    def test_misty_filter_selection_handles_current_sensor_id_schema(self):
        p = default_profile("misty")
        def http(method, path, payload):
            return {"result": [{"nestedProperties": ["SensorId", "Value", "Created"]}]}
        adapter = MistyAdapter(p, http=http, enable_feedback=False)
        adapter._device = {"result": {"sensorCapabilities": {sensor: f"id_{j}" for j, sensor in enumerate(SENSORS)}}}
        filters = adapter._feedback_filters()
        self.assertEqual(filters["head_pitch"]["Property"], "SensorId")
        self.assertEqual(filters["right_arm"]["Value"], "id_4")


if __name__ == "__main__":
    unittest.main()

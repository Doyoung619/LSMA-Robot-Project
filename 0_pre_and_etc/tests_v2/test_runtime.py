"""Execution hazards and no-queue future handoff, with no actual audio/hardware."""
from pathlib import Path
import sys
import threading
import time as real_time
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rita_core import runtime
from rita_core.audio import synthetic_analysis
from rita_core.profiles import default_profile
from rita_core.trajectory import make_segment,add_terminal_hold,validate_segment
from rita_core.adapters.mock import MockAdapter
from rita_core.adapters.base import terminal_hold_start,CapabilityError


class SmallPlanner:
    def __init__(self,analysis,profile,hold=0.,fail_after=None):
        self.analysis,self.profile,self.hold=analysis,profile,hold
        self.calls=0;self.fail_after=fail_after
    def next_segment(self,beat,state):
        self.calls+=1
        if self.fail_after and self.calls>=self.fail_after:
            raise TimeoutError("injected planning deadline")
        knots=np.array([self.analysis.beat_time(beat+k) for k in range(5)])
        offsets=np.array([0,.035,-.035,.035,0])[:,None]*np.linspace(.5,1,self.profile.nj)
        seg=make_segment(knots,self.profile.neutral+offsets,
                         accent_times=knots[1:-1],rhythm_id="test-pattern",
                         metadata={"start_beat":beat,"end_beat":beat+4,"primitive_id":"test-motion"})
        if self.hold: seg=add_terminal_hold(seg,self.hold)
        if not validate_segment(seg,self.profile)["valid"]:
            raise RuntimeError(f"Test reference unexpectedly invalid: {seg.validation}")
        return seg
    def stats(self): return {}


class RuntimeHazards(unittest.TestCase):
    def setUp(self):
        self.analysis=synthetic_analysis(bpm=120,duration=30,phase=.17)
        self.profile=default_profile()

    def test_headless_mock_completes_and_retains_continuous_music_time(self):
        adapter=MockAdapter(self.profile)
        with patch.object(runtime,"make_adapter",return_value=adapter):
            report=runtime.run_session(self.analysis,self.profile,seconds=5,planner=SmallPlanner(self.analysis,self.profile))
        self.assertTrue(report["run"]["completed"],report["run"])
        self.assertEqual(report["segments_executed"],3)
        self.assertTrue(adapter.stopped)
        receipts=report["run"]["dispatch_receipts"]
        self.assertEqual(len(receipts),3)
        for a,b in zip(receipts,receipts[1:]):
            self.assertAlmostEqual(b["scheduled_monotonic"]-a["scheduled_monotonic"],2.)

    def test_feedback_loss_and_disconnect_stop_without_hardware(self):
        for options in ({"feedback_dropout":1.},{"disconnect_at":.5}):
            adapter=MockAdapter(self.profile,**{k:v for k,v in options.items() if k!="disconnect_at"})
            if "disconnect_at" in options:
                # Runtime maps relative failure time to its virtual monotonic epoch.
                original_advance=adapter.advance
                def advance(t):
                    if adapter.disconnect_at is None: adapter.disconnect_at=t+.5
                    original_advance(t)
                adapter.advance=advance
            with patch.object(runtime,"make_adapter",return_value=adapter):
                report=runtime.run_session(self.analysis,self.profile,seconds=5,planner=SmallPlanner(self.analysis,self.profile))
            self.assertFalse(report["run"]["completed"])
            self.assertIsNotNone(report["run"]["error"])
            self.assertTrue(adapter.stopped)
            self.assertEqual(adapter.commands,[])

    def test_planning_error_stops_and_does_not_dispatch_old_command(self):
        adapter=MockAdapter(self.profile)
        with patch.object(runtime,"make_adapter",return_value=adapter):
            report=runtime.run_session(self.analysis,self.profile,seconds=6,
                                       planner=SmallPlanner(self.analysis,self.profile,fail_after=2))
        self.assertFalse(report["run"]["completed"])
        self.assertIn("injected planning deadline",report["run"]["error"])
        self.assertEqual(len(report["run"]["dispatch_receipts"]),1)
        self.assertTrue(adapter.stopped)

    def test_hardware_needs_execute_and_verified_local_workspace(self):
        with self.assertRaises(PermissionError):
            runtime.run_session(self.analysis,self.profile,backend="kinova",headless=False,execute=False)
        with self.assertRaises(PermissionError):
            runtime.run_session(self.analysis,self.profile,backend="kinova",headless=False,execute=True)

    def fake_hardware_run(self,completion_delay=0.,fail_planning=False,hold=.3):
        profile=default_profile()
        profile.local_verified=True
        profile.workspace["verified"]=True
        profile.amax[:]=100
        profile.vmax[:]=5
        profile.max_stationary_seconds=1.
        profile.dispatch_lead_seconds=.1
        profile.latency_mode="timestamp"
        analysis=synthetic_analysis(bpm=120,duration=30,phase=.01)
        elapsed=[0.]
        epoch=1000.
        original_sleep=real_time.sleep
        def now(): return epoch+elapsed[0]
        def sleep(seconds):
            elapsed[0]+=max(0.,seconds)
            original_sleep(.0005) # Give the independent numerical worker CPU time.
        class Clock:
            def __init__(self,*args,**kwargs):
                self.dac_epoch=None;self.finished=threading.Event()
            def start(self): self.dac_epoch=0.
            def clock(self): return elapsed[0]
            def music_to_monotonic(self,t): return epoch+t
            def mapping(self): return {"kind":"test fake output clock, not an acoustic measurement"}
            def close(self): pass
        class Hardware(MockAdapter):
            def __init__(self):
                super().__init__(profile,clock=now)
                self.action_end=-float("inf")
            def get_capabilities(self):
                return {**super().get_capabilities(),"queue":False,"future_execution":True,"hardware":True}
            def ready_for_next_segment(self): return now()>=self.action_end+completion_delay
            def execute_segment(self,seg,scheduled_monotonic=None,now_music=None):
                if self.history and not self.ready_for_next_segment():
                    raise CapabilityError("active no-queue action cannot be replaced")
                if scheduled_monotonic-now()<profile.dispatch_lead_seconds-1e-6:
                    raise CapabilityError("insufficient future lead")
                receipt=super().execute_segment(seg,scheduled_monotonic,now_music)
                self.action_end=scheduled_monotonic+terminal_hold_start(seg)-seg.start
                receipt["controller_action_end_monotonic"]=self.action_end
                return receipt
        adapter=Hardware()
        with patch.object(runtime,"stop_latch_requested",return_value=False),patch.object(runtime,"DACClock",Clock),patch.object(runtime,"decode_audio",return_value="unused.wav"),patch.object(runtime,"make_adapter",return_value=adapter),patch.object(runtime.time,"monotonic",now),patch.object(runtime.time,"sleep",sleep):
            report=runtime.run_session(analysis,profile,backend="kinova",headless=False,execute=True,
                seconds=3,planner=SmallPlanner(analysis,profile,hold=hold,fail_after=2 if fail_planning else None))
        return report,adapter

    def test_noqueue_early_handoff_uses_constant_tail_and_skips_initial_block(self):
        report,adapter=self.fake_hardware_run()
        self.assertTrue(report["run"]["completed"],report["run"])
        self.assertEqual(report["run"]["initial_skipped_beats"],4)
        receipts=report["run"]["dispatch_receipts"]
        self.assertEqual(len(receipts),2) # Already dispatched next phrase is not sent twice.
        self.assertAlmostEqual(receipts[1]["scheduled_monotonic"]-receipts[0]["scheduled_monotonic"],2.)
        self.assertLess(receipts[1]["sent_monotonic"],receipts[1]["scheduled_monotonic"]-.09)
        self.assertGreaterEqual(receipts[1]["sent_monotonic"],receipts[0]["controller_action_end_monotonic"])
        self.assertTrue(adapter.stopped)

    def test_noqueue_late_controller_completion_stops_instead_of_catchup(self):
        report,adapter=self.fake_hardware_run(completion_delay=.3)
        self.assertFalse(report["run"]["completed"])
        self.assertEqual(len(report["run"]["dispatch_receipts"]),1)
        self.assertGreater(report["run"]["deadline_misses"],0)
        self.assertTrue(adapter.stopped)

    def test_realtime_planning_failure_stops_before_future_dispatch(self):
        report,adapter=self.fake_hardware_run(fail_planning=True)
        self.assertFalse(report["run"]["completed"])
        self.assertEqual(len(report["run"]["dispatch_receipts"]),1)
        self.assertTrue(adapter.stopped)

    def test_hardware_stop_latch_is_not_cleared_and_mock_ignores_it(self):
        profile=default_profile()
        profile.local_verified=True
        profile.workspace["verified"]=True
        with patch.object(runtime,"stop_latch_requested",return_value=True):
            with self.assertRaisesRegex(RuntimeError,"STOP_RITA"):
                runtime.run_session(self.analysis,profile,backend="kinova",headless=False,execute=True)
            report=runtime.run_session(self.analysis,profile,seconds=1,planner=SmallPlanner(self.analysis,profile))
        self.assertTrue(report["run"]["completed"])

    def test_failure_invalidates_execution_coverage_but_retains_graph_certificate(self):
        class Planner(SmallPlanner):
            def stats(self): return {"coverage_certificate":{"valid":True,"n":2,"D":1}}
        report=runtime.run_session(self.analysis,self.profile,seconds=5,
            planner=Planner(self.analysis,self.profile,fail_after=2))
        certificate=report["planner"]["coverage_certificate"]
        self.assertFalse(certificate["valid"])
        self.assertEqual(certificate["reason"],"executionfailure")
        self.assertTrue(certificate["reference_graph_certificate"]["valid"])
        self.assertEqual(report["runtime"]["deadline_misses"],1)

    def test_near_end_uses_current_metrical_width_and_completes_at_rest(self):
        analysis=synthetic_analysis(bpm=120,duration=11,phase=.17)
        class Planner(SmallPlanner):
            def next_segment(self,beat,state):
                knots=np.array([self.analysis.beat_time(beat+k*4) for k in range(5)])
                offsets=np.array([0,.15,-.15,.15,0])[:,None]*np.linspace(.5,1,self.profile.nj)
                seg=make_segment(knots,self.profile.neutral+offsets,metrical_level=2,
                    metadata={"start_beat":beat,"end_beat":beat+16,"primitive_id":"wide"})
                validate_segment(seg,self.profile)
                return seg
        report=runtime.run_session(analysis,self.profile,seconds=100,planner=Planner(analysis,self.profile))
        self.assertTrue(report["run"]["completed"],report["run"])
        self.assertEqual(report["segments_executed"],1)
        self.assertAlmostEqual(report["run"]["audio_tail_not_planned_seconds"],2.83)


if __name__=="__main__": unittest.main()

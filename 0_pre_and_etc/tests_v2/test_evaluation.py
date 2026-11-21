import unittest
import numpy as np
from rita_core.audio import synthetic_analysis
from rita_core.profiles import default_profile
from rita_core.trajectory import make_segment,validate_segment
from rita_core.evaluation import evaluate,event_match,feedback_estimates


class EvaluationTests(unittest.TestCase):
    def test_nearly_duplicate_join_times_do_not_create_derivative_spikes(self):
        p=default_profile()
        times=np.r_[np.arange(0,1,.025),1-1e-12,1.,np.arange(1.025,2,.025)]
        positions=np.zeros((len(times),p.nj));positions[:,0]=.1*np.sin(2*np.pi*times)
        estimates=feedback_estimates(times,positions,p)
        self.assertGreater(estimates['coalesced_timestamp_count'],0)
        self.assertEqual(estimates['velocity_excess_estimate'],0)
        self.assertEqual(estimates['acceleration_excess_estimate'],0)
        self.assertAlmostEqual(estimates['max_velocity_estimate_rad_s'],.1*2*np.pi,delta=.03)

    def test_stationary_feedback_is_not_a_pose_passage(self):
        s=make_segment([0,1,2],[[0],[.02],[0]],accent_times=[1],accent_strengths=[1])
        times=np.arange(0,2,.025)
        report=event_match([s],times,np.zeros((len(times),1)))
        self.assertEqual(report['matched_events'],0)
        self.assertEqual(report['missed_events'],1)

    def test_fresh_ids_do_not_hide_repeated_geometry(self):
        p=default_profile();segments=[]
        for i in range(5):
            s=make_segment([i*2,i*2+1,i*2+2],np.array([np.zeros(6),[.12,-.08,.04,0,0,0],np.zeros(6)]),
                           metadata={"primitive_id":f"fresh-{i}"},rhythm_id=f"label-{i}")
            validate_segment(s,p);segments.append(s)
        r,_=evaluate(segments,p,synthetic_analysis(duration=60),[],{},
                     {"seed":7,"backend":"mock","method":"infinite","condition":"test"},[.01])
        self.assertEqual(r["motion_diversity"]["shared_descriptor_feature_cells_visited"],1)
        self.assertGreater(r["motion_diversity"]["short_repeat_ngram_ratio"]["2"],.5)
        self.assertEqual(r["rhythm_diversity"]["event_interval_strength_patterns"],1)
        self.assertIsNone(r["synchronization"]["physical_robot"])


if __name__=="__main__":unittest.main()

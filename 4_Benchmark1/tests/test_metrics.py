import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from metrics import evaluate
from our_adapters import robot_profile
from rita_core.audio import synthetic_analysis
import json


class MetricContracts(unittest.TestCase):
    def setUp(self):
        self.profile=robot_profile();self.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/benchmark.json').read_text())['metrics']
        self.a=synthetic_analysis(duration=60,phase=0)
        self.t=np.arange(0,20,.1)

    def test_stationary_has_no_perfect_alignment(self):
        q=np.tile(self.profile.neutral,(len(self.t),1))
        result,_=evaluate(self.t,q,self.a,self.profile,self.cfg)
        self.assertFalse(result['alignment_valid']);self.assertIsNone(result['permitted_grid_median_error_ms'])
        self.assertEqual(result['stationary_fraction'],1)

    def test_physical_range_normalization_does_not_reward_tiny_jitter(self):
        big=np.zeros((len(self.t),3));big[:,2]=.1*np.sin(2*np.pi*self.t/.7)
        tiny=big*1e-6
        a,_=evaluate(self.t,big,self.a,self.profile,self.cfg);b,_=evaluate(self.t,tiny,self.a,self.profile,self.cfg)
        self.assertGreater(a['window_distance_mean'],b['window_distance_mean']*1000)
        self.assertFalse(b['alignment_valid'])

    def test_raw_violations_and_nonadjacent_recurrence_are_retained(self):
        q=np.zeros((len(self.t),3));q[:,0]=.4*np.sin(2*np.pi*self.t)
        result,_=evaluate(self.t,q,self.a,self.profile,self.cfg)
        self.assertGreater(result['position_violation_rate'],0)
        self.assertGreater(result['velocity_violation_rate'],0)
        self.assertGreater(result['recurrence_rate'],.9)
        self.assertEqual(result['nonadjacent_pair_count'],28)


if __name__=='__main__':unittest.main()

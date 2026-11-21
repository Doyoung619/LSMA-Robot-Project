from copy import deepcopy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reachy_adapter import _loaded,load_upstream,sample_native,adapt_misty


class ReachyTests(unittest.TestCase):
    def test_pinned_pure_upstream_and_cycle(self):
        source=load_upstream()
        self.assertEqual(source['revision'],'6808a976d3ddc11952ece67eb2403934fbb62af9')
        self.assertEqual(source['available_move_count'],20)
        self.assertEqual(source['preview_move_count'],19)
        self.assertEqual(source['preview_cycle_beats'],152)
        self.assertNotIn('reachy_mini',sys.modules)

    def test_official_order_global_phase_and_all_available_primitives(self):
        moves,order,_=_loaded()
        phase=np.arange(len(order))*8+.375
        native=sample_native(np.arange(len(phase),dtype=float),phase)
        self.assertEqual(native['primitive_ids'],order)
        self.assertNotIn('headbanger_combo',native['primitive_ids'])
        for i,name in enumerate(order):
            fn,defaults,_=moves[name]
            expected=fn(float(phase[i]),**deepcopy(defaults))
            np.testing.assert_allclose(native['orientation_rad'][i],expected.orientation_offset,atol=1e-12)
        # Every registered function, including the explicitly preview-excluded
        # one, remains genuinely available and returns finite pure offsets.
        for fn,defaults,_ in moves.values():
            offsets=fn(.375,**deepcopy(defaults))
            self.assertTrue(np.all(np.isfinite(offsets.orientation_offset)))

    def test_default_parameters_and_crop_phase_are_not_mutated(self):
        moves,_,_=_loaded();original={k:deepcopy(v[1]) for k,v in moves.items()}
        phase=np.array([16.3,16.7,17.2]);t=np.array([20.,20.1,20.2])
        a=sample_native(t,phase);b=sample_native(t+30,phase)
        np.testing.assert_array_equal(a['orientation_rad'],b['orientation_rad'])
        for name in moves:
            self.assertEqual(original[name],moves[name][1])

    def test_misty_mapping_preserves_neutral_arms_and_never_uses_antennas(self):
        native={'orientation_rad':np.array([[.1,.2,.3]]),'antenna_rad':np.array([[99.,-99.]])}
        profile=SimpleNamespace(joint_names=('head_pitch','head_roll','head_yaw','left_arm','right_arm'),units='rad',neutral=np.array([0,0,0,.7,.7]))
        result=adapt_misty(native,profile,anchor_first=False)
        np.testing.assert_allclose(result['q'],[[.2,.1,.3,.7,.7]])
        # No clipping: .2rad pitch beyond the example ±.15rad remains visible.
        self.assertEqual(result['q'][0,0],.2)
        anchored=adapt_misty({'orientation_rad':np.array([[.1,.2,.3],[.3,.4,.6]])},profile)
        np.testing.assert_allclose(anchored['q'][0],profile.neutral)
        np.testing.assert_allclose(anchored['q'][1],[.2,.2,.3,.7,.7])
        self.assertEqual(anchored['mapping']['subtracted_first_euler_offset_rad'],[.2,.1,.3])
        profile.joint_names=('head_yaw','head_roll','head_pitch')
        with self.assertRaisesRegex(ValueError,'ordered joints'):adapt_misty(native,profile)

    def test_variable_beat_phase_and_invalid_contracts(self):
        native=sample_native([0,1,2,3],[-1,0,8,16])
        self.assertEqual(native['primitive_ids'][0],'__pre_beat_neutral__')
        np.testing.assert_array_equal(native['orientation_rad'][0],[0,0,0])
        with self.assertRaises(ValueError):sample_native([0,0],[0,1])
        with self.assertRaises(ValueError):sample_native([0,1],[2,1])


if __name__=='__main__':unittest.main()

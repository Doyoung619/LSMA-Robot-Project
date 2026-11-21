"""Integrity checks: missing/invalid outputs must not become fake neutral motion."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import visualize


class SavedTrajectoryVisualizationTests(unittest.TestCase):
    def test_failed_run_does_not_animate_a_stale_success_file(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'trajectories').mkdir()
            np.savez(folder/'trajectories/stage1__our_finite__seed0__beat0.npz',times_seconds=np.arange(121)*.1,q=np.zeros((121,3)))
            (folder/'same_robot_metrics.csv').write_text('stage,method,seed,status,error\nstage1,our_finite,0,failed,injected failure\n',encoding='utf-8')
            loaded,failed=visualize._load_stage1(folder,0,12)
            self.assertNotIn('our_finite',loaded)
            self.assertEqual(failed['our_finite'],'injected failure')

    def test_missing_methods_stay_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded,failed=visualize._load_stage1(Path(directory),.2,12)
        self.assertEqual(loaded,{})
        self.assertEqual(set(failed),set(visualize.METHODS))

    def test_last_control_period_hold_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'trajectories').mkdir()
            times=.2+np.arange(120)*.1
            q=np.zeros((120,3));q[:,0]=.02*np.sin(times)
            np.savez(folder/'trajectories/stage1__our_finite__seed0__beat0.npz',times_seconds=times,q=q)
            loaded,_=visualize._load_stage1(folder,.2,12)
            self.assertIn('our_finite',loaded)
            loaded,failed=visualize._load_stage1(folder,.2,12.2)
            self.assertNotIn('our_finite',loaded)
            self.assertIn('coverage',failed['our_finite'])

    def test_invalid_shape_and_nan_are_not_animated(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'trajectories').mkdir()
            path=folder/'trajectories/stage1__our_infinite__seed0__beat0.npz'
            times=np.arange(121)*.1
            for q in (np.zeros((121,4)),np.full((121,3),np.nan)):
                np.savez(path,times_seconds=times,q=q)
                loaded,failed=visualize._load_stage1(folder,0,12)
                self.assertNotIn('our_infinite',loaded)
                self.assertIn('our_infinite',failed)

    def test_nonfinite_metrics_remain_na(self):
        for value in ('nan','inf','','N/A',None):
            self.assertIsNone(visualize._number(value))


if __name__=='__main__':unittest.main()

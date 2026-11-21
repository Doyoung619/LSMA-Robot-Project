import unittest
import numpy as np
from rita_core.profiles import default_profile
from rita_core.adapters.mock import MockAdapter
from rita_core.trajectory import make_segment


class MockHistoryTests(unittest.TestCase):
    def test_delayed_history_lookup_matches_gap_and_shared_boundary_semantics(self):
        p=default_profile()
        for delay in (0.,.3):
            m=MockAdapter(p,command_delay=delay*.4,tracking_delay=delay*.6)
            for i,start in enumerate((10.,12.,15.,17.)):
                q=np.full(6,i*.03)
                m.commands.append((start,make_segment([0,1,2],[q,q+.08,q])))
            def sequential(now):
                q=m._hold.copy();v=np.zeros_like(q)
                for start,s in m.commands:
                    elapsed=now-start-m.command_delay-m.tracking_delay
                    if elapsed<0:break
                    q,v,_=s.at(min(s.end,s.start+elapsed))
                    if elapsed<=s.duration:break
                return q,v
            for now in list(np.linspace(9,21,97))+[12+delay,15+delay,17+delay]:
                with self.subTest(delay=delay,now=now):
                    expected=sequential(now);actual=m._target(now)
                    np.testing.assert_allclose(actual[0],expected[0],atol=1e-12)
                    np.testing.assert_allclose(actual[1],expected[1],atol=1e-12)


if __name__=='__main__':unittest.main()

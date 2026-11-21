"""Tests tied to the finite planner's geometry, graph and service claims."""
from pathlib import Path
from collections import deque
import sys
import unittest
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(EXPERIMENT/'0_pre_and_etc'),str(EXPERIMENT/'1_finite_version')]
from rita_core.contracts import AudioAnalysis, RobotProfile, BoundaryState
from rita_core.trajectory import validate_segment, descriptor_distance
from rita_finite import FinitePlanner
from rita_finite.graph import MacroAction, CertifiedGraph


def fixture(bpm=120, seconds=1200):
    beats = np.arange(int(seconds*bpm/60)+2)*60/bpm
    return AudioAnalysis('synthetic-click-fixture','synthetic',44100,seconds+2,
                         beats,beats,np.ones(len(beats)),np.ones(len(beats)),
                         .5+.5*(np.arange(len(beats))%4==0))


def profile():
    n=6
    return RobotProfile('mock-six','mock',tuple(f'joint_{i}' for i in range(n)),
                        np.zeros(n),-np.ones(n)*.6,np.ones(n)*.6,np.ones(n)*1.2,np.ones(n)*6)


class FiniteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis,cls.profile = fixture(),profile()
        cls.planner = FinitePlanner(cls.analysis,cls.profile,seed=7)

    def test_48_distinct_profile_mapped_geometries_and_all_motion_joins(self):
        planner = self.planner
        self.assertEqual(len(planner.primitives),48)
        self.assertEqual(planner.projected_duplicates,{})
        descriptions = list(planner.geometry_descriptors.values())
        distances = [descriptor_distance(a,b) for i,a in enumerate(descriptions) for b in descriptions[i+1:]]
        self.assertGreaterEqual(min(distances),planner.config['duplicate_distance'])
        self.assertEqual(len({a.primitive_id for a in planner._component}),48)
        previous,beat = None,0
        for pid in sorted(planner.primitives):
            action = next(a for a in sorted(planner._component) if a.primitive_id == pid)
            segment = planner._make_action(action,beat)
            validation = validate_segment(segment,self.profile,previous=previous)
            self.assertTrue(validation['valid'],(pid,validation))
            np.testing.assert_allclose(segment.controls[::2,0],planner.geometry[pid][:-1])
            np.testing.assert_allclose(segment.controls[1::2,0],planner.midpoints[pid])
            self.assertEqual(len(segment.knots),2*(len(planner.primitives[pid]['normalized_waypoints'])-1)+1)
            previous,beat = segment,segment.metadata['end_beat']
        self.assertFalse(planner.stats()['whole_dense_graph_certified'])
        self.assertGreater(planner.stats()['unknown_edges'],0)

    def test_service_path_is_retained_until_the_target(self):
        planner = self.planner
        hub = max(planner._component,key=lambda a:len(planner._graph.adjacency[a]))
        leaves = [a for a in sorted(planner._component) if a.primitive_id != hub.primitive_id]
        current = leaves[0]
        target = next(a for a in leaves if a.primitive_id != current.primitive_id and
                      planner._graph.edge_status(current,a)=='unknown')
        old_connection = planner._connection
        planner._connection = lambda first,second:(False,{'test':'force sparse base path only'})
        try:
            planner._current = current
            planner._bag = [('motion',target.primitive_id)]
            planner._path = deque()
            first = planner._service(0)
            self.assertEqual(first,hub)
            self.assertTrue(planner._path)
            retained_goal = planner._goal
            second = planner._service(0)
            self.assertEqual(second.primitive_id,target.primitive_id)
            self.assertEqual(planner._goal,retained_goal)
        finally:
            planner._connection = old_connection
            planner._current = None
            planner._path.clear()
            planner._bag.clear()
            planner._goal = None

    def test_ten_minute_reference_service_covers_full_denominators(self):
        planner = FinitePlanner(self.analysis,self.profile,seed=19)
        state = BoundaryState(0,np.zeros(6),np.zeros(6),np.zeros(6))
        beat=0
        while state.t < 600:
            segment = planner.next_segment(beat,state)
            self.assertTrue(segment.validation['valid'])
            self.assertFalse(segment.metadata['geometry_generated_online'])
            beat,state = segment.metadata['end_beat'],segment.end_state()
        result=planner.stats()
        self.assertEqual(result['library_count'],48)
        self.assertEqual(result['feasible_motion_count'],48)
        self.assertEqual(result['scc_motion_count'],48)
        self.assertEqual(result['motion_coverage'],1.)
        self.assertEqual(result['rhythm_coverage'],1.)
        self.assertGreaterEqual(len(result['completed_bags']),1)
        self.assertTrue(all(b['within_bound'] for b in result['completed_bags']))
        self.assertEqual(result['certificate_resets'],0)

    def test_bounded_metrical_adaptation_at_240_bpm(self):
        planner = FinitePlanner(fixture(240),profile(),seed=7)
        self.assertEqual(planner.stats()['metrical_level'],3)
        self.assertEqual(planner.stats()['scc_motion_count'],48)
        self.assertGreaterEqual(planner.stats()['scc_rhythm_count'],12)

    def test_strength_changes_real_speed_without_changing_stored_pose_geometry(self):
        planner=self.planner
        pid=next(pid for pid,p in planner.primitives.items() if 'straight' in p['allowed_rhythms'])
        straight=planner._make_action(MacroAction(pid,'straight',2),0)
        backbeat=planner._make_action(MacroAction(pid,'backbeat',2),0)
        np.testing.assert_allclose(straight.controls[::2,0],backbeat.controls[::2,0])
        np.testing.assert_allclose(straight.controls[1::2,0],backbeat.controls[1::2,0])
        sample=np.linspace(straight.start,straight.end,401)
        self.assertGreater(np.max(np.abs(straight.at(sample)[1]-backbeat.at(sample)[1])),.005)

    def test_terminal_hold_candidates_are_rechecked_and_short_pieces_do_not_abort_search(self):
        planner=FinitePlanner(self.analysis,self.profile,config={'terminal_hold_seconds':.2})
        segment=planner.next_segment(0,BoundaryState(0,np.zeros(6),np.zeros(6),np.zeros(6)))
        self.assertTrue(segment.validation['valid'])
        self.assertAlmostEqual(segment.knots[-1]-segment.knots[-2],.2)
        self.assertAlmostEqual(segment.end,self.analysis.beat_time(segment.metadata['end_beat']))
        self.assertTrue(np.max(np.ptp(segment.controls[-1],axis=0))<1e-12)
        self.assertGreater(planner.stats()['candidate_statuses'].get('timing_construction_rejected',0),0)

    def test_configured_32_and_64_motion_libraries_remain_frozen(self):
        for count in (32,64):
            planner=FinitePlanner(self.analysis,self.profile,config={'library_size':count})
            self.assertEqual(planner.stats()['library_count'],count)
            self.assertEqual(planner.stats()['projected_unique_motion_count'],count)
            self.assertEqual(planner.stats()['feasible_motion_count'],count)


class GraphCertificateTests(unittest.TestCase):
    def test_disconnected_components_and_unknown_edges_are_reported(self):
        graph=CertifiedGraph('fixed-test')
        nodes=[MacroAction(f'm{i}',f'r{i}',0) for i in range(4)]
        for node in nodes: graph.add_node(node,2.)
        for left,right in ((0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)):
            graph.check_edge(nodes[left],nodes[right],True,{'test':'explicit edge witness'})
        selected=graph.choose_component([nodes[0]])
        self.assertEqual(selected,set(nodes[:2]))
        self.assertEqual(len(graph.nodes),4)
        self.assertEqual(len(selected),2)
        self.assertEqual(graph.edge_status(nodes[0],nodes[2]),'unknown')
        self.assertFalse(graph.coverage_certificate(set(nodes))['valid'])
        with self.assertRaises(ValueError):
            graph.shortest_service_path(nodes[0],{nodes[2]},selected)
        cert=graph.coverage_certificate(selected)
        self.assertEqual(cert['goal_count'],4)
        self.assertEqual(cert['directed_diameter'],1)
        self.assertEqual(cert['bag_completion_bound_seconds'],16.)

    def test_conditional_bag_and_revisit_bound_with_positive_cycle_revisit(self):
        graph=CertifiedGraph('fixed-ring')
        nodes=[MacroAction(f'm{i}',f'r{i}',0) for i in range(4)]
        for node in nodes: graph.add_node(node,1.5)
        for i,node in enumerate(nodes): graph.check_edge(node,nodes[(i+1)%4],True)
        component=set(nodes)
        cert=graph.coverage_certificate(component)
        self.assertEqual(cert['directed_diameter'],3)
        self.assertEqual(len(graph.shortest_service_path(nodes[0],{nodes[0]},component)),4)
        rng=np.random.default_rng(9)
        now,current=0.,nodes[0]
        last_visits={}
        intervals=[]
        for _ in range(8):
            bag=[('motion',a.primitive_id) for a in nodes]+[('rhythm',a.rhythm_id) for a in nodes]
            rng.shuffle(bag)
            started=now
            while bag:
                goal=bag[0]
                targets={a for a in component if (a.primitive_id if goal[0]=='motion' else a.rhythm_id)==goal[1]}
                path=graph.shortest_service_path(current,targets,component)
                for action in path:
                    self.assertEqual(graph.edge_status(current,action),'certified')
                    now+=graph.nodes[action]
                    visited=(('motion',action.primitive_id),('rhythm',action.rhythm_id))
                    for hit in visited:
                        if hit in last_visits: intervals.append(now-last_visits[hit])
                        last_visits[hit]=now
                    bag=[g for g in bag if g not in visited]
                    current=action
            self.assertLessEqual(now-started,cert['bag_completion_bound_seconds'])
        self.assertLessEqual(max(intervals),cert['revisit_bound_seconds'])


if __name__=='__main__':
    unittest.main()

"""Audio, planning worker, adapter dispatch, and feedback have separate owners."""
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import time
import numpy as np
from .contracts import BoundaryState,PlannerState,DanceSegment
from .trajectory import validate_segment
from .audio import VirtualClock,DACClock,decode_audio
from .adapters import make_adapter,StaleFeedback
from .adapters.base import require_motion_permission
from .evaluation import evaluate,save_report


STOP_LATCH=Path(__file__).resolve().parents[1]/"STOP_RITA"


def stop_latch_requested():
    return STOP_LATCH.exists()


class BaselinePlanner:
    def __init__(self,analysis,profile,seed,kind,config=None):
        from rita_finite import FinitePlanner
        self.base=FinitePlanner(analysis,profile,seed,{**(config or {}),"no_coverage":True})
        self.profile=profile;self.kind=kind;self.action=None;self.previous=None
    def next_segment(self,start_beat,state):
        if self.kind=="random" or self.action is None:
            seg=self.base.next_segment(start_beat,state);self.action=self.base._current
        else:
            seg=self.base._make_action(self.action,start_beat)
            if not validate_segment(seg,self.profile,self.previous)["valid"]: raise RuntimeError("Fixed-loop action ceased to be feasible")
        self.previous=seg;return seg
    def stats(self):
        result=self.base.stats();result["baseline"]=self.kind;result["coverage_certificate"]={"valid":False,"reason":"baseline has no fair service guarantee"}
        return result


def create_planner(mode,analysis,profile,seed=7,config=None):
    if mode in ("fixed-loop","random"): return BaselinePlanner(analysis,profile,seed,mode,config)
    if mode=="finite":
        from rita_finite import FinitePlanner
        return FinitePlanner(analysis,profile,seed,config)
    if mode=="infinite":
        from rita_infinite import InfinitePlanner
        return InfinitePlanner(analysis,profile,seed,config)
    raise ValueError("unknown planner mode")


def check_feedback(state,profile,now):
    if tuple(state.joint_names)!=tuple(profile.joint_names) or state.units!="rad":
        raise ValueError("Feedback units/joint order differ from profile")
    if not state.fresh or now-state.t>profile.feedback_freshness or state.t>now+.05:
        raise RuntimeError("stale or invalid feedback timestamp; stopping")
    if np.any(state.q<profile.lower-1e-6) or np.any(state.q>profile.upper+1e-6):
        raise RuntimeError("feedback joint position outside configured workspace/limits")


def run_session(analysis,profile,mode="finite",backend="mock",seed=7,seconds=60,
                headless=True,output=None,config=None,mock_options=None,execute=False,
                condition="normal",planner=None):
    if backend!="mock" and headless:
        raise ValueError("Hardware cannot run on an accelerated virtual clock")
    if backend!="mock":
        require_motion_permission(profile,execute)
        if stop_latch_requested():
            raise RuntimeError(f"STOP_RITA latch exists at {STOP_LATCH}; no automatic clear or hardware restart")
    if not np.isfinite(seconds) or seconds<=0:
        raise ValueError("Run duration must be positive finite seconds")
    config=dict(config or {})
    hardware=backend!="mock"
    if hardware:
        if mode=="infinite": config["stop_to_stop"]=True
        # The hold belongs to the certified reference and its activity check.
        # It is NOT inserted by the dispatcher after certification.
        dispatch_latency=profile.latency_seconds if profile.latency_mode=="dispatch" else 0.
        config.setdefault("terminal_hold_seconds",float(profile.endpoint.get(
            "terminal_hold_seconds",max(.2,profile.dispatch_lead_seconds+dispatch_latency+.08))))
    began=time.perf_counter()
    start_beat=0
    if hardware:
        required_lead=profile.dispatch_lead_seconds+profile.latency_seconds+.06
        while analysis.beat_time(start_beat)<required_lead:
            start_beat+=4
            if start_beat+4>=len(analysis.beats):
                raise ValueError("Not enough analyzed music for a safe initial four-beat block and dispatch lead")
    first_start=analysis.beat_time(start_beat)
    planner=planner or create_planner(mode,analysis,profile,seed,config)
    state=BoundaryState(first_start,profile.neutral.copy(),np.zeros(profile.nj),np.zeros(profile.nj))
    shared=PlannerState(state);segments=[];feedback=[];planning=[];receipts=[]
    info={"method":mode,"backend":backend,"seed":seed,"condition":condition,"seconds_requested":seconds,
          "headless_virtual_clock":headless,"planner_config":config,"mock_options":mock_options or {},
          "completed":False,"error":None,"deadline_misses":0,"hardware_verified":False,
          "initial_skipped_beats":start_beat,"initial_dance_music_seconds":first_start,
          "audio_grid_unchanged":True,"planning_during_playback":not headless}
    clock=VirtualClock() if headless else DACClock(decode_audio(analysis.source,Path(output or "outputs")/"audio_cache"),duration=analysis.duration)
    epoch=clock.epoch if headless else None
    options=dict(mock_options or {})
    if backend=="mock":
        options["seed"]=seed
        if options.get("disconnect_at") is not None: options["disconnect_at"]+=epoch if headless else time.monotonic()
    adapter=make_adapter(backend,profile,execute=execute,**options)
    future=None;pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="rita-planning")
    current_receipt=None
    def plan(beat,boundary,terminal=False):
        start=time.perf_counter()
        if terminal:
            finish=getattr(planner,"terminal_suffix",None)
            if finish is None:
                raise RuntimeError("Flowing boundary has no certified terminal suffix; stopping, not claiming successful completion")
            seg=finish(beat,boundary)
            if seg is None:
                raise RuntimeError("Planner did not provide its required terminal suffix")
            info["terminal_suffix_segments"]=info.get("terminal_suffix_segments",0)+1
        else:
            seg=planner.next_segment(beat,boundary)
        if seg.metadata.get("fallback_reason")=="planning_timeout":
            info["deadline_misses"]+=1
        elapsed=time.perf_counter()-start;seg.metadata["dispatch_planning_wall_seconds"]=elapsed
        return seg,elapsed

    def validate_for_dispatch(segment,previous=None):
        if not validate_segment(segment,profile,previous=previous)["valid"]:
            raise RuntimeError(f"Dispatch rejected reference: {segment.validation['errors']}")

    def scheduled_time(segment):
        scheduled=clock.music_to_monotonic(segment.start)
        if profile.latency_mode=="dispatch" and not capabilities.get("future_execution",False):
            scheduled-=profile.latency_seconds
        return scheduled

    def dispatch(segment):
        if hardware and stop_latch_requested():
            raise RuntimeError("STOP_RITA requested; cancel future dispatch and stop without automatic restart")
        now_music=clock.clock()
        if now_music>segment.start+profile.command_period:
            info["deadline_misses"]+=1
            raise RuntimeError("Dispatch deadline missed; catch-up burst forbidden")
        receipt=adapter.execute_segment(segment,scheduled_monotonic=scheduled_time(segment),now_music=now_music)
        receipts.append(receipt)
        return receipt

    try:
        adapter.connect()
        if headless: adapter.advance(epoch)
        capabilities=adapter.get_capabilities()
        info["adapter_capabilities"]=capabilities
        # Subscriptions need time to receive their first samples. This wait is
        # before playback and never sends a movement command.
        initial_deadline=time.perf_counter()+float(profile.endpoint.get("initial_feedback_timeout_seconds",2.))
        while True:
            try:
                initial=adapter.read_state()
                check_feedback(initial,profile,epoch if headless else time.monotonic())
                break
            except StaleFeedback:
                if headless or time.perf_counter()>=initial_deadline: raise
                time.sleep(.01)
        if np.max(np.abs(initial.q-state.q))>float(profile.endpoint.get("max_tracking_error_rad",.07)):
            raise RuntimeError("Actual initial state differs from configured neutral; no automatic home or snap")
        current,elapsed=plan(start_beat,state);planning.append(elapsed)
        if not headless:
            clock.start()
            until=time.monotonic()+2
            while clock.dac_epoch is None and time.monotonic()<until: time.sleep(.002)
            if clock.dac_epoch is None: raise RuntimeError("DAC clock did not start")
        while True:
            if current.end>analysis.beats[-1]+1e-7:
                raise RuntimeError("Planned segment extends beyond analyzed beat timestamps")
            validate_for_dispatch(current,segments[-1] if segments else None)
            shared.reserved=current.end_state();shared.queue=[current]
            next_beat=int(current.metadata["end_beat"])
            # One future phrase is planned during playback; callback never invokes planner.
            future=None
            pending_next=None;pending_receipt=None
            width=int(current.metadata.get("end_beat",next_beat)-current.metadata.get("start_beat",next_beat-4*2**current.metrical_level))
            width=max(4,width)
            enough_grid=next_beat+width<=len(analysis.beats)-1
            normal_next=current.end-first_start<seconds and enough_grid
            boundary=current.end_state()
            flowing=max(np.max(np.abs(boundary.dq)),np.max(np.abs(boundary.ddq)))>1e-7
            terminal_next=not normal_next and flowing
            need_next=normal_next or terminal_next
            if not normal_next:
                info["generation_stop_reason"]="requested_duration_reached" if current.end-first_start>=seconds else "remaining_grid_shorter_than_current_phrase"
            if current_receipt is None:
                current_receipt=dispatch(current)
            if need_next and not headless:
                future=pool.submit(plan,next_beat,current.end_state(),terminal_next)
            times=np.arange(max(clock.clock(),current.start),current.end,profile.command_period)
            if headless:
                for t in times:
                    clock.advance(t);adapter.advance(epoch+t)
                    measured=adapter.read_state();check_feedback(measured,profile,epoch+t)
                    feedback.append({"music_time":float(t),"feedback_monotonic":measured.t,"q":measured.q.tolist(),"dq":measured.dq.tolist()})
                    if np.max(np.abs(measured.q-current.at(t)[0]))>float(profile.endpoint.get("max_tracking_error_rad",.15)):
                        raise RuntimeError("Tracking error exceeded threshold; stop without reference snap")
                clock.advance(current.end);adapter.advance(epoch+current.end)
            else:
                while clock.clock()<current.end:
                    if hardware and stop_latch_requested():
                        raise RuntimeError("STOP_RITA requested during playback; stopping and canceling pending work")
                    t=clock.clock();measured=adapter.read_state();check_feedback(measured,profile,time.monotonic())
                    if clock.finished.is_set() and t<current.end-profile.command_period:
                        raise RuntimeError("Music output ended before the executing reference")
                    if t>=current.start:
                        feedback.append({"music_time":t,"feedback_monotonic":measured.t,"q":measured.q.tolist(),"dq":measured.dq.tolist()})
                        if np.max(np.abs(measured.q-current.at(t)[0]))>float(profile.endpoint.get("max_tracking_error_rad",.15)):
                            raise RuntimeError("Measured tracking error; stop without snapping")
                    if future is not None and future.done() and pending_next is None:
                        pending_next,elapsed=future.result();planning.append(elapsed)
                        validate_for_dispatch(pending_next,current)
                        if abs(pending_next.start-current.end)>1e-6:
                            raise RuntimeError("Planner changed an already reserved execution boundary")
                    if need_next and pending_receipt is None:
                        # Timestamp execution and dispatch lead are separate. A
                        # no-queue adapter may submit only during the certified
                        # constant suffix after its previous action has finished.
                        next_scheduled=clock.music_to_monotonic(current.end)
                        if profile.latency_mode=="dispatch" and not capabilities.get("future_execution",False):
                            next_scheduled-=profile.latency_seconds
                        lead=profile.dispatch_lead_seconds if hardware else 0.
                        latest=next_scheduled-lead
                        ready=bool(capabilities.get("queue",False))
                        if not ready:
                            action_end=current_receipt.get("controller_action_end_monotonic")
                            if action_end is not None and time.monotonic()>=action_end:
                                completed=getattr(adapter,"ready_for_next_segment",None)
                                ready=bool(completed()) if completed is not None else False
                        if pending_next is not None and ready:
                            if time.monotonic()>latest-.002 and hardware:
                                info["deadline_misses"]+=1
                                raise RuntimeError("Insufficient future dispatch lead after prior action completion; stopping")
                            pending_receipt=dispatch(pending_next)
                            shared.queue=[current,pending_next]
                        elif time.monotonic()>=latest-.002:
                            info["deadline_misses"]+=1
                            raise RuntimeError("Planning or controller completion missed the future handoff deadline; stopping without catch-up")
                    time.sleep(min(.01,profile.command_period))
            segments.append(current);shared.current=current.end_state();shared.queue=[]
            if not need_next:
                info["completed"]=True
                info["audio_tail_not_planned_seconds"]=max(0.,analysis.duration-current.end)
                break
            start_beat=next_beat
            if headless:
                current,elapsed=plan(start_beat,shared.current,terminal_next);planning.append(elapsed)
                current_receipt=None
            elif pending_next is not None and pending_receipt is not None:
                current,current_receipt=pending_next,pending_receipt
            else:
                info["deadline_misses"]+=1;raise RuntimeError("Planning worker missed boundary; stop, no accumulation of sleep drift")
        info["clock"]=clock.mapping()
    except (Exception,KeyboardInterrupt) as exc:
        if isinstance(exc,TimeoutError) or "planning_timeout" in str(exc):
            info["deadline_misses"]+=1
        info["error"]=f"{type(exc).__name__}: {exc}";shared.failures.append(info["error"])
    finally:
        try: info["stop_result"]=adapter.stop()
        except Exception as exc:
            info["stop_error"]=str(exc);info["completed"]=False
            if info["error"] is None: info["error"]="Robot-specific stop failed: "+str(exc)
        if future is not None: future.cancel()
        # Motion has already been stopped. Do not postpone stopping for a stuck
        # numerical worker; planner loops have their own bounded search budget.
        pool.shutdown(wait=False,cancel_futures=True)
        try: adapter.close()
        except Exception as exc:
            info["close_error"]=str(exc);info["completed"]=False
            if info["error"] is None: info["error"]="Adapter cleanup failed: "+str(exc)
        if not headless:
            try:clock.close()
            except Exception as exc:
                info["audio_close_error"]=str(exc);info["completed"]=False
                if info["error"] is None:info["error"]="Audio cleanup failed after adapter stop: "+str(exc)
    info["wall_seconds"]=time.perf_counter()-began
    info["dispatch_receipts"]=receipts
    planner_stats=planner.stats()
    planner_stats["visit_statistics_scope"]="Planner selection/reservation state; actual executed coverage is measured from completed segments in motion_diversity."
    if info["error"] is not None and "coverage_certificate" in planner_stats:
        graph_certificate=copy.deepcopy(planner_stats["coverage_certificate"])
        planner_stats["coverage_certificate"]={"valid":False,"reason":"executionfailure",
            "execution_error":info["error"],"reference_graph_certificate":graph_certificate}
        planner_stats["execution_certificate_invalidations"]=1
    report,series=evaluate(segments,profile,analysis,feedback,planner_stats,info,planning)
    if output: save_report(output,report,series)
    return report

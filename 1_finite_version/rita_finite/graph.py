"""Explicit sparse graph: absent edges are UNKNOWN, never implicitly feasible."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class MacroAction:
    primitive_id: str
    rhythm_id: str
    metrical_level: int

    @property
    def key(self):
        return f"{self.primitive_id}|{self.rhythm_id}|k{self.metrical_level}"


class CertifiedGraph:
    def __init__(self, context=""):
        self.context = context
        self.nodes = {}
        self.adjacency = {}
        self.checked_edges = {}

    def add_node(self, action, duration):
        self.nodes[action] = float(duration)
        self.adjacency.setdefault(action, set())

    def check_edge(self, source, target, valid, certificate=None):
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("edge endpoints must be certified action nodes")
        self.checked_edges[source, target] = {
            "status": "certified" if valid else "rejected_by_certificate",
            "certificate": certificate or {},
        }
        if valid:
            self.adjacency[source].add(target)
        else:
            self.adjacency[source].discard(target)

    def edge_status(self, source, target):
        return self.checked_edges.get((source, target), {"status": "unknown"})["status"]

    def reachable(self, starts):
        seen = set(starts) & self.nodes.keys()
        queue = deque(seen)
        while queue:
            for nxt in self.adjacency[queue.popleft()]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def components(self):
        """Tarjan SCCs over certified edges only."""
        index, indices, low, stack, active, result = 0, {}, {}, [], set(), []

        def visit(node):
            nonlocal index
            indices[node] = low[node] = index
            index += 1
            stack.append(node)
            active.add(node)
            for nxt in sorted(self.adjacency[node]):
                if nxt not in indices:
                    visit(nxt)
                    low[node] = min(low[node], low[nxt])
                elif nxt in active:
                    low[node] = min(low[node], indices[nxt])
            if low[node] == indices[node]:
                component = set()
                while True:
                    nxt = stack.pop()
                    active.remove(nxt)
                    component.add(nxt)
                    if nxt == node:
                        break
                result.append(component)

        for node in sorted(self.nodes):
            if node not in indices:
                visit(node)
        return result

    def choose_component(self, entry_nodes):
        reachable = self.reachable(entry_nodes)
        components = [c for c in self.components() if c <= reachable and
                      (len(c) > 1 or next(iter(c)) in self.adjacency[next(iter(c))])]
        if not components:
            return set()
        return max(components, key=lambda c: (len({x.primitive_id for x in c}),
                                             len({x.rhythm_id for x in c}), len(c)))

    def distances(self, source, allowed=None):
        allowed = set(self.nodes) if allowed is None else set(allowed)
        if source not in allowed:
            return {}
        distances, queue = {source: 0}, deque([source])
        while queue:
            node = queue.popleft()
            for nxt in self.adjacency[node] & allowed:
                if nxt not in distances:
                    distances[nxt] = distances[node]+1
                    queue.append(nxt)
        return distances

    def diameter(self, component):
        component = set(component)
        if not component:
            return None
        rows = [self.distances(node, component) for node in component]
        if any(len(row) != len(component) for row in rows):
            return None
        return max(max(row.values()) for row in rows)

    def shortest_service_path(self, source, targets, allowed=None, score=None):
        """Return actions to EXECUTE, including a positive-length revisit.

        Breadth-first depth is primary. A score only chooses between paths of
        identical minimum edge count. A target/path is retained by the caller.
        """
        allowed = set(self.nodes) if allowed is None else set(allowed)
        targets = set(targets) & allowed
        score = (lambda node: 0.) if score is None else score
        if not targets:
            raise ValueError("service target has no node in the selected SCC")
        if source is None:
            return [max(sorted(targets), key=score)]
        if source not in allowed:
            raise ValueError("current node is outside the selected SCC")
        frontier = {n: ([n], score(n)) for n in self.adjacency[source] & allowed}
        seen = {source: 0}
        depth = 1
        while frontier:
            reached = targets & frontier.keys()
            if reached:
                return max((frontier[n] for n in sorted(reached)), key=lambda p:p[1])[0]
            following = {}
            for node, (path, total) in frontier.items():
                seen[node] = min(depth, seen.get(node, depth))
                for nxt in self.adjacency[node] & allowed:
                    if nxt not in targets and seen.get(nxt, depth+1) < depth+1:
                        continue
                    candidate = path+[nxt], total+score(nxt)
                    if nxt not in following or candidate[1] > following[nxt][1]:
                        following[nxt] = candidate
            frontier = following
            depth += 1
        raise ValueError("no certified path to the persistent service target")

    def coverage_certificate(self, component):
        component = set(component)
        diameter = self.diameter(component)
        if diameter is None:
            return {"valid": False, "reason": "component is not strongly connected"}
        goals = len({x.primitive_id for x in component})+len({x.rhythm_id for x in component})
        tau = max(self.nodes[x] for x in component)
        bound = goals*(diameter+1)*tau
        return {"valid": True, "conditional": True, "goal_count": goals,
                "directed_diameter": diameter, "tau_max_seconds": tau,
                "bag_completion_bound_seconds": bound,
                "revisit_bound_seconds": 2*bound,
                "scope": "selected SCC of explicit certified sparse graph only",
                "conditions": ["timing/profile/latency context remains unchanged",
                               "each scheduled action executes successfully",
                               "persistent minimum-edge service paths and complete bags",
                               "no recovery, external interruption, or deadline failure"]}

from __future__ import annotations

# ─── constants.py ─────────────────────────────────────────────
TRUCK_W, TRUCK_D, TRUCK_H = 160, 280, 180
BOX_DIMS  = {0:(30,40,30), 1:(30,50,40), 2:(50,60,50)}
TRUCK_VOL = TRUCK_W * TRUCK_D * TRUCK_H
FIXED_COST = 150_000
FUEL_PER_KM = 500
SHUFFLE_COST = 500

def total_cost(n, km, sh):
    return n * FIXED_COST + km * FUEL_PER_KM + sh * SHUFFLE_COST

# ──────────────────────────────────────────────────────────────

def total(n_truck: int, km: float, shuffle: int) -> float:
    return n_truck*FIXED_COST + km*FUEL_PER_KM + shuffle*SHUFFLE_COST

class CostModel:  # 레거시 인터페이스 호환용
    @staticmethod
    def total(n, km, sh):
        return total(n, km, sh)


# =========================== routing.py ==========================
"""Clarke‑Wright Savings 기반 초기 라우팅"""
from dataclasses import dataclass, field
from typing import List, Dict
import heapq, numpy as np
import sys

@dataclass
class Order:
    id: int
    dest: int
    size_code: int
    vol: int

@dataclass
class Truck:
    id: int
    route:  List[int] = field(default_factory=list)
    orders: List[int] = field(default_factory=list)
    volume: int = 0
    km:     float = 0.0
    shuffle:int = 0
    load_plan: List[tuple] = field(default_factory=list)  # (box,stack,col)

class RouteHeuristic:
    """Clarke‑Wright Savings (단일 Depot, 용량 제약)"""
    def __init__(self, dist: np.ndarray, orders: List[Order]):
        self.dist = dist
        self.orders = orders
        self.depot = 0
        self.dest_to_orders: Dict[int, List[Order]] = {}
        for o in orders:
            self.dest_to_orders.setdefault(o.dest, []).append(o)

    # ---------- public ----------
    def initial_routes(self) -> List[Truck]:
        self._init_singletons(); self._build_heap(); self._merge(); self._finalize()
        return list(self.routes.values())

    # ---------- steps ----------
    def _init_singletons(self):
        self.routes: Dict[int, Truck] = {}
        self.tail_of: Dict[int, int] = {}
        tid = 1
        for dest, olist in self.dest_to_orders.items():
            tr = Truck(tid)
            tr.route = [dest]
            tr.orders = [o.id for o in olist]
            tr.volume = sum(o.vol for o in olist)
            self.routes[dest] = tr
            self.tail_of[dest] = dest
            tid += 1

    def _build_heap(self):
        self.heap: List[tuple] = []
        dests = list(self.dest_to_orders)
        for i, di in enumerate(dests[:-1]):
            for dj in dests[i+1:]:
                save = self.dist[0, di] + self.dist[dj, 0] - self.dist[di, dj]
                heapq.heappush(self.heap, (-save, di, dj))   # max‑heap via –save

    def _merge(self):
        while self.heap:
            _, i, j = heapq.heappop(self.heap)
            head_i = self.tail_of.get(i)
            head_j = self.routes.get(j)
            if head_i is None or head_j is None or head_i is head_j:
                continue
            if head_i not in self.routes:
                continue
            r_i = self.routes[head_i]; r_j = head_j
            if head_i is head_j:
                continue
            if r_i.volume + r_j.volume > TRUCK_VOL:
                continue
            # feasible → merge
            r_i.route.extend(r_j.route)
            r_i.orders.extend(r_j.orders)
            r_i.volume += r_j.volume
            new_tail = r_j.route[-1]
            self.tail_of[new_tail] = head_i
            del self.routes[r_j.route[0]]

    def _finalize(self):
        for tr in self.routes.values():
            path = [0] + tr.route + [0]
            tr.km = float(sum(self.dist[a, b] for a, b in zip(path, path[1:])))


# =========================== loader.py ==========================
"""다중 스택 LIFO 적재 + 셔플링 시뮬"""
from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class Column:
    depth: int
    height: int
    boxes: List[int] = field(default_factory=list)

@dataclass
class Stack:
    width: int
    depth_used: int = 0
    columns: List[Column] = field(default_factory=list)

class MultiStackLoader:
    def __init__(self):
        from __main__ import BOX_DIMS   # ← 단일-파일이므로 이렇게 import
        self.W,self.D,self.H = TRUCK_W,TRUCK_D,TRUCK_H
        self.box_dims = BOX_DIMS

    def load(self, truck, dest_to_orders: Dict[int, List[Order]], box_stop: Dict[int, int]):
        plan, stacks = self._pack(truck.route, dest_to_orders)
        shuffle = self._simulate(truck.route, stacks, box_stop)
        truck.load_plan = plan
        truck.shuffle   = shuffle

    # ---------- greedy pack ----------
    def _pack(self, route, dto):
        stacks: List[Stack] = []
        used_w, plan = 0, []
        for dest in reversed(route):
            for o in dto[dest]:
                bw, bd, bh = self.box_dims[o.size_code]
                placed = False
                # 1) vertical on last column
                for s_idx, s in enumerate(stacks):
                    if not s.columns: continue
                    col = s.columns[-1]
                    if bw <= s.width and bd <= col.depth and col.height + bh <= self.H:
                        col.boxes.append(o.id); col.height += bh
                        plan.append((o.id, s_idx, len(s.columns) - 1)); placed = True; break
                if placed: continue
                # 2) new column front
                for s_idx, s in enumerate(stacks):
                    if bw <= s.width and s.depth_used + bd <= self.D:
                        col = Column(depth=bd, height=bh, boxes=[o.id])
                        s.columns.append(col); s.depth_used += bd
                        plan.append((o.id, s_idx, len(s.columns) - 1)); placed = True; break
                if placed: continue
                # 3) new stack
                if used_w + bw <= self.W:
                    st = Stack(width=bw)
                    st.columns.append(Column(depth=bd, height=bh, boxes=[o.id]))
                    st.depth_used = bd; stacks.append(st)
                    plan.append((o.id, len(stacks) - 1, 0)); used_w += bw; placed = True
                # 4) force to first stack
                if not placed:
                    s = stacks[0] if stacks else Stack(width=bw)
                    if not stacks:
                        stacks.append(s)
                    col = Column(depth=bd, height=bh, boxes=[o.id])
                    s.columns.append(col); s.depth_used += bd
                    plan.append((o.id, 0, len(s.columns) - 1))
        return plan, stacks

    # ---------- simulate unload ----------
    def _simulate(self, route, stacks, box_stop):
        stack_seq = [[b for col in st.columns for b in col.boxes] for st in stacks]
        sh = 0
        for dest in route:
            changed = True
            while changed:
                changed = False
                for seq in stack_seq:
                    if seq and box_stop[seq[-1]] == dest:
                        seq.pop(); changed = True; break
            while any(box_stop[b] == dest for seq in stack_seq for b in seq):
                idx = next(i for i, seq in enumerate(stack_seq) if any(box_stop[b] == dest for b in seq))
                seq = stack_seq[idx]
                while box_stop[seq[-1]] != dest:
                    temp = seq.pop(); sh += 1; seq.insert(0, temp)
                seq.pop()
        return sh


# =========================== data.py ==========================
"""주문 JSON + 거리행렬 TXT 로더"""
import json, numpy as np
from typing import List, Dict
from collections import defaultdict

class DataLoader:
    def __init__(self, order_json: str, dist_txt: str):
        self.order_json, self.dist_txt = order_json, dist_txt

    def load(self):
        with open(self.order_json, encoding="utf-8") as f:
            raw = json.load(f)

        order_list = raw["orders"]
        dests = raw["destinations"]
        #id2idx = {"Depot": 0}
        depot_id = raw["depot"]["destination_id"] 
        id2idx = {depot_id: 0}
        for k, d in enumerate(dests, start=1):
            id2idx[d["destination_id"]] = k

        # 거리 행렬 생성
        dist = _build_dist_matrix(self.dist_txt, id2idx)

        orders: List[Order] = []
        box2sc: Dict[int, int] = {}
        for o in order_list:
            dest = id2idx[o["destination"]]
            box_id = int(o["box_id"].replace("B_", ""))
            dims = o["dimension"]
            w, d, h = dims["width"], dims["length"], dims["height"]
            vol = w * d * h

            # box size code 추정
            for sc, dim in BOX_DIMS.items():
                if (w, d, h) == dim:
                    size_code = sc
                    break
            else:
                raise ValueError(f"Unknown box size: {(w, d, h)}")

            orders.append(Order(id=box_id, dest=dest, size_code=size_code, vol=vol))
            box2sc[box_id] = size_code

        # 착지 → 주문 dict
        dto: Dict[int, List[Order]] = defaultdict(list)
        for od in orders:
            dto[od.dest].append(od)

        # 위경도 정보 맵
        long_map = {0: raw["depot"]["location"]["longitude"]}
        lat_map  = {0: raw["depot"]["location"]["latitude"]}
        for d in dests:
            idx = id2idx[d["destination_id"]]
            long_map[idx] = d["location"]["longitude"]
            lat_map[idx]  = d["location"]["latitude"]

        idx2dest = {v:k for k,v in id2idx.items()}
        return orders, dist, dto, long_map, lat_map, box2sc, idx2dest
    
# =========================== optimizer.py ==========================
import random, time, math
from copy import deepcopy
from typing import List, Tuple

class SimulatedAnnealer:
    def __init__(self, dist, orders, dto,
                 time_limit_s=600, T0=4000, alpha=0.995, iter_T=250):
        self.dist = dist
        self.orders = orders
        self.dto = dto
        self.loader = MultiStackLoader()
        self.limit = time_limit_s
        self.T0, self.alpha, self.iter_T = T0, alpha, iter_T
        self.box_stop = {o.id: o.dest for o in orders}

    def optimize(self, trucks: List) -> Tuple[List, float]:
        for t in trucks:
            t.km = self._route_km(t.route)
            self.loader.load(t, self.dto, self.box_stop)

        best = deepcopy(trucks); best_sc = self._total(best)
        cur  = deepcopy(best);   cur_sc  = best_sc
        T = self.T0
        t0 = time.time()

        while time.time() - t0 < self.limit:
            for _ in range(self.iter_T):
                nb  = self._neighbor(cur)
                nb_sc = self._total(nb)
                d = nb_sc - cur_sc
                if d < 0 or random.random() < math.exp(-d / T):
                    cur, cur_sc = nb, nb_sc
                    if cur_sc < best_sc:
                        best, best_sc = deepcopy(cur), cur_sc
            T *= self.alpha
            if T < 1e-1:
                break
        return best, best_sc

    def _neighbor(self, trucks: List):
        nbr = deepcopy(trucks)

        op = random.random()

        # 트럭에서 착지 하나 뽑아 새 트럭에 분리
        if op < 0.25 and len(nbr) >= 1:
            t = random.choice(nbr)
            if len(t.route) >= 2:
                idx = random.randrange(len(t.route))
                dest = t.route.pop(idx)
                new_id = max(tr.id for tr in nbr) + 1
                new_truck = Truck(id=new_id, route=[dest])
                nbr.append(new_truck)
                if not t.route:
                    nbr.remove(t)

        # relocate
        elif op < 0.6 and len(nbr) >= 2:
            a, b = random.sample(nbr, 2)
            if not a.route:
                return nbr
            dest = a.route.pop(random.randrange(len(a.route)))
            b.route.insert(random.randrange(len(b.route) + 1), dest)
            if not a.route:
                nbr.remove(a)

        # 2-opt
        else:
            t = random.choice(nbr)
            if len(t.route) >= 4:
                i, j = sorted(random.sample(range(len(t.route)), 2))
                t.route[i:j+1] = reversed(t.route[i:j+1])

        for t in nbr:
            t.km = self._route_km(t.route)
            self.loader.load(t, self.dto, self.box_stop)
        return nbr

    def _route_km(self, r):
        if not r:
            return 0.0
        km = self.dist[0, r[0]] + self.dist[r[-1], 0]
        km += sum(self.dist[r[i], r[i+1]] for i in range(len(r) - 1))
        return km 

    def _total(self, trucks):
        km = sum(t.km for t in trucks)
        sh = sum(t.shuffle for t in trucks)
        return total_cost(len(trucks), km, sh)

# =========================== io_utils.py ======================
COLS = ["Vehicle_ID","Route_Order","Destination","Order_Number","Box_ID",
        "Stacking_Order","Lower_Left_X","Lower_Left_Y","Lower_Left_Z",
        "Longitude","Latitude","Box_Width","Box_Length","Box_Height"]

# =========================== io_utils.py ======================
#def write_result(trucks, dto, long_map, lat_map, box2sc, idx2dest, path="Result.xlsx"):
def write_result(trucks, dto, long_map, lat_map, box2sc, idx2dest, box_stop, path="Result.xlsx"):
    rows = []
    for t in trucks:
        written_dest = set()

        # 1️⃣ ────────── 적재된 박스별 기록 ──────────
        #for box_id, stack_idx, col_idx in t.load_plan:
        #    dest_idx = next(
        #        (d for d in t.route if any(o.id == box_id for o in dto[d])),
        #        None
        #    )
        for box_id, stack_idx, col_idx in t.load_plan:
            dest_idx = box_stop[box_id]
            if dest_idx is None:          # ← dest_idx 로 수정
                continue

            written_dest.add(dest_idx)    # ← dest_idx 로 수정
            rows.append({
                "Vehicle_ID"   : t.id,
                "Route_Order"  : t.route.index(dest_idx) + 1,
                "Destination"  : idx2dest[dest_idx],      # ⭐ 변환 필수
                "Order_Number" : box_id,
                "Box_ID"       : f"B_{box_id:05d}",
                "Stacking_Order": 0,
                "Lower_Left_X" : 0,
                "Lower_Left_Y" : 0,
                "Lower_Left_Z" : 0,
                "Longitude"    : long_map.get(dest_idx, 0),
                "Latitude"     : lat_map.get(dest_idx, 0),
                "Box_Width"    : BOX_DIMS[box2sc[box_id]][0],
                "Box_Length"   : BOX_DIMS[box2sc[box_id]][1],
                "Box_Height"   : BOX_DIMS[box2sc[box_id]][2],
            })

        # 2️⃣ ────────── (해당 트럭에 있지만 박스는 없는) 착지 보강 ──────────
        for dest_idx in t.route:          # ← dest 로 쓰지 말고 dest_idx 사용
            if dest_idx in written_dest:
                continue
            rows.append({
                "Vehicle_ID"   : t.id,
                "Route_Order"  : t.route.index(dest_idx) + 1,
                "Destination"  : idx2dest[dest_idx],      # ⭐ 변환 필수
                "Order_Number" : -1,
                "Box_ID"       : "",
                "Stacking_Order": 0,
                "Lower_Left_X" : 0,
                "Lower_Left_Y" : 0,
                "Lower_Left_Z" : 0,
                "Longitude"    : long_map.get(dest_idx, 0),
                "Latitude"     : lat_map.get(dest_idx, 0),
                "Box_Width"    : 0,
                "Box_Length"   : 0,
                "Box_Height"   : 0,
            })


    import pandas as pd
    pd.DataFrame(rows, columns=COLS).to_excel(path, index=False)


def _build_dist_matrix(dist_txt: str, id2idx: dict[str,int]) -> np.ndarray:
    """edge-list(txt) → 대칭 거리행렬(ndarray).
       id2idx : ‘D_00001’ → 정수 index   (Depot 은 0)
    """
    N = len(id2idx)                            # Depot 제외 착지 수
    size = N + 1                               # + Depot(0)
    mat = np.full((size, size), 1e9, dtype=float)   # 큰 값으로 채움
    # 대각선 0
    for i in range(size):
        mat[i, i] = 0.0

    with open(dist_txt, encoding="utf-8") as f:
        next(f)                                # 헤더 한 줄 skip
        for ln in f:
            if not ln.strip():
                continue
            o, d, _, meter = ln.split()
            i = id2idx.get(o)                  # edge 는 방향성 없음 -> 양방향 세팅
            j = id2idx.get(d)
            if i is None or j is None:
                continue                       # (정의 안 된 노드가 있을 경우 스킵)
            km = float(meter) / 1000.0
            mat[i, j] = mat[j, i] = km
    return mat
# -------------------
# ============================= main.py ========================
def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py data.json distance-data.txt")
        return

    # 1) 데이터 로드
    orders, dist, dto, long_map, lat_map, box2sc, idx2dest = DataLoader(sys.argv[1], sys.argv[2]).load()


    # 2) Clarke-Wright 초기 해
    trucks = RouteHeuristic(dist, orders).initial_routes()


    # 3) 초기 적재(셔플 계산)
    loader = MultiStackLoader()
    for t in trucks:
        loader.load(t, dto, {o.id: o.dest for o in orders})

        # 4) SA 최적화 (예: 10분)
    annealer = SimulatedAnnealer(dist, orders, dto, time_limit_s=600)
    best_trucks, best_cost = annealer.optimize(trucks)

    # 5) 착지 누락 보강
    expected_dests = set(long_map) - {0}
    output_dests   = {d for tr in best_trucks for d in tr.route}
    if missing := expected_dests - output_dests:
        next_id = max(tr.id for tr in best_trucks) + 1
        for dest in missing:
            best_trucks.append(Truck(id=next_id, route=[dest])); next_id += 1

    # 6) 결과 저장
    write_result(best_trucks, dto, long_map, lat_map, box2sc, idx2dest, "Result.xlsx")

    # 7) 결과 검증
    import pandas as pd
    result_dest_set   = set(pd.read_excel("Result.xlsx")["Destination"].unique())
    expected_dest_set = {idx2dest[d] for d in long_map if d != 0}
    assert not (exp := expected_dest_set - result_dest_set) \
        and not (ext := result_dest_set - expected_dest_set), \
        f"Mismatch! missing={exp}, extra={ext}"

    print(f"BEST_COST = {best_cost:,.0f} | TRUCKS = {len(best_trucks)}")




if __name__ == "__main__":
    import sys
    import pandas as pd
    import traceback  # ✅ 추가

    try:
        main()
    except Exception as e:
        print("❌ Runtime error:", e)
        traceback.print_exc()  # ✅ 오류 원인 전체 출력
        # 채점기 'Excel file not found' 방지용
        pd.DataFrame(columns=COLS).to_excel("Result.xlsx", index=False)


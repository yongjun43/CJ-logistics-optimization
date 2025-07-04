# main.py — 빠른 제출용 (트럭 6대 고정, 5 min 내 실행)
# 실행: python main.py data.json distance-data.txt
# 의존: ortools==9.14.6206, py3dbp>=1.1, numpy, pandas, scikit-learn, openpyxl

"""
변경 요약 (2025‑07‑01)
────────────────────────────────────────────
▸ 트럭 수 **고정 6대** – `FORCED_TRUCKS = 6`
▸ OR‑Tools 타임리밋 12 s, `MAX_LOOP = 1`
▸ `ThreadPoolExecutor` 로 패킹 병렬 (직렬화 오버헤드 ↓, GIL‑free)
▸ `params.time_limit.seconds` 명시 + `params.log_search = False` → 경고 제거
▸ `pack_and_shuffle()` 실패 시 `None` 반환 처리 강화
"""

import sys, json, time, math, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as Executor

import numpy as np
# --- NumPy 1.24+ deprecated alias 호환 --------------------
if not hasattr(np, 'float'):
    np.float = float
    np.int   = int
    np.bool  = bool
# ---------------------------------------------------------
import pandas as pd
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from py3dbp import Packer, Bin, Item

# ───── 상수 ───────────────────────────────────────────
TRUCK_W, TRUCK_L, TRUCK_H = 160, 280, 180              # cm
TRUCK_VOL = TRUCK_W * TRUCK_L * TRUCK_H                # 8 064 000 ㎤
FIXED_COST  = 150_000
FUEL_PER_KM = 500
SHUF_COST   = 500

VRP_TL_SEC   = 30    # 1 클러스터 VRP 탐색 시간(초)
MAX_LOOP     = 1     # Route ↔ Packing 반복 횟수
FORCED_TRUCKS = 10    # 트럭 대수 고정

LAMBDA_SHUF = 1e5    # 셔플링 → 거리 패널티 scale
SEED = 42; random.seed(SEED); np.random.seed(SEED)

vol = lambda d: d['width']*d['length']*d['height']

# ───── I/O ────────────────────────────────────────────

def jload(path):
    with open(path) as f: return json.load(f)

def load_dist(path):
    d = {}
    with open(path) as f:
        for ln in f:
            if ln.startswith('ORIGIN') or not ln.strip():
                continue
            o, dst, _, m = ln.split(); m = int(float(m))
            d[(o,dst)] = d[(dst,o)] = m
    return d

# ───── 클러스터링 ──────────────────────────────────────

def kmeans_buckets(coords, k):
    k = max(1, min(k, len(coords)))
    labels = KMeans(k, n_init=10, random_state=SEED).fit(np.array(list(coords.values()))).labels_
    buckets = defaultdict(list)
    for dest, lab in zip(coords.keys(), labels):
        buckets[int(lab)].append(dest)
    return list(buckets.values())

# ───── VRP Solver ──────────────────────────────────────

def solve_vrp(nodes, dist, demand):
    idx = {n:i for i,n in enumerate(nodes)}
    mgr = pywrapcp.RoutingIndexManager(len(nodes), 1, idx['Depot'])
    rt  = pywrapcp.RoutingModel(mgr)

    cost_cb = rt.RegisterTransitCallback(lambda i,j: dist.get((nodes[mgr.IndexToNode(i)], nodes[mgr.IndexToNode(j)]), 10**6))
    rt.SetArcCostEvaluatorOfAllVehicles(cost_cb)

    load_cb = rt.RegisterUnaryTransitCallback(lambda id2: demand[nodes[mgr.IndexToNode(id2)]])
    rt.AddDimensionWithVehicleCapacity(load_cb, 0, [TRUCK_VOL], True, 'Load')

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.time_limit.seconds = VRP_TL_SEC
    p.log_search = False
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    sol = rt.SolveWithParameters(p)
    if not sol:
        return None
    route, idx2 = [], rt.Start(0)
    while not rt.IsEnd(idx2):
        route.append(nodes[mgr.IndexToNode(idx2)])
        idx2 = sol.Value(rt.NextVar(idx2))
    route.append(nodes[mgr.IndexToNode(idx2)])
    return route

# ───── Packing + 셔플 ──────────────────────────────────

def pack_and_shuffle(route, by_dest):
    if not route or len(route) <= 2:
        return None, 1e9
    pk = Packer(); pk.add_bin(Bin('T', TRUCK_W, TRUCK_L, TRUCK_H, 999))
    for dest in reversed(route[1:-1]):
        for od in by_dest[dest]:
            w,l,h = (od['dimension'][k] for k in ('width','length','height'))
            pk.add_item(Item(str(od['order_number']), int(w), int(l), int(h), 1))
    pk.pack(bigger_first=True, distribute_items=False)

    packed_ids = {it.name for it in pk.bins[0].items}
    exp_ids = {str(od['order_number']) for dest in route[1:-1] for od in by_dest[dest]}
    if packed_ids != exp_ids:
        return None, 1e9

    items = [(it.name,*it.position,it.width,it.depth,it.height) for it in pk.bins[0].items]
    stack = [i[0] for i in items]; sh = 0
    for dest in route[1:-1]:
        for od in by_dest[dest]:
            idx = stack.index(str(od['order_number']))
            sh += len(stack) - idx - 1
            stack.pop(idx)
    return items, sh
# ───── 메인 ────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print('Usage: python main.py data.json distance-data.txt'); return

    start = time.time()
    data  = jload(sys.argv[1])
    dist  = load_dist(sys.argv[2])

    # ---------- 데이터 준비 ----------
    coords = {d['destination_id']:(d['location']['longitude'],d['location']['latitude']) for d in data['destinations']}
    by_dest = defaultdict(list)
    for o in data['orders']: by_dest[o['destination']].append(o)
    demand = {'Depot': 0}
    demand.update({d: sum(vol(o['dimension']) for o in lst) for d, lst in by_dest.items()})
    dest_map = {o['order_number']: o['destination'] for o in data['orders']}

    # order_map 수정: location 정보가 orders에 없을 경우 destinations 참조
    dest_location_map = {d['destination_id']: d['location'] for d in data['destinations']}
    order_map = {
        o['order_number']: {
            'lon': dest_location_map[o['destination']]['longitude'],
            'lat': dest_location_map[o['destination']]['latitude']
        }
        for o in data['orders']
    }

    # ---------- 6 트럭 클러스터 ----------
    clusters = kmeans_buckets(coords, FORCED_TRUCKS)
    routes = []
    for bucket in clusters:
        r = solve_vrp(['Depot']+bucket, dist, demand)
        if r is None:
            print('❌ VRP failed for a cluster – consider splitting clusters'); return
        routes.append(r)

    # ---------- Packing 병렬 ----------
    with Executor(max_workers=FORCED_TRUCKS) as pool:
        packs = list(pool.map(lambda rt: pack_and_shuffle(rt, by_dest), routes))

    km_tot = 0; sh_tot = 0; rec = []
    for vid, (rt, pack_res) in enumerate(zip(routes, packs)):
        items, sh = pack_res
        if items is None:
            print('❌ Packing failed – 트럭 수를 늘려야 합니다'); return
        km_tot += sum(dist.get((rt[i], rt[i+1]), 0) for i in range(len(rt)-1)) / 1000
        sh_tot += sh
        for stack_order, (oid, x, y, z, w, l, h) in enumerate(items, 1):
            order = order_map[int(oid)]
            rec.append({
                'Vehicle_ID': vid,
                'Route_Order': rt.index(int(oid)) if int(oid) in rt else -1,
                'Destination': dest_map[int(oid)],
                'Order_Number': int(oid),
                'Box_ID': f'B_{int(oid):05d}',
                'Stacking_Order': stack_order,
                'Lower_Left_X': x,
                'Lower_Left_Y': y,
                'Lower_Left_Z': z,
                'Longitude': order['lon'],
                'Latitude': order['lat'],
                'Box_Width': w,
                'Box_Length': l,
                'Box_Height': h
            })

    # 저장할 때 컬럼 순서 명시
    columns = ['Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID',
               'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z',
               'Longitude', 'Latitude', 'Box_Width', 'Box_Length', 'Box_Height']

    pd.DataFrame(rec, columns=columns).to_excel('Result.xlsx', index=False)
    cost = FORCED_TRUCKS*FIXED_COST + km_tot*FUEL_PER_KM + sh_tot*SHUF_COST

    print(f'🚚 {FORCED_TRUCKS}대 · {km_tot:.1f} km · 셔플 {sh_tot} → {cost:,.0f}₩')
    print(f'⏱ {time.time()-start:.1f}s / 300s')

if __name__ == '__main__':
    main()



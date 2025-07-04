# 🚚 CJ-Logistics-Optimization

**“한 번에 싣고, 최소 거리로 달린다.”**  
CVRP(용량 제약 차량경로) + 3-D *LIFO* 적재 문제를 동시에 풀어  
**트럭 수 · 운행 거리 · 셔플링 이동** 3중 비용을 최소화하는 Python 구현입니다.

<div align="center">
  <img src="assets/cj_banner.png" width="750"/>
</div>

---

## 📌 핵심 개념

| 단계 | 알고리즘 | 설명 |
|------|----------|------|
| 1️⃣ 초기 라우팅 | **Clarke–Wright Savings** | 단일 Depot, 용량 제약 고려 |
| 2️⃣ 적재 시뮬 | **Greedy Multi-Stack Loader** | 트럭 적재함 *(160×280×180 cm)* 을 다중 스택으로 가정해 LIFO 적재 |
| 3️⃣ 비용 최적화 | **Simulated Annealing** | 2-opt·Relocate·Split 등 이웃 연산으로 라우트·적재 동시 개선 |

**총비용**  
\[
\text{Cost}=n_{\mathrm{truck}}\times150{,}000 +
\text{km}\times500 +
\text{shuffle}\times500
\]

---

## 🗂️ 코드 & 폴더 구조
```
CJ-logistics-optimization/
├── constants.py          # 전역 상수·비용 함수
├── routing.py            # Clarke–Wright 초기 해
├── loader.py             # 다중 스택 LIFO 적재 + 셔플 계산
├── optimizer.py          # SA 엔진 (Neighbor/Cost/Temp)
├── data.py               # JSON & 거리행렬 로더
├── io_utils.py           # Result.xlsx 작성
├── main.py               # 파이프라인 진입점
├── requirements.txt
└── README.md
```

---

## 📁 입·출력 형식

### 1. `data.json`
```jsonc
{
  "depot": { "destination_id": "D_00000", "location": {"longitude": 126.97,"latitude": 37.56} },
  "destinations": [
    { "destination_id": "D_00001", "location": {"longitude": ..., "latitude": ...} },
    ...
  ],
  "orders": [
    { "box_id":"B_00001", "destination":"D_00003",
      "dimension":{"width":30,"length":40,"height":30} },
    ...
  ]
}
```

### 2. `distance-data.txt`
```
OriginID DestinationID Dummy Distance(m)
D_00000  D_00001    0    42500
D_00000  D_00002    0    33500
...
```

### 3. 결과 `Result.xlsx`
| Vehicle_ID | Route_Order | Destination | Order_Number | Box_ID | … | Longitude | Latitude |
|------------|------------|-------------|--------------|--------|---|-----------|----------|
| 1 | 1 | D_00012 | 37 | B_00037 | … | 127.1 | 37.4 |
| … | … | … | … | … | … | … | … |

---

## ⚙️ 설치 & 실행

```bash
# 1) 의존성
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # numpy, pandas 등

# 2) 최적화 실행
python main.py data.json distance-data.txt

# 3) 결과
ls Result.xlsx          # 트럭별 경로·적재 시트 확인
```

> **런타임 오류가 발생해도** 빈 `Result.xlsx` 를 출력해 채점기 오류를 방지합니다.

---

## 🔑 주요 모듈 내부

### `constants.py`
```python
TRUCK_W, TRUCK_D, TRUCK_H = 160, 280, 180          # cm
BOX_DIMS = {0:(30,40,30), 1:(30,50,40), 2:(50,60,50)}
FIXED_COST, FUEL_PER_KM, SHUFFLE_COST = 150_000, 500, 500
```

### `routing.py`
* **_init_singletons → _build_heap → _merge**  
  Savings 값을 최대 heap 에 넣어 두 라우트를 병합  
* 용량 초과 시 병합 거부 → 트럭 수 최소화 보장 X → SA에서 보정

### `loader.py`
* 트럭을 **Stack 목록**으로 보고 *뒤에서 앞으로* 주문을 적재  
* unload 시 목적지 박스가 스택 바닥에 있으면 앞쪽으로 이동시키며 `shuffle++`

### `optimizer.py`
| 이웃 연산 | 설명 |
|-----------|------|
| Split | 한 트럭에서 착지 1곳 분리 → 새 트럭 |
| Relocate | 착지 1곳을 다른 트럭 임의 위치로 이동 |
| 2-opt | 한 트럭 내 구간을 뒤집어 순서 변경 |

온도 ☞ `T0=4000`, `alpha=0.995`, `iter_T=250` *(10분 실행 시 권장)*

---

## 📈 사용 팁
| 목적 | 파라미터 |
|------|----------|
| 실행 시간 단축 | `SimulatedAnnealer(time_limit_s=120)` |
| 초기 온도 조정 | `T0` 값 낮추면 탐색 범위 ↓ |
| 셔플 패널티 조정 | `constants.py → SHUFFLE_COST` |

---

## 📝 라이선스
MIT License – 자유 변경·배포 가능. 기업·학술 활용 시 **레포 링크**를 꼭 남겨주세요.

---

> Made with **Python 3.11** & 🧠  
> 문의: yongjun43 (at) gmail.com

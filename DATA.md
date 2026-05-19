# Data explanation
**minimal toy scene** end-to-end: one lane, one car, one cone, one traffic light.

---

## 0. Scene (before any encoding)

```
        y
        ^
        |     [vehicle]  (10 m ahead, moving forward)
        |         □
        |    ─────────────  centerline
        |   |   lane    |
        |   |  60 km/h  |
        |   ─────────────
        |  🚦 (controls this lane connector — RED now)
        |
   ego ●──────> x
  (0,0), heading 0°, 5 m/s
        |
        |  ▲ cone at (3, -2), 0.5×0.5 m
```

- **Ego**: rear axle at `(0, 0)`, heading `0`, speed `5 m/s`.
- **Lane**: straight, speed limit `16.7 m/s` (~60 km/h), on route.
- **TL**: status `RED` on this lane’s connector for the last 2 s (2 history steps + present → 3 values in a tiny example; real run uses **21**).
- **Vehicle**: center `(10, 0)`, heading `0`, velocity `(8, 0)`, box `2×4 m`.
- **Cone**: center `(3, -2)`, heading `0`, box `0.5×0.5 m`.

---

## 1. Feature builder output (numpy, map frame)

### Map — one polygon token (the lane)

`sample_points = 20` → each polyline has **20** segments.

| Tensor | Shape (this example) | Meaning |
|--------|----------------------|---------|
| `point_position[0]` | `(3, 20, 2)` | 3 polylines × 20 points × (x,y) |
| `polygon_type[0]` | `0` | `LANE` |
| `polygon_on_route[0]` | `True` | on mission route |
| `polygon_speed_limit[0]` | `16.7` | m/s |
| `polygon_has_speed_limit[0]` | `True` | |
| `polygon_tl_status[0]` | `(21,)` e.g. `[…, RED, RED, RED]` | TL history on **this lane id** |
| `polygon_center[0]` | `(3,)` e.g. `(5, 0, 0)` | mid-centerline (x,y,θ) |

**Centerline only (polyline 0), first 3 points:**

```text
point_position[0, 0, 0] = (0.0, 0.0)    # start of lane near ego
point_position[0, 0, 1] = (1.0, 0.0)
point_position[0, 0, 2] = (2.0, 0.0)
...
point_vector[0, 0, 0]   = (1.0, 0.0)    # segment to next point
```

Left/right boundaries are parallel polylines at ±1.75 m in `y` (not shown).

### Agent — ego (index 0) + vehicle (index 1)

`T = 21` timesteps (past + now). Only **present** matters for “who is in the scene”; history fills masks.

| | ego `[0]` | vehicle `[1]` |
|---|-----------|----------------|
| `position[:, 20]` | `(0,0)` at present | `(10,0)` at present |
| `velocity[:, 20]` | `(5,0)` in ego frame | `(8,0)` global → rotated after normalize |
| `heading[:, 20]` | `0` | `0` |
| `shape[:, 20]` | `(2, 4.5)` ego L×W | `(2, 4)` |
| `category` | `0` EGO | `1` VEHICLE |
| `valid_mask[:, 20]` | all `True` | `True` only when track exists |

### Static object — cone (index 0)

| Field | Value |
|-------|--------|
| `position[0]` | `(3, -2)` |
| `heading[0]` | `0` |
| `shape[0]` | `(0.5, 0.5)` |
| `category[0]` | `2` → `TRAFFIC_CONE` |
| `valid_mask[0]` | `True` |

**No TL xyz here** — TL is only on the map row for the lane.

---

## 2. After `PlutoFeature.normalize` (ego frame)

Ego rear axle → origin, rotate so ego heading = 0:

```text
current_state[:3]  →  (0, 0, 0)

agent[0] position  →  still ~(0, 0)
agent[1] position  →  still ~(10, 0)   # ahead on x-axis
cone position      →  (3, -2)          # right of ego

map centerline     →  same numbers if ego was already aligned
polygon_tl_status  →  unchanged (discrete enum, not a vector)
```

Crop: keep polygons with **any** centerline point in `|x|,|y| < radius` (e.g. 100 m). Our lane stays; far-away lanes drop.

---

## 3. Map token — geometry then pooling

**One lane → one vector of dim `D` (e.g. 128).**

### Step A — per centerline point (20 points), local 10-D feature

Example for **point 0** (start of centerline):

```text
rel_to_center     = point_pos[0,0,0] - polygon_center[:2]
                  ≈ (0,0) - (5,0) = (-5, 0)

segment_vector    = (1, 0)
heading           = (cos 0, sin 0) = (1, 0)

offset_left       = left[0] - center[0]  ≈ (0, 1.75)
offset_right      = right[0] - center[0] ≈ (0, -1.75)

→ one row of shape (10,) for that point
```

`PointsEncoder` runs MLP on each valid point, **max-pools** over 20 → `geom_emb` shape `(D,)`.

### Step B — add scalar / discrete signals

```text
map_token[0] = geom_emb
             + type_emb(0)              # LANE
             + on_route_emb(1)          # on route
             + tl_emb([…,RED,RED,RED])  # 21-step history → D-dim
             + fourier(16.7)            # speed limit
```

**Traffic light in this example:** not “a point at the mast,” but **“lane token 0 has been RED for the last N frames.”**

---

## 4. Agent token — deltas over time

For the **vehicle** at present step, the encoder uses **changes** between consecutive valid steps, not raw `(10,0)` every frame.

Tiny `T=3` illustration (real `T=21`):

| Step | position | velocity |
|------|----------|----------|
| t=0 | (8, 0) | (7, 0) |
| t=1 | (9, 0) | (7.5, 0) |
| t=2 | (10, 0) | (8, 0) |

Features fed to `SequenceEncoder` (per step pair):

```text
Δpos     = (1, 0), (1, 0)
Δvel     = (0.5, 0), (0.5, 0)
Δheading → (cos 0, sin 0) = (1, 0), (1, 0)
shape    = (2, 4) at t=1, t=2
valid    = 1

→ 9 numbers per interval × sequence → one agent token (D,) + type_emb(VEHICLE)
```

**Ego** (many checkpoints): often **not** this history path; instead `current_state` → `[0,0,0, 5, accel, steer, yaw_rate]` → MLP → replaces `agent[0]` token.

---

## 5. Cone token — box, not polyline

```text
shape_emb  = FourierEmbedding([0.5, 0.5])
type_emb   = Embedding(TRAFFIC_CONE)
cone_token = shape_emb + type_emb          # shape (D,)

cone_pose  = (3, -2, 0)                    # for transformer position only
```

No 20-point polyline for the cone in the **default** pipeline.

---

## 6. Transformer sees three token types

```text
Token index   Type        Position for attention     Content dim
─────────────────────────────────────────────────────────────
0             ego         (0, 0, 0)                  D  (from current_state or history)
1             vehicle     (10, 0, 0)                 D
2             lane        (5, 0, 0)  polygon_center   D  (geom + type + route + TL + speed)
3             cone        (3, -2, 0)                 D  (size + category)

Each token:  x_token + FourierEmbedding(x, y, θ)
Then:        self-attention (ego can attend lane TL state, vehicle, cone, …)
```

So in one forward pass the model can learn patterns like: **“lane token says RED + vehicle ahead + cone on the right”** without ever placing a TL at `(x_tl, y_tl)`.

---

## 7. One-line “recipe” per object type

| Object | Built as | Encoded as |
|--------|----------|------------|
| Lane | 3×20 polylines + metadata | Pool points → + type, route, **TL history**, speed |
| Vehicle | 21× (pos, vel, θ, box) | Δ features over time + category |
| Cone | (x,y,θ), (w,l), type | Fourier(w,l) + category; pose separate |
| Traffic light | Status on **lane id** × time | Conv over 21 statuses → added to **lane** token |


```python
m, a = 0, 1  # first lane, first non-ego agent
print("lane type:", snap["map"]["polygon_type"][m])
print("lane pt0:", snap["map"]["point_position"][m, 0, 0])
print("speed:", snap["map"]["polygon_speed_limit"][m], snap["map"]["polygon_has_speed_limit"][m])
print("TL last 5:", snap["map"]["polygon_tl_status"][m, -5:])
print("vehicle pos now:", snap["agent"]["position"][a, builder.history_samples])
print("vehicle shape:", snap["agent"]["shape"][a, builder.history_samples])
if "static_objects" in snap:
    print("cone:", snap["static_objects"]["position"][0], snap["static_objects"]["category"][0])
```

```python
 agent                                       
   position                                  (1, 3, 101, 2)          float32
   heading                                   (1, 3, 101)             float32
   velocity                                  (1, 3, 101, 2)          float32
   shape                                     (1, 3, 101, 2)          float32
   category                                  (1, 3)                  int8
   valid_mask                                (1, 3, 101)             bool
   turn_signals                              (1, 3, 101)             int8
   target                                    (1, 3, 80, 3)           float32
 map                                         
   point_position                            (1, 110, 3, 20, 2)      float32
   point_vector                              (1, 110, 3, 20, 2)      float32
   point_orientation                         (1, 110, 3, 20)         float32
   point_side                                (1, 110, 3)             int8
   polygon_center                            (1, 110, 3)             float32
   polygon_position                          (1, 110, 2)             float32
   polygon_orientation                       (1, 110)                float32
   polygon_type                              (1, 110)                int8
   polygon_on_route                          (1, 110)                bool
   polygon_tl_status                         (1, 110, 21)            int8
   polygon_has_speed_limit                   (1, 110)                bool
   polygon_speed_limit                       (1, 110)                float32
   polygon_road_block_id                     (1, 110)                int32
   border_type                               (1, 110, 20, 2)         int8
   access_type                               (1, 110)                int8
   valid_mask                                (1, 110, 20)            bool
 reference_line                              
   position                                  (1, 1, 120, 2)          float32
   vector                                    (1, 1, 120, 2)          float32
   orientation                               (1, 1, 120)             float32
   valid_mask                                (1, 1, 120)             bool
   future_projection                         (1, 1, 8, 2)            float32
 current_state                               (1, 7)                  float32
 origin                                      (1, 2)                  float32
 angle                                       (1,)                    float32
 cost_maps                                   (1, 600, 600, 1)        float16
 obstacle                                    
   point_position                            (1, 3, 4, 2)            float32
   point_vector                              (1, 3, 4, 2)            float32
   point_orientation                         (1, 3, 4)               float32
   point_valid_mask                          (1, 3, 4)               bool
   polygon_center                            (1, 3, 3)               float32
   polygon_type                              (1, 3)                  int8
```

## Batching:
```python
sample @ iter 0: agents=3, map polygons=110, ref_lines=1
sample @ iter 5: agents=3, map polygons=110, ref_lines=1
batched shapes: agent.position (2, 3, 101, 2), map.point_position (2, 110, 3, 20, 2)
  -> batch dim B=2; agent axis max_A=3, map axis max_M=110
Padding: extra agent/map slots are zeros; encoder key_padding_mask marks them ignored.
```

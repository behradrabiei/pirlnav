"""Login-node probe of the ORACLE_REVEAL pipeline (renderer-less, CPU).

1. Pure-math unit tests of OracleRevealCloud frustum gates and ego_snapshot.
2. cast_ray smoke test (does bullet return hits at all?).
3. Replay recorded demo poses through OracleRevealCloud exactly like
   EgoObjectCloudSensor does, and report reveal behaviour + through-wall
   leak checks + goal-revealed-by-end stats.
"""
import gzip
import json
import os
import sys

import numpy as np
import quaternion

sys.path.insert(0, "/workspace/pirlnav")

from pirlnav.task.object_cloud import OracleRevealCloud  # noqa: E402

SCENES_DIR = "/projects/bgon/brabiei/MP3D/scenes/mp3d"
CACHE = "/workspace/pirlnav/data/object_clouds_ply/mp3d"
CONTENT = ("/projects/bgon/brabiei/MP3D/demo_episodes/data/datasets/objectnav/"
           "objectnav_mp3d_thda_70k_21cat/objectnav/objectnav_mp3d_thda_70k_21cat/"
           "train/content")
CAT = ['chair','table','picture','cabinet','cushion','sofa','bed',
       'chest_of_drawers','plant','sink','toilet','stool','towel','tv_monitor',
       'shower','bathtub','counter','fireplace','gym_equipment','seating','clothes']

HFOV, ASPECT = 79.0, 480.0 / 640.0
GATES = dict(min_dist=0.5, max_dist=5.0, min_angular_size=0.0145,
             occlusion_margin=0.75)


def unit_tests():
    print("=== unit tests (no sim) ===")
    free_ray = lambda o, d, m: None  # noqa: E731
    pos = np.array([[0, 0, -3.0],   # straight ahead
                    [0, 0, +3.0],   # behind
                    [3.0, 0, 0],    # to the right
                    [0, 0, -8.0],   # ahead but beyond max_dist
                    [0, 0, -0.2]])  # too close
    tids = np.arange(5); radii = np.full(5, 0.3)
    rc = OracleRevealCloud(pos, tids, radii, HFOV, ASPECT,
                           cast_ray_fn=free_ray, **GATES)
    new = rc.update(np.zeros(3), quaternion.one)  # identity: facing -Z
    print("identity-facing reveal (expect [0]):", new)

    rc.reset()
    q_left = quaternion.from_rotation_vector([0, np.pi / 2, 0])  # yaw 90 CCW
    new = rc.update(np.zeros(3), q_left)  # now facing -X
    print("yaw+90 reveal (expect [] since -X has nothing):", new)

    # object on -X axis should reveal when facing left
    rc2 = OracleRevealCloud(np.array([[-3.0, 0, 0]]), np.array([0]),
                            np.array([0.3]), HFOV, ASPECT,
                            cast_ray_fn=free_ray, **GATES)
    print("yaw+90 sees -X object (expect [0]):", rc2.update(np.zeros(3), q_left))

    # ego_snapshot conventions
    rc3 = OracleRevealCloud(np.array([[1.0, 0, -2.0]]), np.array([0]),
                            np.array([0.3]), HFOV, ASPECT,
                            cast_ray_fn=free_ray, **GATES)
    rc3.update(np.zeros(3), quaternion.one)
    ego, _ = rc3.ego_snapshot(np.zeros(3), quaternion.one)
    print("ego identity (expect [[1,0,-2]]):", ego)
    ego, _ = rc3.ego_snapshot(np.zeros(3), q_left)
    print("ego yaw+90 (expect approx [[-2,0,-1]]):", np.round(ego, 3))

    # blocked ray => no reveal
    blocked = lambda o, d, m: 0.5  # noqa: E731  wall 0.5m away
    rc4 = OracleRevealCloud(np.array([[0, 0, -3.0]]), np.array([0]),
                            np.array([0.3]), HFOV, ASPECT,
                            cast_ray_fn=blocked, **GATES)
    print("blocked reveal (expect []):", rc4.update(np.zeros(3), quaternion.one))


def make_sim(scene):
    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = os.path.join(SCENES_DIR, scene, f"{scene}.glb")
    cfg.scene_dataset_config_file = os.path.join(
        SCENES_DIR, "mp3d.scene_dataset_config.json")
    cfg.create_renderer = False
    cfg.enable_physics = True
    agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=[])
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


def probe_scene(scene, n_episodes=8):
    import habitat_sim
    print(f"\n=== scene {scene} ===")
    sim = make_sim(scene)

    def cast(origin, direction, max_dist):
        res = sim.cast_ray(habitat_sim.geo.Ray(origin, direction), max_dist)
        return float(res.hits[0].ray_distance) if res.has_hits() else None

    # cast_ray smoke test: straight down from a navigable point must hit floor
    p = sim.pathfinder.get_random_navigable_point()
    down = cast(np.asarray(p) + [0, 0.88, 0], np.array([0.0, -1.0, 0.0]), 10.0)
    up = cast(np.asarray(p) + [0, 0.88, 0], np.array([0.0, 1.0, 0.0]), 50.0)
    print(f"cast_ray down from navigable+0.88m: {down} (expect ~0.88); up: {up}")
    if down is None:
        print("!!! cast_ray returns NO HITS -- physics/collision mesh broken; "
              "every in-frustum object would reveal through walls")

    z = np.load(os.path.join(CACHE, scene, f"{scene}.npz"))
    obj_pos, task_ids, radii, inst = (z["obj_pos"], z["task_ids"],
                                      z["radii"], z["instance_ids"])
    inst_to_idx = {int(i): k for k, i in enumerate(inst)}
    print(f"cloud: {len(task_ids)} objects, radii median "
          f"{np.median(radii):.2f} p90 {np.percentile(radii,90):.2f} "
          f"max {radii.max():.2f}")

    d = json.load(gzip.open(os.path.join(CONTENT, f"{scene}.json.gz")))
    goals_by_cat = d["goals_by_category"]

    rng = np.random.RandomState(0)
    eps = [d["episodes"][i] for i in
           rng.choice(len(d["episodes"]), size=n_episodes, replace=False)]

    stats = []
    for ep in eps:
        rc = OracleRevealCloud(obj_pos, task_ids, radii, HFOV, ASPECT,
                               cast_ray_fn=cast, **GATES)
        replay = ep["reference_replay"]
        goal_key = f"{scene}.glb_{ep['object_category']}"
        goal_idx = [inst_to_idx[int(g["object_id"])]
                    for g in goals_by_cat[goal_key]
                    if int(g["object_id"]) in inst_to_idx]
        step0 = None
        for t, step in enumerate(replay):
            st = step["agent_state"]
            cam_pos = np.asarray(st["position"], dtype=np.float64) + [0, 0.88, 0]
            r = st["rotation"]
            cam_rot = np.quaternion(r[3], r[0], r[1], r[2])
            rc.update(cam_pos, cam_rot)
            if step0 is None:
                step0 = int(rc.revealed.sum())
        n_rev = int(rc.revealed.sum())
        goal_rev = any(rc.revealed[g] for g in goal_idx)
        stats.append((len(replay), step0, n_rev, len(task_ids), goal_rev,
                      ep["object_category"]))
    print("per-episode: steps, revealed@step0, revealed@end/total, goal-revealed, category")
    for s in stats:
        print("  ", s)
    print(f"goal revealed by episode end: {sum(s[4] for s in stats)}/{len(stats)}")
    sim.close()


if __name__ == "__main__":
    unit_tests()
    for scene in sys.argv[1:] or ["17DRP5sb8fy"]:
        probe_scene(scene)

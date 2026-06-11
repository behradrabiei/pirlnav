"""Dump per-scene object clouds with *vertex-based* centroids from MP3D
semantic meshes.

Why not ``dump_scene_object_clouds.py`` (annotation AABB centers)? The
``*_semantic.ply`` mesh is the geometry source of truth: it is what the
SEMANTIC_SENSOR renders, and its per-instance vertex centroids agree with
the annotation boxes on well-behaved instances (median gap ~0.07m across
the train split). Beyond removing any annotation-box risk, the cache also
carries per-instance radii + instance ids, which the reveal gates and the
comparison tooling need and the AABB dump lacks. (The online pipeline this
replaces had its own artifacts -- 5m MAX_DEPTH clamp, semantic-mesh holes
depositing phantom objects on occluders -- diagnosed per-frame 2026-06-09.)

For every scene directory under --scenes-dir containing
``<scene>_semantic.ply``, this script:

  1. opens a renderer-less habitat-sim (CPU-only) to read the instance ->
     task-id mapping from the semantic annotations (same 21-class filter as
     the online cloud),
  2. parses the ply directly with numpy (packed binary; vertex = xyz f4 +
     rgb u1, face = u1 count + 3 i4 indices + i4 object_id),
  3. auto-detects the ply -> habitat world frame (identity vs +/-90deg about
     X) by aligning per-instance vertex centroids against the annotation
     centers of well-behaved instances,
  4. writes ``<output-dir>/<scene>/<scene>.npz`` with obj_pos (vertex
     centroids), task_ids, radii (max vertex distance), instance_ids --
     the schema EgoObjectCloudSensor's ORACLE_REVEAL cache mode loads.

Run inside the container (no GPU needed):
  python scripts/dump_ply_object_clouds.py                  # all scenes
  python scripts/dump_ply_object_clouds.py --scenes 17DRP5sb8fy 1LXtFkjw3qL
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

VERTEX_DTYPE = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
FACE_DTYPE = np.dtype([("n", "u1"), ("idx", "<i4", 3), ("oid", "<i4")])

# Candidate ply -> habitat world rotations (habitat is y-up; raw scans z-up).
FRAME_CANDIDATES = {
    "identity": np.eye(3),
    "x-90": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "x+90": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", default="data/scene_datasets/mp3d")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="Scene ids; default = every dir with a semantic ply.")
    p.add_argument("--output-dir", default="data/object_clouds_ply/mp3d")
    return p.parse_args()


def read_semantic_ply(path):
    """Return (vertices (V, 3) float64, face_vertex_idx (F, 3), face_oid (F,))."""
    with open(path, "rb") as f:
        counts = {}
        line = f.readline()
        while line.strip() != b"end_header":
            parts = line.split()
            if parts[0] == b"element":
                counts[parts[1].decode()] = int(parts[2])
            line = f.readline()
        verts = np.frombuffer(
            f.read(counts["vertex"] * VERTEX_DTYPE.itemsize), VERTEX_DTYPE
        )
        faces = np.frombuffer(
            f.read(counts["face"] * FACE_DTYPE.itemsize), FACE_DTYPE
        )
    assert (faces["n"] == 3).all(), f"{path}: non-triangular faces"
    return verts["xyz"].astype(np.float64), faces["idx"], faces["oid"]


def annotation_info(scenes_dir, scene):
    """instance id -> (task_id, aabb_center) from a renderer-less sim."""
    import habitat_sim
    from pirlnav.task.semantic_map import MPCAT40_TO_TASK

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.join(scenes_dir, scene, f"{scene}.glb")
    sim_cfg.scene_dataset_config_file = os.path.join(
        scenes_dir, "mp3d.scene_dataset_config.json"
    )
    sim_cfg.create_renderer = False
    agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=[])
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        info = {}
        for obj in sim.semantic_scene.objects:
            if obj is None or obj.category is None:
                continue
            try:
                inst_id = int(obj.id.split("_")[-1])
                mp = int(obj.category.index("mpcat40"))
            except (ValueError, AttributeError):
                continue
            task = (
                int(MPCAT40_TO_TASK[mp]) if 0 <= mp < len(MPCAT40_TO_TASK) else -1
            )
            if task >= 0:
                info[inst_id] = (task, np.asarray(obj.aabb.center, np.float64))
        return info
    finally:
        sim.close()


def instance_centroids(vertices, face_idx, face_oid, instance_ids):
    """Per-instance vertex centroid + bounding radius, in ply coordinates."""
    out = {}
    for inst in instance_ids:
        vert_ids = np.unique(face_idx[face_oid == inst])
        if vert_ids.size == 0:
            continue
        pts = vertices[vert_ids]
        centroid = pts.mean(axis=0)
        out[inst] = (centroid, float(np.linalg.norm(pts - centroid, axis=1).max()))
    return out


def detect_frame(ply_clouds, annotations):
    """Pick the rotation that best aligns ply centroids with annotation
    centers (median distance; robust to the broken-box minority)."""
    best = None
    for name, R in FRAME_CANDIDATES.items():
        dists = [
            np.linalg.norm(R @ c - annotations[inst][1])
            for inst, (c, _) in ply_clouds.items()
        ]
        med = float(np.median(dists))
        if best is None or med < best[2]:
            best = (name, R, med)
    return best


def dump_scene(scene, args):
    ply_path = os.path.join(args.scenes_dir, scene, f"{scene}_semantic.ply")
    if not os.path.isfile(ply_path):
        print(f"[ply-dump] {scene}: SKIP no semantic ply")
        return False
    annotations = annotation_info(args.scenes_dir, scene)
    if not annotations:
        print(f"[ply-dump] {scene}: SKIP no goal-class annotations")
        return False

    vertices, face_idx, face_oid = read_semantic_ply(ply_path)
    clouds = instance_centroids(
        vertices, face_idx, face_oid, sorted(annotations)
    )
    if not clouds:
        print(f"[ply-dump] {scene}: SKIP no goal-class geometry in ply")
        return False

    frame, R, med = detect_frame(clouds, annotations)
    if med > 1.0:
        print(f"[ply-dump] {scene}: FAIL best frame {frame} still "
              f"median-misaligned by {med:.2f}m")
        return False

    insts = sorted(clouds)
    obj_pos = np.stack([R @ clouds[i][0] for i in insts]).astype(np.float32)
    radii = np.array([clouds[i][1] for i in insts], dtype=np.float32)
    task_ids = np.array([annotations[i][0] for i in insts], dtype=np.int64)
    instance_ids = np.array(insts, dtype=np.int64)

    out_dir = os.path.join(args.output_dir, scene)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}.npz")
    np.savez(out_path, obj_pos=obj_pos, task_ids=task_ids, radii=radii,
             instance_ids=instance_ids)
    print(f"[ply-dump] {scene}: wrote {len(insts)} objects "
          f"(frame={frame}, median annotation gap {med:.2f}m) -> {out_path}")
    return True


def main():
    args = parse_args()
    scenes = args.scenes or sorted(
        d for d in os.listdir(args.scenes_dir)
        if os.path.isfile(
            os.path.join(args.scenes_dir, d, f"{d}_semantic.ply")
        )
    )
    print(f"[ply-dump] {len(scenes)} scene(s)")
    results = [dump_scene(s, args) for s in scenes]
    n_ok = sum(results)
    print(f"[ply-dump] summary: {n_ok}/{len(results)} scenes dumped")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

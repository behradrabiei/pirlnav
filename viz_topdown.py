import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

SCENE = "17DRP5sb8fy"
PLY   = f"/data/hm3d_datasets/MP3D/v1/tasks/mp3d_habitat/mp3d/{SCENE}/{SCENE}_semantic.ply"
OUT   = "scene_topdown.png"

CEILING_FRAC = 0.7   # drop the top 30% of vertical range (ceilings/roof)

print(f"Loading {PLY} ...")
mesh = trimesh.load(PLY, process=False)

verts = np.asarray(mesh.vertices)
faces = np.asarray(mesh.faces)
vc    = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0

# MP3D meshes are Y-up; figure out the up axis by picking the one with the
# smallest "footprint" (room outlines look best looking down that axis)
ranges = verts.max(0) - verts.min(0)
up_axis = int(np.argmin(ranges))
plane_axes = [a for a in range(3) if a != up_axis]
print(f"Up axis: {'XYZ'[up_axis]}  (range = {ranges[up_axis]:.2f})")

# drop ceiling: keep faces whose centroid is below the cutoff
y = verts[:, up_axis]
cutoff = y.min() + CEILING_FRAC * (y.max() - y.min())
face_centroid_y = y[faces].mean(axis=1)
keep = face_centroid_y < cutoff
faces = faces[keep]
print(f"Keeping {keep.sum()} / {len(keep)} faces below ceiling cutoff")

# build 2D triangles + per-face colour, sorted so lower faces draw first
tri_2d   = verts[faces][:, :, plane_axes]
tri_col  = vc[faces].mean(axis=1)
tri_h    = y[faces].mean(axis=1)
order    = np.argsort(tri_h)              # bottom -> top
tri_2d   = tri_2d[order]
tri_col  = tri_col[order]

fig, ax = plt.subplots(figsize=(10, 10))
ax.add_collection(PolyCollection(tri_2d, facecolors=tri_col, edgecolors="none"))
ax.set_aspect("equal")
ax.autoscale_view()
ax.set_axis_off()
ax.set_title(f"MP3D scene {SCENE} — top-down")
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved {OUT}")

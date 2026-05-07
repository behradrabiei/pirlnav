import trimesh
import plotly.graph_objects as go
import numpy as np

SCENE = "17DRP5sb8fy"
PLY   = f"/data/hm3d_datasets/MP3D/v1/tasks/mp3d_habitat/mp3d/{SCENE}/{SCENE}_semantic.ply"
OUT   = "scene_mesh.html"

print(f"Loading {PLY} ...")
mesh = trimesh.load(PLY, process=False)

verts  = np.array(mesh.vertices)
faces  = np.array(mesh.faces)

# vertex colours (0-255 → 0-1)
if hasattr(mesh.visual, "vertex_colors"):
    vc = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
    color = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in vc]
else:
    color = "lightgray"

fig = go.Figure(go.Mesh3d(
    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    vertexcolor=color if isinstance(color, list) else None,
    color=color if isinstance(color, str) else None,
    opacity=1.0,
    flatshading=False,
))

fig.update_layout(
    title=f"MP3D scene: {SCENE}",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.write_html(OUT)
print(f"Saved interactive mesh to {OUT}  —  open it in a browser to explore.")

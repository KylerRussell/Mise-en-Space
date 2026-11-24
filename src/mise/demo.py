"""A synthetic furnished bedroom, so the editor, backend and MCP server run without a scan.

Objects use ``asset_kind="box"``. Replace them with real splats from Stage 2.
"""

from __future__ import annotations

from .scene.models import Dims, Opening, Pose, Room, Scene, SceneObject


def demo_scene(scene_id: str = "demo-bedroom") -> Scene:
    room = Room(
        floor_polygon=[(0, 0), (4.2, 0), (4.2, 3.6), (0, 3.6)],
        ceiling_height=2.6,
        openings=[
            # wall 1: (0,0)->(4.2,0); wall 2: (4.2,0)->(4.2,3.6); wall 3: (4.2,3.6)->(0,3.6); wall 4: (0,3.6)->(0,0)
            Opening(id="door_1", kind="door", wall="1", center=0.7, width=0.85, top=2.05, swing="in", hinge="right"),
            Opening(id="window_1", kind="window", wall="3", center=2.1, width=1.4, bottom=0.9, top=2.2),
            Opening(id="door_2", kind="door", wall="2", center=3.0, width=0.75, top=2.05, swing="out"),
        ],
    )
    objs = [
        SceneObject(id="bed_1", name="Bed", category="bed", dims=Dims(w=1.6, d=2.05, h=0.95),
                    pose=Pose(x=1.025, y=2.4, yaw=-90), color="#c9b79c", locked=True),
        SceneObject(id="nightstand_1", name="Nightstand", category="nightstand", dims=Dims(w=0.45, d=0.4, h=0.55),
                    pose=Pose(x=0.2, y=1.3, yaw=-90), color="#8a6f52"),
        SceneObject(id="lamp_1", name="Table lamp", category="lamp", dims=Dims(w=0.25, d=0.25, h=0.45),
                    pose=Pose(x=0.2, y=1.3, z=0.55), support="nightstand_1", color="#e8dcc0"),
        SceneObject(id="wardrobe_1", name="Wardrobe", category="wardrobe", dims=Dims(w=1.0, d=0.6, h=2.0),
                    pose=Pose(x=3.0, y=0.3, yaw=0), color="#6d5a48"),
        SceneObject(id="desk_1", name="Desk", category="desk", dims=Dims(w=1.2, d=0.6, h=0.75),
                    pose=Pose(x=3.9, y=1.5, yaw=90), color="#a68a64"),
        SceneObject(id="chair_1", name="Desk chair", category="chair", dims=Dims(w=0.5, d=0.5, h=0.9),
                    pose=Pose(x=3.3, y=1.5, yaw=-90), color="#3f4a52"),
        SceneObject(id="rug_1", name="Rug", category="rug", dims=Dims(w=2.0, d=1.4, h=0.01),
                    pose=Pose(x=2.2, y=2.0), color="#b5a08a"),
    ]
    return Scene(id=scene_id, name="Demo bedroom", room=room, objects=objs)

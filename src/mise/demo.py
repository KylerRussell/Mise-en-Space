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


def demo_apartment(scene_id: str = "demo-apartment") -> Scene:
    """Four spaces: open-plan living room + kitchen, and a bedroom and bathroom behind 12 cm walls."""
    from .scene.models import Space

    t = 0.12  # interior wall thickness
    spaces = [
        Space(id="living", name="Living room", kind="living", polygon=[(0, 0), (5, 0), (5, 4), (0, 4)], open_to=["kitchen"]),
        Space(id="kitchen", name="Kitchen", kind="kitchen", polygon=[(5, 0), (8, 0), (8, 4), (5, 4)]),
        Space(id="bedroom", name="Bedroom", kind="bedroom", polygon=[(0, 4 + t), (4, 4 + t), (4, 7.5), (0, 7.5)]),
        Space(id="bathroom", name="Bathroom", kind="bathroom", ceiling_height=2.3,
              polygon=[(4 + t, 4 + t), (6.5, 4 + t), (6.5, 6.5), (4 + t, 6.5)]),
    ]
    room = Room(spaces=spaces, ceiling_height=2.6)
    room.openings = [
        # living.1 runs (0,0)->(5,0); bedroom.1 (0,4.12)->(4,4.12); bathroom.1 (4.12,4.12)->(6.5,4.12)
        Opening(id="front_door", kind="door", wall="living.1", center=1.0, width=0.9, top=2.05, swing="in", hinge="left"),
        Opening(id="bedroom_door", kind="door", wall="bedroom.1", center=3.0, width=0.8, top=2.05, swing="in", hinge="right"),
        Opening(id="bathroom_door", kind="door", wall="bathroom.1", center=1.4, width=0.75, top=2.05, swing="in", hinge="left"),
        Opening(id="living_window", kind="window", wall="living.3", center=2.0, width=1.6, bottom=0.8, top=2.1),
    ]
    objs = [
        SceneObject(id="sofa_1", name="Sofa", category="sofa", dims=Dims(w=2.1, d=0.9, h=0.85),
                    pose=Pose(x=2.5, y=0.6, yaw=0), color="#7d8a96"),
        SceneObject(id="coffee_table_1", name="Coffee table", category="coffee_table", dims=Dims(w=1.0, d=0.55, h=0.42),
                    pose=Pose(x=2.5, y=1.8), color="#a68a64"),
        SceneObject(id="counter_1", name="Kitchen counter", category="cabinet", dims=Dims(w=2.6, d=0.62, h=0.92),
                    pose=Pose(x=6.6, y=0.31, yaw=0), color="#d8d4cc", locked=True),
        SceneObject(id="island_1", name="Kitchen island", category="table", dims=Dims(w=1.6, d=0.8, h=0.92),
                    pose=Pose(x=5.6, y=2.0, yaw=90), color="#c9c2b6"),
        SceneObject(id="bed_1", name="Bed", category="bed", dims=Dims(w=1.6, d=2.05, h=0.95),
                    pose=Pose(x=1.2, y=6.45, yaw=180), color="#c9b79c"),
        SceneObject(id="toilet_1", name="Toilet", category="toilet", dims=Dims(w=0.4, d=0.7, h=0.8),
                    pose=Pose(x=6.15, y=5.9, yaw=90), color="#f2f2f2", locked=True),
    ]
    return Scene(id=scene_id, name="Demo apartment", room=room, objects=objs)

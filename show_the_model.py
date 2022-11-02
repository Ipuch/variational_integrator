"""
This file is to display the human model into bioviz
"""
import os
import bioviz


biorbd_viz = bioviz.Viz(
    "pendulum.bioMod",
    show_gravity_vector=False,
    show_floor=False,
    show_local_ref_frame=False,
    show_global_ref_frame=False,
    show_markers=False,
    show_mass_center=False,
    show_global_center_of_mass=False,
    show_segments_center_of_mass=False,
    mesh_opacity=1,
    background_color=(1, 1, 1),
)

biorbd_viz.exec()
print("Done")

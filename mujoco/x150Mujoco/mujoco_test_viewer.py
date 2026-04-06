import mujoco
import mujoco.viewer
import numpy as np

# Before running set: export MUJOCO_GL=osmesa

def offscreen_render(m):
  d = mujoco.MjData(m)
  print("D: ", d)
  render = mujoco.Renderer(m, 480, 640)
  mujoco.mj_step(m, d)
  render.update_scene(d)
  img = render.render()
  print("Offscreen render image shape:", img.shape)

m = mujoco.MjModel.from_xml_path('rx150.xml')
print("Model loaded successfully, number of joints:", m.njnt)
# try:
#   mujoco.viewer.launch_from_path('rx150.xml')
# except Exception as e:
#   print("Failed to launch viewer:", e)
#   # if segfault but model loaded OK, WSL2 issue
#   if Exception("Segmentation fault") in str(e):
#     if m.njnt > 0:
#       print("Model loaded successfully (WSL2 issue)")
offscreen_render(m)
    # else:
    #   print("Model failed to load")
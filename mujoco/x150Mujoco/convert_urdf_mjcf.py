import mujoco

# converts the URDF file to an MJCF file, which can be loaded by mujoco
m = mujoco.MjModel.from_xml_path("rx150.urdf")
mujoco.mj_saveLastXML("rx150.xml", m)

# check that conversion worked
try:
  m2 = mujoco.MjModel.from_xml_path("rx150.xml")
  print("Conversion successful")
except Exception as e:
  print("Conversion failed:", e)
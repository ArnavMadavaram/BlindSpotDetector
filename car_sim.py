import carla
import os
import time
import numpy as np
import random
import shutil

# === CLEAN OUTPUT FOLDERS ===
output_folders = [
    "output/rgb",
    "output/depth_png",
    "output/depth_npy",
    "output/segmentation",
    "output/segmentation_npy",
    "output/dvs_npy"
]
for folder in output_folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# === CONNECT TO CARLA ===
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# === WEATHER CONDITIONS ===
world.set_weather(carla.WeatherParameters(
    cloudiness=90.0,
    precipitation=80.0,
    fog_density=85.0,
    sun_altitude_angle=10.0
))

blueprints = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# === FORCE EGO VEHICLE TO A KNOWN BUSY LOCATION ===
central_spawn = spawn_points[10] if len(spawn_points) > 10 else random.choice(spawn_points)
vehicle_bp = blueprints.filter('vehicle.tesla.model3')[0]
vehicle = world.try_spawn_actor(vehicle_bp, central_spawn)
if not vehicle:
    raise RuntimeError("Failed to spawn ego vehicle.")
vehicle.set_autopilot(True)

# === SPAWN TRAFFIC NEARBY ===
traffic_bps = blueprints.filter('vehicle.*')
spawned_vehicles = []
used_points = set()
used_points.add(central_spawn)

nearby_spawns = [pt for pt in spawn_points if pt.location.distance(central_spawn.location) < 60.0 and pt not in used_points]
random.shuffle(nearby_spawns)

for i in range(min(25, len(nearby_spawns))):
    bp = random.choice(traffic_bps)
    if bp.has_attribute('color'):
        bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
    try:
        npc = world.try_spawn_actor(bp, nearby_spawns[i])
        if npc:
            npc.set_autopilot(True)
            spawned_vehicles.append(npc)
    except:
        continue

# === SENSOR TRANSFORM ===
transform = carla.Transform(carla.Location(x=1.5, z=2.4))

# === RGB CAMERA ===
rgb_bp = blueprints.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '800')
rgb_bp.set_attribute('image_size_y', '600')
rgb_bp.set_attribute('fov', '90')
rgb = world.spawn_actor(rgb_bp, transform, attach_to=vehicle)
rgb.listen(lambda img: img.save_to_disk(f"output/rgb/rgb_{img.frame}.png"))

# === DEPTH CAMERA ===
def process_depth(img):
    img.save_to_disk(f"output/depth_png/depth_{img.frame}.png")
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    depth = (arr[:, :, 0] + arr[:, :, 1]*256 + arr[:, :, 2]*256*256) / (256**3 - 1)
    np.save(f"output/depth_npy/depth_{img.frame}.npy", depth)

depth_bp = blueprints.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', '800')
depth_bp.set_attribute('image_size_y', '600')
depth_bp.set_attribute('fov', '90')
depth_sensor = world.spawn_actor(depth_bp, transform, attach_to=vehicle)
depth_sensor.listen(process_depth)

# === SEMANTIC SEGMENTATION ===
def process_seg(img):
    img.save_to_disk(f"output/segmentation/seg_{img.frame}.png", carla.ColorConverter.CityScapesPalette)
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    class_ids = arr[:, :, 2]
    np.save(f"output/segmentation_npy/seg_{img.frame}.npy", class_ids)

seg_bp = blueprints.find('sensor.camera.semantic_segmentation')
seg_bp.set_attribute('image_size_x', '800')
seg_bp.set_attribute('image_size_y', '600')
seg_bp.set_attribute('fov', '90')
seg = world.spawn_actor(seg_bp, transform, attach_to=vehicle)
seg.listen(process_seg)

# === DVS CAMERA ===
def process_dvs(dvs_array):
    events = dvs_array.to_dvs_event_array()
    np.save(f"output/dvs_npy/dvs_{dvs_array.frame}.npy", events)

dvs_bp = blueprints.find('sensor.camera.dvs')
dvs_bp.set_attribute('fov', '90')
dvs = world.spawn_actor(dvs_bp, transform, attach_to=vehicle)
dvs.listen(process_dvs)

# === RECORD ===
print("📸 Recording in busy area for 60 seconds...")
time.sleep(60)

# === CLEANUP ===
for sensor in [rgb, depth_sensor, seg, dvs]:
    sensor.stop()
    sensor.destroy()

vehicle.destroy()
for npc in spawned_vehicles:
    npc.destroy()

print("✅ Done! Check the output/ folder for images and raw data.")

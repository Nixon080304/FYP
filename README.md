# Semantic Exploration and Language-Guided Navigation

This repository contains a ROS 2 workspace for autonomous frontier exploration, camera-based semantic mapping, and language-guided navigation with a TurtleBot3. The system detects unexplored map boundaries, selects safe navigation goals, detects objects with YOLO, projects detections into the map frame, maintains persistent semantic landmarks, and accepts commands such as `go to the chair`.

## System overview

The project has two complementary exploration paths:

- `frontier_detector` provides a focused frontier-cell detector and RViz marker output.
- `exploration_planner` clusters frontiers, scores reachable goals, sends Nav2 goals, tracks coverage, and applies recovery logic.
- `semantic_mapper/semantic_mapper_node.py` provides an experimental semantic frontier scorer based on unknown-space gain, free-space openness, obstacle density, and map boundaries.

The semantic perception path is:

```text
camera image
    -> YOLO object detector
    -> 2D detections
    -> ground-plane projection in the map frame
    -> persistent semantic tracks
    -> language command
    -> Nav2 NavigateToPose goal
```

Supported semantic classes are `person`, `bicycle`, `car`, `bed`, `chair`, and `dining table`.

## Packages

| Package | Purpose |
| --- | --- |
| `frontier_detector` | Detects free cells adjacent to unknown space and publishes frontier markers. |
| `exploration_planner` | Clusters frontiers, checks reachability and clearance, monitors coverage, and sends Nav2 goals. |
| `semantic_interfaces` | Defines projected-object and persistent semantic-map ROS messages. |
| `semantic_mapper` | Runs YOLO detection, image-to-ground projection, semantic tracking, semantic frontier scoring, and language navigation. |
| `simulation_bringup` | Starts a TurtleBot3 Gazebo world with a camera transform and semantic objects. |
| `evaluation_logger` | Reserved scaffold for future evaluation logging. |

## Requirements

- Ubuntu with ROS 2 and `colcon`
- TurtleBot3 packages and Gazebo Classic integration
- Nav2 and a SLAM or map source that publishes `/map`
- ROS packages used by the workspace: `rclpy`, `nav_msgs`, `geometry_msgs`, `sensor_msgs`, `vision_msgs`, `visualization_msgs`, `tf2_ros`, `cv_bridge`, and `nav2_msgs`
- Python packages: `ultralytics`, `opencv-python`, and `numpy`
- Gazebo models referenced by `semantic_rect_world.world`: Bed, Chair, person, dining table, Prius, and bicycle models

The code was developed around the ROS 2 Python APIs used by ROS 2 Humble. Other ROS 2 releases may require package or Gazebo changes.

## Build

Place the repository in a ROS 2 workspace:

```bash
mkdir -p ~/fyp_ws/src
cd ~/fyp_ws/src
git clone https://github.com/Nixon080304/FYP.git
cd ..

source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install ultralytics opencv-python numpy
colcon build --symlink-install
source install/setup.bash
```

## Run the simulation

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch simulation_bringup semantic_rect_world.launch.py
```

The launch file starts Gazebo, spawns the selected TurtleBot3 model, and publishes camera transforms. Start Nav2 and SLAM separately so that `/map`, TF, and the `navigate_to_pose` action are available.

## Run exploration

After sourcing the workspace, run the planner:

```bash
ros2 run exploration_planner exploration_planner_node
```

Useful outputs include:

- `/frontier_centroids`
- `/selected_frontier`
- `/debug_reachable_cells`
- `/debug_rejected_goals`
- `/debug_recovery_goal`
- `/exploration_finished`
- `/exploration_status`

For frontier visualization without autonomous goal selection:

```bash
ros2 run frontier_detector frontier_detector_node
```

## Run semantic mapping

Start each node in a sourced terminal:

```bash
ros2 run semantic_mapper object_detector_node
ros2 run semantic_mapper semantic_projection_node
ros2 run semantic_mapper semantic_map_node
```

The detector downloads or loads `yolov8n.pt` through Ultralytics by default. Main topic flow:

| Topic | Message | Description |
| --- | --- | --- |
| `/camera/camera/image_raw` | `sensor_msgs/Image` | Camera input. |
| `/yolo/detections` | `vision_msgs/Detection2DArray` | Filtered 2D object detections. |
| `/yolo/annotated_image` | `sensor_msgs/Image` | Detection preview. |
| `/semantic_objects/projected` | `semantic_interfaces/ProjectedSemanticObjectArray` | Ground-projected detections in the map frame. |
| `/semantic_map/objects` | `semantic_interfaces/SemanticMapObjectArray` | Confirmed and tracked semantic landmarks. |

Most node settings are ROS parameters and can be overridden with `--ros-args -p`. For example:

```bash
ros2 run semantic_mapper object_detector_node --ros-args \
  -p model_path:=yolov8s.pt \
  -p conf_threshold:=0.5
```

## Language-guided navigation

Run the language navigation node after semantic objects and Nav2 are available:

```bash
ros2 run semantic_mapper language_navigation_node
```

Publish a command:

```bash
ros2 topic pub --once /language_command std_msgs/msg/String \
  "{data: 'go to the chair'}"
```

The node matches a supported object name, chooses the highest-confidence confirmed landmark, and sends a goal near that object through Nav2.

## Repository layout

```text
FYP/
├── evaluation_logger/       # Evaluation package scaffold
├── exploration_planner/     # Autonomous frontier exploration
├── frontier_detector/       # Frontier detection and visualization
├── semantic_interfaces/     # Custom ROS 2 messages
├── semantic_mapper/         # Detection, projection, mapping, and language navigation
└── simulation_bringup/       # Gazebo launch file, world, and camera descriptions
```

## Current limitations

- The simulation launch file does not start SLAM, Nav2, RViz, or the semantic nodes.
- Ground projection assumes detected objects touch a flat ground plane.
- Language handling is phrase matching, not a general natural-language parser.
- Several package manifests still contain placeholder metadata, and `evaluation_logger` has no executable yet.
- Some Gazebo models must already exist in the local Gazebo model path.

## License

No repository-level license has been added. Individual packages may declare their own license metadata. Without a repository license, normal copyright restrictions apply.

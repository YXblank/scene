# ros-semantic-scene-recognition

Python > 3.6
Ubuntu > 18.04

# How to use:

* First clone the repository from the link in catkin workspace
```
git clone ros-semantic-mapper.git
```
* Run setup file which contains necessary steps to download model files and configure paths
```
./setup.sh
```
* In subsequent runs you can use
```
./ros_launch.sh
```
* Play the bagfile you downloaded in separate terminal
```
source devel/setup.sh
rosbag play office.bag
```
* Check the topics available in separate terminal
```
source devel/setup.sh
rostopic list
```
* Note the pointcloud semantic_mapper/cloud
```
rosrun rviz rviz  
```
If you want to train the network with your own data
Execute "train.py" 
The code trained from scratch will be gradually released in the subsequent stages


## Localization in a pointcloud map(pcd)



This operation will send a init pose to topic `/initialpose`.

play the rosbag:

```bash
rosbag play KAIST02-small.bag --clock
```



The final localization msg will send to `/ndt_pose` topic:

```proto
---
header: 
  seq: 1867
  stamp: 
    secs: 1566536121
    nsecs: 251423898
  frame_id: "map"
pose: 
  position: 
    x: -94.8022766113
    y: 544.097351074
    z: 42.5747337341
  orientation: 
    x: 0.0243843578881
    y: 0.0533175268768
    z: -0.702325920272
    w: 0.709437048124
---
```

The localizer also publish a tf of `base_link` to `map`:

```
---
transforms: 
  - 
    header: 
      seq: 0
      stamp: 
        secs: 1566536121
        nsecs: 251423898
      frame_id: "map"
    child_frame_id: "base_link"
    transform: 
      translation: 
        x: -94.8022766113
        y: 544.097351074
        z: 42.5747337341
      rotation: 
        x: 0.0243843578881
        y: 0.0533175268768
        z: -0.702325920272
        w: 0.709437048124
```

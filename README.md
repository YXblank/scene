# ros-semantic-mapper



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

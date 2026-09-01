# ros-semantic-mapper



# How to use:

* First clone the repository from the link in catkin workspace
```
git clone https://github.com/fdayoub/ros-semantic-mapper.git
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
* in rviz display: map, odom, semantic_mapper/cloud
* call the following service to get a semantic map that code the probability distribution of a place over a covered area: rosservice call /semantic_mapper_node/get_semantic_map "label_id: x" replace x by a number between 1 to 11 to select which label you want from my_cats.txt file (of course different list in my_cats.txt will means different x range)
* Display the served map (/oneLabel_cloud) in rvis
* Also check the topic semantic_label
* The image /sem_label_image display the probability distribution over all place labels in the current view.  

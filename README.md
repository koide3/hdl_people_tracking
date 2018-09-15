# hdl_people_tracking
***hdl_people_tracking*** is a ROS package for real-time people tracking using a 3D LIDAR. It first performs <cite>[Haselich's clustering technique][1]</cite> to detect human candidate clusters, and then applies <cite>[Kidono's person classifier][2]</cite> to eliminate false detections. The detected clusters are tracked by using Kalman filter with a contant velocity model.

[1]:https://userpages.uni-koblenz.de/~agas/Documents/Haeselich2014CBP.pdf
[2]:http://www.aisl.cs.tut.ac.jp/~jun/pdffiles/kidono-iv2011.pdf

Video:<br>
[![hdl_people_tracking](http://img.youtube.com/vi/x1WfCYhLfQA/0.jpg)](http://www.youtube.com/watch?v=x1WfCYhLfQA)

## Requirements
***hdl_people_tracking*** requires the following libraries:
- OpenMP
- PCL 1.7

The following ros packages are required:
- pcl_ros
- <a href="https://github.com/koide3/ndt_omp">ndt_omp</a>
- <a href="https://github.com/koide3/hdl_localization">hdl_localization</a>

## Example

Bag file (recorded in an outdoor environment): 
- [hdl_400.bag.tar.gz](http://www.aisl.cs.tut.ac.jp/databases/hdl_graph_slam/hdl_400.bag.tar.gz) (933MB)

```bash
rosparam set use_sim_time true
roslaunch hdl_people_tracking hdl_people_tracking.launch
```

```bash
roscd hdl_localization/rviz
rviz -d hdl_localization.rviz
```

```bash
rosbag play --clock hdl_400.bag
```

If it doesn't work well, change *ndt_neighbor_search_method* in *hdl_localization.launch* to "DIRECT1". It makes the scan matching significantly fast, but a little bit unstable.

## Related packages

- <a href="https://github.com/koide3/hdl_graph_slam">hdl_graph_slam</a>
- <a href="https://github.com/koide3/hdl_localization">hdl_localization</a>
- <a href="https://github.com/koide3/hdl_people_tracking">hdl_people_tracking</a>

<img src="data/figs/packages.png"/>

## Papers
Kenji Koide, Jun Miura, and Emanuele Menegatti, A Portable 3D LIDAR-based System for Long-term and Wide-area People Behavior Measurement, IEEE Transactions on Human-Machine Systems (under review) [PDF].

## Contact
Kenji Koide, Active Intelligent Systems Laboratory, Toyohashi University of Technology <a href="http://www.aisl.cs.tut.ac.jp">[URL]</a> <br>
koide@aisl.cs.tut.ac.jp

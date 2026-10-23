# RDKX5 物流小车视觉

ROS 2视觉源码，包含相机采集/去畸变和圆环、物料、转盘、码垛识别。
当前初始导入保留圆环fast-v3实现，作为改用参考圆环算法前的备份。

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true
```

需安装ROS 2、cv_bridge、OpenCV、NumPy和pyserial；相机标定及数字/颜色模型保留在RDK的`~/logistics_vision_data`，不包含在本仓库。
圆环历史更新说明见[包内文档](src/logistics_ring_v2/RING_LATENCY_UPDATE.md)。

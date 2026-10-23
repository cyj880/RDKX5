# RDKX5 物流小车视觉

当前圆环默认使用640×480全图霍夫圆检测（hough），保留模式控制修复、固定编号、Yaw及STM32协议。

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true
```

开机自启动用户编译后重启X5。模型和标定沿用`~/logistics_vision_data`；无需烧录STM32。
[当前部署说明](src/logistics_ring_v2/RING_REFERENCE_UPDATE.md)。
参考算法移植前的历史快照标签为`rdk-before-reference-rings`，不是此次全图霍夫默认配置。

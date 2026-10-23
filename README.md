# RDKX5 物流小车视觉

ROS 2视觉源码：相机采集/去畸变、圆环、物料、转盘和码垛识别。
默认圆环已改用参考Vision项目的霍夫/轮廓、同心圆合并、空心验证与椭圆精修；取消旧版运动关联和8帧输出等待。
现有固定ID、Yaw及STM32串口协议保留。

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true
```

模型/标定沿用RDK的`~/logistics_vision_data`，不包含在仓库。
[部署、验证与回退说明](src/logistics_ring_v2/RING_REFERENCE_UPDATE.md)。
修改前备份标签：`rdk-before-reference-rings`。无需烧录STM32。

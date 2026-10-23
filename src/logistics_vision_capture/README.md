# logistics_vision_capture

用于 RDK X5 / Ubuntu 22.04 + ROS 2 Humble 的 USB 摄像头采集和棋盘格标定。

## 依赖

```bash
sudo apt install ros-humble-cv-bridge python3-opencv v4l-utils
```

确认摄像头设备：

```bash
ls /dev/video*
v4l2-ctl --list-formats-ext -d /dev/video0
```

## 构建

```bash
cd <工作区根目录>
source /opt/ros/humble/setup.bash
colcon build --packages-select logistics_vision_capture
source install/setup.bash
```

## 采集标定图和空场图

默认使用 `/dev/video0`、`1280x720`、`30 FPS`，数据保存到 `~/logistics_vision_data`：

```bash
ros2 run logistics_vision_capture camera_capture
```

预览窗口按键：

- `C`：检测当前画面；检测到完整 `9x6` 棋盘格时保存标定图片。
- `S`：保存当前原始画面到 `empty_site_images`。
- `Q` 或 `Esc`：退出。

可指定场景标签：

```bash
ros2 run logistics_vision_capture camera_capture --ros-args -p scene_label:=rough_empty
ros2 run logistics_vision_capture camera_capture --ros-args -p scene_label:=temporary_empty
```

## 生成标定 YAML

采集至少 20 张有效棋盘格图片后执行：

```bash
ros2 run logistics_vision_capture camera_calibrate
```

默认参数为 9x6 内角点、18 mm 单格边长，输出类似：

```text
~/logistics_vision_data/calibration/camera_640x480.yaml
```

如果实际打开分辨率不是 640x480，标定结果会按实际分辨率命名，并且后续必须使用该分辨率重新加载对应 YAML。

## 实时去畸变

先在终端 1 启动采集节点，再在终端 2 启动去畸变节点：

```bash
ros2 run logistics_vision_capture camera_undistort
```

节点订阅 `/vision/camera/image_raw`，发布 `/vision/camera/image_undistorted`。
默认读取 `~/logistics_vision_data/calibration/camera_640x480.yaml`，因此采集节点必须使用 `640x480`。默认窗口只显示去畸变图；按 `U` 保存去畸变图，按 `Q` 或 `Esc` 退出。

在 RDK X5 无桌面环境运行时关闭窗口：

```bash
ros2 run logistics_vision_capture camera_undistort --ros-args -p display:=false
```

## 独立圆环识别与定位

圆环识别节点与数字识别节点相互独立，只订阅去畸变图像。当前为「找黑 + 圆拟合 + 同心圆分组」调试版：先按 `processing_scale` 参数把图像压缩、转灰度，再用全局大津法(Otsu)把黑色抠成前景，对二值图找轮廓、逐个做最小二乘圆拟合，按半径范围、圆形度、圆周覆盖率筛选；再按圆心距离把圆聚类成同心圆组，组圆心取成员圆心平均值。窗口里各成员圆画绿色、最内环画黑色、组圆心画十字；结果 JSON 发布到 `/vision/camera/ring_results`，二值图发布到 `/vision/camera/ring_binary`；数字命名和残缺圆识别后续再加。

```bash
ros2 run logistics_vision_capture ring_detect
```

输入图像为 `/vision/camera/image_undistorted`（`640x480`、`bgr8`），标注图发布到 `/vision/camera/ring_annotated`，JSON 结果发布到 `/vision/camera/ring_results`，消息类型为 `std_msgs/msg/String`。查看结果：

```bash
ros2 topic echo /vision/camera/ring_results
```

RDK X5 无桌面环境可关闭预览窗口，节点仍会发布标注图像和 JSON：

```bash
ros2 run logistics_vision_capture ring_detect --ros-args -p display:=false
```

检测参数可通过 ROS 2 参数覆盖，现场调试时主要调整 `min_radius_px`/`max_radius_px`、`max_radial_error`、`min_coverage`、`center_cluster_distance_px` 和 `min_concentric_circles`。

## 数字 1-3 识别与定位

数字识别节点订阅去畸变图像，只识别工位数字 `1`、`2`、`3`，并发布带标注的图像：

```bash
ros2 run logistics_vision_capture number_detect
```

运行时建议让去畸变节点关闭预览窗口：

```bash
ros2 run logistics_vision_capture camera_undistort --ros-args -p display:=false
```

识别节点订阅 `/vision/camera/image_undistorted`，发布 `/vision/camera/number_annotated`。窗口显示数字外接框、数字中心像素坐标和模板匹配置信度；终端仅在连续 5 帧稳定且识别结果变化时打印结果。

节点优先从 `~/logistics_vision_data/digit_templates/` 加载现场模板：

```text
digit_1.png
digit_2.png
digit_3.png
```

模板缺失时节点会提示并使用备用字体模板，适合先验证流程；现场运行前应替换为相同相机、补光和分辨率下的实拍数字模板。

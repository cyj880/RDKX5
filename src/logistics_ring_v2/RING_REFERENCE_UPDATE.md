# 参考圆环检测版 reference-v4

默认圆环识别已改用用户提供的`Vision/pi_vision/detectors/ring_detector.py`检测方式：
CLAHE → 高斯滤波 → 霍夫检圆；霍夫无结果时使用自适应阈值与轮廓兜底；
同心圆聚类、半径合并、空心验证、椭圆圆心精修后输出。

## 与旧版本的区别

- 每帧重新检测，不走旧版25像素跟踪关联和8次命中确认。完整三环当帧通过几何检查即可输出；大幅移动不会仅因关联失败而重新等待8帧。
- 参考算法使用`dp=1.5`、ALT质量门槛0.8、空心对比阈值15、椭圆长短轴比最多3。不是V3的半分辨率候选/局部边缘拟合算法。
- 适配现有相机：直接处理原生640×480，不沿用参考默认放大到至少960×720；候选半径20～60、中心聚类距离20、半径合并容差3像素，每组至少两个不同半径。
- 不使用参考默认最大半径400，避免大范围搜索与相邻工位混并。以上参数是针对本项目环尺寸的适配，不保证与参考项目不同相机高度下的结果相同。
- 检测器自身不平滑，保留现有最终结果单层EMA alpha=0.8，避免叠加参考配置alpha=0.25造成拖尾。
- 保留三环几何校验、完整三环左到右ID1/2/3、缺环数字身份识别；参考算法中的`ring_no=1`不是本站固定编号，不能直接用于驱动车辆。
- 缺数字模型时，完整三环仍工作，缺环身份未知标记0。目标消失立即输出空结果，不使用缓存中心。旧的“8帧确认”只保留在可选fast_v3后端中。
- 图像与坐标同帧、最新帧覆盖、模式切换隔离、源帧300ms过期保护、最多30Hz圆环串口发送均保留。
- 相机、物料、转盘、码垛、STM32源代码及串口帧格式均不修改。STM32原有5个新结果完成判据仍由固件执行。

参考算法保留在`reference_ring_detector.py`，文件注明来源。无需安装参考项目、YOLO或其串口/UI模块。

## 部署

修改前源码已推送到Git标签`rdk-before-reference-rings`；最新main为参考算法版。
在RDK停止原launch后，可以将Git仓库直接作为一个新工作区使用（不要同时启动旧工作区程序）：

```bash
git clone https://github.com/cyj880/RDKX5.git
cd RDKX5
colcon build --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true
```

如已有此仓库，确认没有未保存改动后执行`git pull --ff-only`，然后重新编译并source。
沿用RDK的ROS环境、cv_bridge、OpenCV、NumPy、pyserial及`~/logistics_vision_data`模型/标定文件；仓库不包含这些设备数据。
没有显示屏时使用`display:=false`，设备编号继续通过原来的`camera_device`、`serial_port`等参数指定。无需烧录STM32。

如果使用交付ZIP包，则在现有工作区外备份旧包，将`logistics_ring_v2`解压到工作区src目录，再执行：

```bash
colcon build --packages-select logistics_ring_v2 --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true
```

启动日志应有`backend=reference`；结果JSON有`detector_version: reference-v4`和`search_mode: reference`。
最新算法不再使用ROI/快速边缘路径；旧的`ring_fast_enabled`及`ring_roi_enabled`只控制可选fast_v3后端。

```bash
ros2 topic echo /vision/ring2/sites --once
ros2 topic hz /vision/ring2/sites
```

默认后端是reference。比较上一版算法时，停止launch后重新启动：

```bash
ros2 launch logistics_ring_v2 vision_all.launch.py ring_detector_backend:=fast_v3 display:=true
```

## 验证与回退

65项本地测试通过：包括首次当帧三环输出、40像素移动后不重新等待、0～3环、缺模型身份、正负Yaw、轮廓兜底、空心筛选、椭圆精修、既有最新帧线程/过期保护及物料/码垛回归。
本地没有ROS运行环境，ROS节点接口使用测试替身；检测、线程和协议使用实际代码。Python编码到STM32实际C接收解析器另行验证。
未连接X5，未进行ROS/colcon实机编译或现场识别速度、误检与跟随精度验收；本次算法替换不能宣称已经在X5上提速。

测试环境有ROS时，在仓库根目录执行：

```bash
python3 -m unittest discover -s src/logistics_ring_v2/test -p 'test_ring*.py'
```

要完整回退，停止launch，先提交/备份本地更改，再切换到修改前标签并重新编译：

```bash
git switch --detach rdk-before-reference-rings
colcon build --symlink-install
source install/setup.bash
```

恢复新版使用`git switch main`后重新编译。参考文件的公开Git发布仅包含本次用户指定的RDK代码及检测算法适配，不包含参考目录中的模型、数据集、相机标定与其他文件。

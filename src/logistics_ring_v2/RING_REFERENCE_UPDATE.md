# 参考圆环检测版 reference-v4

## 串口/键盘收到命令但仍停在RAW的补充修复

截图的`MODE: RAW / GATE: STOP`与`input=0 process=0 search=waiting`表示检测尚未启动；input计数只统计run期间提交的帧，不能仅凭input=0断言相机没图。
Qt窗口底部鼠标坐标和R/G/B读数属于OpenCV自带状态栏，与是否加载物料识别无关。

本次取消模式控制对所有其他识别节点的就绪依赖。相机模式可立即初始化；切换圆环只等待圆环订阅者。
如果目标节点缺失，会关闭旧模式、保持raw，并每5秒明确提示缺失的目标；目标就绪后自动执行最近命令，raw/stop可取消等待。
键盘与串口均走相同门控，键盘请求未完成时窗口显示REQUEST等待提示。串口桥订阅实际active_mode，避免键盘已切换但回传还停留在raw。

新增真实回调链测试，75项本地测试通过（ROS接口替身，真实算法/协议/线程）；尚未在X5实机验证。
运行日志分别提供：串口收到命令、当前激活模式、圆环收到run；启动还打印源码路径，便于确认运行进程。
没有修改识别算法参数、STM32固件或串口帧。补充修复包：`RDK_reference_v4_mode_execution_fix.zip`。

**开机自启动用户：替换包后，在现有工作区根目录重新编译并重启RDK。**
运行中的旧进程不会因colcon build自动重启。没有必要为本修复烧录STM32。

```bash
colcon build --packages-select logistics_ring_v2 --symlink-install
```

重启后发送原来的STM32独立校准帧，或点击图像区域后按r，应看到`MODE: RING / GATE: RUN`。
只有模式确实变成RING后，再根据检测数量与processing_ms定位实际检圆问题；不应把RAW状态当作算法检测失败。


## 串口模式启动修复

修正mode_gate仍等待旧六节点圆环链订阅者的启动条件。默认圆环已合并为一个节点，
现在每个模式有至少一个订阅者即可就绪，启动前收到的有效串口模式命令会在就绪后执行。
新增实际串口解析→模式门控→圆环工作线程激活测试；修改前4项失败，修复后69项通过。
串口帧和STM32固件不变。更新RDK源码包、重新编译并重启launch；Git仓库用户执行git pull --ff-only后重新编译。


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

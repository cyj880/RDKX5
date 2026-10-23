# 圆环高速版 fast-v3 — 2026-10-23

本节为最新行为，替代下面旧版本关于“每帧原图霍夫、10Hz圆环回传”的描述。
此更新只涉及圆环算法、圆环串口回传分支与相关启动参数。无需烧录STM32。

## 本次改动

1. 全局找环时，在320×240灰度图上完成模糊、CLAHE和霍夫候选检测，搜索像素为原来的四分之一；半径范围10～30、最小间距30按比例对应原图20～60、60像素。
2. 候选映射回640×480原图后，从局部原图边缘进行圆心/半径拟合和覆盖检查，输出仍为原图像素，不直接拿缩小图坐标控制车。
3. 三环确认后，普通帧使用三个局部原图区域的轮廓/边缘重新拟合，不跑霍夫，不做全图CLAHE或全图Canny。虚线断边可使用当前帧半径邻域的边缘稳健拟合。
4. 局部拟合失败或不满足三环几何关系时，同一帧重新进行320×240全局搜索。每15个处理帧主动全局复查一次。结果必须有当前帧边缘支持；不预测、不发送旧圆心。
5. 保留中心聚类距离、三环间距/等距/共线校验、固定编号规则和8次命中确认。新拟合至少要求9/12扇区边缘支持，边缘覆盖检查允许1像素栅格误差。该路径与旧版逐像素后验并非完全相同，需要实拍验证误检率。
6. 默认OpenCV使用2个线程，仅影响独立圆环进程，避免高线程数开销。可通过`ring_opencv_threads:=1/2/4`比较X5效果，参数需重启launch生效。
7. 圆环串口发送上限由10Hz提高到30Hz，每个源帧只发送一次；转盘继续使用原10Hz定时器。0x13格式、300ms源帧过期保护、未知编号停车规则均不变。实际回传频率受相机和有效识别频率限制，不会复制坐标凑30Hz。

首次寻找仍需8次命中确认；本次没有放宽确认次数。相比前两个版本，半分辨率候选和原图局部边缘拟合可能改变细线、远处、小圆环、强反光和低对比度场景的检出表现，用户已选择速度优先，需现场验收。

## 更新与启动

将`RDK_ring_fast_v3_update.zip`复制到X5。停止原launch和串口桥，在**现有ROS工作区根目录**执行；下面假设包目录是`src/logistics_ring_v2`、压缩包在`~/Downloads`，按实际路径调整。

```bash
cp -a src/logistics_ring_v2 ../logistics_ring_v2_before_fast_v3
unzip -o ~/Downloads/RDK_ring_fast_v3_update.zip -d src
colcon build --packages-select logistics_ring_v2 --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true
```

无屏运行用`display:=false`；保留原有相机编号、串口设备参数。模型、标定文件、STM32均不更新。
停止旧节点后再启动，不能同时打开两份串口桥或两套圆环链。切换到圆环模式后，应看到：

```text
圆环fast-v3：320x240候选+原图精修，稳定帧局部边缘跟踪，OpenCV线程=2。
RING PERF ... search=edge_track stage_ms={'gray': ..., 'detection': ..., 'validation': ...}
```

每5秒的日志包括真实输入/处理频率和处理P95；`search=edge_track`是无需霍夫的快速路径，`fast_search`是初始/周期全局搜索，`fast_recovery`是局部失败后全局恢复。
JSON新增`detector_version: fast-v3`、`refine_pixels`和`stage_ms`。稳定帧`hough_pixels=0`，全局搜索帧为76800。标注图显示同一帧原灰度图；原来的CLAHE调试窗口在高速路径下显示的是原图灰度，因为不再生成全图CLAHE。

另外开终端（先source工作区）查看：

```bash
ros2 topic echo /vision/ring2/sites --once
ros2 topic hz /vision/ring2/sites
```

比较线程设置时，停止launch再重新启动，例如：

```bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true ring_opencv_threads:=1
```

如需临时比较上次原图ROI检圆算法，停止launch后用`ring_fast_enabled:=false`启动；此开关只回退检测算法，不回退30Hz圆环串口定时器。完整回退见下一节。

## 回退到上次ROI版本

停止launch，在工作区根目录执行：

```bash
mv src/logistics_ring_v2 ../logistics_ring_v2_after_fast_v3
cp -a ../logistics_ring_v2_before_fast_v3 src/logistics_ring_v2
colcon build --packages-select logistics_ring_v2 --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true
```

如没有现场备份，将交付的`RDK_ring_fast_v3_rollback.zip`解压到src恢复上次版本；先移走当前源码目录，避免留下新增文件，随后重新编译。

## 本地验证与实机边界

56项测试通过：实际OpenCV处理、线程/最新帧、模式隔离、同帧绘图、拆包/协议、固定编号、缺环、正负Yaw、目标消失、细/粗/虚线圆环、原图圆心精修、局部跟踪不调用霍夫、同帧全局恢复、30Hz圆环/10Hz转盘独立定时器。ROS接口在Windows用替身，未进行X5 ROS实机联调。
Python编码到现有STM32 USART3实际C解析器的兼容测试通过，未修改STM32源代码。

相同150帧合成静止图像，与上次ROI-v2源码对比，本机均使用2个OpenCV线程：平均处理约2.83→1.47ms；150帧移动图像约2.52→1.42ms，三环数量一致，平均X误差约0.22像素。这是Windows/OpenCV4.11合成测试，不能视为X5提速比例或实拍识别率承诺。
X5实际帧率、CPU占用、运动跟随和薄线/反光漏检率待现场验证。相机源仍无拍摄时间戳；300ms保护从进入圆环节点开始计算，不涵盖相机驱动/上游缓存延迟。


---

# 圆环识别延迟优化 — 2026-10-23

## 第二次更新：三环局部跟踪试用版

首次进入仍用640×480全图搜索，并保留8次命中确认。确认完整三环后，改在包围三环的矩形区域内检圆：区域包含最大允许半径60像素及32像素移动余量，直接裁取原图，不缩放。
预处理、CLAHE、Canny与后验仍使用完整原图，避免改变局部对比度和原筛选阈值。局部圆心立即加上裁剪偏移，回到原640×480坐标系；下一帧窗口使用本帧实测中心更新，不使用平滑后的中心。

局部候选未通过三环几何检查时，在同一帧扩大移动余量至64像素重试，仍失败则全图恢复。三次尝试只给跟踪器一次最终观测，不会凑够确认帧数。无三环时如实回传实际数量/空结果并关闭局部跟踪，不发送旧坐标。突然大幅移动的新目标仍需原有确认过程。
每15个处理帧至少执行一次全图复查；退出圆环模式时清空局部窗口。恢复搜索那一帧可能比普通局部帧耗时更长。

新版更新包为`RDK_ring_roi_tracking_update.zip`。部署方式与下文一致，将解压命令中的压缩包名称替换为新版名称。先在工作区之外备份当前源码包，更新后重新执行colcon build；无需烧录STM32。
试用版回退包`RDK_ring_roi_tracking_rollback.zip`恢复到上一次合并节点、全图检测版本。回退时先移走当前包再恢复旧包，最后重新编译。

默认启动方式不变。为方便对比，可停止当前launch后，仅通过启动参数关闭局部跟踪：

```bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=true ring_roi_enabled:=false
```

启用时用`ring_roi_enabled:=true`或省略该参数。每次比较均重启launch；不使用运行中param set修改。单独圆环launch也支持同名开关。

结果JSON额外提供：

| 字段 | 含义 |
|---|---|
| `search_mode` | `roi`局部、`roi_expanded`扩大局部、`full_recovery`同帧全图恢复、`full`首次/周期全图 |
| `detection_region` | 最终检圆区域`[x0,y0,x1,y1]`，右/下边界不包含；坐标输出始终为全图坐标 |
| `hough_pixels` | 本帧所有检圆尝试的像素总量，恢复帧可能超过全图面积 |

日志`RING PERF`追加`search=...`，可以确认已进入局部跟踪。下文的源帧过期检查、去重、话题和串口格式全部沿用。

本地47项测试通过（ROS接口替身，实际OpenCV/线程/节点回调）；新增覆盖局部偏移还原、窗口边界、扩大搜索、同帧全图恢复、丢失立即清空、周期全图复查和确认计数不被重试累加。
与上一次全图版本比较的120帧合成静止目标：112帧局部、8帧全图，检圆平均面积约为全图的34.2%，检测数量与坐标一致。本机处理均值约7.09→3.94ms，属于Windows/OpenCV4.11合成数据，不能当作X5实测提速比例。
首次搜索耗时与8帧确认等待仍保留；本次主要改善已找到三环后的持续识别。X5实机速度、真实圆环漏检率及移动跟随效果待部署验证。

---

此次只更新RDK圆环识别链与串口桥的圆环结果分支，无需烧录STM32。
相机、去畸变、转盘、物料、码垛、模式门控及底盘速度没有修改。

## 使用效果与兼容性

- 默认使用单个`ring2_pipeline`节点，替代六节点之间的图像传输。
- 保持640×480全图、霍夫参数、后验覆盖与三环几何阈值；完整三环编号及缺环数字识别规则不变。
- 一个后台检测线程，最多一个正在处理帧和一个待处理帧；新帧覆盖旧待处理帧。
- 图像、候选、边缘、数字裁剪与绘图同帧。跟踪层不平滑坐标，仅保留最终EMA，默认alpha=0.8。
- 仍需8次命中确认圆环，进入模式时可能短暂显示无目标；不通过放宽误检门槛提速。
- 关闭或重新进入模式时清空待处理、跟踪与平滑状态，旧会话的在途计算不得发布。
- 无订阅者且关闭调试时不生成标注图；`ring2.launch.py display_debug:=true`仅打开同帧CLAHE调试图。
- 原有六个调试节点命令保留，但它们不提供新时效字段，不再用于向新版串口桥提供控制结果。实车使用默认合并节点，不能同时启动两套圆环链。
- `/vision/ring2/sites`和`/vision/ring2/annotated`话题保留，UART模式帧及0x13回传格式均不变。

## 部署

先在RDK上停止原来的launch及串口桥。打开终端，进入**现有ROS工作区根目录**，即有`src`和`install`的目录。
下面假设更新包放在`~/Downloads`，包目录在`src/logistics_ring_v2`；如果工作区嵌套一层src，按实际包位置调整解压路径。
备份必须放在工作区src之外，避免colcon发现重复包。

```bash
cp -a src/logistics_ring_v2 ../logistics_ring_v2_before_latency_20261023
unzip -o ~/Downloads/RDK_ring_low_latency_update.zip -d src
colcon build --packages-select logistics_ring_v2 --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py start_mode:=raw display:=false
```

有显示屏需要窗口时，将最后一行的`display:=false`改成`display:=true`。
如果使用了其他串口设备或相机编号，继续附加原来的`serial_port`、`serial_baudrate`、`camera_device`参数。
启动指令与串口调试帧继续沿用原来的。Yaw/位置校准和成功后自动退出、鸣叫由现有STM32固件控制，本次不改。

确认ROS输出的源码来自新安装目录：

```bash
ros2 pkg prefix logistics_ring_v2
ros2 pkg executables logistics_ring_v2
ros2 node list
```

应包含`ring_pipeline`可执行项与`/ring2_pipeline`节点；默认不再出现`/ring2_gray_node`等六个旧处理节点。
更新包包含完整Python包源码，未包含或覆盖`~/logistics_vision_data`下的模型、标定数据。

## 时效与诊断

结果JSON增加以下字段：

| 字段 | 含义 |
|---|---|
| `frame_sequence` | 圆环节点接收帧序号，可能跳号，因为中间待处理帧会被主动丢弃 |
| `ingress_monotonic_ns` | 图像进入圆环节点接收回调时的单调时钟，纳秒 |
| `processing_ms` | 本帧转换与检测耗时，不包含随后绘图和DDS/串口等待 |

圆环串口桥与识别节点必须运行在**同一台RDK**。桥接发送前检查源帧年龄：超过300ms、时间信息缺失/非法时发送现有空结果；重复、倒序源帧不会重新触发目标发送。每帧最多作为一次新观测，发送上限仍为10Hz。

**当前相机消息没有拍摄时间戳。**本次年龄只表示进入圆环节点到发送的时间，不含相机内部及去畸变上游缓存延迟。输入Header原样传给对应标注图，不伪造拍摄时间。

每5秒可见汇总日志：

```text
RING PERF input=...Hz process=...Hz overwritten=... discarded=... process_p95=...ms
RING TX source_age_p95=...ms samples=...
```

`overwritten`增加是主动跳过旧帧，并非串口丢包；`discarded`表示退出/切换会话后丢弃的在途结果。
P95使用最近一个5秒窗口内最多512个样本。没有新处理或没有发送目标时，不把无样本状态当作性能达标。

实机进入圆环模式后，在另外的终端（先source工作区）观察：

```bash
ros2 topic hz /vision/camera/image_undistorted
ros2 topic hz /vision/ring2/sites
ros2 topic echo /vision/ring2/sites --once
```

有窗口订阅标注图时，可另测`ros2 topic hz /vision/ring2/annotated`。无屏且无订阅者时，此话题不生成图像属于正常行为。
先连续运行60秒，再缓慢移动相机/小车：确认标记跟随、延迟不随运行时间增长；遮挡目标或停止输入后应停车。
验收目标为有效完整三环结果接近现有10Hz串口频率、源帧进入圆环节点至发送P95低于200ms。需同时核对`count=3`，不能只以空结果频率衡量识别性能。

## 测试与实测边界

本地使用Python 3.12、OpenCV 4.11。没有ROS运行环境，ROS消息、节点初始化接口通过测试替身提供；实际OpenCV算法、线程调度、节点回调、编码器和串口桥处理使用真实代码。
覆盖最新帧替换、单线程、退出后旧结果隔离、同帧绘图、时效、去重、0～3目标、正负Yaw、缺模型、原协议、转盘回传、码垛和物料算法回归。

相同35帧合成圆形目标序列与**修改前备份代码**比较：目标数量逐帧一致，第8帧开始确认；移动段平均X误差由5.51px降到0.49px。这是合成数据的平滑滞后比较，不代表真实圆环识别率或X5帧率。
未提供实拍视频；实机图像识别率、CPU负载、相机内部延迟、X5实际频率与P95仍待现场验证。

在装有ROS的RDK工作区中可运行：

```bash
python3 -m unittest discover -s src/logistics_ring_v2/test -p 'test_ring*.py'
```

## 回退

先停止新launch及串口桥，然后在工作区根目录执行（备份路径与上面的部署步骤一致）：

```bash
mv src/logistics_ring_v2 ../logistics_ring_v2_after_latency_20261023
cp -a ../logistics_ring_v2_before_latency_20261023 src/logistics_ring_v2
colcon build --packages-select logistics_ring_v2 --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py start_mode:=raw display:=false
```

如没有现场备份，可将交付的`RDK_ring_low_latency_rollback.zip`解压到src恢复旧包；必须将识别链、launch与串口桥一起回退。

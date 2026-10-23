# 当前版本：reference-v4

默认圆环识别改为参考Vision算法，取消旧8帧等待和25像素关联门槛。
部署与兼容说明见 [RING_REFERENCE_UPDATE.md](RING_REFERENCE_UPDATE.md)。
以下为历史版本记录，以最新说明为准。

# 圆环高速版 fast-v3（2026-10-23）

默认320×240候选检圆＋原图精修，稳定后局部边缘跟踪；圆环串口上限30Hz。
部署、关闭快速路径和回退方式见 [RING_LATENCY_UPDATE.md](RING_LATENCY_UPDATE.md)。

# 圆环局部跟踪试用版（2026-10-23）

确认三环后局部检圆，丢失自动扩大并恢复全图；每15帧全图复查。
默认启用，可通过launch参数 `ring_roi_enabled:=false` 关闭以比较效果。
部署、回退与实测边界见 [RING_LATENCY_UPDATE.md](RING_LATENCY_UPDATE.md)。

# 圆环延迟优化更新（2026-10-23）

默认圆环链已合并为单个同帧处理节点，避免旧帧积压及坐标错帧。
部署、时效字段、性能检查与回退步骤见 [RING_LATENCY_UPDATE.md](RING_LATENCY_UPDATE.md)。
本次不改STM32、相机和其他识别模式。

# logistics_ring_v2

## 串口模式切换

统一启动 `vision_all.launch.py` 时，系统默认显示相机画面(`raw`)。串口桥接节点独占 `/dev/ttyS1`，默认波特率为 `115200`，接收以下帧：

```text
AA 55 10 LEN PAYLOAD CHECK 0D 0A
CHECK = (0x10 + LEN + sum(PAYLOAD)) & 0xFF
```

模式负载如下：

```text
00        相机画面(raw)
01        普通圆环(ring)
02        全物料识别(material_all)
03 COLOR  物料追踪(material_track)，COLOR 为 1~6
04        转盘识别(turntable)
05        码垛识别(stacked)
```

颜色编号保持现有调试窗口约定：`1` 红、`2` 黄、`3` 蓝、`4` 绿、`5` 黑、`6` 浅蓝。

例如，`AA 55 10 01 00 11 0D 0A` 会返回相机模式，`AA 55 10 01 01 12 0D 0A` 会进入普通圆环模式。模式切换最终由 `/vision/mode_command` 和 `mode_gate` 统一处理；当前模式可从 `/vision/active_mode` 获取。

启动时可覆盖串口参数：

```bash
ros2 launch logistics_ring_v2 vision_all.launch.py \
  serial_port:=/dev/ttyS1 serial_baudrate:=115200 start_mode:=raw
```

## 圆环低速位置校准（2026-10-20）

## 到站流程

两批粗加工区（作业朝向270°）、第一批暂存区（0°）对准后停车，等待当前编码器读取结束，
暂停RS485轮询，发送 `AA 55 10 01 01 12 0D 0A` 进入圆环识别。
第二批暂存区保持相机模式及原停顿。用户已有上电丝杆校准流程不修改。

方向确认前LCD显示SET DIR，保持停车。本次上电确认Yaw方向后，进入自动校准：

| 有效固定编号 | 动作 |
|---|---|
| 只有1、或1和2 | 车体后退，2RPM |
| 只有3、或2和3 | 车体前进，2RPM |
| 无目标、ID0、只有2、只有1和3、编号重复/倒序、旧0x12 | 停车等待 |
| 完整1、2、3且Yaw超出±2° | 只调整Yaw |
| 完整1、2、3且Yaw在±2°内 | 调整三个中心的平均位置 |

三环平均X在315～325、平均Y在235～245、Yaw有效且在−2°～+2°，同时成立且连续确认5个新识别结果，
LCD显示DONE。平均值用三坐标之和比较，不因整数除法把5.33像素误差判为到位。
校准结束后保持停车、圆环识别与编码器暂停，不放料、不回程、不自动走下一段，
也不因画面变化自行再次移动。OP0可重新校准。

校准先Yaw后平移。平移中Yaw再超差，立即停平移重新调整Yaw。
图像平均Y−240控制左右（系数+1），图像平均X−320控制前后（系数−1），沿用转盘实测交叉映射。
平移速度为当前转盘速度的50%（整数取整前缩放），约2～13RPM，速度变化斜率60RPM/s；
Yaw比例1RPM/度，限幅1～3RPM，变化斜率10RPM/s。切换找环/Yaw/居中阶段先停车。

坐标超过300ms未更新、无目标或身份不明立即停车并清除到位累计/输出状态，恢复可靠结果后重新判断。
从启动或重新确认方向起，校准和等待最长60秒；超时显示TIMEOUT并保持停车和编码器暂停，OP0重试。
UART7发送 `AA 55 06 06 0D 0A` 中止整条当前路线，恢复编码器读取并返回raw相机画面。
异常退出和重新启动路线也恢复读取。不保存回程目标，不执行里程计回程；编码器硬件仍累计实际位移。

## 调试帧：STM32 UART7，115200、8N1

```text
AA 55 15 OP VALUE CHECK 0D 0A
CHECK = (0x15 + OP + VALUE字节) & 0xFF
VALUE: +1=01，-1=FF，0=00
```

| 功能 | HEX帧 |
|---|---|
| OP0，独立启动/重新校准 | `AA 55 15 00 00 15 0D 0A` |
| OP1，正旋转输出2RPM测试500ms | `AA 55 15 01 01 17 0D 0A` |
| OP1，负旋转输出2RPM测试500ms | `AA 55 15 01 FF 15 0D 0A` |
| OP2，正视觉Yaw使用正旋转输出 | `AA 55 15 02 01 18 0D 0A` |
| OP2，正视觉Yaw使用负旋转输出 | `AA 55 15 02 FF 16 0D 0A` |
| OP2，清除方向确认并停车 | `AA 55 15 02 00 17 0D 0A` |
| 06，中止/退出 | `AA 55 06 06 0D 0A` |

步骤：

1. OP0独立启动，不用先跑路线。LCD显示SET DIR，等待完整三环。
2. 用OP1正/负输出做500ms测试，测试只允许圆环阶段当前停车且完整三环数据新鲜；运行中不能重复测试。
3. 结束后停在TEST STOP，不自动校准。LCD的 `Y:前 > 后` 单位为0.01°，例如+500表示+5.00°。
   后值必须来自测试停车后新收到的完整三环结果。测试期间断流或缺环也立即结束旋转。
4. 确定正视觉Yaw应使用哪种输出符号后发对应OP2，此时开始自动校准。
   若测试前Yaw为负，要选择使角度向0靠近的方向，再换算正Yaw对应的相反输出符号。
   正/负输出只是底盘速度指令符号，不预先声明实际顺/逆时针方向。
5. DONE后保持停车；06退出。方向只保存在RAM，上电后需重新确认。

有效且接受的15帧原样ACK，忙时/条件不满足不ACK。OP0可在当前圆环阶段重启（包括TIMEOUT/DONE），
其他路线行驶、转盘定位、机械臂动作或停顿执行期间拒绝独立启动。
OP2可在空闲预设方向或在圆环阶段确认；清除确认立即停车。
新版《帧生成器.html》增加“圆环位置校准(15)”，提供上述操作和参数校验。

## 固定编号及RDK → STM32协议

完整三环通过原几何校验后，从左到右赋固定ID1、2、3。只见1～2个环时，
使用环内数字HOG/SVM识别固定身份，置信度默认0.6；不明确时ID0，仍可显示坐标但不驱动车。
模型默认路径为 `~/logistics_vision_data/digit_model/digits_svm.yml`。
模型缺失/加载失败会日志提示，完整三环仍可识别，缺环不能确定身份时停车等待。
缺环数字重复或顺序矛盾时该帧身份视为未知；不能把剩下一个环自动叫作1。

ROS `/vision/ring2/sites` 的每个site增加整数 `id`（0～3），仍包含count、sites、station_yaw_deg、yaw_valid。
桥接最多10Hz发送尚未发送的新检测结果；同一缓存不会重复计为新观测。
超过300ms无新结果时发送空结果。转盘七字节格式保持兼容。

新增结果帧：

```text
AA 55 13 LEN N FLAGS YAW_L YAW_H [ID X_L X_H Y_L Y_H]×N CHECK 0D 0A
LEN = 4 + 5*N
CHECK = (0x13 + LEN + sum(PAYLOAD)) & 0xFF
```

- N为0～3；总长度11、16、21或26字节。
- FLAGS bit0为Yaw有效，其余为0；Yaw为int16小端、单位0.01°，无效填0。
- 坐标为640×480绝对像素，X右Y下，uint16小端；从左到右排序，ID跟随对应坐标。
- Yaw为图像首尾中心连线的偏角：向右下倾为正、向右上倾为负；至少两个不同中心才有效。
- 非零ID不可重复；非法长度/数量/标志/Yaw/越界坐标/校验/尾帧不覆盖上一有效结果。
- 原0x12仍可解析显示，`identified=0`，禁止用于校准运动；新0x13设置`identified=1`。
- UART7启动路线命令0x13与USART3识别结果0x13属于不同串口和帧结构，不混用。

示例：

| 结果 | HEX帧 |
|---|---|
| 无圆环 | `AA 55 13 04 00 00 00 00 17 0D 0A` |
| 只有固定ID1 | `AA 55 13 09 01 00 00 00 01 64 00 F0 00 72 0D 0A` |
| 只有固定ID3 | `AA 55 13 09 01 00 00 00 03 F4 01 F0 00 05 0D 0A` |
| ID1和2 | `AA 55 13 0E 02 01 00 00 01 64 00 F0 00 02 FA 00 F0 00 65 0D 0A` |
| 完整三环，均值(320,240)、Yaw0° | `AA 55 13 13 03 01 00 00 01 AA 00 F0 00 02 40 01 F0 00 03 D6 01 F0 00 C2 0D 0A` |

STM32 `RDK_RingResult.points[i].id` 和 `identified` 用于安全区分固定身份与旧帧。
`RingAlign_Start/Update/Stop/Command` 负责纯视觉控制；编码器暂停和恢复仍由MotorTask统一持有。
LCD显示SET DIR、WAIT DATA、WAIT ID、BACK、FORWARD、YAW、CENTER、SETTLE、DONE、TIMEOUT、TEST/TEST STOP，
同时显示ID坐标、Yaw和原串口B/F/G/E/O诊断。

## 部署

STM32烧录 `STM32_RDK_ring_alignment.hex`。RDK解压更新包，将其中 `logistics_ring_v2`
替换到现有ROS工作空间src目录；保留已有相机包和数字模型数据。停止旧launch和串口桥后，在工作空间根目录执行：

```bash
colcon build --packages-select logistics_ring_v2 --symlink-install
source install/setup.bash
ros2 launch logistics_ring_v2 vision_all.launch.py display:=false start_mode:=raw serial_port:=/dev/ttyS1 serial_baudrate:=115200
```

可用 `ros2 topic echo /vision/ring2/sites` 查看ID、坐标和Yaw。串口仍只有统一桥接节点持有。
新自动校准需要两端一起更新；旧RDK的0x12帧只显示、不会让车自动移动。
无外接显示屏可正常运行。数字模型识别效果、Yaw实车方向与停止精度需现场确认。
本次交付未烧录、未在实物RDK/ROS/相机环境中验证，主机测试使用实际协议、控制及回调代码。

"""Ring V2 pipeline step 6: three-station relation check and final output.

圆环组识别第二版 · 第六步(三组关系校验): 订阅后验节点输出的环组, 从 N 组
中选出满足"共线 + 等间距 + 期望组距"的三组作为工位结果, 发布组心/内环/
站位角度; 观察窗口画出三组连线与被淘汰组。

设计要点(判据全部为比值, 与相机高度无关):
    等间距  min(s12,s23)/max(s12,s23) >= spacing_ratio_min(实测三真组 0.99);
    共线    中间组到端点连线的垂距/跨度 <= max_collinearity(实测约 0.003);
    组距先验 期望组距 = expected_spacing_mm * mm_per_px(295mm/291 ≈ 1.014,
            150mm → 约 152px), 容差 ±20%, 防画面里偶现的共线等距假组。
    假组(车体圆孔/齿轮, 覆盖分帧间波动可破单组覆盖门槛)与任何两组的组合
    必然违反上述判据, 在此被统一剔除。
    站位角度 station_yaw_deg = atan2(组3心 - 组1心), 单位度, 供底盘纠偏;
    车停正时三组水平排列, 该角即车身偏角。

    输出层 EMA 跨帧平滑: 对 S1/S2/S3 的 (x, y, r_inner) 与站位角做指数滑动
    平均 smoothed = α·新值 + (1-α)·旧值(α=smooth_alpha), 消除 Hough 峰值
    量化与合成输入集变化带来的 ±1px 级抖动; 新值偏离平滑值超过
    snap_dist_px 时视为车被挪动/严重误检, 直接跳变重置不拖尾。校验失败帧
    维持"发 count:0 不发旧值"语义, 平滑状态冻结, 下次成功帧继续。

    缺组降级分支(N=1/2 时开启): 如实输出已通过后验筛选的圆环中心,
    用环内数字确认固定ID1/2/3；模型缺失或身份不明则ID0，主控停车等待。
    完整三环按画面从左到右赋固定ID；部分观测不能重新编号。
    两个不同中心可计算排列偏角, 单中心时 yaw 无效。
    N≥3 路径保持纯几何校验, 不使用数字。输出统一 status: full/partial/none。

处理门控: 订阅 /vision/ring2/command, run 开始处理 / stop 跳过处理(省 CPU)。
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from .ring_processing import RingValidation, annotate_ring
from .ring_serial_protocol import ring_yaw

WINDOW_NAME = 'Ring V2 Final'


class ValidateNode(Node, RingValidation):
    """从环组中校验并选出三工位组, 发布最终结果。"""

    def __init__(self) -> None:
        super().__init__('ring2_validate')
        self.declare_parameter('groups_topic', '/vision/ring2/groups')
        self.declare_parameter('image_topic', '/vision/ring2/clahe')
        self.declare_parameter('output_topic', '/vision/ring2/sites')
        self.declare_parameter('annotated_topic', '/vision/ring2/annotated')
        self.declare_parameter('expected_spacing_mm', 150.0)
        self.declare_parameter('mm_per_px', 1.014)
        self.declare_parameter('spacing_tolerance', 0.2)
        self.declare_parameter('spacing_ratio_min', 0.85)
        self.declare_parameter('max_collinearity', 0.08)
        self.declare_parameter('smooth_alpha', 0.3)
        self.declare_parameter('snap_dist_px', 15.0)
        self.declare_parameter(
            'model_path', '~/logistics_vision_data/digit_model/digits_svm.yml')
        self.declare_parameter('digit_confidence_min', 0.6)
        self.declare_parameter('display', False)
        self.declare_parameter('command_topic', '/vision/ring2/command')
        self.declare_parameter('start_mode', 'run')

        # 期望组距的像素先验由(mm, mm/px)现算, 现场改高度只改 mm_per_px。
        self.expected_spacing_px = (
            float(self.get_parameter('expected_spacing_mm').value)
            * float(self.get_parameter('mm_per_px').value)
        )
        self.spacing_tolerance = float(self.get_parameter('spacing_tolerance').value)
        self.spacing_ratio_min = float(self.get_parameter('spacing_ratio_min').value)
        self.max_collinearity = float(self.get_parameter('max_collinearity').value)
        self.smooth_alpha = float(self.get_parameter('smooth_alpha').value)
        self.snap_dist_px = float(self.get_parameter('snap_dist_px').value)
        self.display = bool(self.get_parameter('display').value)
        # 处理门控: run 才处理, stop 跳过(进程常驻, 省算力)
        self.running = str(self.get_parameter('start_mode').value).lower() == 'run'
        self.window_created = False
        # EMA 平滑状态: 按左中右位次维护; 校验失败帧冻结, 跳变时重置
        self.smoothed_sites: Optional[List[tuple]] = None
        self.smoothed_yaw: Optional[float] = None
        self.digit_conf_min = float(self.get_parameter('digit_confidence_min').value)
        self.digit_available = False
        self.load_digit_model(str(self.get_parameter('model_path').value), self.get_logger())

        self.bridge = CvBridge()
        self.latest_gray: Optional[np.ndarray] = None

        groups_topic = str(self.get_parameter('groups_topic').value)
        image_topic = str(self.get_parameter('image_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.publisher = self.create_publisher(String, output_topic, 10)
        self.annotated_publisher = self.create_publisher(
            Image, str(self.get_parameter('annotated_topic').value), 10)
        self.create_subscription(String, groups_topic, self._groups_callback, 10)
        self.create_subscription(Image, image_topic, self._image_callback, 10)
        self.create_subscription(String, command_topic, self._command_callback, 10)
        self.get_logger().info(
            'Ring V2 校验节点已启动: 组 %s, 发布 %s, 期望组距 %.0fpx(±%.0f%%), '
            '等间距比 >= %.2f, 共线度 <= %.2f, 门控 %s(%s), 观察窗口 %s。'
            % (groups_topic, output_topic, self.expected_spacing_px,
               self.spacing_tolerance * 100, self.spacing_ratio_min,
               self.max_collinearity, command_topic,
               'run' if self.running else 'stop', '开' if self.display else '关')
        )

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == 'run' and not self.running:
            self.running = True
            self.get_logger().info('收到 run: 开始处理。')
        elif command == 'stop' and self.running:
            self.running = False
            self.latest_gray = None
            self.smoothed_sites = None
            self.smoothed_yaw = None
            self.get_logger().info('收到 stop: 跳过处理, 节省算力。')
        elif command not in ('run', 'stop'):
            self.get_logger().warn('无法识别命令 "%s"。可用: run / stop' % message.data)

    def _poll_quit(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令，正在关闭校验节点。')
            rclpy.shutdown()

    def _image_callback(self, message: Image) -> None:
        if not self.running:
            # stop 门控: 跳过处理省 CPU; 已开的窗口保持响应(q 可退出)
            if self.display and self.window_created:
                self._poll_quit()
            return
        try:
            self.latest_gray = self.bridge.imgmsg_to_cv2(message, 'mono8')
        except CvBridgeError as error:
            self.get_logger().error('无法将 ROS 图像转换为 mono8: %s' % error)
            return
        if self.display and self.window_created:
            self._poll_quit()

    @staticmethod
    def _relation(triple: Tuple[dict, ...]):
        """三组关系: 端点排序后返回 (s12, s23, 共线度, yaw 角)。"""
        p = [(g['x'], g['y']) for g in triple]
        d01 = math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1])
        d02 = math.hypot(p[2][0] - p[0][0], p[2][1] - p[0][1])
        d12 = math.hypot(p[2][0] - p[1][0], p[2][1] - p[1][1])
        spans = {(0, 1): d01, (0, 2): d02, (1, 2): d12}
        (first, second), span = max(spans.items(), key=lambda kv: kv[1])
        middle = 3 - first - second
        line = np.asarray(p[second]) - np.asarray(p[first])
        offset = np.asarray(p[middle]) - np.asarray(p[first])
        # np.cross 已含 |line|*|offset|*sinθ = span*垂距, 再除 span^2 得垂距/跨度
        collinearity = abs(float(np.cross(line, offset))) / (span * span)
        # 连线方向统一为"从左指向右", 消除端点顺序带来的 180° 歧义;
        # 站位角取值 (-90, 90), 车停正时三组水平排列, 该角即车身偏角。
        if line[0] < 0:
            line = -line
        yaw = math.degrees(math.atan2(line[1], line[0]))
        # 端点各自到中间组的间距
        s_first = math.hypot(p[middle][0] - p[first][0],
                             p[middle][1] - p[first][1])
        s_second = math.hypot(p[second][0] - p[middle][0],
                              p[second][1] - p[middle][1])
        return s_first, s_second, collinearity, yaw

    def _score(self, s_first, s_second, collinearity):
        """组合打分: 越小越好(间距均衡 + 共线好 + 贴近期望组距)。"""
        ratio = abs(math.log(max(s_first, s_second)
                             / max(1e-6, min(s_first, s_second))))
        spacing_err = abs((s_first + s_second) / 2 - self.expected_spacing_px)
        return ratio + collinearity + spacing_err / self.expected_spacing_px


    def _groups_callback(self, message: String) -> None:
        if not self.running:
            # stop 门控: 环组 JSON 到了也不处理
            return
        try:
            groups = json.loads(message.data).get('groups', [])
            if not isinstance(groups, list):
                raise ValueError('groups must be a list')
            groups = [dict(g, x=float(g['x']), y=float(g['y']), r_inner=float(g['r_inner']))
                      for g in groups if isinstance(g, dict)]
            groups = [g for g in groups if math.isfinite(g['x']) and math.isfinite(g['y']) and
                      math.isfinite(g['r_inner']) and g['r_inner'] > 0 and
                      0 <= g['x'] < 640 and 0 <= g['y'] < 480]
            groups.sort(key=lambda g: (float(g['x']), float(g['y'])))
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as error:
            self.get_logger().warn('环组 JSON 解析失败: %s' % error)
            self._publish('none', [], [], None)
            return

        result, rejected = self.evaluate(groups)
        self._publish(result['status'], result['sites'], [], result['station_yaw_deg'])
        self._publish_annotated(result['status'], result['sites'], [],
                                result['station_yaw_deg'], rejected)

    def _publish(self, status: str, sites: List[dict], missing: List[int],
                 yaw: Optional[float]) -> None:
        payload = {
            'status': status,
            'count': len(sites),
            'sites': sites,
            'missing': missing,
            'station_yaw_deg': yaw,
            'yaw_valid': yaw is not None,
        }
        output = String()
        output.data = json.dumps(payload, separators=(',', ':'))
        self.publisher.publish(output)

    def _show(self, status: str, sites: List[dict], missing: List[int],
              yaw: Optional[float], rejected: List[dict]) -> None:
        if self.latest_gray is None:
            return
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        annotated = cv2.cvtColor(self.latest_gray, cv2.COLOR_GRAY2BGR)
        for g in rejected:
            cv2.circle(annotated, (int(g['x']), int(g['y'])),
                       int(g['r_inner']), (0, 0, 160), 1)
        if sites:
            points = [(int(round(g['x'])), int(round(g['y']))) for g in sites]
            if len(points) >= 2:
                cv2.line(annotated, points[0], points[-1], (255, 255, 0), 1,
                         cv2.LINE_AA)
            for index, (g, pt) in enumerate(zip(sites, points), start=1):
                cv2.circle(annotated, pt, int(round(g['r_inner'])), (0, 255, 0), 2)
                cv2.drawMarker(annotated, pt, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
                label = 'ID%d (%.0f,%.0f)' % (g.get('id', 0), g['x'], g['y'])
                cv2.putText(annotated, label, (pt[0] + 8, pt[1] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                            cv2.LINE_AA)
            if status == 'full':
                cv2.putText(annotated, 'SITES OK  yaw %.2f deg' % yaw,
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(annotated,
                            'PARTIAL %d yaw=%s' % (len(sites), yaw if yaw is not None else '--'),
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 165, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, 'NO SITE (%d groups seen)' % len(rejected),
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, annotated)
        self._poll_quit()

    def _publish_annotated(self, status: str, sites: List[dict],
                           missing: List[int], yaw: Optional[float],
                           rejected: List[dict]) -> None:
        if self.latest_gray is None:
            return
        annotated = annotate_ring(self.latest_gray, status, sites, yaw, rejected)
        message = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.annotated_publisher.publish(message)

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[ValidateNode] = None
    try:
        node = ValidateNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('ring2_validate 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

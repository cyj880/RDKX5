"""Ring V2 pipeline step 5: geometric posterior on Hough circles.

圆环组识别第二版 · 第五步(几何后验): 订阅霍夫候选圆与 CLAHE 图, 做
三件事——圆心聚类成环组、逐候选 12 扇区圆周覆盖统计、组内合成输出,
发布组心与内环, 同时在观察窗口画出筛选结果。

设计要点:
    扇区覆盖(12 扇区, 每扇区 30 采样点, 命中任一点即该扇区有效)借鉴
    cyqm: 虚线圆每段划各占若干扇区, 覆盖仍可达标; 断续假边缘凑不满多数
    扇区。覆盖达标的候选才参与组心合成(按覆盖加权平均, 同组候选共享物理
    圆心, 合成消单圆估计噪声); 内环取组内半径最小的合格候选——半径是
    离散档位, 不能跨档平均。
    图像与候选圆两路话题相差 1~2 帧, 场景静止时组心不动, 可接受。

    组级持续性跟踪: 真圆环组位置稳定(实测 std<1px 且帧帧存在), 车体
    假组间歇乱跳(实测各位置仅 10~13% 帧出现)。新组须连续命中
    track_confirm_hits 帧才 confirmed 并发布, 假组因连续性不足永远停在
    candidate——时域关口在源头拦截, 下游几何校验与数字分支只看到稳定组。
    连续丢失 track_max_misses 帧以上命中史清零(防间歇假组攒命中数),
    超过 track_remove_misses 帧删除(防旧位置残留)。

处理门控: 订阅 /vision/ring2/command, run 开始处理 / stop 跳过处理(省 CPU)。
"""

import json
import math
from typing import List, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from .ring_geometry import circle_groups
from .ring_processing import RingTracker


WINDOW_NAME = 'Ring V2 Posterior'


class PosteriorNode(Node, RingTracker):
    """候选圆聚类、扇区覆盖筛选、组心/内环合成与发布。"""

    def __init__(self) -> None:
        super().__init__('ring2_posterior')
        self.declare_parameter('circles_topic', '/vision/ring2/circles')
        self.declare_parameter('image_topic', '/vision/ring2/clahe')
        self.declare_parameter('output_topic', '/vision/ring2/groups')
        self.declare_parameter('cluster_dist_px', 20.0)
        self.declare_parameter('sector_count', 12)
        self.declare_parameter('samples_per_sector', 30)
        self.declare_parameter('min_coverage', 0.7)
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)
        self.declare_parameter('track_match_dist_px', 25.0)
        self.declare_parameter('track_confirm_hits', 8)
        self.declare_parameter('track_max_misses', 3)
        self.declare_parameter('track_remove_misses', 8)
        self.declare_parameter('track_smooth_alpha', 0.5)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('display', False)
        self.declare_parameter('command_topic', '/vision/ring2/command')
        self.declare_parameter('start_mode', 'run')

        self.cluster_dist = float(self.get_parameter('cluster_dist_px').value)
        self.sector_count = int(self.get_parameter('sector_count').value)
        self.samples_per_sector = int(self.get_parameter('samples_per_sector').value)
        self.min_coverage = float(self.get_parameter('min_coverage').value)
        self.canny_low = int(self.get_parameter('canny_low').value)
        self.canny_high = int(self.get_parameter('canny_high').value)
        self.track_match_dist = float(self.get_parameter('track_match_dist_px').value)
        self.track_confirm_hits = int(self.get_parameter('track_confirm_hits').value)
        self.track_max_misses = int(self.get_parameter('track_max_misses').value)
        self.track_remove_misses = int(self.get_parameter('track_remove_misses').value)
        self.track_smooth_alpha = float(self.get_parameter('track_smooth_alpha').value)
        self.image_size = (
            int(self.get_parameter('image_width').value),
            int(self.get_parameter('image_height').value),
        )
        self.display = bool(self.get_parameter('display').value)
        # 处理门控: run 才处理, stop 跳过(进程常驻, 省算力)
        self.running = str(self.get_parameter('start_mode').value).lower() == 'run'
        self.window_created = False

        self.bridge = CvBridge()
        self.latest_gray: Optional[np.ndarray] = None
        self.latest_edges: Optional[np.ndarray] = None
        # 组跟踪器状态: 每项 {x, y, r_inner, coverage, n, hits, misses}
        self.tracked: List[dict] = []
        # 扇区采样角度表: (sector_count, samples_per_sector) 弧度
        sector_edges = np.arange(self.sector_count + 1) * (2 * np.pi / self.sector_count)
        offsets = (sector_edges[:-1, None]
                   + np.linspace(0, 2 * np.pi / self.sector_count,
                                 self.samples_per_sector, endpoint=False)[None, :])
        self.sector_angles = offsets  # 供按半径生成采样点

        circles_topic = str(self.get_parameter('circles_topic').value)
        image_topic = str(self.get_parameter('image_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        command_topic = str(self.get_parameter('command_topic').value)
        self.publisher = self.create_publisher(String, output_topic, 10)
        self.create_subscription(String, circles_topic, self._circles_callback, 10)
        self.create_subscription(Image, image_topic, self._image_callback, 10)
        self.create_subscription(String, command_topic, self._command_callback, 10)
        self.get_logger().info(
            'Ring V2 后验节点已启动: 候选 %s + 图像 %s, 发布 %s, 聚类距 %.0fpx, '
            '%d 扇区, 覆盖门槛 %.2f, 门控 %s(%s), 观察窗口 %s。'
            % (circles_topic, image_topic, output_topic, self.cluster_dist,
               self.sector_count, self.min_coverage, command_topic,
               'run' if self.running else 'stop', '开' if self.display else '关')
        )

    def _command_callback(self, message: String) -> None:
        command = message.data.strip().lower()
        if command == 'run' and not self.running:
            self.running = True
            self.get_logger().info('收到 run: 开始处理。')
        elif command == 'stop' and self.running:
            self.running = False
            self.tracked.clear()
            self.latest_gray = None
            self.latest_edges = None
            self.get_logger().info('收到 stop: 跳过处理, 节省算力。')
        elif command not in ('run', 'stop'):
            self.get_logger().warn('无法识别命令 "%s"。可用: run / stop' % message.data)

    def _poll_quit(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            self.get_logger().info('收到退出指令，正在关闭后验节点。')
            rclpy.shutdown()

    def _image_callback(self, message: Image) -> None:
        if not self.running:
            # stop 门控: 跳过处理省 CPU; 已开的窗口保持响应(q 可退出)
            if self.display and self.window_created:
                self._poll_quit()
            return
        try:
            gray = self.bridge.imgmsg_to_cv2(message, desired_encoding='mono8')
        except CvBridgeError as error:
            self.get_logger().error('无法将 ROS 图像转换为 mono8: %s' % error)
            return
        height, width = gray.shape[:2]
        if (width, height) != self.image_size:
            return
        self.latest_gray = gray
        self.latest_edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        if self.display and self.window_created:
            # 窗口绘制放在 circles 回调; 这里仅保持窗口响应按键
            self._poll_quit()

    def _coverage(self, x: float, y: float, r: float) -> Tuple[float, np.ndarray]:
        """圆周 12 扇区覆盖: 返回 (覆盖比, 各扇区命中布尔)。"""
        edges = self.latest_edges
        if edges is None:
            return 0.0, np.zeros(self.sector_count, dtype=bool)
        height, width = edges.shape[:2]
        angles = self.sector_angles
        px = np.rint(x + r * np.cos(angles)).astype(int)
        py = np.rint(y + r * np.sin(angles)).astype(int)
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        hits = np.zeros(angles.shape, dtype=bool)
        hits[valid] = edges[py[valid], px[valid]] != 0
        per_sector = hits.reshape(self.sector_count, self.samples_per_sector).any(axis=1)
        return float(per_sector.mean()), per_sector

    def _cluster(self, circles: List[tuple]) -> List[List[tuple]]:
        """按圆心距贪心聚类(与 ring_detect 同风格): 近者归组。"""
        clusters: List[List[tuple]] = []
        for circle in circles:
            target = None
            for cluster in clusters:
                gx = float(np.mean([c[0] for c in cluster]))
                gy = float(np.mean([c[1] for c in cluster]))
                if np.hypot(circle[0] - gx, circle[1] - gy) <= self.cluster_dist:
                    target = cluster
                    break
            if target is None:
                clusters.append([circle])
            else:
                target.append(circle)
        return clusters


    def _circles_callback(self, message: String) -> None:
        if not self.running:
            # stop 门控: 候选圆 JSON 到了也不处理
            return
        try:
            payload = json.loads(message.data)
            circles = [(float(c['x']), float(c['y']), float(c['r']))
                       for c in payload.get('circles', [])]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            self.get_logger().warn('候选圆 JSON 解析失败: %s' % error)
            return
        if self.latest_edges is None:
            return

        accepted, rejected = self._posterior(circles)
        confirmed, candidates = self._update_tracks(accepted)
        self._publish(confirmed)

        if self.display:
            self._show(confirmed, candidates, rejected)

    def _posterior(self, circles: List[tuple]):
        """聚类 + 覆盖筛选 + 合成。返回 (合格组列表, 被拒候选列表)。"""
        accepted, rejected = circle_groups(circles, self.latest_edges, self.cluster_dist,
                                            self.sector_count, self.samples_per_sector, self.min_coverage)
        for group in accepted:
            group.pop('r_outer')
        return accepted, rejected

    def _publish(self, groups: List[dict]) -> None:
        output = String()
        output.data = json.dumps({'groups': groups}, separators=(',', ':'))
        self.publisher.publish(output)

    def _show(self, confirmed: List[dict], candidates: List[dict],
              rejected: List[tuple]) -> None:
        if self.latest_gray is None:
            return
        if not self.window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_created = True
        annotated = cv2.cvtColor(self.latest_gray, cv2.COLOR_GRAY2BGR)
        for x, y, r, cov in rejected:
            cv2.circle(annotated, (int(round(x)), int(round(y))),
                       int(round(r)), (0, 0, 160), 1)
        for c in candidates:
            # 未确认组(持续性不足): 灰色小圈, 命中数标注
            cv2.circle(annotated, (int(round(c['x'])), int(round(c['y']))),
                       int(round(c['r_inner'])), (160, 160, 160), 1)
            cv2.putText(annotated, 'cand %d' % c['hits'],
                        (int(round(c['x'])) + 6, int(round(c['y'])) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1,
                        cv2.LINE_AA)
        for g in confirmed:
            center = (int(round(g['x'])), int(round(g['y'])))
            cv2.circle(annotated, center, int(round(g['r_inner'])), (0, 255, 0), 2)
            cv2.drawMarker(annotated, center, (255, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(annotated, 'cov %.2f n%d' % (g['coverage'], g['n']),
                        (center[0] + 10, center[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                        cv2.LINE_AA)
        cv2.putText(annotated,
                    'groups: %d (+%d candidate)' % (len(confirmed), len(candidates)),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, annotated)
        self._poll_quit()

    def destroy_node(self) -> bool:
        if self.window_created:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[PosteriorNode] = None
    try:
        node = PosteriorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        print('ring2_posterior 启动失败: %s' % error)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

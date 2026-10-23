"""Route public mode commands to mutually exclusive detector gates.

模式全集: raw(空转) + ring / material_all / material_track /
turntable / stacked 五个并列检测门(物料按"逐个识别→逐个抓取"工作,
指定颜色定位节点已删除; idle 不是节点——material idle 等价于关闭全部
物料门回到 raw)。

公共命令(/vision/mode_command):
    raw | stop                          全部停止
    ring | turntable | stacked          激活对应模式(run)
    material_all                        全物料识别
    material_track [c]                  单目标追踪(缺省用节点参数)
    material [all | track c | idle]     旧语法兼容路由
旧话题入口(/vision/ring2/command 等)继续兼容, 经同样路由。
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String


SINGLE_MODES = ('ring', 'turntable', 'stacked')


class ModeGateNode(Node):
    def __init__(self):
        super().__init__('mode_gate')
        self.declare_parameter('start_mode', 'raw')
        self.declare_parameter('start_material_mode', 'idle')
        self.declare_parameter('start_ring_mode', 'stop')
        self.declare_parameter('start_turntable_mode', 'stop')
        self.declare_parameter('start_stacked_mode', 'stop')
        self.start_command = str(self.get_parameter('start_mode').value)
        legacy = []
        material = str(self.get_parameter('start_material_mode').value)
        if material != 'idle':
            legacy.append('material ' + material)
        for mode in ('ring', 'turntable', 'stacked'):
            if self.get_parameter('start_%s_mode' % mode).value == 'run':
                legacy.append(mode)
        if len(legacy) > 1 or (legacy and self.start_command != 'raw'):
            raise ValueError('只能指定一个启动模式，请使用 start_mode')
        if legacy:
            self.start_command = legacy[0]
        self.active = 'raw'
        # 物料两节点互斥: 同一时刻只开一个门, 默认全部 stop
        self.commands = {'ring': 'stop', 'material_all': 'stop',
                         'material_track': 'stop',
                         'turntable': 'stop', 'stacked': 'stop'}
        self.gate_publishers = {
            mode: self.create_publisher(String, '/vision/gated/%s/command' % mode, 10)
            for mode in self.commands
        }
        retained = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, '/vision/active_mode', retained)
        self.create_subscription(String, '/vision/mode_command', self._command, 10)
        for mode, topic in (('ring', 'ring2'), ('material', 'material2'),
                            ('turntable', 'turntable'), ('stacked', 'stacked')):
            self.create_subscription(String, '/vision/%s/command' % topic,
                                     lambda msg, source=mode: self._legacy(source, msg), 10)
        self.ready = False
        self.pending_command = self.start_command
        self.pending_selection = None
        self.next_wait_log = 0.0
        self.timer = self.create_timer(0.2, self._startup)
        self.get_logger().info('模式控制按目标节点就绪状态切换，源码=%s。' % __file__)

    def _startup(self):
        # Initialize raw/stop immediately. An unrelated missing detector must
        # never prevent keyboard or serial commands from reaching a healthy one.
        if not self.ready:
            self.ready = True
            command = self.pending_command
            self.pending_command = ''
            self._command(String(data=command))
        elif self.pending_selection is not None:
            mode, command = self.pending_selection
            if self.gate_publishers[mode].get_subscription_count() >= 1:
                self._select(mode, command)
            else:
                self._log_waiting(mode)

    def _log_waiting(self, mode):
        now = time.monotonic()
        if now >= self.next_wait_log:
            self.get_logger().warning(
                '模式%s等待目标节点：/vision/gated/%s/command 订阅者为0。'
                '请检查该节点是否启动失败；其他模式不受影响。' % (mode, mode))
            self.next_wait_log = now + 5.0

    def _valid_mode_tokens(self, tokens) -> bool:
        """未就绪期的命令白名单校验: 合法才缓存, 防止垃圾命令抢占启动命令。"""
        mode = tokens[0]
        if mode in ('raw', 'stop', 'ring', 'turntable', 'stacked', 'material_all'):
            return len(tokens) == 1
        if mode == 'material_track':
            rest = tokens[1:]
            if not rest:
                return True  # 无颜色参数 = 用节点参数默认值
            return len(rest) == 1 and rest[0].isdigit() and 1 <= int(rest[0]) <= 6
        if mode == 'material':
            sub = tokens[1:]
            if not sub or sub == ['idle']:
                return True
            return self._valid_material(' '.join(sub))
        return False

    def _command(self, msg):
        tokens = msg.data.lower().split()
        if not tokens:
            return
        mode = tokens[0]
        if not self.ready:
            if self._valid_mode_tokens(tokens):
                self.pending_command = msg.data
            return
        if mode in ('raw', 'stop') and len(tokens) == 1:
            self._select('raw')
        elif mode in SINGLE_MODES and len(tokens) == 1:
            self._select(mode, 'run')
        elif mode == 'material_all' and len(tokens) == 1:
            self._select('material_all', 'run')
        elif mode == 'material_track' and self._valid_mode_tokens(tokens):
            self._select(mode, self._direct_command(tokens))
        elif mode == 'material':
            self._route_material(tokens[1:])
        else:
            self.get_logger().warn('无效模式命令: %s' % msg.data)

    @staticmethod
    def _direct_command(tokens):
        """material_track 直达命令转成节点命令: 无色参=run, 有色参=track <色>。"""
        rest = tokens[1:]
        if not rest:
            return 'run'
        return '%s %s' % (tokens[0].split('_', 1)[1], ' '.join(rest))

    def _legacy(self, source, msg):
        command = ' '.join(msg.data.lower().split())
        if not self.ready:
            if source == 'material':
                sub = command.split()
                if sub == ['idle']:
                    self.pending_command = 'raw'
                elif (sub and self._valid_material(command)) or not sub:
                    self.pending_command = 'material ' + command
            elif command == 'run':
                self.pending_command = source
            elif command == 'stop' and self.pending_command == source:
                self.pending_command = 'raw'
            return
        if source != 'material':
            if command == 'run':
                self._select(source, command)
            elif command == 'stop' and (self.active == source or
                    (self.pending_selection and self.pending_selection[0] == source)):
                self._select('raw')
            return
        self._route_material(command.split())

    def _route_material(self, sub):
        """物料命令路由(旧语法 `all/track c/idle` → 两个新门; locate 已删除)。"""
        if not sub or sub == ['all']:
            self._select('material_all', 'run')
            return
        command = ' '.join(sub)
        if sub[0] == 'track' and len(sub) == 2 and sub[1].isdigit() and 1 <= int(sub[1]) <= 6:
            self._select('material_track', 'track ' + sub[1])
        elif sub == ['locate'] or (sub and sub[0] == 'locate'):
            self.get_logger().warn('locate 模式已删除(逐个抓取无需指定颜色定位): %s' % command)
        elif sub == ['idle']:
            if self.active.startswith('material'):
                self._select('raw')
            # 非物料模式下的 idle 忽略(与旧逻辑一致)
        elif self._valid_material(command):
            # 文法合法但动词未路由到(理论不可达), 防御性告警
            self.get_logger().warn('物料命令未路由: %s' % command)
        else:
            self.get_logger().warn('无效物料命令: %s' % command)

    @staticmethod
    def _valid_material(command):
        tokens = command.split()
        if command == 'all':
            return True
        if tokens and tokens[0] == 'track':
            return (len(tokens) == 2 and tokens[1].isdigit()
                    and 1 <= int(tokens[1]) <= 6)
        return False

    def _select(self, mode, command=None):
        if mode != 'raw' and self.gate_publishers[mode].get_subscription_count() < 1:
            # Close the current gate first, then retry only this requested gate.
            self._select('raw')
            self.pending_selection = (mode, command)
            self.next_wait_log = 0.0
            self._log_waiting(mode)
            return
        self.pending_selection = None
        self.active = mode
        # 与 __init__ 的 self.commands 保持同一份键集(新增模式两处都要改)
        self.commands = {'ring': 'stop', 'material_all': 'stop',
                         'material_track': 'stop',
                         'turntable': 'stop', 'stacked': 'stop'}
        if mode != 'raw':
            self.commands[mode] = command
        # Close other gates before sending the selected mode's activation.
        for source in [s for s in self.commands if s != mode] + ([mode] if mode != 'raw' else []):
            self.gate_publishers[source].publish(String(data=self.commands[source]))
        self.state_pub.publish(String(data=mode))
        self.get_logger().info('当前激活模式: %s' % mode)


def main(args=None):
    rclpy.init(args=args)
    node = ModeGateNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

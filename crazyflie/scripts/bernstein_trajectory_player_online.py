#!/usr/bin/env python3

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class BernsteinTrajectoryPlayer(Node):
    def __init__(self) -> None:
        super().__init__('bernstein_trajectory_player')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter('x_topic', '/x_vector')
        self.declare_parameter('y_topic', '/y_vector')
        self.declare_parameter('z_topic', '/z_vector')
        self.declare_parameter('psi_topic', '/psi_vector')

        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('path_topic', '/bernstein_mpc_path')
        self.declare_parameter('current_pose_topic', '/bernstein_mpc_current_pose')

        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('hold_last_trajectory', True)

        self.declare_parameter('t0', 0.0)
        self.declare_parameter('tf', 0.9)
        self.declare_parameter('num_dense_points', 20)

        self.declare_parameter('expected_vector_length', 6)
        self.declare_parameter('strict_vector_length', True)

        self.x_topic = self.get_parameter('x_topic').get_parameter_value().string_value
        self.y_topic = self.get_parameter('y_topic').get_parameter_value().string_value
        self.z_topic = self.get_parameter('z_topic').get_parameter_value().string_value
        self.psi_topic = self.get_parameter('psi_topic').get_parameter_value().string_value

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.current_pose_topic = self.get_parameter('current_pose_topic').get_parameter_value().string_value

        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.hold_last_trajectory = self.get_parameter('hold_last_trajectory').get_parameter_value().bool_value

        self.t0 = self.get_parameter('t0').get_parameter_value().double_value
        self.tf = self.get_parameter('tf').get_parameter_value().double_value
        self.num_dense_points = self.get_parameter('num_dense_points').get_parameter_value().integer_value

        self.expected_vector_length = self.get_parameter('expected_vector_length').get_parameter_value().integer_value
        self.strict_vector_length = self.get_parameter('strict_vector_length').get_parameter_value().bool_value

        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn('publish_rate_hz must be > 0. Using 20.0 Hz.')
            self.publish_rate_hz = 20.0

        if self.tf <= self.t0:
            self.get_logger().warn('tf must be > t0. Using t0=0.0, tf=0.9.')
            self.t0 = 0.0
            self.tf = 0.9

        if self.num_dense_points < 2:
            self.get_logger().warn('num_dense_points must be >= 2. Using 20.')
            self.num_dense_points = 20

        # ------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.current_pose_pub = self.create_publisher(PoseStamped, self.current_pose_topic, 10)

        # ------------------------------------------------------------
        # Incoming vector queues
        # ------------------------------------------------------------
        self.queues: Dict[str, Deque[Dict[str, Any]]] = {
            'x': deque(),
            'y': deque(),
            'z': deque(),
            'psi': deque(),
        }

        # ------------------------------------------------------------
        # Active trajectory state
        # ------------------------------------------------------------
        self.active_path_points: List[Dict[str, float]] = []
        self.active_traj_start_ns: Optional[int] = None
        self.active_tf: float = self.tf
        self.active_order: Optional[int] = None
        self.bundle_counter: int = 0

        # ------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------
        self.sub_x = self.create_subscription(
            Float64MultiArray,
            self.x_topic,
            self._make_callback('x'),
            qos
        )

        self.sub_y = self.create_subscription(
            Float64MultiArray,
            self.y_topic,
            self._make_callback('y'),
            qos
        )

        self.sub_z = self.create_subscription(
            Float64MultiArray,
            self.z_topic,
            self._make_callback('z'),
            qos
        )

        self.sub_psi = self.create_subscription(
            Float64MultiArray,
            self.psi_topic,
            self._make_callback('psi'),
            qos
        )

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.timer_callback)

        self.get_logger().info('---------------------------------------------')
        self.get_logger().info('bernstein_trajectory_player started')
        self.get_logger().info(f'x_topic               : {self.x_topic}')
        self.get_logger().info(f'y_topic               : {self.y_topic}')
        self.get_logger().info(f'z_topic               : {self.z_topic}')
        self.get_logger().info(f'psi_topic             : {self.psi_topic}')
        self.get_logger().info(f'frame_id              : {self.frame_id}')
        self.get_logger().info(f'path_topic            : {self.path_topic}')
        self.get_logger().info(f'current_pose_topic    : {self.current_pose_topic}')
        self.get_logger().info(f'publish_rate_hz       : {self.publish_rate_hz}')
        self.get_logger().info(f't0                    : {self.t0}')
        self.get_logger().info(f'tf                    : {self.tf}')
        self.get_logger().info(f'num_dense_points      : {self.num_dense_points}')
        self.get_logger().info(f'expected_vector_length: {self.expected_vector_length}')
        self.get_logger().info('This node computes Bernstein trajectories ONLINE from live control-point topics.')
        self.get_logger().info('---------------------------------------------')

    # ============================================================
    # Bernstein utilities (ported from your C++ code)
    # ============================================================
    def _linspace(self, start: float, stop: float, num: int) -> List[float]:
        if num <= 1:
            return [start]
        step = (stop - start) / float(num - 1)
        return [start + i * step for i in range(num)]

    def _bernstein_matrix_a2b(
        self,
        N: int,
        t: List[float],
        tmin: float = 0.0,
        tmax: float = 1.0
    ) -> List[List[float]]:
        if N < 0:
            return []

        if abs(tmax - tmin) < 1e-12 and N > 0:
            raise ValueError('tmax and tmin must be different for N > 0.')

        # Same binomial-precompute pattern as your C++ code
        B = [1.0] * (N + 1)
        for j in range(1, N // 2 + 1):
            B[j] = B[j - 1] * (N + 1 - j) / j
            B[N - j] = B[j]

        denom = (tmax - tmin) ** N if N > 0 else 1.0
        b = [[1.0 for _ in range(N + 1)] for _ in range(len(t))]

        for i, ti in enumerate(t):
            tp = ti - tmin
            ttp = tmax - ti

            T = [1.0] * (N + 1)
            TT = [1.0] * (N + 1)

            for j in range(1, N + 1):
                T[j] = tp * T[j - 1]
                TT[N - j] = ttp * TT[N - j + 1]
                T[j - 1] *= B[j - 1]

            for j in range(0, N + 1):
                b[i][j] = T[j] * TT[j] / denom

        return b

    def _bernstein_poly(
        self,
        cp: List[float],
        t: List[float],
        t0: float = 0.0,
        tf: float = 1.0
    ) -> List[float]:
        if len(cp) == 0:
            return []

        N = len(cp) - 1
        B = self._bernstein_matrix_a2b(N, t, t0, tf)

        poly_t = [0.0 for _ in range(len(t))]
        for i in range(len(t)):
            value = 0.0
            for k in range(N + 1):
                value += cp[k] * B[i][k]
            poly_t[i] = value

        return poly_t

    # ============================================================
    # Topic callbacks / bundle collection
    # ============================================================
    def _make_callback(self, key: str):
        def callback(msg: Float64MultiArray) -> None:
            now_ns = self.get_clock().now().nanoseconds

            self.queues[key].append({
                'values': list(msg.data),
                'recv_time_ns': now_ns,
            })

            self._try_process_bundles()

        return callback

    def _all_queues_have_data(self) -> bool:
        return (
            len(self.queues['x']) > 0 and
            len(self.queues['y']) > 0 and
            len(self.queues['z']) > 0 and
            len(self.queues['psi']) > 0
        )

    def _try_process_bundles(self) -> None:
        while self._all_queues_have_data():
            x_msg = self.queues['x'].popleft()
            y_msg = self.queues['y'].popleft()
            z_msg = self.queues['z'].popleft()
            psi_msg = self.queues['psi'].popleft()

            x_vals = x_msg['values']
            y_vals = y_msg['values']
            z_vals = z_msg['values']
            psi_vals = psi_msg['values']

            # Basic checks
            lengths = [len(x_vals), len(y_vals), len(z_vals), len(psi_vals)]
            if min(lengths) == 0:
                self.get_logger().warn('Received empty vector bundle, skipping.')
                continue

            if not (len(x_vals) == len(y_vals) == len(z_vals) == len(psi_vals)):
                self.get_logger().warn(
                    f'Incoming vector lengths mismatch: x={len(x_vals)}, y={len(y_vals)}, '
                    f'z={len(z_vals)}, psi={len(psi_vals)}. Skipping bundle.'
                )
                continue

            if self.strict_vector_length and len(x_vals) != self.expected_vector_length:
                self.get_logger().warn(
                    f'Incoming vector length={len(x_vals)} does not match expected_vector_length='
                    f'{self.expected_vector_length}. Skipping bundle.'
                )
                continue

            if len(x_vals) < 2:
                self.get_logger().warn(
                    f'Need at least 2 control points, got {len(x_vals)}. Skipping bundle.'
                )
                continue

            now_ns = max(
                x_msg['recv_time_ns'],
                y_msg['recv_time_ns'],
                z_msg['recv_time_ns'],
                psi_msg['recv_time_ns']
            )

            t_dense = self._linspace(self.t0, self.tf, self.num_dense_points)

            # Bernstein evaluation ONLINE
            x_dense = self._bernstein_poly(x_vals, t_dense, self.t0, self.tf)
            y_dense = self._bernstein_poly(y_vals, t_dense, self.t0, self.tf)
            z_dense = self._bernstein_poly(z_vals, t_dense, self.t0, self.tf)
            psi_dense = self._bernstein_poly(psi_vals, t_dense, self.t0, self.tf)

            self.active_path_points = []
            for i in range(self.num_dense_points):
                self.active_path_points.append({
                    'dense_index': i,
                    'dense_time_s': t_dense[i],
                    'x': x_dense[i],
                    'y': y_dense[i],
                    'z': z_dense[i],
                    'psi': psi_dense[i],
                })

            self.active_traj_start_ns = now_ns
            self.active_tf = self.tf
            self.active_order = len(x_vals) - 1
            self.bundle_counter += 1

            self.get_logger().info(
                f'Activated online Bernstein trajectory #{self.bundle_counter} '
                f'with order N={self.active_order}, control_points={len(x_vals)}, '
                f'dense_points={self.num_dense_points}'
            )

            # Publish immediately on new bundle
            self._publish_active_trajectory(now_ns)

    # ============================================================
    # Publishing
    # ============================================================
    def _yaw_to_quaternion(self, yaw: float):
        qx = 0.0
        qy = 0.0
        qz = math.sin(0.5 * yaw)
        qw = math.cos(0.5 * yaw)
        return qx, qy, qz, qw

    def _build_path_msg(self, now_msg_time) -> Path:
        path_msg = Path()
        path_msg.header.stamp = now_msg_time
        path_msg.header.frame_id = self.frame_id

        for p in self.active_path_points:
            pose = PoseStamped()
            pose.header.stamp = now_msg_time
            pose.header.frame_id = self.frame_id

            pose.pose.position.x = p['x']
            pose.pose.position.y = p['y']
            pose.pose.position.z = p['z']

            qx, qy, qz, qw = self._yaw_to_quaternion(p['psi'])
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path_msg.poses.append(pose)

        return path_msg

    def _build_current_pose_msg(self, now_ns: int, now_msg_time) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = now_msg_time
        pose.header.frame_id = self.frame_id

        if self.active_traj_start_ns is None:
            return pose

        if len(self.active_path_points) == 0:
            return pose

        local_time_s = (now_ns - self.active_traj_start_ns) / 1e9
        if local_time_s < self.t0:
            local_time_s = self.t0
        if local_time_s > self.active_tf:
            local_time_s = self.active_tf

        closest_idx = min(
            range(len(self.active_path_points)),
            key=lambda i: abs(self.active_path_points[i]['dense_time_s'] - local_time_s)
        )

        p = self.active_path_points[closest_idx]

        pose.pose.position.x = p['x']
        pose.pose.position.y = p['y']
        pose.pose.position.z = p['z']

        qx, qy, qz, qw = self._yaw_to_quaternion(p['psi'])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def _publish_active_trajectory(self, now_ns: Optional[int] = None) -> None:
        if len(self.active_path_points) == 0:
            return

        if now_ns is None:
            now_ns = self.get_clock().now().nanoseconds

        now_msg_time = self.get_clock().now().to_msg()

        path_msg = self._build_path_msg(now_msg_time)
        current_pose_msg = self._build_current_pose_msg(now_ns, now_msg_time)

        self.path_pub.publish(path_msg)
        self.current_pose_pub.publish(current_pose_msg)

    def timer_callback(self) -> None:
        if len(self.active_path_points) == 0:
            return

        if not self.hold_last_trajectory and self.active_traj_start_ns is not None:
            now_ns = self.get_clock().now().nanoseconds
            local_time_s = (now_ns - self.active_traj_start_ns) / 1e9
            if local_time_s > self.active_tf:
                return

        self._publish_active_trajectory()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = BernsteinTrajectoryPlayer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().error(f'Fatal error: {e}')
        else:
            print(f'Fatal error: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

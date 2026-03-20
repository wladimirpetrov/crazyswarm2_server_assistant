#!/usr/bin/env python3

import csv
import math
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Any, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class BernsteinTrajectoryPlayer(Node):
    def __init__(self) -> None:
        super().__init__('bernstein_trajectory_player')

        scripts_dir = os.path.dirname(os.path.abspath(__file__))

        default_control_csv = os.path.join(scripts_dir, 'control_point_vectors.csv')
        default_bernstein_csv = os.path.join(scripts_dir, 'bernstein_linespaced_vectors.csv')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter('control_csv_file', default_control_csv)
        self.declare_parameter('bernstein_csv_file', default_bernstein_csv)

        self.declare_parameter('x_topic', '/x_vector')
        self.declare_parameter('y_topic', '/y_vector')
        self.declare_parameter('z_topic', '/z_vector')
        self.declare_parameter('psi_topic', '/psi_vector')

        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('path_topic', '/bernstein_mpc_path')
        self.declare_parameter('current_pose_topic', '/bernstein_mpc_current_pose')

        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('match_tolerance', 1e-4)
        self.declare_parameter('hold_last_trajectory', True)

        self.control_csv_file = self.get_parameter('control_csv_file').get_parameter_value().string_value
        self.bernstein_csv_file = self.get_parameter('bernstein_csv_file').get_parameter_value().string_value

        self.x_topic = self.get_parameter('x_topic').get_parameter_value().string_value
        self.y_topic = self.get_parameter('y_topic').get_parameter_value().string_value
        self.z_topic = self.get_parameter('z_topic').get_parameter_value().string_value
        self.psi_topic = self.get_parameter('psi_topic').get_parameter_value().string_value

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.current_pose_topic = self.get_parameter('current_pose_topic').get_parameter_value().string_value

        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.match_tolerance = self.get_parameter('match_tolerance').get_parameter_value().double_value
        self.hold_last_trajectory = self.get_parameter('hold_last_trajectory').get_parameter_value().bool_value

        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn('publish_rate_hz must be > 0. Using 20.0 Hz.')
            self.publish_rate_hz = 20.0

        if self.match_tolerance <= 0.0:
            self.get_logger().warn('match_tolerance must be > 0. Using 1e-4.')
            self.match_tolerance = 1e-4

        # ------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.current_pose_pub = self.create_publisher(PoseStamped, self.current_pose_topic, 10)

        # ------------------------------------------------------------
        # Incoming vector queues
        # Each item is:
        # {
        #   'values': [...],
        #   'recv_time_ns': int
        # }
        # ------------------------------------------------------------
        self.queues: Dict[str, Deque[Dict[str, Any]]] = {
            'x': deque(),
            'y': deque(),
            'z': deque(),
            'psi': deque(),
        }

        # ------------------------------------------------------------
        # Samples loaded from CSV
        #
        # self.samples_by_id[sample_id] = {
        #   'sample_id': int,
        #   'x_cp': [...],
        #   'y_cp': [...],
        #   'z_cp': [...],
        #   'psi_cp': [...],
        #   'bundle_time_ns': int,
        #   'bundle_time_s': float,
        #   'elapsed_time_s': float,
        #   'tf': float,
        #   'dense_length': int,
        #   'points': [
        #       {
        #         'dense_index': int,
        #         'dense_time_s': float,
        #         'x': float,
        #         'y': float,
        #         'z': float,
        #         'psi': float
        #       }, ...
        #   ]
        # }
        # ------------------------------------------------------------
        self.samples_by_id: Dict[int, Dict[str, Any]] = {}
        self.sample_ids_in_order: List[int] = []

        self.active_sample_id: Optional[int] = None
        self.active_sample_start_ns: Optional[int] = None

        self.last_path_publish_sample_id: Optional[int] = None

        self._load_all_csv_data()

        if len(self.samples_by_id) == 0:
            self.get_logger().error('No valid joined samples were loaded.')
            raise RuntimeError('No valid samples found after loading CSV files.')

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
        self.get_logger().info(f'control_csv_file    : {self.control_csv_file}')
        self.get_logger().info(f'bernstein_csv_file  : {self.bernstein_csv_file}')
        self.get_logger().info(f'x_topic             : {self.x_topic}')
        self.get_logger().info(f'y_topic             : {self.y_topic}')
        self.get_logger().info(f'z_topic             : {self.z_topic}')
        self.get_logger().info(f'psi_topic           : {self.psi_topic}')
        self.get_logger().info(f'frame_id            : {self.frame_id}')
        self.get_logger().info(f'path_topic          : {self.path_topic}')
        self.get_logger().info(f'current_pose_topic  : {self.current_pose_topic}')
        self.get_logger().info(f'publish_rate_hz     : {self.publish_rate_hz}')
        self.get_logger().info(f'match_tolerance     : {self.match_tolerance}')
        self.get_logger().info(f'joined samples      : {len(self.samples_by_id)}')
        self.get_logger().info('This node switches trajectory when it sees a matching live')
        self.get_logger().info('x/y/z/psi control-point bundle on the ROS topics.')
        self.get_logger().info('---------------------------------------------')

    # ============================================================
    # CSV loading
    # ============================================================
    def _safe_int(self, value) -> int:
        if value is None or value == '':
            return 0
        return int(float(value))

    def _safe_float(self, value) -> float:
        if value is None or value == '':
            return 0.0
        return float(value)

    def _load_all_csv_data(self) -> None:
        control_samples = self._load_control_points_csv(self.control_csv_file)
        bernstein_samples = self._load_bernstein_csv(self.bernstein_csv_file)

        common_ids = sorted(set(control_samples.keys()).intersection(set(bernstein_samples.keys())))
        if len(common_ids) == 0:
            self.get_logger().error(
                'No common sample_id values found between control_point_vectors.csv '
                'and bernstein_linespaced_vectors.csv'
            )
            return

        joined: Dict[int, Dict[str, Any]] = {}
        for sid in common_ids:
            c = control_samples[sid]
            b = bernstein_samples[sid]

            joined[sid] = {
                'sample_id': sid,
                'x_cp': c['x_cp'],
                'y_cp': c['y_cp'],
                'z_cp': c['z_cp'],
                'psi_cp': c['psi_cp'],
                'bundle_time_ns': c['bundle_time_ns'],
                'bundle_time_s': c['bundle_time_s'],
                'elapsed_time_s': c['elapsed_time_s'],
                'tf': b['tf'],
                'dense_length': b['dense_length'],
                'points': b['points'],
            }

        self.samples_by_id = joined
        self.sample_ids_in_order = common_ids

        self.get_logger().info(
            f'Loaded {len(control_samples)} control-point samples, '
            f'{len(bernstein_samples)} dense-trajectory samples, '
            f'joined {len(common_ids)} samples.'
        )

    def _load_control_points_csv(self, csv_file: str) -> Dict[int, Dict[str, Any]]:
        if not os.path.isfile(csv_file):
            self.get_logger().error(f'Control-point CSV file does not exist: {csv_file}')
            return {}

        grouped: Dict[int, Dict[str, Any]] = {}

        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)

            required_columns = {
                'row_type',
                'sample_id',
                'bundle_time_ns',
                'bundle_time_s',
                'elapsed_time_s',
                'point_index',
                'x_value',
                'y_value',
                'z_value',
                'psi_value',
            }

            if reader.fieldnames is None:
                self.get_logger().error('Control-point CSV has no header.')
                return {}

            missing = required_columns.difference(set(reader.fieldnames))
            if missing:
                self.get_logger().error(f'Control-point CSV missing columns: {sorted(missing)}')
                return {}

            total_rows = 0
            kept_rows = 0

            for row in reader:
                total_rows += 1

                if row.get('row_type', '') != 'control_point':
                    continue

                sid = self._safe_int(row['sample_id'])
                if sid not in grouped:
                    grouped[sid] = {
                        'sample_id': sid,
                        'bundle_time_ns': self._safe_int(row['bundle_time_ns']),
                        'bundle_time_s': self._safe_float(row['bundle_time_s']),
                        'elapsed_time_s': self._safe_float(row['elapsed_time_s']),
                        'rows': []
                    }

                grouped[sid]['rows'].append({
                    'point_index': self._safe_int(row['point_index']),
                    'x': self._safe_float(row['x_value']),
                    'y': self._safe_float(row['y_value']),
                    'z': self._safe_float(row['z_value']),
                    'psi': self._safe_float(row['psi_value']),
                })
                kept_rows += 1

        samples: Dict[int, Dict[str, Any]] = {}

        for sid, sample in grouped.items():
            rows = sorted(sample['rows'], key=lambda r: r['point_index'])

            x_cp = [r['x'] for r in rows]
            y_cp = [r['y'] for r in rows]
            z_cp = [r['z'] for r in rows]
            psi_cp = [r['psi'] for r in rows]

            if len(x_cp) == 0:
                continue

            samples[sid] = {
                'sample_id': sid,
                'bundle_time_ns': sample['bundle_time_ns'],
                'bundle_time_s': sample['bundle_time_s'],
                'elapsed_time_s': sample['elapsed_time_s'],
                'x_cp': x_cp,
                'y_cp': y_cp,
                'z_cp': z_cp,
                'psi_cp': psi_cp,
            }

        self.get_logger().info(
            f'Parsed {kept_rows} control_point rows from {total_rows} total rows in {csv_file}'
        )

        return samples

    def _load_bernstein_csv(self, csv_file: str) -> Dict[int, Dict[str, Any]]:
        if not os.path.isfile(csv_file):
            self.get_logger().error(f'Bernstein CSV file does not exist: {csv_file}')
            return {}

        grouped: Dict[int, Dict[str, Any]] = {}

        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)

            required_columns = {
                'row_type',
                'sample_id',
                'tf',
                'dense_length',
                'dense_index',
                'dense_time_s',
                'x_value',
                'y_value',
                'z_value',
                'psi_value',
            }

            if reader.fieldnames is None:
                self.get_logger().error('Bernstein CSV has no header.')
                return {}

            missing = required_columns.difference(set(reader.fieldnames))
            if missing:
                self.get_logger().error(f'Bernstein CSV missing columns: {sorted(missing)}')
                return {}

            total_rows = 0
            kept_rows = 0

            for row in reader:
                total_rows += 1

                if row.get('row_type', '') != 'bernstein_dense':
                    continue

                sid = self._safe_int(row['sample_id'])

                if sid not in grouped:
                    grouped[sid] = {
                        'sample_id': sid,
                        'tf': self._safe_float(row['tf']),
                        'dense_length': self._safe_int(row['dense_length']),
                        'points': []
                    }

                grouped[sid]['points'].append({
                    'dense_index': self._safe_int(row['dense_index']),
                    'dense_time_s': self._safe_float(row['dense_time_s']),
                    'x': self._safe_float(row['x_value']),
                    'y': self._safe_float(row['y_value']),
                    'z': self._safe_float(row['z_value']),
                    'psi': self._safe_float(row['psi_value']),
                })
                kept_rows += 1

        samples: Dict[int, Dict[str, Any]] = {}

        for sid, sample in grouped.items():
            points_sorted = sorted(sample['points'], key=lambda p: p['dense_index'])

            unique_points = []
            seen = set()
            for p in points_sorted:
                idx = p['dense_index']
                if idx in seen:
                    continue
                seen.add(idx)
                unique_points.append(p)

            if len(unique_points) == 0:
                continue

            samples[sid] = {
                'sample_id': sid,
                'tf': sample['tf'],
                'dense_length': sample['dense_length'],
                'points': unique_points,
            }

        self.get_logger().info(
            f'Parsed {kept_rows} bernstein_dense rows from {total_rows} total rows in {csv_file}'
        )

        return samples

    # ============================================================
    # Subscriptions and bundle processing
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

            match_id, score, maxdiff = self._find_matching_sample_id(x_vals, y_vals, z_vals, psi_vals)

            if match_id is None:
                self.get_logger().warn(
                    'No matching sample found for incoming vector bundle. '
                    f'Lengths: x={len(x_vals)}, y={len(y_vals)}, z={len(z_vals)}, psi={len(psi_vals)}'
                )
                continue

            now_ns = max(
                x_msg['recv_time_ns'],
                y_msg['recv_time_ns'],
                z_msg['recv_time_ns'],
                psi_msg['recv_time_ns']
            )

            self.active_sample_id = match_id
            self.active_sample_start_ns = now_ns

            self.get_logger().info(
                f'Activated sample_id={match_id}, score={score:.12f}, maxdiff={maxdiff:.12f}'
            )

            # Publish immediately on switch
            self._publish_active_trajectory(now_ns)

    def _find_matching_sample_id(
        self,
        x_vals: List[float],
        y_vals: List[float],
        z_vals: List[float],
        psi_vals: List[float]
    ) -> Tuple[Optional[int], float, float]:
        best_id: Optional[int] = None
        best_score = float('inf')
        best_maxdiff = float('inf')

        for sid in self.sample_ids_in_order:
            sample = self.samples_by_id[sid]

            x_cp = sample['x_cp']
            y_cp = sample['y_cp']
            z_cp = sample['z_cp']
            psi_cp = sample['psi_cp']

            if not (
                len(x_vals) == len(x_cp) and
                len(y_vals) == len(y_cp) and
                len(z_vals) == len(z_cp) and
                len(psi_vals) == len(psi_cp)
            ):
                continue

            maxdiff = 0.0
            score = 0.0
            valid = True

            for a, b in zip(x_vals, x_cp):
                d = abs(a - b)
                score += d
                if d > maxdiff:
                    maxdiff = d
                if d > self.match_tolerance:
                    valid = False
                    break
            if not valid:
                continue

            for a, b in zip(y_vals, y_cp):
                d = abs(a - b)
                score += d
                if d > maxdiff:
                    maxdiff = d
                if d > self.match_tolerance:
                    valid = False
                    break
            if not valid:
                continue

            for a, b in zip(z_vals, z_cp):
                d = abs(a - b)
                score += d
                if d > maxdiff:
                    maxdiff = d
                if d > self.match_tolerance:
                    valid = False
                    break
            if not valid:
                continue

            for a, b in zip(psi_vals, psi_cp):
                d = abs(a - b)
                score += d
                if d > maxdiff:
                    maxdiff = d
                if d > self.match_tolerance:
                    valid = False
                    break
            if not valid:
                continue

            if score < best_score:
                best_id = sid
                best_score = score
                best_maxdiff = maxdiff

        return best_id, best_score, best_maxdiff

    # ============================================================
    # Publishing
    # ============================================================
    def _yaw_to_quaternion(self, yaw: float):
        qx = 0.0
        qy = 0.0
        qz = math.sin(0.5 * yaw)
        qw = math.cos(0.5 * yaw)
        return qx, qy, qz, qw

    def _build_path_msg(self, sample: Dict[str, Any], now_msg_time) -> Path:
        path_msg = Path()
        path_msg.header.stamp = now_msg_time
        path_msg.header.frame_id = self.frame_id

        for p in sample['points']:
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

    def _build_current_pose_msg(
        self,
        sample: Dict[str, Any],
        now_ns: int,
        now_msg_time
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = now_msg_time
        pose.header.frame_id = self.frame_id

        if self.active_sample_start_ns is None:
            return pose

        points = sample['points']
        if len(points) == 0:
            return pose

        tf = sample['tf']
        if tf <= 0.0:
            tf = 0.9

        local_time_s = (now_ns - self.active_sample_start_ns) / 1e9
        if local_time_s < 0.0:
            local_time_s = 0.0
        if local_time_s > tf:
            local_time_s = tf

        closest_idx = min(
            range(len(points)),
            key=lambda i: abs(points[i]['dense_time_s'] - local_time_s)
        )

        p = points[closest_idx]

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
        if self.active_sample_id is None:
            return

        if self.active_sample_id not in self.samples_by_id:
            return

        if now_ns is None:
            now_ns = self.get_clock().now().nanoseconds

        sample = self.samples_by_id[self.active_sample_id]
        now_msg_time = self.get_clock().now().to_msg()

        path_msg = self._build_path_msg(sample, now_msg_time)
        current_pose_msg = self._build_current_pose_msg(sample, now_ns, now_msg_time)

        self.path_pub.publish(path_msg)
        self.current_pose_pub.publish(current_pose_msg)

        if self.last_path_publish_sample_id != self.active_sample_id:
            self.get_logger().info(
                f'Publishing trajectory for sample_id={self.active_sample_id}, '
                f'points={len(sample["points"])}'
            )
            self.last_path_publish_sample_id = self.active_sample_id

    def timer_callback(self) -> None:
        if self.active_sample_id is None:
            return

        if not self.hold_last_trajectory and self.active_sample_start_ns is not None:
            sample = self.samples_by_id[self.active_sample_id]
            tf = sample['tf']
            if tf <= 0.0:
                tf = 0.9

            now_ns = self.get_clock().now().nanoseconds
            local_time_s = (now_ns - self.active_sample_start_ns) / 1e9
            if local_time_s > tf:
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

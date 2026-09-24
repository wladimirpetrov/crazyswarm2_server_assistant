#!/usr/bin/env python3

import csv
import math
import os
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


Point3 = Tuple[float, float, float]


class ExperimentSosMppOverObs(Node):
    """
    Scenario 0 visualization node.

    Publishes:
      1. Scenario-0 trajectory read from python_run.csv using:
           point_id, init_px, init_py, init_pz
         for point_id = 0..99.
      2. A front tail:
           [1.8, 4.2, 0.6] -> rounded corner near [-1.8, 4.2, 0.6]
           -> CSV point_id 0.
      3. A final extension:
           CSV point_id 99 -> [1.2, 1.8, 0.6].
      4. A rectangular wall/obstacle for scenario 0.

    This node intentionally publishes on its own topic so it does not conflict
    with the dynamic scenario-1/scenario-2 obstacle topic.
    """

    def __init__(self) -> None:
        super().__init__('experiment_sos_mpp_over_obs')

        # ============================================================
        # General settings
        # ============================================================
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('marker_topic', '/experiment_sos_mpp_over_obs_markers')
        self.declare_parameter('publish_rate_hz', 20.0)

        # CSV can be absolute or relative. If relative, this node tries:
        #   current working directory
        #   crazyflie package share directory
        #   crazyflie package share/data directory
        self.declare_parameter('csv_file', 'python_run.csv')
        self.declare_parameter('csv_package_name', 'crazyflie')
        self.declare_parameter('point_id_min', 0)
        self.declare_parameter('point_id_max', 99)

        # ============================================================
        # Scenario-0 trajectory styling
        # ============================================================
        self.declare_parameter('trajectory_color_r', 0.0)
        self.declare_parameter('trajectory_color_g', 0.3)
        self.declare_parameter('trajectory_color_b', 1.0)
        self.declare_parameter('trajectory_color_a', 0.35)
        self.declare_parameter('trajectory_line_width', 0.018)

        # ============================================================
        # Front tail and final extension
        # ============================================================
        self.declare_parameter('tail_enabled', True)
        self.declare_parameter('tail_start_x', 1.8)
        self.declare_parameter('tail_start_y', 4.2)
        self.declare_parameter('tail_start_z', 0.6)
        self.declare_parameter('tail_corner_x', -1.8)
        self.declare_parameter('tail_corner_y', 4.2)
        self.declare_parameter('tail_corner_z', 0.6)
        self.declare_parameter('tail_corner_radius', 0.10)
        self.declare_parameter('tail_arc_points', 12)

        self.declare_parameter('final_extension_enabled', True)
        self.declare_parameter('final_target_x', 1.2)
        self.declare_parameter('final_target_y', 1.8)
        self.declare_parameter('final_target_z', 0.6)

        # ============================================================
        # Scenario-0 rectangular wall
        # Centerline endpoints in XY:
        #   end 1 = [-1.4, 2.6]
        #   end 2 = [-1.4, 3.6]
        # width/thickness = 0.22 m, height = 0.65 m, length = 1.0 m
        # ============================================================
        self.declare_parameter('wall_enabled', True)
        self.declare_parameter('wall_end1_x', -1.4)
        self.declare_parameter('wall_end1_y', 2.6)
        self.declare_parameter('wall_end2_x', -1.4)
        self.declare_parameter('wall_end2_y', 3.6)
        self.declare_parameter('wall_base_z', 0.0)
        self.declare_parameter('wall_width', 0.22)
        self.declare_parameter('wall_height', 0.65)
        self.declare_parameter('wall_color_r', 0.75)
        self.declare_parameter('wall_color_g', 0.75)
        self.declare_parameter('wall_color_b', 0.75)
        self.declare_parameter('wall_color_a', 0.85)

        # ============================================================
        # Read parameters
        # ============================================================
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self.csv_file = self.get_parameter('csv_file').get_parameter_value().string_value
        self.csv_package_name = self.get_parameter('csv_package_name').get_parameter_value().string_value
        self.point_id_min = self.get_parameter('point_id_min').get_parameter_value().integer_value
        self.point_id_max = self.get_parameter('point_id_max').get_parameter_value().integer_value

        self.trajectory_color_r = self.get_parameter('trajectory_color_r').get_parameter_value().double_value
        self.trajectory_color_g = self.get_parameter('trajectory_color_g').get_parameter_value().double_value
        self.trajectory_color_b = self.get_parameter('trajectory_color_b').get_parameter_value().double_value
        self.trajectory_color_a = self.get_parameter('trajectory_color_a').get_parameter_value().double_value
        self.trajectory_line_width = self.get_parameter('trajectory_line_width').get_parameter_value().double_value

        self.tail_enabled = self.get_parameter('tail_enabled').get_parameter_value().bool_value
        self.tail_start = (
            self.get_parameter('tail_start_x').get_parameter_value().double_value,
            self.get_parameter('tail_start_y').get_parameter_value().double_value,
            self.get_parameter('tail_start_z').get_parameter_value().double_value,
        )
        self.tail_corner = (
            self.get_parameter('tail_corner_x').get_parameter_value().double_value,
            self.get_parameter('tail_corner_y').get_parameter_value().double_value,
            self.get_parameter('tail_corner_z').get_parameter_value().double_value,
        )
        self.tail_corner_radius = self.get_parameter('tail_corner_radius').get_parameter_value().double_value
        self.tail_arc_points = self.get_parameter('tail_arc_points').get_parameter_value().integer_value

        self.final_extension_enabled = self.get_parameter('final_extension_enabled').get_parameter_value().bool_value
        self.final_target = (
            self.get_parameter('final_target_x').get_parameter_value().double_value,
            self.get_parameter('final_target_y').get_parameter_value().double_value,
            self.get_parameter('final_target_z').get_parameter_value().double_value,
        )

        self.wall_enabled = self.get_parameter('wall_enabled').get_parameter_value().bool_value
        self.wall_end1 = (
            self.get_parameter('wall_end1_x').get_parameter_value().double_value,
            self.get_parameter('wall_end1_y').get_parameter_value().double_value,
        )
        self.wall_end2 = (
            self.get_parameter('wall_end2_x').get_parameter_value().double_value,
            self.get_parameter('wall_end2_y').get_parameter_value().double_value,
        )
        self.wall_base_z = self.get_parameter('wall_base_z').get_parameter_value().double_value
        self.wall_width = self.get_parameter('wall_width').get_parameter_value().double_value
        self.wall_height = self.get_parameter('wall_height').get_parameter_value().double_value
        self.wall_color_r = self.get_parameter('wall_color_r').get_parameter_value().double_value
        self.wall_color_g = self.get_parameter('wall_color_g').get_parameter_value().double_value
        self.wall_color_b = self.get_parameter('wall_color_b').get_parameter_value().double_value
        self.wall_color_a = self.get_parameter('wall_color_a').get_parameter_value().double_value

        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn('publish_rate_hz must be > 0. Using 20.0.')
            self.publish_rate_hz = 20.0

        if self.point_id_max < self.point_id_min:
            raise RuntimeError('point_id_max must be >= point_id_min.')

        self.csv_path = self._resolve_csv_path(self.csv_file)
        self.csv_trajectory_points = self._load_csv_trajectory_points(self.csv_path)
        self.path_points = self._build_complete_path(self.csv_trajectory_points)

        if len(self.path_points) < 2:
            raise RuntimeError('Scenario-0 path must contain at least 2 points.')

        self.pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.timer_callback)

        self.get_logger().info('---------------------------------------------')
        self.get_logger().info('experiment_sos_mpp_over_obs started')
        self.get_logger().info(f'frame_id              : {self.frame_id}')
        self.get_logger().info(f'marker_topic          : {self.marker_topic}')
        self.get_logger().info(f'csv_path              : {self.csv_path}')
        self.get_logger().info(f'point_id range        : {self.point_id_min}..{self.point_id_max}')
        self.get_logger().info(f'csv trajectory points : {len(self.csv_trajectory_points)}')
        self.get_logger().info(f'complete path points  : {len(self.path_points)}')
        self.get_logger().info(f'tail_enabled          : {self.tail_enabled}')
        self.get_logger().info(f'final_extension       : {self.final_extension_enabled}')
        self.get_logger().info(f'wall_enabled          : {self.wall_enabled}')
        self.get_logger().info('---------------------------------------------')

    # ============================================================
    # CSV loading
    # ============================================================
    def _resolve_csv_path(self, csv_file: str) -> str:
        candidates = []

        if os.path.isabs(csv_file):
            candidates.append(csv_file)
        else:
            candidates.append(os.path.abspath(csv_file))

            if get_package_share_directory is not None:
                try:
                    share_dir = get_package_share_directory(self.csv_package_name)
                    candidates.append(os.path.join(share_dir, csv_file))
                    candidates.append(os.path.join(share_dir, 'data', csv_file))
                    candidates.append(os.path.join(share_dir, 'config', csv_file))
                except Exception:
                    pass

        for path in candidates:
            if path and os.path.exists(path):
                return path

        msg = 'Could not find CSV file. Tried:\n' + '\n'.join(f'  - {p}' for p in candidates)
        raise RuntimeError(msg)

    def _load_csv_trajectory_points(self, csv_path: str) -> List[Point3]:
        rows = []

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            required = ['point_id', 'init_px', 'init_py', 'init_pz']

            if reader.fieldnames is None:
                raise RuntimeError('CSV has no header row.')

            missing = [name for name in required if name not in reader.fieldnames]
            if missing:
                raise RuntimeError(f'CSV is missing required columns: {missing}')

            for row in reader:
                try:
                    point_id = int(float(row['point_id']))
                except Exception:
                    continue

                if point_id < self.point_id_min or point_id > self.point_id_max:
                    continue

                try:
                    x = float(row['init_px'])
                    y = float(row['init_py'])
                    z = float(row['init_pz'])
                except Exception:
                    self.get_logger().warn(f'Skipping bad CSV row for point_id={point_id}')
                    continue

                rows.append((point_id, x, y, z))

        rows.sort(key=lambda item: item[0])

        if len(rows) == 0:
            raise RuntimeError(
                f'No CSV rows found for point_id range {self.point_id_min}..{self.point_id_max}.'
            )

        expected = self.point_id_max - self.point_id_min + 1
        if len(rows) != expected:
            self.get_logger().warn(
                f'Expected {expected} CSV trajectory rows, but loaded {len(rows)}.'
            )

        return [(x, y, z) for (_, x, y, z) in rows]

    # ============================================================
    # Path geometry helpers
    # ============================================================
    def _build_complete_path(self, csv_points: List[Point3]) -> List[Point3]:
        points: List[Point3] = []

        if self.tail_enabled and len(csv_points) > 0:
            tail_points = self._make_rounded_tail(
                previous_pt=self.tail_start,
                corner_pt=self.tail_corner,
                next_pt=csv_points[0],
                radius=self.tail_corner_radius,
                arc_points=max(3, self.tail_arc_points),
            )
            points.extend(tail_points[:-1])

        points.extend(csv_points)

        if self.final_extension_enabled:
            points.append(self.final_target)

        return points

    def _make_rounded_tail(
        self,
        previous_pt: Point3,
        corner_pt: Point3,
        next_pt: Point3,
        radius: float,
        arc_points: int,
    ) -> List[Point3]:
        """
        Creates a rounded 2D corner in XY with constant Z.

        Output order:
          previous_pt -> tangent on incoming segment -> arc -> tangent on outgoing segment -> next_pt
        """
        px, py, pz = previous_pt
        cx, cy, cz = corner_pt
        nx, ny, nz = next_pt

        v1x = px - cx
        v1y = py - cy
        v2x = nx - cx
        v2y = ny - cy

        l1 = math.hypot(v1x, v1y)
        l2 = math.hypot(v2x, v2y)

        if radius <= 0.0 or l1 < 1e-9 or l2 < 1e-9:
            return [previous_pt, corner_pt, next_pt]

        u1x = v1x / l1
        u1y = v1y / l1
        u2x = v2x / l2
        u2y = v2y / l2

        dot = max(-1.0, min(1.0, u1x * u2x + u1y * u2y))
        theta = math.acos(dot)

        # No useful rounded corner for almost straight or almost 180-degree reversal.
        if theta < 1e-3 or abs(math.pi - theta) < 1e-3:
            return [previous_pt, corner_pt, next_pt]

        tangent_dist = radius / math.tan(0.5 * theta)
        tangent_dist = min(tangent_dist, 0.49 * l1, 0.49 * l2)

        # Effective radius after clamping tangent distance.
        effective_radius = tangent_dist * math.tan(0.5 * theta)

        t1 = (cx + u1x * tangent_dist, cy + u1y * tangent_dist, cz)
        t2 = (cx + u2x * tangent_dist, cy + u2y * tangent_dist, cz)

        bx = u1x + u2x
        by = u1y + u2y
        bl = math.hypot(bx, by)
        if bl < 1e-9:
            return [previous_pt, corner_pt, next_pt]

        bx /= bl
        by /= bl

        center_dist = effective_radius / math.sin(0.5 * theta)
        center_x = cx + bx * center_dist
        center_y = cy + by * center_dist

        start_angle = math.atan2(t1[1] - center_y, t1[0] - center_x)
        end_angle = math.atan2(t2[1] - center_y, t2[0] - center_x)

        delta = end_angle - start_angle
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi

        arc: List[Point3] = []
        for i in range(arc_points):
            s = i / float(arc_points - 1)
            a = start_angle + s * delta
            arc.append(
                (
                    center_x + effective_radius * math.cos(a),
                    center_y + effective_radius * math.sin(a),
                    cz,
                )
            )

        # Force exact tangent endpoints to avoid numerical drift.
        arc[0] = t1
        arc[-1] = t2

        return [previous_pt] + arc + [next_pt]

    def _make_point(self, x: float, y: float, z: float) -> Point:
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    def _yaw_to_quaternion(self, yaw: float):
        return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)

    # ============================================================
    # Marker creation
    # ============================================================
    def make_trajectory_marker(self, marker_id: int) -> Marker:
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'scenario_zero_trajectory'
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = self.trajectory_line_width

        m.color.r = self.trajectory_color_r
        m.color.g = self.trajectory_color_g
        m.color.b = self.trajectory_color_b
        m.color.a = self.trajectory_color_a

        m.points = [self._make_point(x, y, z) for x, y, z in self.path_points]

        m.lifetime.sec = 0
        m.frame_locked = False
        return m

    def make_wall_marker(self, marker_id: int) -> Marker:
        x1, y1 = self.wall_end1
        x2, y2 = self.wall_end2

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            length = 1.0
            yaw = 0.0
        else:
            yaw = math.atan2(dy, dx)

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'scenario_zero_rectangular_wall'
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = 0.5 * (x1 + x2)
        m.pose.position.y = 0.5 * (y1 + y2)
        m.pose.position.z = self.wall_base_z + 0.5 * self.wall_height

        qx, qy, qz, qw = self._yaw_to_quaternion(yaw)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        # local X axis = wall length, local Y axis = wall width/thickness
        m.scale.x = length
        m.scale.y = self.wall_width
        m.scale.z = self.wall_height

        m.color.r = self.wall_color_r
        m.color.g = self.wall_color_g
        m.color.b = self.wall_color_b
        m.color.a = self.wall_color_a

        m.lifetime.sec = 0
        m.frame_locked = False
        return m

    def timer_callback(self) -> None:
        arr = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.frame_id
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        marker_id = 0

        arr.markers.append(self.make_trajectory_marker(marker_id))
        marker_id += 1

        if self.wall_enabled:
            arr.markers.append(self.make_wall_marker(marker_id))
            marker_id += 1

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentSosMppOverObs()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

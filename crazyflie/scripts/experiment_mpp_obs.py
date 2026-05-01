#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class ExperimentZ2D(Node):
    def __init__(self) -> None:
        super().__init__('experiment_z_2d')

        self.show_station_numbers = False

        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('marker_topic', '/experiment_z_2d_markers')
        self.declare_parameter('publish_rate_hz', 20.0)

        # ============================================================
        # Station cube settings
        # ============================================================

        self.declare_parameter('station_cube_size_xy', 0.10)
        self.declare_parameter('station_cube_down_z', 0.005)
        self.declare_parameter('station_cube_up_z', 0.03)

        # Start station color: blue
        self.declare_parameter('station_color_r', 0.0)
        self.declare_parameter('station_color_g', 0.3)
        self.declare_parameter('station_color_b', 1.0)
        self.declare_parameter('station_color_a', 0.25)

        # Target station color: green
        self.declare_parameter('target_station_color_r', 0.0)
        self.declare_parameter('target_station_color_g', 1.0)
        self.declare_parameter('target_station_color_b', 0.2)
        self.declare_parameter('target_station_color_a', 0.45)

        # ============================================================
        # Desired path settings
        # ============================================================

        self.declare_parameter('path_color_r', 0.0)
        self.declare_parameter('path_color_g', 0.3)
        self.declare_parameter('path_color_b', 1.0)
        self.declare_parameter('path_color_a', 1.0)
        self.declare_parameter('path_line_width', 0.015)

        # ============================================================
        # Label settings
        # ============================================================

        self.declare_parameter('label_scale_z', 0.08)
        self.declare_parameter('label_offset_x', -0.05)
        self.declare_parameter('label_offset_y', 0.05)
        self.declare_parameter('label_offset_z', 0.032)
        self.declare_parameter('label_color_r', 1.0)
        self.declare_parameter('label_color_g', 1.0)
        self.declare_parameter('label_color_b', 1.0)
        self.declare_parameter('label_color_a', 1.0)

        # ============================================================
        # Obstacle settings
        #
        # Two vertical cylinders:
        #   Obstacle 1 center: x = -0.6, y = 2.0
        #   Obstacle 2 center: x =  0.6, y = 1.8
        #
        # Diameter: 0.30 m
        # Height:   1.20 m
        # Base z:   0.00 m
        # ============================================================

        self.declare_parameter(
            'obstacles_xy',
            [
                -0.6, 2.0,
                 0.6, 1.8,
            ]
        )

        self.declare_parameter('obstacle_diameter', 0.30)
        self.declare_parameter('obstacle_height', 1.20)
        self.declare_parameter('obstacle_base_z', 0.0)

        self.declare_parameter('obstacle_color_r', 1.0)
        self.declare_parameter('obstacle_color_g', 0.2)
        self.declare_parameter('obstacle_color_b', 0.0)
        self.declare_parameter('obstacle_color_a', 0.65)

        # ============================================================
        # Two stations / waypoints
        #
        # Station 1: start station, blue
        # Station 2: target station, green
        #
        # These two points are connected by the LINE_STRIP marker.
        # ============================================================

        self.declare_parameter(
            'waypoints_xyz',
            [
                -1.2, 1.8, 0.6,
                 1.2, 1.8, 0.6,
            ]
        )

        self.declare_parameter('waypoint_yaw_deg', [0.0, 0.0])

        # ============================================================
        # Read parameters
        # ============================================================

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self.station_cube_size_xy = self.get_parameter('station_cube_size_xy').get_parameter_value().double_value
        self.station_cube_down_z = self.get_parameter('station_cube_down_z').get_parameter_value().double_value
        self.station_cube_up_z = self.get_parameter('station_cube_up_z').get_parameter_value().double_value

        self.station_color_r = self.get_parameter('station_color_r').get_parameter_value().double_value
        self.station_color_g = self.get_parameter('station_color_g').get_parameter_value().double_value
        self.station_color_b = self.get_parameter('station_color_b').get_parameter_value().double_value
        self.station_color_a = self.get_parameter('station_color_a').get_parameter_value().double_value

        self.target_station_color_r = self.get_parameter('target_station_color_r').get_parameter_value().double_value
        self.target_station_color_g = self.get_parameter('target_station_color_g').get_parameter_value().double_value
        self.target_station_color_b = self.get_parameter('target_station_color_b').get_parameter_value().double_value
        self.target_station_color_a = self.get_parameter('target_station_color_a').get_parameter_value().double_value

        self.path_color_r = self.get_parameter('path_color_r').get_parameter_value().double_value
        self.path_color_g = self.get_parameter('path_color_g').get_parameter_value().double_value
        self.path_color_b = self.get_parameter('path_color_b').get_parameter_value().double_value
        self.path_color_a = self.get_parameter('path_color_a').get_parameter_value().double_value
        self.path_line_width = self.get_parameter('path_line_width').get_parameter_value().double_value

        self.label_scale_z = self.get_parameter('label_scale_z').get_parameter_value().double_value
        self.label_offset_x = self.get_parameter('label_offset_x').get_parameter_value().double_value
        self.label_offset_y = self.get_parameter('label_offset_y').get_parameter_value().double_value
        self.label_offset_z = self.get_parameter('label_offset_z').get_parameter_value().double_value
        self.label_color_r = self.get_parameter('label_color_r').get_parameter_value().double_value
        self.label_color_g = self.get_parameter('label_color_g').get_parameter_value().double_value
        self.label_color_b = self.get_parameter('label_color_b').get_parameter_value().double_value
        self.label_color_a = self.get_parameter('label_color_a').get_parameter_value().double_value

        self.obstacle_diameter = self.get_parameter('obstacle_diameter').get_parameter_value().double_value
        self.obstacle_height = self.get_parameter('obstacle_height').get_parameter_value().double_value
        self.obstacle_base_z = self.get_parameter('obstacle_base_z').get_parameter_value().double_value

        self.obstacle_color_r = self.get_parameter('obstacle_color_r').get_parameter_value().double_value
        self.obstacle_color_g = self.get_parameter('obstacle_color_g').get_parameter_value().double_value
        self.obstacle_color_b = self.get_parameter('obstacle_color_b').get_parameter_value().double_value
        self.obstacle_color_a = self.get_parameter('obstacle_color_a').get_parameter_value().double_value

        waypoint_flat = list(self.get_parameter('waypoints_xyz').value)
        yaw_deg_list = list(self.get_parameter('waypoint_yaw_deg').value)
        obstacle_flat = list(self.get_parameter('obstacles_xy').value)

        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 20.0

        self.waypoints = self._parse_waypoints(waypoint_flat)

        if len(self.waypoints) == 0:
            raise RuntimeError('No valid waypoints provided.')

        self.yaw_rad = self._parse_yaws(yaw_deg_list, len(self.waypoints))
        self.obstacles_xy = self._parse_obstacles(obstacle_flat)

        self.pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.timer_callback)

        self.get_logger().info('---------------------------------------------')
        self.get_logger().info('experiment_z_2d started')
        self.get_logger().info(f'frame_id            : {self.frame_id}')
        self.get_logger().info(f'marker_topic        : {self.marker_topic}')
        self.get_logger().info(f'publish_rate_hz     : {self.publish_rate_hz}')
        self.get_logger().info(f'waypoint_count      : {len(self.waypoints)}')
        self.get_logger().info(f'obstacle_count      : {len(self.obstacles_xy)}')
        self.get_logger().info(f'obstacle_diameter   : {self.obstacle_diameter}')
        self.get_logger().info(f'obstacle_height     : {self.obstacle_height}')
        self.get_logger().info(f'obstacle_base_z     : {self.obstacle_base_z}')
        self.get_logger().info(f'show_station_numbers: {self.show_station_numbers}')
        self.get_logger().info('---------------------------------------------')

    def _parse_waypoints(self, flat: List[float]) -> List[Tuple[float, float, float]]:
        pts = []
        usable = (len(flat) // 3) * 3

        for i in range(0, usable, 3):
            pts.append(
                (
                    float(flat[i]),
                    float(flat[i + 1]),
                    float(flat[i + 2]),
                )
            )

        return pts

    def _parse_obstacles(self, flat: List[float]) -> List[Tuple[float, float]]:
        obstacles = []
        usable = (len(flat) // 2) * 2

        for i in range(0, usable, 2):
            obstacles.append(
                (
                    float(flat[i]),
                    float(flat[i + 1]),
                )
            )

        return obstacles

    def _parse_yaws(self, yaw_deg_list: List[float], n: int) -> List[float]:
        yaws = [0.0] * n

        for i in range(min(n, len(yaw_deg_list))):
            yaws[i] = math.radians(float(yaw_deg_list[i]))

        return yaws

    def _make_point(self, x: float, y: float, z: float) -> Point:
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p

    def _yaw_to_quaternion(self, yaw: float):
        qx = 0.0
        qy = 0.0
        qz = math.sin(0.5 * yaw)
        qw = math.cos(0.5 * yaw)

        return qx, qy, qz, qw

    def make_station_cube_marker(
        self,
        marker_id: int,
        x: float,
        y: float,
        z: float,
        yaw: float,
        is_target_station: bool,
    ) -> Marker:
        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        if is_target_station:
            m.ns = 'target_station_cube'
        else:
            m.ns = 'station_cubes'

        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD

        z_center = z + 0.5 * (self.station_cube_up_z - self.station_cube_down_z)

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z_center

        qx, qy, qz, qw = self._yaw_to_quaternion(yaw)

        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        m.scale.x = self.station_cube_size_xy
        m.scale.y = self.station_cube_size_xy
        m.scale.z = self.station_cube_down_z + self.station_cube_up_z

        if is_target_station:
            m.color.r = self.target_station_color_r
            m.color.g = self.target_station_color_g
            m.color.b = self.target_station_color_b
            m.color.a = self.target_station_color_a
        else:
            m.color.r = self.station_color_r
            m.color.g = self.station_color_g
            m.color.b = self.station_color_b
            m.color.a = self.station_color_a

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    def make_obstacle_cylinder_marker(
        self,
        marker_id: int,
        x: float,
        y: float,
    ) -> Marker:
        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'vertical_cylinder_obstacles'
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = self.obstacle_base_z + 0.5 * self.obstacle_height

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = self.obstacle_diameter
        m.scale.y = self.obstacle_diameter
        m.scale.z = self.obstacle_height

        m.color.r = self.obstacle_color_r
        m.color.g = self.obstacle_color_g
        m.color.b = self.obstacle_color_b
        m.color.a = self.obstacle_color_a

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    def make_label_marker(
        self,
        marker_id: int,
        x: float,
        y: float,
        z: float,
        number: int,
    ) -> Marker:
        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'station_labels'
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD

        top_of_cube = z + self.station_cube_up_z

        m.pose.position.x = x + self.label_offset_x
        m.pose.position.y = y + self.label_offset_y
        m.pose.position.z = top_of_cube + self.label_offset_z

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.z = self.label_scale_z

        m.color.r = self.label_color_r
        m.color.g = self.label_color_g
        m.color.b = self.label_color_b
        m.color.a = self.label_color_a

        m.text = str(number)

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    def make_path_marker(self, marker_id: int) -> Marker:
        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'desired_path'
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.scale.x = self.path_line_width

        m.color.r = self.path_color_r
        m.color.g = self.path_color_g
        m.color.b = self.path_color_b
        m.color.a = self.path_color_a

        m.points = [
            self._make_point(x, y, z)
            for (x, y, z) in self.waypoints
        ]

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    def timer_callback(self) -> None:
        now = self.get_clock().now()
        arr = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.frame_id
        delete_all.header.stamp = now.to_msg()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        marker_id = 0

        # ============================================================
        # Draw two stations:
        #   station 1 = blue start station
        #   station 2 = green target station
        # ============================================================

        for i, (pt, yaw) in enumerate(zip(self.waypoints, self.yaw_rad), start=1):
            x, y, z = pt

            is_target_station = i == len(self.waypoints)

            arr.markers.append(
                self.make_station_cube_marker(
                    marker_id=marker_id,
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    is_target_station=is_target_station,
                )
            )
            marker_id += 1

            if self.show_station_numbers:
                arr.markers.append(
                    self.make_label_marker(
                        marker_id=marker_id,
                        x=x,
                        y=y,
                        z=z,
                        number=i,
                    )
                )
                marker_id += 1

        # ============================================================
        # Draw two vertical cylinder obstacles
        # ============================================================

        for x_obs, y_obs in self.obstacles_xy:
            arr.markers.append(
                self.make_obstacle_cylinder_marker(
                    marker_id=marker_id,
                    x=x_obs,
                    y=y_obs,
                )
            )
            marker_id += 1

        # ============================================================
        # Draw desired path connecting the two stations
        # ============================================================

        arr.markers.append(self.make_path_marker(marker_id))

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)

    node = ExperimentZ2D()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

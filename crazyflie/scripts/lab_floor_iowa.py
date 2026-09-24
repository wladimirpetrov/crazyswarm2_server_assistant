#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class LabFloorIowa(Node):
    def __init__(self) -> None:
        super().__init__('lab_floor_iowa')

        # ============================================================
        # GENERAL PARAMETERS
        # ============================================================

        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('topic', '/lab_floor_markers')
        self.declare_parameter('airflow_topic', '/fan_airflow_markers_change')
        self.declare_parameter('publish_rate_hz', 1.0)

        # ============================================================
        # FLOOR TILE PARAMETERS
        # ============================================================

        self.declare_parameter('floor_tile_size', 0.6)
        self.declare_parameter('floor_tile_thickness', 0.01)

        # ============================================================
        # WALL TILE PARAMETERS
        # ============================================================

        self.declare_parameter('wall_tile_size', 0.3)
        self.declare_parameter('wall_tile_thickness', 0.02)

        # ============================================================
        # FLOOR ORIGIN
        # ============================================================

        self.declare_parameter('floor_origin_x', -1.8)
        self.declare_parameter('floor_origin_y', 3.0)
        self.declare_parameter('floor_origin_z', -0.01)

        # ============================================================
        # WALL ORIGIN
        # ============================================================

        self.declare_parameter('wall_origin_x', 4.2)
        self.declare_parameter('wall_origin_y', -2.7)
        self.declare_parameter('wall_origin_z', 0.75)

        # CAS wall rotation around Z axis
        self.declare_parameter('wall_yaw_deg', -90.0)

        # ============================================================
        # FAN SELECTION
        #
        # 0 = no fan
        # 1 = original small fan
        # 2 = new large fan
        # ============================================================

        self.declare_parameter('fan_selection', 0)

        # ============================================================
        # FAN 1 - ORIGINAL SMALL FAN
        # ============================================================

        self.declare_parameter('fan1_width', 0.15)
        self.declare_parameter('fan1_depth', 0.10)
        self.declare_parameter('fan1_height', 1.20)

        # ============================================================
        # FAN 2 - NEW LARGE FAN
        #
        # Width  = 45 inches = 1.1430 m
        # Depth  = 31 inches = 0.7874 m
        # Height = 75 inches = 1.9050 m
        # ============================================================

        self.declare_parameter('fan2_width', 1.1430)
        self.declare_parameter('fan2_depth', 0.7874)
        self.declare_parameter('fan2_height', 1.9050)

        # ============================================================
        # FAN POSITION
        #
        # x/y position is identical for both fans.
        #
        # fan_base_z is the BOTTOM of the fan.
        # This means both fans sit on the floor correctly even though
        # they have different heights.
        # ============================================================

        self.declare_parameter('fan_x', -2.10)
        self.declare_parameter('fan_y', -0.900)
        self.declare_parameter('fan_base_z', 0.0)

        # Fan rotation around Z axis
        self.declare_parameter('fan_yaw_deg', 315.0)

        # ============================================================
        # AIRFLOW
        # ============================================================

        self.declare_parameter('airflow_enabled', True)
        self.declare_parameter('airflow_offset', 0.02)
        self.declare_parameter('airflow_length', 3.20)
        self.declare_parameter('airflow_width_scale', 2.5)
        self.declare_parameter('airflow_height_scale', 1.1)

        # Airflow color
        self.declare_parameter('airflow_r', 17.0)
        self.declare_parameter('airflow_g', 199.0)
        self.declare_parameter('airflow_b', 255.0)
        self.declare_parameter('airflow_a', 0.05)

        # ============================================================
        # MAIN SCENE COLOR
        # ============================================================

        self.declare_parameter('yellow_r', 1.0)
        self.declare_parameter('yellow_g', 0.85)
        self.declare_parameter('yellow_b', 0.0)
        self.declare_parameter('yellow_a', 1.0)

        # ============================================================
        # READ GENERAL PARAMETERS
        # ============================================================

        self.frame_id = (
            self.get_parameter('frame_id')
            .get_parameter_value()
            .string_value
        )

        self.topic = (
            self.get_parameter('topic')
            .get_parameter_value()
            .string_value
        )

        self.airflow_topic = (
            self.get_parameter('airflow_topic')
            .get_parameter_value()
            .string_value
        )

        self.publish_rate_hz = (
            self.get_parameter('publish_rate_hz')
            .get_parameter_value()
            .double_value
        )

        # ============================================================
        # READ FLOOR PARAMETERS
        # ============================================================

        self.floor_tile_size = (
            self.get_parameter('floor_tile_size')
            .get_parameter_value()
            .double_value
        )

        self.floor_tile_thickness = (
            self.get_parameter('floor_tile_thickness')
            .get_parameter_value()
            .double_value
        )

        self.floor_origin_x = (
            self.get_parameter('floor_origin_x')
            .get_parameter_value()
            .double_value
        )

        self.floor_origin_y = (
            self.get_parameter('floor_origin_y')
            .get_parameter_value()
            .double_value
        )

        self.floor_origin_z = (
            self.get_parameter('floor_origin_z')
            .get_parameter_value()
            .double_value
        )

        # ============================================================
        # READ WALL PARAMETERS
        # ============================================================

        self.wall_tile_size = (
            self.get_parameter('wall_tile_size')
            .get_parameter_value()
            .double_value
        )

        self.wall_tile_thickness = (
            self.get_parameter('wall_tile_thickness')
            .get_parameter_value()
            .double_value
        )

        self.wall_origin_x = (
            self.get_parameter('wall_origin_x')
            .get_parameter_value()
            .double_value
        )

        self.wall_origin_y = (
            self.get_parameter('wall_origin_y')
            .get_parameter_value()
            .double_value
        )

        self.wall_origin_z = (
            self.get_parameter('wall_origin_z')
            .get_parameter_value()
            .double_value
        )

        self.wall_yaw_deg = (
            self.get_parameter('wall_yaw_deg')
            .get_parameter_value()
            .double_value
        )

        # ============================================================
        # READ FAN SELECTION
        # ============================================================

        self.fan_selection = (
            self.get_parameter('fan_selection')
            .get_parameter_value()
            .integer_value
        )

        if self.fan_selection not in (0, 1, 2):
            self.get_logger().warning(
                f'Invalid fan_selection={self.fan_selection}. '
                f'Using 0 (no fan).'
            )
            self.fan_selection = 0

        # ============================================================
        # READ FAN 1 DIMENSIONS
        # ============================================================

        self.fan1_width = (
            self.get_parameter('fan1_width')
            .get_parameter_value()
            .double_value
        )

        self.fan1_depth = (
            self.get_parameter('fan1_depth')
            .get_parameter_value()
            .double_value
        )

        self.fan1_height = (
            self.get_parameter('fan1_height')
            .get_parameter_value()
            .double_value
        )

        # ============================================================
        # READ FAN 2 DIMENSIONS
        # ============================================================

        self.fan2_width = (
            self.get_parameter('fan2_width')
            .get_parameter_value()
            .double_value
        )

        self.fan2_depth = (
            self.get_parameter('fan2_depth')
            .get_parameter_value()
            .double_value
        )

        self.fan2_height = (
            self.get_parameter('fan2_height')
            .get_parameter_value()
            .double_value
        )

        # ============================================================
        # READ FAN POSITION / ORIENTATION
        # ============================================================

        self.fan_x = (
            self.get_parameter('fan_x')
            .get_parameter_value()
            .double_value
        )

        self.fan_y = (
            self.get_parameter('fan_y')
            .get_parameter_value()
            .double_value
        )

        self.fan_base_z = (
            self.get_parameter('fan_base_z')
            .get_parameter_value()
            .double_value
        )

        self.fan_yaw_deg = (
            self.get_parameter('fan_yaw_deg')
            .get_parameter_value()
            .double_value
        )

        # ============================================================
        # READ AIRFLOW PARAMETERS
        # ============================================================

        self.airflow_enabled = (
            self.get_parameter('airflow_enabled')
            .get_parameter_value()
            .bool_value
        )

        self.airflow_offset = (
            self.get_parameter('airflow_offset')
            .get_parameter_value()
            .double_value
        )

        self.airflow_length = (
            self.get_parameter('airflow_length')
            .get_parameter_value()
            .double_value
        )

        self.airflow_width_scale = (
            self.get_parameter('airflow_width_scale')
            .get_parameter_value()
            .double_value
        )

        self.airflow_height_scale = (
            self.get_parameter('airflow_height_scale')
            .get_parameter_value()
            .double_value
        )

        self.airflow_r = (
            self.get_parameter('airflow_r')
            .get_parameter_value()
            .double_value
        )

        self.airflow_g = (
            self.get_parameter('airflow_g')
            .get_parameter_value()
            .double_value
        )

        self.airflow_b = (
            self.get_parameter('airflow_b')
            .get_parameter_value()
            .double_value
        )

        self.airflow_a = (
            self.get_parameter('airflow_a')
            .get_parameter_value()
            .double_value
        )

        # ============================================================
        # READ COLOR PARAMETERS
        # ============================================================

        self.yellow_r = (
            self.get_parameter('yellow_r')
            .get_parameter_value()
            .double_value
        )

        self.yellow_g = (
            self.get_parameter('yellow_g')
            .get_parameter_value()
            .double_value
        )

        self.yellow_b = (
            self.get_parameter('yellow_b')
            .get_parameter_value()
            .double_value
        )

        self.yellow_a = (
            self.get_parameter('yellow_a')
            .get_parameter_value()
            .double_value
        )

        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 1.0

        # ============================================================
        # ROS PUBLISHERS
        # ============================================================

        self.pub = self.create_publisher(
            MarkerArray,
            self.topic,
            10
        )

        self.airflow_pub = self.create_publisher(
            MarkerArray,
            self.airflow_topic,
            10
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.timer_callback
        )

        # ============================================================
        # BUILD STATIC TILES
        # ============================================================

        self.word_tiles = self.build_iowa_tiles()
        self.border_tiles = self.build_checker_border(self.word_tiles)
        self.cas_tiles = self.build_cas_tiles()

        # ============================================================
        # LOGGING
        # ============================================================

        self.get_logger().info('---------------------------------------------')
        self.get_logger().info('lab_floor_iowa started')
        self.get_logger().info(f'topic            : {self.topic}')
        self.get_logger().info(f'airflow_topic    : {self.airflow_topic}')
        self.get_logger().info(f'frame_id         : {self.frame_id}')
        self.get_logger().info(f'fan_selection    : {self.fan_selection}')

        if self.fan_selection == 0:
            self.get_logger().info('selected fan     : NONE')

        elif self.fan_selection == 1:
            self.get_logger().info('selected fan     : FAN 1 - original')
            self.get_logger().info(
                f'dimensions       : '
                f'W={self.fan1_width}, '
                f'D={self.fan1_depth}, '
                f'H={self.fan1_height}'
            )

        elif self.fan_selection == 2:
            self.get_logger().info('selected fan     : FAN 2 - large')
            self.get_logger().info(
                f'dimensions       : '
                f'W={self.fan2_width}, '
                f'D={self.fan2_depth}, '
                f'H={self.fan2_height}'
            )

        self.get_logger().info(
            f'fan position     : '
            f'x={self.fan_x}, '
            f'y={self.fan_y}, '
            f'base_z={self.fan_base_z}'
        )

        self.get_logger().info(
            f'fan yaw          : {self.fan_yaw_deg} deg'
        )

        self.get_logger().info('---------------------------------------------')

    # ================================================================
    # GET SELECTED FAN DIMENSIONS
    # ================================================================

    def get_selected_fan_dimensions(self):
        if self.fan_selection == 1:
            return (
                self.fan1_width,
                self.fan1_depth,
                self.fan1_height
            )

        if self.fan_selection == 2:
            return (
                self.fan2_width,
                self.fan2_depth,
                self.fan2_height
            )

        return None

    # ================================================================
    # RECTANGLE OUTLINE
    # ================================================================

    def rect_outline_tiles(
        self,
        col0: int,
        row0: int,
        width: int,
        height: int
    ):
        tiles = set()

        for c in range(col0, col0 + width):
            tiles.add((c, row0))
            tiles.add((c, row0 + height - 1))

        for r in range(row0, row0 + height):
            tiles.add((col0, r))
            tiles.add((col0 + width - 1, r))

        return tiles

    # ================================================================
    # IOWA ON FLOOR
    # ================================================================

    def build_letter_i(self, col0: int, row0: int):
        tiles = set()

        width = 5
        height = 7

        for c in range(col0, col0 + width):
            tiles.add((c, row0))
            tiles.add((c, row0 + height - 1))

        center_col = col0 + width // 2

        for r in range(row0, row0 + height):
            tiles.add((center_col, r))

        return tiles

    def build_letter_o(self, col0: int, row0: int):
        return self.rect_outline_tiles(
            col0,
            row0,
            6,
            7
        )

    def build_letter_w(self, col0: int, row0: int):
        tiles = set()

        c_left = col0
        c_mid = col0 + 3
        c_right = col0 + 6

        for r in range(row0, row0 + 7):
            tiles.add((c_left, r))
            tiles.add((c_mid, r))
            tiles.add((c_right, r))

        for c in range(col0, col0 + 7):
            tiles.add((c, row0))

        return tiles

    def build_letter_a(self, col0: int, row0: int):
        tiles = set()

        width = 6
        height = 7

        left = col0
        right = col0 + width - 1
        top = row0 + height - 1
        mid = row0 + 3

        for r in range(row0, row0 + height):
            tiles.add((left, r))
            tiles.add((right, r))

        for c in range(col0 + 1, col0 + width - 1):
            tiles.add((c, top))

        for c in range(col0 + 1, col0 + width - 1):
            tiles.add((c, mid))

        return tiles

    def build_iowa_tiles(self):
        all_tiles = set()

        row0 = 0
        col = 0

        i_tiles = self.build_letter_i(col, row0)
        all_tiles.update(i_tiles)
        col += 5 + 1

        o_tiles = self.build_letter_o(col, row0)
        all_tiles.update(o_tiles)
        col += 6 + 1

        w_tiles = self.build_letter_w(col, row0)
        all_tiles.update(w_tiles)
        col += 7 + 1

        a_tiles = self.build_letter_a(col, row0)
        all_tiles.update(a_tiles)

        return sorted(all_tiles)

    def build_checker_border(self, word_tiles):
        cols = [t[0] for t in word_tiles]
        rows = [t[1] for t in word_tiles]

        min_col = min(cols)
        max_col = max(cols)
        min_row = min(rows)
        max_row = max(rows)

        gap = 1
        border_thickness = 2

        inner_min_col = min_col - gap
        inner_max_col = max_col + gap
        inner_min_row = min_row - gap
        inner_max_row = max_row + gap

        outer_min_col = inner_min_col - border_thickness
        outer_max_col = inner_max_col + border_thickness
        outer_min_row = inner_min_row - border_thickness
        outer_max_row = inner_max_row + border_thickness

        border_tiles = set()

        for c in range(
            outer_min_col,
            outer_max_col + 1
        ):
            for r in range(
                outer_min_row,
                outer_max_row + 1
            ):

                in_inner_gap_zone = (
                    inner_min_col <= c <= inner_max_col and
                    inner_min_row <= r <= inner_max_row
                )

                if not in_inner_gap_zone:
                    if (c + r) % 2 == 0:
                        border_tiles.add((c, r))

        border_tiles.difference_update(word_tiles)

        return sorted(border_tiles)

    # ================================================================
    # CAS ON WALL
    # ================================================================

    def build_letter_c_wall(self, col0: int, row0: int):
        tiles = set()

        width = 3
        height = 5

        for r in range(row0, row0 + height):
            tiles.add((col0, r))

        for c in range(col0, col0 + width):
            tiles.add((c, row0 + height - 1))
            tiles.add((c, row0))

        return tiles

    def build_letter_a_wall(self, col0: int, row0: int):
        tiles = set()

        left = col0
        right = col0 + 2
        top = row0 + 4
        mid = row0 + 2

        for r in range(row0, row0 + 5):
            tiles.add((left, r))
            tiles.add((right, r))

        for c in range(col0, col0 + 3):
            tiles.add((c, top))
            tiles.add((c, mid))

        return tiles

    def build_letter_s_wall(self, col0: int, row0: int):
        tiles = set()

        width = 3

        for c in range(col0, col0 + width):
            tiles.add((c, row0 + 4))
            tiles.add((c, row0 + 2))
            tiles.add((c, row0))

        tiles.add((col0, row0 + 3))
        tiles.add((col0 + 2, row0 + 1))

        return tiles

    def build_cas_tiles(self):
        all_tiles = set()

        row0 = 0
        col = 0

        c_tiles = self.build_letter_c_wall(col, row0)
        all_tiles.update(c_tiles)
        col += 3 + 1

        a_tiles = self.build_letter_a_wall(col, row0)
        all_tiles.update(a_tiles)
        col += 3 + 1

        s_tiles = self.build_letter_s_wall(col, row0)
        all_tiles.update(s_tiles)

        return sorted(all_tiles)

    # ================================================================
    # POINT HELPER
    # ================================================================

    def make_point(
        self,
        x: float,
        y: float,
        z: float
    ) -> Point:

        p = Point()

        p.x = x
        p.y = y
        p.z = z

        return p

    # ================================================================
    # FLOOR TILE MARKER
    # ================================================================

    def make_floor_tile_marker(
        self,
        marker_id: int,
        col: int,
        row: int
    ) -> Marker:

        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'iowa_floor'
        m.id = marker_id

        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = (
            self.floor_origin_x
            + row * self.floor_tile_size
            + 0.5 * self.floor_tile_size
        )

        m.pose.position.y = (
            self.floor_origin_y
            - col * self.floor_tile_size
            - 0.5 * self.floor_tile_size
        )

        m.pose.position.z = (
            self.floor_origin_z
            + 0.5 * self.floor_tile_thickness
        )

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = self.floor_tile_size
        m.scale.y = self.floor_tile_size
        m.scale.z = self.floor_tile_thickness

        m.color.r = self.yellow_r
        m.color.g = self.yellow_g
        m.color.b = self.yellow_b
        m.color.a = self.yellow_a

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    # ================================================================
    # WALL TILE MARKER
    # ================================================================

    def make_wall_tile_marker(
        self,
        marker_id: int,
        col: int,
        row: int
    ) -> Marker:

        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'cas_wall'
        m.id = marker_id

        m.type = Marker.CUBE
        m.action = Marker.ADD

        local_x = (
            col * self.wall_tile_size
            + 0.5 * self.wall_tile_size
        )

        local_y = -0.5 * self.wall_tile_thickness

        local_z = (
            row * self.wall_tile_size
            + 0.5 * self.wall_tile_size
        )

        yaw = math.radians(self.wall_yaw_deg)

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        x_rot = cy * local_x - sy * local_y
        y_rot = sy * local_x + cy * local_y
        z_rot = local_z

        m.pose.position.x = (
            self.wall_origin_x + x_rot
        )

        m.pose.position.y = (
            self.wall_origin_y + y_rot
        )

        m.pose.position.z = (
            self.wall_origin_z + z_rot
        )

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)

        m.scale.x = self.wall_tile_size
        m.scale.y = self.wall_tile_thickness
        m.scale.z = self.wall_tile_size

        m.color.r = self.yellow_r
        m.color.g = self.yellow_g
        m.color.b = self.yellow_b
        m.color.a = self.yellow_a

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    # ================================================================
    # FAN MARKER
    # ================================================================

    def make_fan_marker(
        self,
        marker_id: int,
        fan_width: float,
        fan_depth: float,
        fan_height: float
    ) -> Marker:

        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'fan'
        m.id = marker_id

        m.type = Marker.CUBE
        m.action = Marker.ADD

        # ------------------------------------------------------------
        # Same X/Y position for both fans.
        #
        # Z is automatically calculated from fan height so the
        # bottom remains at fan_base_z.
        # ------------------------------------------------------------

        fan_center_z = (
            self.fan_base_z
            + 0.5 * fan_height
        )

        m.pose.position.x = self.fan_x
        m.pose.position.y = self.fan_y
        m.pose.position.z = fan_center_z

        yaw = math.radians(self.fan_yaw_deg)

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)

        m.scale.x = fan_width
        m.scale.y = fan_depth
        m.scale.z = fan_height

        m.color.r = self.yellow_r
        m.color.g = self.yellow_g
        m.color.b = self.yellow_b
        m.color.a = self.yellow_a

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    # ================================================================
    # AIRFLOW CONE
    # ================================================================

    def make_airflow_cone_marker(
        self,
        marker_id: int,
        fan_width: float,
        fan_depth: float,
        fan_height: float
    ) -> Marker:

        m = Marker()

        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()

        m.ns = 'fan_airflow_cone'
        m.id = marker_id

        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD

        yaw = math.radians(self.fan_yaw_deg)

        # ------------------------------------------------------------
        # Local fan axes in world coordinates
        #
        # local x = width direction
        # local y = airflow direction
        # ------------------------------------------------------------

        ux_x = math.cos(yaw)
        ux_y = math.sin(yaw)

        uy_x = -math.sin(yaw)
        uy_y = math.cos(yaw)

        # ------------------------------------------------------------
        # Fan vertical center
        # ------------------------------------------------------------

        fan_center_z = (
            self.fan_base_z
            + 0.5 * fan_height
        )

        # ------------------------------------------------------------
        # Near face center
        # ------------------------------------------------------------

        near_cx = (
            self.fan_x
            + (
                0.5 * fan_depth
                + self.airflow_offset
            ) * uy_x
        )

        near_cy = (
            self.fan_y
            + (
                0.5 * fan_depth
                + self.airflow_offset
            ) * uy_y
        )

        near_cz = fan_center_z

        # ------------------------------------------------------------
        # Far face center
        # ------------------------------------------------------------

        far_cx = (
            near_cx
            + self.airflow_length * uy_x
        )

        far_cy = (
            near_cy
            + self.airflow_length * uy_y
        )

        far_cz = near_cz

        # ------------------------------------------------------------
        # Near and far dimensions
        # ------------------------------------------------------------

        near_half_w = 0.5 * fan_width
        near_half_h = 0.5 * fan_height

        far_half_w = (
            near_half_w
            * self.airflow_width_scale
        )

        far_half_h = (
            near_half_h
            * self.airflow_height_scale
        )

        # ------------------------------------------------------------
        # NEAR RECTANGLE
        # ------------------------------------------------------------

        nbl = self.make_point(
            near_cx - near_half_w * ux_x,
            near_cy - near_half_w * ux_y,
            near_cz - near_half_h
        )

        nbr = self.make_point(
            near_cx + near_half_w * ux_x,
            near_cy + near_half_w * ux_y,
            near_cz - near_half_h
        )

        ntl = self.make_point(
            near_cx - near_half_w * ux_x,
            near_cy - near_half_w * ux_y,
            near_cz + near_half_h
        )

        ntr = self.make_point(
            near_cx + near_half_w * ux_x,
            near_cy + near_half_w * ux_y,
            near_cz + near_half_h
        )

        # ------------------------------------------------------------
        # FAR RECTANGLE
        # ------------------------------------------------------------

        fbl = self.make_point(
            far_cx - far_half_w * ux_x,
            far_cy - far_half_w * ux_y,
            far_cz - far_half_h
        )

        fbr = self.make_point(
            far_cx + far_half_w * ux_x,
            far_cy + far_half_w * ux_y,
            far_cz - far_half_h
        )

        ftl = self.make_point(
            far_cx - far_half_w * ux_x,
            far_cy - far_half_w * ux_y,
            far_cz + far_half_h
        )

        ftr = self.make_point(
            far_cx + far_half_w * ux_x,
            far_cy + far_half_w * ux_y,
            far_cz + far_half_h
        )

        pts = []

        # Left side
        pts.extend([nbl, ntl, ftl])
        pts.extend([nbl, ftl, fbl])

        # Right side
        pts.extend([nbr, fbr, ftr])
        pts.extend([nbr, ftr, ntr])

        # Bottom side
        pts.extend([nbl, fbl, fbr])
        pts.extend([nbl, fbr, nbr])

        # Top side
        pts.extend([ntl, ntr, ftr])
        pts.extend([ntl, ftr, ftl])

        # Far face
        pts.extend([fbl, ftl, ftr])
        pts.extend([fbl, ftr, fbr])

        m.points = pts

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        m.color.r = self.airflow_r
        m.color.g = self.airflow_g
        m.color.b = self.airflow_b
        m.color.a = self.airflow_a

        m.lifetime.sec = 0
        m.frame_locked = False

        return m

    # ================================================================
    # TIMER CALLBACK
    # ================================================================

    def timer_callback(self) -> None:

        # ------------------------------------------------------------
        # MAIN SCENE
        # ------------------------------------------------------------

        arr = MarkerArray()

        delete_all = Marker()

        delete_all.header.frame_id = self.frame_id
        delete_all.header.stamp = (
            self.get_clock().now().to_msg()
        )

        delete_all.action = Marker.DELETEALL

        arr.markers.append(delete_all)

        # ------------------------------------------------------------
        # AIRFLOW
        # ------------------------------------------------------------

        air_arr = MarkerArray()

        air_delete_all = Marker()

        air_delete_all.header.frame_id = self.frame_id
        air_delete_all.header.stamp = (
            self.get_clock().now().to_msg()
        )

        air_delete_all.action = Marker.DELETEALL

        air_arr.markers.append(air_delete_all)

        marker_id = 0

        # ------------------------------------------------------------
        # IOWA WORD
        # ------------------------------------------------------------

        for col, row in self.word_tiles:

            arr.markers.append(
                self.make_floor_tile_marker(
                    marker_id,
                    col,
                    row
                )
            )

            marker_id += 1

        # ------------------------------------------------------------
        # CHECKER BORDER
        # ------------------------------------------------------------

        for col, row in self.border_tiles:

            arr.markers.append(
                self.make_floor_tile_marker(
                    marker_id,
                    col,
                    row
                )
            )

            marker_id += 1

        # ------------------------------------------------------------
        # CAS WALL
        # ------------------------------------------------------------

        for col, row in self.cas_tiles:

            arr.markers.append(
                self.make_wall_tile_marker(
                    marker_id,
                    col,
                    row
                )
            )

            marker_id += 1

        # ------------------------------------------------------------
        # SELECT FAN
        # ------------------------------------------------------------

        fan_dimensions = (
            self.get_selected_fan_dimensions()
        )

        # fan_selection == 0:
        # fan_dimensions is None
        # -> no fan and no airflow

        if fan_dimensions is not None:

            fan_width, fan_depth, fan_height = (
                fan_dimensions
            )

            # --------------------------------------------------------
            # FAN
            # --------------------------------------------------------

            arr.markers.append(
                self.make_fan_marker(
                    marker_id,
                    fan_width,
                    fan_depth,
                    fan_height
                )
            )

            marker_id += 1

            # --------------------------------------------------------
            # AIRFLOW
            # --------------------------------------------------------

            if self.airflow_enabled:

                air_arr.markers.append(
                    self.make_airflow_cone_marker(
                        0,
                        fan_width,
                        fan_depth,
                        fan_height
                    )
                )

        # ------------------------------------------------------------
        # PUBLISH
        # ------------------------------------------------------------

        self.pub.publish(arr)
        self.airflow_pub.publish(air_arr)


def main(args=None):

    rclpy.init(args=args)

    node = LabFloorIowa()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

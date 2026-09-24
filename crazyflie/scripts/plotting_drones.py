#!/usr/bin/env python3

import math
import xml.etree.ElementTree as ET
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from std_msgs.msg import Float64, String
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster


class DroneState:
    def __init__(self) -> None:
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.z: Optional[float] = None
        self.psi: Optional[float] = None

    def ready(self) -> bool:
        return None not in (self.x, self.y, self.z, self.psi)


class PlottingDrones(Node):
    def __init__(self) -> None:
        super().__init__('plotting_drones')

        # ============================================================
        # PARAMETERS
        # ============================================================

        self.declare_parameter('fixed_frame', 'world')

        # ------------------------------------------------------------
        # Original Crazyflie
        # ------------------------------------------------------------
        self.declare_parameter(
            'source_robot_description_topic',
            '/cf231/robot_description'
        )

        self.declare_parameter(
            'source_robot_name',
            'cf231'
        )

        self.declare_parameter(
            'description_replace_token',
            'cf231'
        )

        self.declare_parameter(
            'source_root_frame',
            'cf231'
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # OLD:
        #   desired    = p_ref
        #   augmented  = p_aug
        #
        # NEW:
        #   desired    = p_cmd
        #   augmented  = p_ref
        #
        # These names are used inside the cloned URDF and TF tree.
        # ------------------------------------------------------------

        self.declare_parameter(
            'desired_robot_name',
            'p_cmd'
        )

        self.declare_parameter(
            'augmented_robot_name',
            'p_ref'
        )

        self.declare_parameter(
            'desired_root_frame',
            ''
        )

        self.declare_parameter(
            'augmented_root_frame',
            ''
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # Keep the OLD robot_description topics.
        #
        # This allows your existing RViz configuration and existing
        # robot_state_publisher setup to continue listening to exactly
        # the same topics as before.
        #
        # /p_ref/robot_description contains the NEW p_cmd URDF
        # /p_aug/robot_description contains the NEW p_ref URDF
        # ------------------------------------------------------------

        self.declare_parameter(
            'desired_robot_description_topic',
            '/p_ref/robot_description'
        )

        self.declare_parameter(
            'augmented_robot_description_topic',
            '/p_aug/robot_description'
        )

        # ------------------------------------------------------------
        # Commanded/original reference topics
        #
        # These used to drive p_ref.
        # They now drive p_cmd.
        # ------------------------------------------------------------

        self.declare_parameter(
            'desired_x_topic',
            '/xend'
        )

        self.declare_parameter(
            'desired_y_topic',
            '/yend'
        )

        self.declare_parameter(
            'desired_z_topic',
            '/zend'
        )

        self.declare_parameter(
            'desired_psi_topic',
            '/psiend'
        )

        # ------------------------------------------------------------
        # Augmented/L1 reference topics
        #
        # These used to drive p_aug.
        # They now drive p_ref.
        # ------------------------------------------------------------

        self.declare_parameter(
            'augmented_x_topic',
            '/x_augment'
        )

        self.declare_parameter(
            'augmented_y_topic',
            '/y_augment'
        )

        self.declare_parameter(
            'augmented_z_topic',
            '/z_augment'
        )

        self.declare_parameter(
            'augmented_psi_topic',
            '/psi_augment'
        )

        # ------------------------------------------------------------
        # Pose output topics
        #
        # Here I use the NEW logical names:
        #
        # /p_cmd/pose
        # /p_ref/pose
        #
        # These pose topics are independent of RobotModel loading.
        # ------------------------------------------------------------

        self.declare_parameter(
            'desired_pose_topic',
            '/p_cmd/pose'
        )

        self.declare_parameter(
            'augmented_pose_topic',
            '/p_ref/pose'
        )

        self.declare_parameter(
            'publish_rate_hz',
            50.0
        )

        self.declare_parameter(
            'publish_pose_topics',
            True
        )

        self.declare_parameter(
            'yaw_in_degrees',
            False
        )

        self.declare_parameter(
            'republish_robot_description_period_s',
            1.0
        )

        # ============================================================
        # READ PARAMETERS
        # ============================================================

        self.fixed_frame = (
            self.get_parameter('fixed_frame')
            .get_parameter_value()
            .string_value
        )

        self.source_description_topic = (
            self.get_parameter('source_robot_description_topic')
            .get_parameter_value()
            .string_value
        )

        self.source_robot_name = (
            self.get_parameter('source_robot_name')
            .get_parameter_value()
            .string_value
        )

        self.description_replace_token = (
            self.get_parameter('description_replace_token')
            .get_parameter_value()
            .string_value
        )

        self.source_root_frame = (
            self.get_parameter('source_root_frame')
            .get_parameter_value()
            .string_value
        )

        self.desired_robot_name = (
            self.get_parameter('desired_robot_name')
            .get_parameter_value()
            .string_value
        )

        self.augmented_robot_name = (
            self.get_parameter('augmented_robot_name')
            .get_parameter_value()
            .string_value
        )

        desired_root_frame_param = (
            self.get_parameter('desired_root_frame')
            .get_parameter_value()
            .string_value
        )

        augmented_root_frame_param = (
            self.get_parameter('augmented_root_frame')
            .get_parameter_value()
            .string_value
        )

        # ============================================================
        # ROOT TF FRAMES
        #
        # source_root_frame = cf231
        #
        # desired:
        #   cf231 -> p_cmd
        #
        # augmented:
        #   cf231 -> p_ref
        # ============================================================

        self.desired_root_frame = (
            desired_root_frame_param
            if desired_root_frame_param
            else self._replace_token(
                self.source_root_frame,
                self.desired_robot_name
            )
        )

        self.augmented_root_frame = (
            augmented_root_frame_param
            if augmented_root_frame_param
            else self._replace_token(
                self.source_root_frame,
                self.augmented_robot_name
            )
        )

        # ============================================================
        # ROBOT DESCRIPTION OUTPUT TOPICS
        # ============================================================

        self.desired_description_topic = (
            self.get_parameter('desired_robot_description_topic')
            .get_parameter_value()
            .string_value
        )

        self.augmented_description_topic = (
            self.get_parameter('augmented_robot_description_topic')
            .get_parameter_value()
            .string_value
        )

        # ============================================================
        # INPUT TOPICS
        # ============================================================

        self.desired_x_topic = (
            self.get_parameter('desired_x_topic')
            .get_parameter_value()
            .string_value
        )

        self.desired_y_topic = (
            self.get_parameter('desired_y_topic')
            .get_parameter_value()
            .string_value
        )

        self.desired_z_topic = (
            self.get_parameter('desired_z_topic')
            .get_parameter_value()
            .string_value
        )

        self.desired_psi_topic = (
            self.get_parameter('desired_psi_topic')
            .get_parameter_value()
            .string_value
        )

        self.augmented_x_topic = (
            self.get_parameter('augmented_x_topic')
            .get_parameter_value()
            .string_value
        )

        self.augmented_y_topic = (
            self.get_parameter('augmented_y_topic')
            .get_parameter_value()
            .string_value
        )

        self.augmented_z_topic = (
            self.get_parameter('augmented_z_topic')
            .get_parameter_value()
            .string_value
        )

        self.augmented_psi_topic = (
            self.get_parameter('augmented_psi_topic')
            .get_parameter_value()
            .string_value
        )

        # ============================================================
        # POSE OUTPUT TOPICS
        # ============================================================

        self.desired_pose_topic = (
            self.get_parameter('desired_pose_topic')
            .get_parameter_value()
            .string_value
        )

        self.augmented_pose_topic = (
            self.get_parameter('augmented_pose_topic')
            .get_parameter_value()
            .string_value
        )

        publish_rate_hz = (
            self.get_parameter('publish_rate_hz')
            .get_parameter_value()
            .double_value
        )

        self.publish_pose_topics = (
            self.get_parameter('publish_pose_topics')
            .get_parameter_value()
            .bool_value
        )

        self.yaw_in_degrees = (
            self.get_parameter('yaw_in_degrees')
            .get_parameter_value()
            .bool_value
        )

        republish_period_s = (
            self.get_parameter(
                'republish_robot_description_period_s'
            )
            .get_parameter_value()
            .double_value
        )

        if publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be > 0')

        if republish_period_s <= 0.0:
            raise ValueError(
                'republish_robot_description_period_s must be > 0'
            )

        # ============================================================
        # COLORS
        #
        # p_cmd = blue
        # p_ref = red
        # ============================================================

        self.desired_rgba = '0.0 0.0 1.0 1.0'
        self.augmented_rgba = '1.0 0.0 0.0 1.0'

        # ============================================================
        # STATE
        # ============================================================

        self.desired_state = DroneState()
        self.augmented_state = DroneState()

        self.source_robot_description: Optional[str] = None

        self.desired_robot_description: Optional[str] = None
        self.augmented_robot_description: Optional[str] = None

        # ============================================================
        # QoS
        # ============================================================

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        default_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ============================================================
        # ROBOT DESCRIPTION PUBLISHERS
        #
        # IMPORTANT:
        #
        # p_cmd URDF -> /p_ref/robot_description
        #
        # p_ref URDF -> /p_aug/robot_description
        #
        # This intentionally preserves the OLD topic names.
        # ============================================================

        self.desired_description_pub = self.create_publisher(
            String,
            self.desired_description_topic,
            latched_qos
        )

        self.augmented_description_pub = self.create_publisher(
            String,
            self.augmented_description_topic,
            latched_qos
        )

        # ============================================================
        # POSE PUBLISHERS
        # ============================================================

        self.desired_pose_pub = None
        self.augmented_pose_pub = None

        if self.publish_pose_topics:

            self.desired_pose_pub = self.create_publisher(
                PoseStamped,
                self.desired_pose_topic,
                default_qos
            )

            self.augmented_pose_pub = self.create_publisher(
                PoseStamped,
                self.augmented_pose_topic,
                default_qos
            )

        # ============================================================
        # SOURCE ROBOT DESCRIPTION SUBSCRIBER
        # ============================================================

        self.create_subscription(
            String,
            self.source_description_topic,
            self._robot_description_cb,
            latched_qos
        )

        # ============================================================
        # COMMAND / p_cmd SUBSCRIBERS
        # ============================================================

        self.create_subscription(
            Float64,
            self.desired_x_topic,
            self._desired_x_cb,
            default_qos
        )

        self.create_subscription(
            Float64,
            self.desired_y_topic,
            self._desired_y_cb,
            default_qos
        )

        self.create_subscription(
            Float64,
            self.desired_z_topic,
            self._desired_z_cb,
            default_qos
        )

        self.create_subscription(
            Float64,
            self.desired_psi_topic,
            self._desired_psi_cb,
            default_qos
        )

        # ============================================================
        # AUGMENTED / p_ref SUBSCRIBERS
        # ============================================================

        self.create_subscription(
            Float64,
            self.augmented_x_topic,
            self._augmented_x_cb,
            default_qos
        )

        self.create_subscription(
            Float64,
            self.augmented_y_topic,
            self._augmented_y_cb,
            default_qos
        )

        self.create_subscription(
            Float64,
            self.augmented_z_topic,
            self._augmented_z_cb,
            default_qos
        )

        self.create_subscription(
            Float64,
            self.augmented_psi_topic,
            self._augmented_psi_cb,
            default_qos
        )

        # ============================================================
        # TF BROADCASTER
        # ============================================================

        self.tf_broadcaster = TransformBroadcaster(self)

        # ============================================================
        # TIMERS
        # ============================================================

        self.create_timer(
            1.0 / publish_rate_hz,
            self._publish_cycle
        )

        self.create_timer(
            republish_period_s,
            self._republish_robot_descriptions
        )

        # ============================================================
        # STARTUP INFORMATION
        # ============================================================

        self.get_logger().info(
            'plotting_drones node started.'
        )

        self.get_logger().info(
            f'Fixed frame: {self.fixed_frame}'
        )

        self.get_logger().info(
            f'Source robot_description topic: '
            f'{self.source_description_topic}'
        )

        self.get_logger().info(
            f'Source root frame: {self.source_root_frame}'
        )

        self.get_logger().info(
            f'Command drone name: {self.desired_robot_name}'
        )

        self.get_logger().info(
            f'Reference drone name: {self.augmented_robot_name}'
        )

        self.get_logger().info(
            f'Command root frame: {self.desired_root_frame}'
        )

        self.get_logger().info(
            f'Reference root frame: {self.augmented_root_frame}'
        )

        self.get_logger().info(
            f'Command URDF published on: '
            f'{self.desired_description_topic}'
        )

        self.get_logger().info(
            f'Reference URDF published on: '
            f'{self.augmented_description_topic}'
        )

    # ================================================================
    # STRING REPLACEMENT
    # ================================================================

    def _replace_token(
        self,
        text: str,
        new_robot_name: str
    ) -> str:

        return text.replace(
            self.description_replace_token,
            new_robot_name
        )

    # ================================================================
    # URDF RECOLORING
    # ================================================================

    def _recolor_urdf(
        self,
        urdf_xml: str,
        rgba: str,
        material_name: str
    ) -> str:

        try:
            root = ET.fromstring(urdf_xml)

        except ET.ParseError as e:

            self.get_logger().error(
                f'Failed to parse URDF for recoloring: {e}'
            )

            return urdf_xml

        # ------------------------------------------------------------
        # Top-level material definitions
        # ------------------------------------------------------------

        for material in root.findall('material'):

            if (
                'name' not in material.attrib
                or not material.attrib['name']
            ):
                material.set(
                    'name',
                    material_name
                )

            color = material.find('color')

            if color is None:
                color = ET.SubElement(
                    material,
                    'color'
                )

            color.set(
                'rgba',
                rgba
            )

        # ------------------------------------------------------------
        # Materials inside individual visual elements
        # ------------------------------------------------------------

        for visual in root.iter('visual'):

            material = visual.find('material')

            if material is None:

                material = ET.SubElement(
                    visual,
                    'material'
                )

                material.set(
                    'name',
                    material_name
                )

            else:

                if (
                    'name' not in material.attrib
                    or not material.attrib['name']
                ):
                    material.set(
                        'name',
                        material_name
                    )

            color = material.find('color')

            if color is None:

                color = ET.SubElement(
                    material,
                    'color'
                )

            color.set(
                'rgba',
                rgba
            )

        return ET.tostring(
            root,
            encoding='unicode'
        )

    # ================================================================
    # ORIGINAL URDF CALLBACK
    # ================================================================

    def _robot_description_cb(
        self,
        msg: String
    ) -> None:

        self.source_robot_description = msg.data

        # ------------------------------------------------------------
        # ORIGINAL:
        #
        # cf231 -> p_ref
        # cf231 -> p_aug
        #
        # NEW:
        #
        # cf231 -> p_cmd
        # cf231 -> p_ref
        # ------------------------------------------------------------

        desired_xml = self._replace_token(
            msg.data,
            self.desired_robot_name
        )

        augmented_xml = self._replace_token(
            msg.data,
            self.augmented_robot_name
        )

        # ------------------------------------------------------------
        # p_cmd = blue
        # ------------------------------------------------------------

        self.desired_robot_description = self._recolor_urdf(
            desired_xml,
            self.desired_rgba,
            f'{self.desired_robot_name}_material'
        )

        # ------------------------------------------------------------
        # p_ref = red
        # ------------------------------------------------------------

        self.augmented_robot_description = self._recolor_urdf(
            augmented_xml,
            self.augmented_rgba,
            f'{self.augmented_robot_name}_material'
        )

        if self.description_replace_token not in msg.data:

            self.get_logger().warning(
                f"description_replace_token="
                f"'{self.description_replace_token}' "
                f"was not found in the incoming robot_description."
            )

        self._publish_robot_descriptions_once()

    # ================================================================
    # ROBOT DESCRIPTION PUBLISHING
    # ================================================================

    def _publish_robot_descriptions_once(
        self
    ) -> None:

        if self.desired_robot_description is not None:

            msg = String()
            msg.data = self.desired_robot_description

            self.desired_description_pub.publish(
                msg
            )

        if self.augmented_robot_description is not None:

            msg = String()
            msg.data = self.augmented_robot_description

            self.augmented_description_pub.publish(
                msg
            )

    def _republish_robot_descriptions(
        self
    ) -> None:

        self._publish_robot_descriptions_once()

    # ================================================================
    # p_cmd CALLBACKS
    #
    # /xend
    # /yend
    # /zend
    # /psiend
    # ================================================================

    def _desired_x_cb(
        self,
        msg: Float64
    ) -> None:

        self.desired_state.x = msg.data

    def _desired_y_cb(
        self,
        msg: Float64
    ) -> None:

        self.desired_state.y = msg.data

    def _desired_z_cb(
        self,
        msg: Float64
    ) -> None:

        self.desired_state.z = msg.data

    def _desired_psi_cb(
        self,
        msg: Float64
    ) -> None:

        self.desired_state.psi = (
            math.radians(msg.data)
            if self.yaw_in_degrees
            else msg.data
        )

    # ================================================================
    # p_ref CALLBACKS
    #
    # /x_augment
    # /y_augment
    # /z_augment
    # /psi_augment
    # ================================================================

    def _augmented_x_cb(
        self,
        msg: Float64
    ) -> None:

        self.augmented_state.x = msg.data

    def _augmented_y_cb(
        self,
        msg: Float64
    ) -> None:

        self.augmented_state.y = msg.data

    def _augmented_z_cb(
        self,
        msg: Float64
    ) -> None:

        self.augmented_state.z = msg.data

    def _augmented_psi_cb(
        self,
        msg: Float64
    ) -> None:

        self.augmented_state.psi = (
            math.radians(msg.data)
            if self.yaw_in_degrees
            else msg.data
        )

    # ================================================================
    # YAW -> QUATERNION
    # ================================================================

    def _yaw_to_quaternion(
        self,
        yaw: float
    ):

        half = 0.5 * yaw

        qx = 0.0
        qy = 0.0
        qz = math.sin(half)
        qw = math.cos(half)

        return qx, qy, qz, qw

    # ================================================================
    # TF PUBLISHING
    # ================================================================

    def _publish_transform(
        self,
        child_frame: str,
        state: DroneState
    ) -> None:

        qx, qy, qz, qw = (
            self._yaw_to_quaternion(
                state.psi
            )
        )

        t = TransformStamped()

        t.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        t.header.frame_id = self.fixed_frame

        t.child_frame_id = child_frame

        t.transform.translation.x = float(
            state.x
        )

        t.transform.translation.y = float(
            state.y
        )

        t.transform.translation.z = float(
            state.z
        )

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(
            t
        )

    # ================================================================
    # POSE PUBLISHING
    # ================================================================

    def _publish_pose(
        self,
        pub,
        state: DroneState
    ) -> None:

        if pub is None:
            return

        qx, qy, qz, qw = (
            self._yaw_to_quaternion(
                state.psi
            )
        )

        msg = PoseStamped()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = self.fixed_frame

        msg.pose.position.x = float(
            state.x
        )

        msg.pose.position.y = float(
            state.y
        )

        msg.pose.position.z = float(
            state.z
        )

        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        pub.publish(
            msg
        )

    # ================================================================
    # MAIN PUBLISHING CYCLE
    # ================================================================

    def _publish_cycle(
        self
    ) -> None:

        # ------------------------------------------------------------
        # BLUE drone
        #
        # OLD p_ref
        # NEW p_cmd
        #
        # Driven by:
        # /xend /yend /zend /psiend
        # ------------------------------------------------------------

        if self.desired_state.ready():

            self._publish_transform(
                self.desired_root_frame,
                self.desired_state
            )

            self._publish_pose(
                self.desired_pose_pub,
                self.desired_state
            )

        # ------------------------------------------------------------
        # RED drone
        #
        # OLD p_aug
        # NEW p_ref
        #
        # Driven by:
        # /x_augment /y_augment /z_augment /psi_augment
        # ------------------------------------------------------------

        #if self.augmented_state.ready():

         #   self._publish_transform(
          #      self.augmented_root_frame,
           #     self.augmented_state
            #)

           # self._publish_pose(
            #    self.augmented_pose_pub,
             #   self.augmented_state
            #)


def main(args=None) -> None:

    rclpy.init(args=args)

    node = PlottingDrones()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

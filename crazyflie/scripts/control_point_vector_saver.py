#!/usr/bin/env python3

import csv
import os
from collections import deque
from typing import Deque, Dict, List, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray


class ControlPointVectorSaver(Node):
    def __init__(self) -> None:
        super().__init__('control_point_vector_saver')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('x_topic', '/x_vector')
        self.declare_parameter('y_topic', '/y_vector')
        self.declare_parameter('z_topic', '/z_vector')
        self.declare_parameter('psi_topic', '/psi_vector')

        self.declare_parameter('output_csv', 'control_point_vectors.csv')
        self.declare_parameter('flush_every_sample', True)
        self.declare_parameter('log_every_n_samples', 10)

        self.x_topic = self.get_parameter('x_topic').get_parameter_value().string_value
        self.y_topic = self.get_parameter('y_topic').get_parameter_value().string_value
        self.z_topic = self.get_parameter('z_topic').get_parameter_value().string_value
        self.psi_topic = self.get_parameter('psi_topic').get_parameter_value().string_value

        self.output_csv = self.get_parameter('output_csv').get_parameter_value().string_value
        self.flush_every_sample = self.get_parameter('flush_every_sample').get_parameter_value().bool_value
        self.log_every_n_samples = self.get_parameter('log_every_n_samples').get_parameter_value().integer_value

        # Make output path absolute relative to current working directory if needed
        self.output_csv = os.path.abspath(self.output_csv)
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        # ------------------------------------------------------------------
        # Internal storage
        # Each queue element is a dict:
        # {
        #   'values': [...],
        #   'receive_time_ns': int,
        #   'update_count': int
        # }
        # ------------------------------------------------------------------
        self.queues: Dict[str, Deque[Dict[str, Any]]] = {
            'x': deque(),
            'y': deque(),
            'z': deque(),
            'psi': deque(),
        }

        self.update_counts: Dict[str, int] = {
            'x': 0,
            'y': 0,
            'z': 0,
            'psi': 0,
        }

        self.sample_id = 0
        self.first_bundle_time_ns = None

        # ------------------------------------------------------------------
        # CSV setup
        # We use a row-per-control-point layout.
        # Later you can append Bernstein rows into the SAME csv
        # by using row_type='bernstein'.
        # ------------------------------------------------------------------
        self.csv_file = open(self.output_csv, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'row_type',              # control_point now; bernstein later
            'sample_id',

            'bundle_time_ns',
            'bundle_time_s',
            'elapsed_time_s',

            'x_msg_time_ns',
            'y_msg_time_ns',
            'z_msg_time_ns',
            'psi_msg_time_ns',

            'x_update_count',
            'y_update_count',
            'z_update_count',
            'psi_update_count',

            'vector_length',
            'x_length',
            'y_length',
            'z_length',
            'psi_length',

            'point_index',

            'x_value',
            'y_value',
            'z_value',
            'psi_value'
        ])
        self.csv_file.flush()

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100
        )

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
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

        self.get_logger().info('---------------------------------------------')
        self.get_logger().info('control_point_vector_saver started')
        self.get_logger().info(f'x_topic      : {self.x_topic}')
        self.get_logger().info(f'y_topic      : {self.y_topic}')
        self.get_logger().info(f'z_topic      : {self.z_topic}')
        self.get_logger().info(f'psi_topic    : {self.psi_topic}')
        self.get_logger().info(f'output_csv   : {self.output_csv}')
        self.get_logger().info('---------------------------------------------')
        self.get_logger().info(
            'If you want timestamps tied to rosbag simulated time, run this node with use_sim_time:=true '
            'and play the bag with clock.'
        )

    def _make_callback(self, key: str):
        def callback(msg: Float64MultiArray) -> None:
            now_ns = self.get_clock().now().nanoseconds

            self.update_counts[key] += 1

            snapshot = {
                'values': list(msg.data),
                'receive_time_ns': now_ns,
                'update_count': self.update_counts[key],
            }

            self.queues[key].append(snapshot)

            self._try_write_samples()

        return callback

    def _all_queues_have_data(self) -> bool:
        return (
            len(self.queues['x']) > 0 and
            len(self.queues['y']) > 0 and
            len(self.queues['z']) > 0 and
            len(self.queues['psi']) > 0
        )

    def _try_write_samples(self) -> None:
        while self._all_queues_have_data():
            x_msg = self.queues['x'].popleft()
            y_msg = self.queues['y'].popleft()
            z_msg = self.queues['z'].popleft()
            psi_msg = self.queues['psi'].popleft()

            x_values = x_msg['values']
            y_values = y_msg['values']
            z_values = z_msg['values']
            psi_values = psi_msg['values']

            x_len = len(x_values)
            y_len = len(y_values)
            z_len = len(z_values)
            psi_len = len(psi_values)

            max_len = max(x_len, y_len, z_len, psi_len)

            if not (x_len == y_len == z_len == psi_len):
                self.get_logger().warn(
                    f'Length mismatch in sample {self.sample_id}: '
                    f'x={x_len}, y={y_len}, z={z_len}, psi={psi_len}. '
                    f'Saving with blanks where needed.'
                )

            bundle_time_ns = max(
                x_msg['receive_time_ns'],
                y_msg['receive_time_ns'],
                z_msg['receive_time_ns'],
                psi_msg['receive_time_ns']
            )

            if self.first_bundle_time_ns is None:
                self.first_bundle_time_ns = bundle_time_ns

            bundle_time_s = bundle_time_ns / 1e9
            elapsed_time_s = (bundle_time_ns - self.first_bundle_time_ns) / 1e9

            for point_index in range(max_len):
                x_value = x_values[point_index] if point_index < x_len else ''
                y_value = y_values[point_index] if point_index < y_len else ''
                z_value = z_values[point_index] if point_index < z_len else ''
                psi_value = psi_values[point_index] if point_index < psi_len else ''

                self.csv_writer.writerow([
                    'control_point',
                    self.sample_id,

                    bundle_time_ns,
                    f'{bundle_time_s:.9f}',
                    f'{elapsed_time_s:.9f}',

                    x_msg['receive_time_ns'],
                    y_msg['receive_time_ns'],
                    z_msg['receive_time_ns'],
                    psi_msg['receive_time_ns'],

                    x_msg['update_count'],
                    y_msg['update_count'],
                    z_msg['update_count'],
                    psi_msg['update_count'],

                    max_len,
                    x_len,
                    y_len,
                    z_len,
                    psi_len,

                    point_index,

                    x_value,
                    y_value,
                    z_value,
                    psi_value
                ])

            if self.flush_every_sample:
                self.csv_file.flush()

            self.sample_id += 1

            if self.log_every_n_samples > 0 and (self.sample_id % self.log_every_n_samples == 0):
                self.get_logger().info(
                    f'Saved {self.sample_id} samples to {self.output_csv}'
                )

    def close_file(self) -> None:
        if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f'CSV file closed: {self.output_csv}')

    def destroy_node(self):
        self.close_file()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlPointVectorSaver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

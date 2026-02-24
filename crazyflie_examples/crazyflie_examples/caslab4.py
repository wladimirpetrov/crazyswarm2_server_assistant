#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np


def main():
    #import rclpy
    #rclpy.init()
    #node = rclpy.create_node("startup_wait")
    #while node.get_clock().now().nanoseconds == 0:
    #     rclpy.spin_once(node, timeout_sec=0.1)
    #node.destroy_node()

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    
    # enable logging
    allcfs.setParam('usd.logging', 1)
    timeHelper.sleep(2.5)
    TRIALS = 1
    TIMESCALE = 1.0
    for i in range(TRIALS):
        
        allcfs.takeoff(targetHeight=1.0, duration=2.0)
        timeHelper.sleep(2.5)
        cf.goTo(np.array([1.0,1.0,0]),0,2.0)
        
        timeHelper.sleep(2.5)
        cf.goTo(np.array([0,1.0,0]),0,2.0)
        timeHelper.sleep(2.5)
        cf.goTo(np.array([0,0,0]),0,2.0)
        timeHelper.sleep(2.5)
        cf.goTo(np.array([1.0,0,0]),0,2.0)
        #for cf in allcfs.crazyflies:
        #    pos = np.array(cf.initialPosition) + np.array([0, 0, 1.0])
        #    cf.goTo(pos, 0, 2.0)
        timeHelper.sleep(2.5)

        allcfs.land(targetHeight=0.06, duration=2.0)
        timeHelper.sleep(3.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()

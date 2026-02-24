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
    trajs = []
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    
    # enable logging
    allcfs.setParam('usd.logging', 1)
    timeHelper.sleep(2.5)
    print("banana")
    TRIALS = 1
    TIMESCALE = 1.0
    for i in range(TRIALS):
        cf1 = swarm.allcfs.crazyflies[0]
        cf2 = swarm.allcfs.crazyflies[1]
        cf3 = swarm.allcfs.crazyflies[2]
        cf4 = swarm.allcfs.crazyflies[3]
        
        allcfs.takeoff(targetHeight=1.0, duration=2.0)
        timeHelper.sleep(2.0)
        cf1.goTo(np.array([-.4,.5,1]), 0, 2.0)
        cf2.goTo(np.array([.5861,.653,1]),0,2.0)
        cf3.goTo(np.array([-.8375,1,1]),0,2.0)
        cf4.goTo(np.array([0,0.7919,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf1.goTo(np.array([-.3,.625,1]),0,2.0)
        cf2.goTo(np.array([.4972,.7111,1]),0,2.0)
        cf3.goTo(np.array([-.675,1,1]),0,2.0)
        cf4.goTo(np.array([0,0.9358,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf1.goTo(np.array([-.2,.75,1]), 0, 2.0)
        cf2.goTo(np.array([.1831,.7111,1]),0,2.0)
        cf3.goTo(np.array([-.5125,1,1]),0,2.0)
        cf4.goTo(np.array([.0407,1,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf1.goTo(np.array([-.1,.875,1]),0,2.0)
        cf2.goTo(np.array([.0512,.7111,1]),0,2.0)
        cf3.goTo(np.array([-.35,1,1]),0,2.0)
        cf4.goTo(np.array([.497,1,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf1.goTo(np.array([0,1.0,1]),0,2.0)
        cf2.goTo(np.array([0,.7919,1]),0,2.0)
        cf3.goTo(np.array([-.1875,1,1]),0,2.0)
        cf4.goTo(np.array([.586,0.936,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf1.goTo(np.array([.1,1,1]), 0, 2.0)
        cf2.goTo(np.array([0,.9358,1]),0,2.0)
        cf3.goTo(np.array([-.025,1,1]),0,2.0)
        cf4.goTo(np.array([.586,0.792,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf1.goTo(np.array([.2,1,1]),0,2.0)
        cf2.goTo(np.array([.0407,1,1]),0,2.0)
        cf3.goTo(np.array([.1375,1,1]),0,2.0)
        cf4.goTo(np.array([0.4972,0.7111,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf1.goTo(np.array([.4,1,1]), 0, 2.0)
        cf2.goTo(np.array([.2,1,1]),0,2.0)
        cf3.goTo(np.array([.3,1,1]),0,2.0)
        cf4.goTo(np.array([.0512,0.7111,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf4.goTo(np.array([0,0.7919,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf4.goTo(np.array([0,.9358,1]),0,2.0)
        timeHelper.sleep(1.0)
        cf4.goTo(np.array([.1,1,1]),0,2.0)
        timeHelper.sleep(2.0)
        #for cf in allcfs.crazyflies:
        #    pos = np.array(cf.initialPosition) + np.array([0.0, 0.0, 1.0])
        #    cf.goTo(pos, 0, 2.0)

        #timeHelper.sleep(3.0)
        #allcfs.startTrajectory(0, timescale=TIMESCALE)
        #timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)
        #timeHelper.sleep(max([t.duration for t in trajs]) * TIMESCALE + 2.0)
        # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
        # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

        allcfs.land(targetHeight=0.06, duration=2.0)
        timeHelper.sleep(3.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()

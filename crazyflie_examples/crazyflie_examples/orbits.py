#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # Load trajectories
    traj1 = Trajectory()
    traj1.loadcsv(Path(__file__).parent / 'data/traj_orbit_1.csv')

    traj2 = Trajectory()
    traj2.loadcsv(Path(__file__).parent / 'data/traj_orbit_2.csv')

    # Enable onboard logging
    allcfs.setParam('usd.logging', 1)

    TIMESCALE = 1.0

    # Use only the first two enabled Crazyflies
    cfs = swarm.allcfs.crazyflies
    cf1, cf2 = cfs[0], cfs[1]

    print("Uploading trajectory 1...")
    cf1.uploadTrajectory(0, 0, traj1)

    print("Uploading trajectory 2...")
    cf2.uploadTrajectory(0, 0, traj2)

    print("Both trajectories uploaded")

    # Match this height to the z value in your MATLAB CSV generator
    allcfs.takeoff(targetHeight=0.8, duration=2.0)
    timeHelper.sleep(2.5)

    print("Starting trajectories...")
    cf1.startTrajectory(0, timescale=TIMESCALE)
    cf2.startTrajectory(0, timescale=TIMESCALE)

    timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

    print("Landing...")
    allcfs.land(targetHeight=0.06, duration=2.0)
    timeHelper.sleep(3.0)

    # Disable onboard logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()

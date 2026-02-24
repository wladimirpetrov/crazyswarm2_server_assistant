#!/usr/bin/env python

import time
import zmq
import numpy as np
import threading
from crazyflie_py import Crazyswarm

BASE_DURATION = 1.5
TARGET_START_POSITIONS = {
    "cf231": np.array([-0.14, 0.03, 0.0]),
    "cf233": np.array([-0.14, 0.43, 0.0]),
}

def fly_stepwise(cf, waypoints, drone_id, publisher, pub_lock, barrier):
    start_pos = TARGET_START_POSITIONS[drone_id]
    in_air = False
    waypoint_count = 0

    def publish(msg):
        with pub_lock:
            publisher.send_string(msg)
            print(msg)

    publish(f"{drone_id} status: Starting at {start_pos}")
    for i, wp in enumerate(waypoints):
        command_alt = wp[2]
        full_pos = start_pos + np.array(wp)

        # Barrier before deciding step
        barrier.wait()

        # Determine action
        if not in_air and command_alt > 0:
            # Takeoff
            publish(f"{drone_id} status: Taking off to {command_alt}")
            cf.takeoff(targetHeight=command_alt, duration=BASE_DURATION)
            time.sleep(BASE_DURATION + 0.5)
            in_air = True
        elif in_air and command_alt == 0:
            # Landing
            publish(f"{drone_id} status: Landing")
            cf.land(targetHeight=0.0, duration=BASE_DURATION * 2)
            time.sleep(BASE_DURATION * 2 + 0.5)
            in_air = False
        elif in_air:
            # Move
            publish(f"{drone_id} status: Moving to {full_pos}")
            cf.goTo(full_pos, yaw=0.0, duration=BASE_DURATION)
            time.sleep(BASE_DURATION + 0.5)
        else:
            # Idle on ground
            publish(f"{drone_id} status: Idle on ground")

        waypoint_count += 1
        publish(f"{drone_id} reached waypoint {waypoint_count}")

        # Barrier at the end of step
        barrier.wait()

    # Final landing if needed
    barrier.wait()
    if in_air:
        publish(f"{drone_id} final landing")
        cf.land(targetHeight=0.0, duration=BASE_DURATION * 2)
        time.sleep(BASE_DURATION * 2 + 0.5)
    barrier.wait()

def main():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:5555")
    time.sleep(5)
    pub_lock = threading.Lock()

    swarm = Crazyswarm()
    cfs = swarm.allcfs.crazyflies
    if len(cfs) < 2:
        print("Need at least 2 drones")
        return

    cf1, cf2 = cfs[0], cfs[1]
    #cf1 = cfs[0]
    barrier = threading.Barrier(2)

    waypoints1 = [(0.14, -0.03, 0.2), (0.24, -0.03, 0.2), (0.24, -0.03, 0.0)]
    waypoints2 = [(0.14, -0.43, 0.2), (0.24, -0.43, 0.2), (0.24, -0.43, 0.0)]

    t1 = threading.Thread(target=fly_stepwise, args=(cf1, waypoints1, "cf231", publisher, pub_lock, barrier))
    t2 = threading.Thread(target=fly_stepwise, args=(cf2, waypoints2, "cf233", publisher, pub_lock, barrier))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    publisher.close()
    context.term()

if __name__ == '__main__':
    main()

import sys
from mavlink.pymavlink import mavutil


def wait_for_msg(mavlink_msg):
    # Wait for a 'SYS_STATUS' message with the specified values.
    msg = drone.recv_match(type=mavlink_msg, blocking=True)
    if not msg:
        print('not msg')
    if msg.get_type() == "BAD_DATA":
        if mavutil.all_printable(msg.data):
            sys.stdout.write(msg.data)
            sys.stdout.flush()
    else:
        # Message is valid
        # Use the attribute
        print('Mode: %s' % msg)


if __name__ == '__main__':
    # Import ardupilotmega module for MAVLink 1
    # from pymavlink.dialects.v20 import development as mavlink2
    # from pymavlink.dialects.v10.development import
    # # mavlink2.MAVLINK_MSG_ID_FIREFLY_DELTA
    # mavlink2.MAVLink_firefly_ctrlalloc_message

    # Start a connection listening on a UDP port
    # drone = mavutil.mavlink_connection("/dev/ttyUSB0", "921600")
    # drone = mavutil.mavlink_connection(device="/dev/ttyACM0", baudrate="921600")
    mavutil.set_dialect('development')
    # mavutil.set_dialect('ardupilotmega')
    drone = mavutil.mavlink_connection(device="/dev/ttyACM0", baudrate="921600", dialect='development')
    # drone = mavutil.mavlink_connection(device="/dev/ttyUSB1", baudrate="921600", dialect='development')
    # drone = mavutil.mavlink_connection(device="/dev/ttyUSB1", baudrate="921600")
    print(f'drone {drone}')

    # for _ in range(2000):
    #     try:
    #         print(drone.recv_match().to_dict())
    #     except:
    #         pass

    # Send heartbeat from a MAVLink application.
    # drone.mav.heartbeat_send(
    #     mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    #                 Request a data stream.
    #
    #                 target_system             : The target requested to send the message stream. (type:uint8_t)
    #                 target_component          : The target requested to send the message stream. (type:uint8_t)
    #                 req_stream_id             : The ID of the requested data stream (type:uint8_t)
    #                 req_message_rate          : The requested message rate [Hz] (type:uint16_t)
    #                 start_stop                : 1 to start sending, 0 to stop sending. (type:uint8_t)
    # drone.mav.request_data_stream_send(drone.target_system, drone.target_component,
    #                                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 30, 1)

    print("Waiting for heartbeat (system %u component %u)" % (drone.target_system, drone.target_component))
    # Wait for the first heartbeat
    drone.wait_heartbeat()
    print('The heartbeat was received')

    # devnum = 10
    # cmd = "mavlink stream -d /dev/ttyS1 -s FIREFLY_CTRLALLOC -r 30"
    # buf = [ord(x) for x in cmd]
    # n = len(buf)
    # drone.mav.serial_control_send(
    #     devnum, mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE | mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND,
    #     0, 0, n, buf)

    # # Once connected, use 'drone' to get and send messages
    # msg = drone.messages['ODOMETRY']
    # print(f'msg {msg}')
    # msg = drone.messages['FIREFLY_CTRLALLOC']
    # print(f'msg {msg}')

    # Wait for a 'SYS_STATUS' message with the specified values.
    wait_for_msg('SYS_STATUS')
    wait_for_msg('ODOMETRY')
    wait_for_msg('FIREFLY_CTRLALLOC')
    # wait_for_msg('POSITION_TARGET_LOCAL_NED')

    # mavlink stream -d /dev/ttyS1 -s FIREFLY_CTRLALLOC -r 30
    # mavlink stream -d /dev/ttyS1 -s POSITION_TARGET_LOCAL_NED -r 30

    # mavlink start -d /dev/ttyACM0 -b 2000000 -r 800000 -x
    # mavlink start -d /dev/ttyS2 -b 115200 -r 2000 -x

from pymavlink import mavutil

# Start a connection listening to a UDP port
mvlink_conn = mavutil.mavlink_connection(device='/dev/ttyUSB0', baud=921600, autoreconnect=True)

# Wait for the first heartbeat
#   This sets the system and component ID of remote system for the link
mvlink_conn.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (mvlink_conn.target_system, mvlink_conn.target_component))

mvlink_conn.write('firefly nsh_command 0 0 1')


from mavsdk import System
import mavsdk
import asyncio


class MavFirefly:
    @staticmethod
    async def initialize_drone():
        drone = System()
        # await drone.connect(system_address="serial:///dev/ttyACM0")
        await drone.connect(system_address="serial:///dev/ttyUSB0:921600")

        print("Waiting for drone to connect...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"-- Connected to drone!")
                break

        async for health in drone.telemetry.health():
            print(f"-- health {health}")
            if health.is_accelerometer_calibration_ok:
                print(f"-- is_accelerometer_calibration_ok {health.is_accelerometer_calibration_ok}")
                break

        # async for flight_mode in drone.telemetry.flight_mode():
        #     print("FlightMode:", flight_mode)

        try:
            data = await drone.info.get_flight_information()
            print("get_flight_information:", data)
        except mavsdk.info.InfoError as error:
            print(f'mavsdk.info.InfoError {error}')

        try:
            data = await drone.info.get_identification()
            print("get_identification:", data)
        except mavsdk.info.InfoError as error:
            print(f'mavsdk.info.InfoError {error}')

        data = drone.telemetry.actuator_control_target()
        print("actuator_control_target:", data)
        data = drone.telemetry.actuator_output_status()
        print("actuator_output_status:", data)
        data = drone.telemetry.armed()
        print("armed:", data)
        data = drone.telemetry.attitude_angular_velocity_body()
        print("attitude_angular_velocity_body:", data)
        data = drone.telemetry.attitude_euler()
        print("attitude_euler:", data)
        data = drone.telemetry.position_velocity_ned()
        print("position_velocity_ned:", data)
        data = drone.telemetry.velocity_ned()
        print("velocity_ned:", data)
        data = drone.telemetry.vtol_state()
        print("vtol_state:", data)
        data = drone.telemetry.armed()
        print("armed:", data)
        data = drone.telemetry.armed()
        print("armed:", data)
        data = drone.telemetry.armed()
        print("armed:", data)

        try:
            print(f"-- Calling drone.action.arm()")
            data = await drone.action.arm()
            print("data:", data)
        except mavsdk.action.ActionError as error:
            print(f'error {error}')

        try:
            print(f"-- Calling drone.info.get_flight_information()")
            data = await drone.info.get_flight_information()
            print("data:", data)
        except mavsdk.info.InfoError as error:
            print(f'error {error}')

        try:
            print(f"-- Calling drone.telemetry.set_rate_velocity_ned(10)")
            data = await drone.telemetry.set_rate_velocity_ned(10)
            print("data:", data)
        except mavsdk.telemetry.TelemetryError as error:
            print(f'error {error}')

        # Start the tasks
        print(f"-- Starting tasks")
        # asyncio.ensure_future(MavFirefly.print_battery(drone))
        # asyncio.ensure_future(MavFirefly.print_gps_info(drone))
        # asyncio.ensure_future(MavFirefly.print_in_air(drone))
        # asyncio.ensure_future(MavFirefly.print_position(drone))
        # asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.position_velocity_ned))
        # asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.velocity_ned))
        # asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.odometry))
        # asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.armed))
        # asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.attitude_euler))
        # asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.attitude_angular_velocity_body))
        asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.actuator_output_status))
        # asyncio.ensure_future(MavFirefly.print_generic(drone, drone.telemetry.actuator_control_target))

        # info = await drone.info.get_version()
        # print(info)

        # async for flight_mode in drone.telemetry.flight_mode():
        #     print("FlightMode:", flight_mode)

        # info = await drone.telemetry.EulerAngle()
        # print("info:", info)

        # info = Info(drone)
        # finfo = await drone.info.get_flight_information()
        # print(f"finfo {finfo}")

        return drone

    @staticmethod
    async def print_generic(drone, method):
        async for data in method():
            print(f'{method.__name__}: {data}')

    @staticmethod
    async def print_battery(drone):
        async for battery in drone.telemetry.battery():
            print(f"battery.remaining_percent: {battery.remaining_percent}")

    @staticmethod
    async def print_gps_info(drone):
        async for gps_info in drone.telemetry.gps_info():
            print(f"gps_info: {gps_info}")

    @staticmethod
    async def print_in_air(drone):
        async for in_air in drone.telemetry.in_air():
            print(f"in_air: {in_air}")

    @staticmethod
    async def print_position(drone):
        async for position in drone.telemetry.position():
            print(f'position: {position}')

# def test_mavsdk():
#     drone = System('serial:///dev/ttyACM0', port=0)
#     # drone = System('serial:///dev/ttyACM1', port=1)
#     await drone.connect()


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(MavFirefly.initialize_drone())

    # Start the main function
    asyncio.ensure_future(MavFirefly.initialize_drone())

    # Runs the event loop until the program is canceled with e.g. CTRL-C
    asyncio.get_event_loop().run_forever()

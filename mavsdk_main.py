import asyncio


async def main():
    print('Hello ...')
    await asyncio.sleep(1)
    print('... World!')


if __name__ == '__main__':
    asyncio.run(main())

# from mavsdk import System
#
#
# def test_mavsdk():
#     drone = System('serial:///dev/ttyACM0', port=0)
#     # drone = System('serial:///dev/ttyACM1', port=1)
#     await drone.connect()
#
#
# if __name__ == '__main__':
#     test_mavsdk()



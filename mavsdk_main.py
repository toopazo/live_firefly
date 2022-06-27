from mavsdk import System


def test_mavsdk():
    drone = System()
    await drone.connect()


if __name__ == '__main__':
    test_mavsdk()



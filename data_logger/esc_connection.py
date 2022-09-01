from live_esc.kde_uas85uvc.kdecan_interface import KdeCanIface


class EscIfaceWrapper:
    def __init__(self):
        self.esc = KdeCanIface()
        self.esc_arr = list(range(11, 19))

    def get_data(self):
        # get data
        resp_arr = self.esc.get_data_esc_arr(self.esc_arr)

        # for each targetid, get data
        log_data_final = ""
        for resp in resp_arr:
            [telap, targetid,
             voltage, current, rpm, temp, warning,
             inthrottle, outthrottle] = resp

            log_data = "%s, %s, %04.2f, %s, %07.2f, %s, %s, %s, %s " % \
                       (telap, targetid,
                        voltage, current, rpm, temp, warning,
                        inthrottle, outthrottle)
            if len(log_data_final) == 0:
                log_data_final = log_data
            else:
                # log_data_final = log_data_final + "\r\n" + log_data
                log_data_final = log_data_final + ", " + log_data
            # self.log_fd.write(log_data + "\r\n")
        return log_data_final

    def close(self):
        self.esc.kdecan_bus.shutdown()


class SensorIfaceWrapper:
    def __init__(self):
        self.esc = EscIfaceWrapper()

    def get_data(self):
        esc_data = self.esc.get_data()
        log_data = f"{esc_data}"
        return log_data

    def close(self):
        # self.ars.close()
        self.esc.close()

    @staticmethod
    def parse_sensor_data(sensor_data):

        parsed_data_arr = []
        for e in sensor_data.split(','):
            try:
                val = float(e.strip())
            except ValueError:
                val = e.strip()
            parsed_data_arr.append(val)

        parsed_data_dict = {}

        num_rotors = 8
        for i in range(0, num_rotors):
            indx = 9 * i
            escid = str(10 + i + 1)

            try:
                parsed_data_dict[f'time_{escid}'] = parsed_data_arr[
                    indx + 0]  # 20 + 9*0 = 20, # 20 + 9*1 = 29
                parsed_data_dict[f'escid_{escid}'] = parsed_data_arr[indx + 1]
                parsed_data_dict[f'voltage_{escid}'] = parsed_data_arr[indx + 2]
                parsed_data_dict[f'current_{escid}'] = parsed_data_arr[indx + 3]
                parsed_data_dict[f'angVel_{escid}'] = parsed_data_arr[indx + 4]
                parsed_data_dict[f'temp_{escid}'] = parsed_data_arr[indx + 5]
                parsed_data_dict[f'warning_{escid}'] = parsed_data_arr[indx + 6]
                parsed_data_dict[f'inthtl_{escid}'] = parsed_data_arr[indx + 7]
                parsed_data_dict[f'outthtl_{escid}'] = parsed_data_arr[indx + 8]
            except IndexError:
                pass

        return parsed_data_dict

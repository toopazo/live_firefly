import pprint


class FireflyOptimizer:

    @staticmethod
    def filter_parsed_data(parsed_data):
        old_dict = parsed_data
        your_keys = [
            'cur1', 'cur2', 'cur3', 'cur4', 'cur5', 'cur6', 'cur7', 'cur8',
            'rpm1', 'rpm2', 'rpm3', 'rpm4', 'rpm5', 'rpm6', 'rpm7', 'rpm8',
        ]
        num_rotors = 8
        for i in range(0, num_rotors):
            escid = str(10 + i + 1)
            # your_keys.append(f'time_{escid}')
            # your_keys.append(f'escid_{escid}')
            your_keys.append(f'voltage_{escid}')
            your_keys.append(f'current_{escid}')
            your_keys.append(f'angVel_{escid}')
            # your_keys.append(f'temp_{escid}')
            # your_keys.append(f'warning_{escid}')
            # your_keys.append(f'inthtl_{escid}')
            # your_keys.append(f'outthtl_{escid}')

        dict_you_want = {your_key: old_dict[your_key] for your_key in your_keys}
        return dict_you_want

    @staticmethod
    def parse_sensor_data(sensor_data):
        # fields = ["sps", "mills", "secs", "dtmills",
        #           "cur1", "cur2", "cur3", "cur4",
        #           "cur5", "cur6", "cur7", "cur8",
        #           "rpm1", "rpm2", "rpm3", "rpm4",
        #           "rpm5", "rpm6", "rpm7", "rpm8"]
        # ars_header = ", ".join(fields)
        # esc_header = "time s, escid, " \
        #              "voltage V, current A, angVel rpm, temp degC, warning, " \
        #              "inthtl us, outthtl perc"
        # sensor_header = f"{ars_header}, {esc_header}"
        parsed_data_arr = []
        for e in sensor_data.split(','):
            try:
                val = float(e.strip())
            except ValueError:
                val = e.strip()
            parsed_data_arr.append(val)

        parsed_data_dict = {
            'sps': parsed_data_arr[0],
            'mills': parsed_data_arr[1],
            'secs': parsed_data_arr[2],
            'dtmills': parsed_data_arr[3],
            'cur1': parsed_data_arr[4],
            'cur2': parsed_data_arr[5],
            'cur3': parsed_data_arr[6],
            'cur4': parsed_data_arr[7],
            'cur5': parsed_data_arr[8],
            'cur6': parsed_data_arr[9],
            'cur7': parsed_data_arr[10],
            'cur8': parsed_data_arr[11],
            'rpm1': parsed_data_arr[12],
            'rpm2': parsed_data_arr[13],
            'rpm3': parsed_data_arr[14],
            'rpm4': parsed_data_arr[15],
            'rpm5': parsed_data_arr[16],
            'rpm6': parsed_data_arr[17],
            'rpm7': parsed_data_arr[18],
            'rpm8': parsed_data_arr[19],
        }
        num_rotors = 8
        for i in range(0, num_rotors):
            indx = 20 + 9*i
            escid = str(10+i+1)
            # print(f'indx {indx} escid {escid}')

            try:
                parsed_data_dict[f'time_{escid}'] = parsed_data_arr[indx+0]   # 20 + 9*0 = 20, # 20 + 9*1 = 29
                parsed_data_dict[f'escid_{escid}'] = parsed_data_arr[indx + 1]
                parsed_data_dict[f'voltage_{escid}'] = parsed_data_arr[indx+2]
                parsed_data_dict[f'current_{escid}'] = parsed_data_arr[indx+3]
                parsed_data_dict[f'angVel_{escid}'] = parsed_data_arr[indx+4]
                parsed_data_dict[f'temp_{escid}'] = parsed_data_arr[indx+5]
                parsed_data_dict[f'warning_{escid}'] = parsed_data_arr[indx+6]
                parsed_data_dict[f'inthtl_{escid}'] = parsed_data_arr[indx+7]
                parsed_data_dict[f'outthtl_{escid}'] = parsed_data_arr[indx+8]
            except IndexError:
                pass
        return parsed_data_dict

    @staticmethod
    def sensor_data_to_cost_fnct(sensor_data):
        parsed_data = FireflyOptimizer.parse_sensor_data(sensor_data)
        # print(f'parsed_data {parsed_data}')
        try:
            pprint.pprint(FireflyOptimizer.filter_parsed_data(parsed_data))
        except KeyError:
            print('[sensor_data_to_cost_fnct] error parsing data')

        cost_arr = []
        num_rotors = 8
        for i in range(0, num_rotors):
            escid = str(10 + i + 1)
            try:
                # cost = parsed_data[f'voltage_{escid}'] * parsed_data[f'current_{escid}']
                cost = parsed_data[f'voltage_{escid}'] * parsed_data[f'current_{escid}']
                # cur1 corresponds to motor3
                # cur3 corresponds to motor4
                # cur4 corresponds to motor7
            except KeyError:
                cost = None
            cost_arr.append(cost)
        return cost_arr



class FireflyOptimizer:
    @staticmethod
    def print_sensor_data(sensor_data):
        fields = ["sps", "mills", "secs", "dtmills",
                  "cur1", "cur2", "cur3", "cur4",
                  "cur5", "cur6", "cur7", "cur8",
                  "rpm1", "rpm2", "rpm3", "rpm4",
                  "rpm5", "rpm6", "rpm7", "rpm8"]
        ars_header = ", ".join(fields)
        esc_header = "time s, escid, " \
                     "voltage V, current A, angVel rpm, temp degC, warning, " \
                     "inthtl us, outthtl perc"
        # sensor_header = f"{ars_header}, {esc_header}"

        FireflyOptimizer.sensor_data_to_cost_fnct(sensor_data)

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
        parsed_data = []
        for e in sensor_data.split(','):
            try:
                val = float(e.strip())
            except ValueError:
                val = e.strip()
            parsed_data.append(val)

        parsed_data2 = {
            'sps': parsed_data[0],
            'mills': parsed_data[1],
            'secs': parsed_data[2],
            'dtmills': parsed_data[3],
            'cur1': parsed_data[4],
            'cur2': parsed_data[5],
            'cur3': parsed_data[6],
            'cur4': parsed_data[7],
            'cur5': parsed_data[8],
            'cur6': parsed_data[8],
            'cur7': parsed_data[10],
            'cur8': parsed_data[11],
            'rpm1': parsed_data[12],
            'rpm2': parsed_data[13],
            'rpm': parsed_data[14],
            'rpm4': parsed_data[15],
            'rpm5': parsed_data[16],
            'rpm6': parsed_data[17],
            'rpm7': parsed_data[18],
            'rpm8': parsed_data[19],
        }
        for i in range(0, 8):
            indx = 20 + 8*i
            escid = str(10+i+1)
            print(f'indx {indx} escid {escid}')
            parsed_data2[f'time_{escid}'] = parsed_data[indx+0]   # 20 + 8*0 = 20, # 20 + 8*1 = 28
            parsed_data2[f'voltage_{escid}'] = parsed_data[indx+1]
            parsed_data2[f'current_{escid}'] = parsed_data[indx+2]
            parsed_data2[f'angVel_{escid}'] = parsed_data[indx+3]
            parsed_data2[f'temp_{escid}'] = parsed_data[indx+4]
            parsed_data2[f'warning_{escid}'] = parsed_data[indx+5]
            parsed_data2[f'inthtl_{escid}'] = parsed_data[indx+6]
            parsed_data2[f'outthtl_{escid}'] = parsed_data[indx+7]

        # sensor_data = [float(e.strip()) for e in sensor_data.split(',')]
        # print(f'parsed_data {parsed_data2}')
        return parsed_data2

    @staticmethod
    def sensor_data_to_cost_fnct(sensor_data):
        parsed_data = FireflyOptimizer.parse_sensor_data(sensor_data)
        # sensor_data = [float(e.strip()) for e in sensor_data.split(',')]
        print(f'parsed_data {parsed_data}')
        m1_curr = parsed_data[4]
        m1_rpm = parsed_data[12]
        cost = m1_curr
        return cost

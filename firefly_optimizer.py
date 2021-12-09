

class FireflyOptimizer:
    @staticmethod
    def sensor_data_to_cost_fnct(sensor_data):
        # fields = ["sps", "mills", "secs", "dtmills",
        #           "cur1", "cur2", "cur3", "cur4",
        #           "cur5", "cur6", "cur7", "cur8",
        #           "rpm1", "rpm2", "rpm3", "rpm4",
        #           "rpm5", "rpm6", "rpm7", "rpm8"]
        # ars_header = ", ".join(fields)
        # esc_header = "time s, escid, "\
        #              "voltage V, current A, angVel rpm, temp degC, warning, "\
        #              "inthtl us, outthtl perc"
        # sensor_header = f"{ars_header}, {esc_header}"
        parsed_data = []
        for e in sensor_data.split(','):
            try:
                val = float(e.strip())
            except ValueError:
                val = e.strip()
            parsed_data.append(val)
        # sensor_data = [float(e.strip()) for e in sensor_data.split(',')]
        print(f'parsed_data {parsed_data}')
        m1_curr = parsed_data[4]
        m1_rpm = parsed_data[12]
        cost = m1_curr
        return cost

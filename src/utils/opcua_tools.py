import asyncio
import pandas as pd
from datetime import datetime
from opcua import Client as syncClient
from asyncua import Client as asyncClient


def search_result(answer, select_var):
    if not answer:
        return pd.DataFrame()

    results = []
    date_time = []

    # Execute the function 10 times
    for i in range(len(answer)):
        ans_out = answer[i]
        results.append(ans_out.Value.Value)
        date_time.append(ans_out.SourceTimestamp)

        # print(f"Cycle {i}: value {ans_out.Value.Value} at {ans_out.SourceTimestamp}")

    complete_dataset = pd.DataFrame({"Time": date_time, select_var: results})
    return complete_dataset


def _get_address(machine):
    switch_dict = {
        "pump_vacuum": lambda: "opc.tcp://172.31.111.114:4868",
        "compressor": lambda: "opc.tcp://172.31.111.114:4869"
        }

    # Use the walrus operator to assign the value from the dictionary and return it
    return switch_dict.get(machine, lambda: "opc.tcp://172.31.111.114:4868")()


def _get_nodeid(machine, variable):
    if machine == "pump_vacuum":
        answer = _get_nodeid_vacuum_pump(variable)
    elif machine == "compressor":
        answer = _get_nodeid_compressor(variable)
    else:
        answer = _get_nodeid_vacuum_pump(variable)

    return answer


def _get_nodeid_vacuum_pump(variable):
    switch_dict = {
        "InletPressure": lambda: "ns=2;i=10015",
        "OutletPressure": lambda: "ns=2;i=10078",
        "phaseA_voltage": lambda: "ns=2;i=8929",
        "phaseB_voltage": lambda: "ns=2;i=9286",
        "phaseC_voltage": lambda: "ns=2;i=9643",
        "phaseA_current": lambda: "ns=2;i=8992",
        "phaseB_current": lambda: "ns=2;i=9349",
        "phaseC_current": lambda: "ns=2;i=9706",
        "OAVelocity_x": lambda: "ns=2;i=8062",       # VerticalVibration-OverallVibrationVelocity
        "OAVelocity_y": lambda: "ns=2;i=7639",       # HorizontalVibration-OverallVibrationVelocity
        "OAVelocity_z": lambda: "ns=2;i=8485",       # AxialVibration-OverallVibrationVelocity
        "temperature": lambda: "ns=2;i=10156",      # InletTemperature
        }

    # Use the walrus operator to assign the value from the dictionary and return it
    return switch_dict.get(variable, lambda: "ns=2;i=8929")()


def _get_nodeid_compressor(variable):
    switch_dict = {
        "Pressure": lambda: "ns=2;i=10545",
        "phaseA_voltage": lambda: "ns=2;i=8929",
        "phaseB_voltage": lambda: "ns=2;i=9466",
        "phaseC_voltage": lambda: "ns=2;i=10003",
        "phaseA_current": lambda: "ns=2;i=8992",
        "phaseB_current": lambda: "ns=2;i=9529",
        "phaseC_current": lambda: "ns=2;i=10066",
        "OAVelocity_x": lambda: "ns=2;i=8062",       # VerticalVibration-OverallVibrationVelocity
        "OAVelocity_y": lambda: "ns=2;i=7639",       # HorizontalVibration-OverallVibrationVelocity
        "OAVelocity_z": lambda: "ns=2;i=8485",       # AxialVibration-OverallVibrationVelocity
        }

    # Use the walrus operator to assign the value from the dictionary and return it
    return switch_dict.get(variable, lambda: "ns=2;i=8929")()


def _calendar_adjustment(date, date_pos):
    if not date:
        if date_pos == "begin":
            date = "01.01.2023 10:00:00"
        elif date_pos == "end":
            date = "16.06.2023 10:00:00"
        return datetime.strptime(date, "%d.%m.%Y %H:%M:%S")
    if isinstance(date, datetime):
        return date
    elif isinstance(date, str):
        return datetime.strptime(date, "%d.%m.%Y %H:%M:%S")


async def _get_historized_values(machine_server, variable_nodeid, start_date, end_date):
    client = asyncClient(machine_server, timeout=400)
    await client.connect()

    counter = client.get_node(variable_nodeid)

    data = await counter.read_raw_history(
        starttime=start_date,
        endtime=end_date,
        numvalues=1000)
    # print(data)

    await client.disconnect()

    return data


def get_historized_values(machine: str, variable: str, start_date: datetime, end_date: datetime):
    machine_in = _get_address(machine)
    variable_in = _get_nodeid(machine, variable)
    start_date_in = _calendar_adjustment(start_date, "begin")
    end_date_in = _calendar_adjustment(end_date, "end")

    # print(machine_in)
    # print(variable_in)
    # print(start_date_in)
    # print(end_date_in)

    task = asyncio.run(_get_historized_values(machine_in, variable_in, start_date_in, end_date_in))
    dataset = search_result(task, variable)

    return dataset


def get_single_value(parameter):
    # Specify the address of the OPC-UA server
    opcua_server_address = "opc.tcp://172.31.111.114:4868"  # Replace with the actual server address

    # Connect to the OPC-UA server
    client = syncClient(url=opcua_server_address, timeout=4000)
    client.connect()

    try:
        # Browse the server's address space to find the available parameters
        zero_lvl_node = client.get_root_node()
        first_lvl_node = zero_lvl_node.get_child(["Objects"])
        second_lvl_node = first_lvl_node.get_child(["2:AASAssetAdministrationShell"])
        third_lvl_node = second_lvl_node.get_child(["2:Submodel:ConditionMonitoring"])

        if parameter in "phaseA_current":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseA"])
            var_node = sub_var_node.get_child(["2:Current"])
        elif parameter == "phaseA_voltage":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseA"])
            var_node = sub_var_node.get_child(["2:Voltage"])
        elif parameter == "phaseB_current":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseB"])
            var_node = sub_var_node.get_child(["2:Current"])
        elif parameter == "phaseB_voltage":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseB"])
            var_node = sub_var_node.get_child(["2:Voltage"])
        elif parameter == "phaseC_current":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseC"])
            var_node = sub_var_node.get_child(["2:Current"])
        elif parameter == "phaseC_voltage":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseC"])
            var_node = sub_var_node.get_child(["2:Voltage"])
        else:
            print("Wrong parameter")
            exit()

        # Open the available nodes
        # open_first_lvl_node = first_lvl_node.get_children()
        # open_second_lvl_node = second_lvl_node.get_children()
        # open_third_lvl_node = third_lvl_node.get_children()

        # open_top_var_node = top_var_node.get_children()
        # open_sub_var_node = sub_var_node.get_children()
        open_var_node = var_node.get_children()

        # Print the names of the parameters
        # print(sub_var_node.get_display_name())
        value_single = open_var_node[4].get_value()
        time_single = open_var_node[4].get_data_value().SourceTimestamp

        # print("Var: ", value_single)
        # print("Time: ", time_single)

    finally:
        # Disconnect from the OPC-UA server
        client.disconnect()

    return [value_single, time_single]


if __name__ == '__main__':
    # These code lines won't run if this file is imported.
    get_single_value("phaseA_voltage")
    get_historized_values(machine="pump_vacuum", variable="phaseA_voltage",
                          start_date=datetime(2023, 1, 1, 0, 0, 0),
                          end_date=datetime(2023, 7, 24, 23, 59, 59))

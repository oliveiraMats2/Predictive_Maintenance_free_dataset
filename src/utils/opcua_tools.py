from datetime import datetime
from opcua import Client


def get_single(parameter):
    # Specify the address of the OPC-UA server
    opcua_server_address = "opc.tcp://172.31.111.114:4868"  # Replace with the actual server address

    # Connect to the OPC-UA server
    client = Client(url=opcua_server_address)  # , timeout=4)
    client.connect()

    try:
        # Browse the server's address space to find the available parameters
        zero_lvl_node = client.get_root_node()
        first_lvl_node = zero_lvl_node.get_child(["Objects"])
        second_lvl_node = first_lvl_node.get_child(["2:AASAssetAdministrationShell"])
        third_lvl_node = second_lvl_node.get_child(["2:Submodel:ConditionMonitoring"])

        if parameter in "Phase":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseA"])
            var_node = sub_var_node.get_child(["2:Current"])
        elif parameter == "PhaseA-voltage":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseA"])
            var_node = sub_var_node.get_child(["2:Voltage"])
        elif parameter == "PhaseB-current":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseB"])
            var_node = sub_var_node.get_child(["2:Current"])
        elif parameter == "PhaseB-voltage":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseB"])
            var_node = sub_var_node.get_child(["2:Voltage"])
        elif parameter == "PhaseC-current":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseC"])
            var_node = sub_var_node.get_child(["2:Current"])
        elif parameter == "PhaseC-voltage":
            top_var_node = third_lvl_node.get_child(["2:Electrical"])
            sub_var_node = top_var_node.get_child(["2:PhaseC"])
            var_node = sub_var_node.get_child(["2:Voltage"])
        else:
            print("Wrong parameter")
            exit()

        # Open the available nodes
        open_first_lvl_node = first_lvl_node.get_children()
        open_second_lvl_node = second_lvl_node.get_children()
        open_third_lvl_node = third_lvl_node.get_children()

        open_top_var_node = top_var_node.get_children()
        open_sub_var_node = sub_var_node.get_children()
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
    # This code won't run if this file is imported.
    get_single("PhaseA-voltage")

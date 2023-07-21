from datetime import datetime
from tools import opcua_tools as op

# Use maquina em ingles, vacuum_pump ou compressor, se errar vou usar o vacuum_pump
# use_param seria a variavel a usar, usa o nome original, se falhar uso o phaseA_voltage
# data de inicio e fim, usa a estrutura ano, mes, dia, hora, min, sec

results = []
date_time = []

start_date = datetime(2023, 1, 1, 0, 0, 0)
end_date = datetime(2023, 7, 24, 23, 59, 59)

machines = ["pump_vacuum", "compressor"]
all_param = [
    "InletPressure", "OutletPressure", "temperature",
    "phaseA_voltage", "phaseB_voltage", "phaseC_voltage",
    "phaseA_current", "phaseB_current", "phaseC_current",
    "OAVelocity_x", "OAVelocity_y", "OAVelocity_z", "Pressure"]

for j in range(2):
    machine = machines[j]
    print(f"Results for {machine}")
    for i in range(13):
        use_param = all_param[i]
        if (machine == "compressor" and (i == 0 or i == 1)) or\
            (machine == "pump_vacuum" and i == 12):
            print(f"For {machine}: {use_param} was skipped")
        else:
            dataset = op.get_historized_values(machine, use_param, start_date, end_date)
            print(dataset.head())

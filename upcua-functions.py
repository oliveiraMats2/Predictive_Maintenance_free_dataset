from classical_methods_forecast.neural_prophet.upcua_instants_value import UpcuaInstantValues

upcua_instant_values = UpcuaInstantValues("PhaseA-voltage")
df = upcua_instant_values.actual_dataframe(100)

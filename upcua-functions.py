import time
import pandas as pd
from src.utils import opcua_tools as op
from upcua_instants_value import UpcuaInstantValues

upcua_instant_values = UpcuaInstantValues("PhaseA-voltage")
df = upcua_instant_values.actual_dataframe(100)

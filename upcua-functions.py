import time
import pandas as pd
from src.utils import opcua_tools as op


# Execute the function 10 times
results = []
date_time = []
use_param = "PhaseA-voltage"

for i in range(10):
    output = op.get_single(use_param)
    results.append(output[0])
    date_time.append(output[1])

    print(f"Cycle {i}: value {output[0]} at {output[1]}")

    # Wait for 1 minute
    time.sleep(1)

# Convert the results into a dataset (Pandas DataFrame)
dataset = pd.DataFrame({"Time": date_time, use_param: results})

# Print the dataset
print(dataset)

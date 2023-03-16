import torch
from pytorch_forecasting.models import TemporalFusionTransformer
import torch.optim as optim
from pytorch_forecasting.metrics.quantile import QuantileLoss

# Define optimizer


# Define input data
X = torch.randn(10, 24, 5)  # input shape: (batch_size, sequence_length, num_features)
y = torch.randn(10, 12)  # target shape: (batch_size, prediction_length)

# Define model architecture
model = TemporalFusionTransformer(
    output_size=y.shape[-1],  # number of output features
    hidden_size=32,
    dropout=0.1,
    loss=QuantileLoss(),
)

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass through the model
output = model(X)

# Compute loss
loss = model.loss(output, y)

# Backward pass and optimization step
loss.backward()
optimizer.step()

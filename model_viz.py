import torch
from model import NeutronDetectorCNN
from torchviz import make_dot

# Instantiate your model
model = NeutronDetectorCNN(image_size=10)

# Create a dummy input consistent with the input dimensions of your model
dummy_input = torch.randn(1, 1, 10, 10)

# Forward pass through the model
output = model(dummy_input)

# Visualize the graph
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("neutron_detector_cnn", format="png")

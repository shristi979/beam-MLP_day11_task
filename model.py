
import torch
import torch.nn as nn



# 1. DEFINE THE NEURAL NETWORK


class BeamMLP(nn.Module):

    def __init__(self):

        # Initialize parent class
        super().__init__()

        # First hidden layer
        self.layer1 = nn.Linear(5, 64)

        # Second hidden layer
        self.layer2 = nn.Linear(64, 64)

        # Output layer
        self.output = nn.Linear(64, 1)

        # Activation function
        self.relu = nn.ReLU()


    
    # 2. FORWARD PROPAGATION
    

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.output(x)

        return x



# 3. TEST THE MODEL


if __name__ == "__main__":

    # Create model
    model = BeamMLP()

    # Create dummy input
    sample_input = torch.randn(10, 5)

    # Pass input through network
    prediction = model(sample_input)

    print("Input shape:", sample_input.shape)
    print("Output shape:", prediction.shape)

    print("\nModel architecture:")
    print(model)

    # Count trainable parameters
    total_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal trainable parameters:", total_params)


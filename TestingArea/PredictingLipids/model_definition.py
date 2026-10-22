import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class ModelArchitecture(nn.Module):
    def __init__(self, input_size, output_size, l1, l2, l3):   # Define the layers and building blocks of the network
        super().__init__()  # Call the parent class constructor
        self.fc1 = nn.Linear(input_size, l1)            # First fully connected layer
        self.fc2 = nn.Linear(l1, l2)                    # Second fully connected layer
        self.fc3 = nn.Linear(l2, l3)                    # Third fully connected layer
        self.fc4 = nn.Linear(l3, output_size)           # Fourth fully connected layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer with a dropout probability of 0.2

    def forward(self, x):                                       # Define the forward pass, i.e. how layers are connected
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out) 
        out = F.softplus(out)                           # Ensure positive outputs for densities and ensure smoothness for gradients
        return out
    
    
    
class GlobalLipidConstraint(nn.Module):
    """Applies the global compossition constraint on raw model outputs.
    Same code path at train time and inference time -> nothing to keep in sync."""

    def __init__(self, eps=1e-8, k_clamp=1e3):
        super().__init__()
        self.eps = eps
        self.k_clamp = k_clamp

    def _find_scaling(self, phi, target_composition, P_A, dH):
        predicted_composition = (phi * P_A * dH).sum()
        if torch.isnan(target_composition):
            return torch.zeros((), device=phi.device, dtype=phi.dtype)
        # sign-preserving clamp so we don't divide by (near) zero
        pc = torch.clamp(predicted_composition.abs(), min=self.eps) * torch.sign(predicted_composition)
        K = target_composition / pc
        K = torch.clamp(K, min=-self.k_clamp, max=self.k_clamp)
        return K

    def forward(self, y_pred, P_A, compositions, dH):
        K1 = self._find_scaling(y_pred[:, 0], compositions[0], P_A, dH)
        K2 = self._find_scaling(y_pred[:, 1], compositions[1], P_A, dH)

        y_pred_scaled = y_pred.clone()
        y_pred_scaled[:, 0] = y_pred[:, 0] * K1
        y_pred_scaled[:, 1] = y_pred[:, 1] * K2
        return y_pred_scaled
    
    
    
class ConstrainedPINN(nn.Module):
    def __init__(self, base_model, scaling_timer=500):
        super().__init__()
        self.base_model = base_model
        self.constraint = GlobalLipidConstraint()
        self.scaling_timer = scaling_timer

    def forward(self, x, P_A=None, compositions=None, dH=None, apply_constraint=True):
        y_pred = self.base_model(x)
        if apply_constraint:
            assert P_A is not None and compositions is not None and dH is not None, \
                "P_A, compositions, and dH are required to apply the global composition constraint."
            y_pred = self.constraint(y_pred, P_A, compositions, dH)
        return y_pred
    
    
def create_model(input_size, output_size, l1, l2, l3, scaling_timer=500):
    """Creates a ConstrainedPINN model with the specified architecture and scaling timer."""
    
    base_model = ModelArchitecture(input_size, output_size, l1, l2, l3)
    model = ConstrainedPINN(base_model=base_model, scaling_timer=scaling_timer)
    return model
    
def save_model(model, config, HP_config, scaler):
    """
    Save the model's architecture, parameters, and scaler to a file.

    Parameters:
    - model: The PyTorch model to save.
    - config: The configuration object containing feature columns and other settings.
    - HP_config: The hyperparameter configuration dictionary containing layer sizes and other hyperparameters.
    - scaler: The fitted StandardScaler object used for input normalization.
    """
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_size": config.n_features,
            "output_size": config.n_targets,
            "l1": HP_config["l1"], "l2": HP_config["l2"], "l3": HP_config["l3"],
            "scaling_timer": model.scaling_timer,
        },
        "model_input_params": config.feature_columns,
        # Features for the standard scaler
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }, f"{config.model_name}.pth")
    
def load_model(model_name, device):
    """
    Load the model's architecture, parameters, and scaler.
    
    Returns:
    - model: The loaded PyTorch model.
    - device: The device on which the model is loaded (CPU or GPU).
    """
    # Load checkpoint
    checkpoint = torch.load(f"{model_name}.pth", map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    input_params = checkpoint["model_input_params"]
    
    # Construct scaler
    scaler_X = StandardScaler()
    scaler_X.mean_ = checkpoint["scaler_mean"]
    scaler_X.scale_ = checkpoint["scaler_scale"]

    # Construct the model
    base_model = ModelArchitecture(cfg["input_size"], cfg["output_size"], cfg["l1"], cfg["l2"], cfg["l3"])
    model = ConstrainedPINN(base_model, scaling_timer=cfg["scaling_timer"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    
    return {"model": model, "scaler_X": scaler_X, "feature_columns": input_params}
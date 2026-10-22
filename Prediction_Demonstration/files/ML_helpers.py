import os

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

    
def load_model(model_folder, model_name, device):
    """
    Load the model's architecture, parameters, and scaler.
    
    Returns:
    - model: The loaded PyTorch model.
    - device: The device on which the model is loaded (CPU or GPU).
    """
    # Load checkpoint
    checkpoint = torch.load(os.path.join(model_folder, f"{model_name}.pth"), map_location=device, weights_only=False)
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



import numpy as np
from scipy import stats


def calc_statistical_descriptors(shape, area, n_bins=100, bin_range=(-0.7, 0.5)):
    """A huge function for calculating a variety of statistical descriptors for a given shape and area.
    Hopefully used to generate features for machine learning."""
    # Construct a histogram of the shape data weighted by area
    pdf, bin_edges = np.histogram(shape, bins=n_bins, range=bin_range, weights=area, density=True)
    p = pdf / np.sum(pdf)  # Normalize to get a probability distribution
    entropy = stats.entropy(p)  # Calculate the entropy of the distribution
    
    # Calculate statistical descriptors for the given shape and area
    # Estimate a CDF and find the quantiles using that
    quantiles = np.percentile(shape, [5, 95, 25, 50, 75], weights=area, method='inverted_cdf')
    Q_5 = quantiles[0]
    Q_95 = quantiles[1]
    Q_25 = quantiles[2]
    Q_50 = quantiles[3]
    Q_75 = quantiles[4]

    # Quantile-based descriptors
    percentile_span = Q_95 - Q_5
    IQR = Q_75 - Q_25
    Bowley_skewness = (Q_75 - Q_50) - (Q_50 - Q_25) / (Q_75 - Q_25)

    # Calculating moments (Membrane descriptors)
    mean_H = np.average(shape, weights=area)                                        # First moment
    var_H = np.average(shape*shape, weights=area) - mean_H**2                       # Second Central moment (variance)
    skew_H = np.average((shape - mean_H)**3, weights=area) / (var_H)**(3/2)         # Third standardized moment (skewness) (Also called the Fisher-Pearson coefficient of skewness)
    kurtosis_H = np.average((shape - mean_H)**4, weights=area) / (var_H)**(2) - 3   # Fourth standardized moment (kurtosis)
                                                                                    # Also called the excess kurtosis, which is the kurtosis minus 3 (the kurtosis of a normal distribution)
    
    # Bimodality coefficient (From a finite sample)
    n = len(shape)
    bimodality_coefficient = (skew_H**2 + 1) / (kurtosis_H + (3 * (n - 1)**2) / ((n -2) * (n - 3)))
    
    
    return {
        "entropy": entropy,
        "Q_5": Q_5,
        "Q_25": Q_25,
        "Q_50": Q_50,
        "Q_75": Q_75,
        "Q_95": Q_95,
        "percentile_span": percentile_span,
        "IQR": IQR,
        "Bowley_skewness": Bowley_skewness,
        "mean_H": mean_H,
        "var_H": var_H,
        "skew_H": skew_H,
        "kurtosis_H": kurtosis_H,
        "bimodality_coefficient": bimodality_coefficient,
    }





### ML Helpers

def stat_to_input(stat_container, feature_columns, C_POPC, C_CDL, n_bins, bin_range):
    input_params = {key: None for key in feature_columns}

    # Add all the statistical descriptors to the input_params dictionary
    for i, key in enumerate(feature_columns):
        if key in stat_container:
            input_params[key] = stat_container[key]

    # Construct the bin_centers for the input data
    bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    input_params['H_i'] = bin_centers

    # Construct the concentration array for the input data
    input_params["C_ratio"] = C_CDL/C_POPC
    return input_params


def construct_ML_input(input_params):
    """Converts the filled input_params dictionary into a tensor suitable for the ML model."""
    n_bins = len(input_params['H_i'])
    input_tensor = torch.zeros((1, n_bins, len(input_params)))
    
    # fill the tensor with the values from input_params
    for i, key in enumerate(input_params):
        input_tensor[0, :, i] = torch.tensor(input_params[key], dtype=torch.float32)            
    return input_tensor.squeeze(0)  # Remove the first dimension to match the expected input shape for the model



def convert_to_prob(densities):
    """
    Convert densities to probabilities by normalizing each bin across lipids.
    """
    probabilites = []
    for data in densities:
        totals = data.sum(axis=0)  # total composition per bin, summed over lipids

        lipid_prob = np.divide(
            data, totals,
            out=np.zeros_like(data, dtype=float),
            where=totals > 0
        )
        probabilites.append(lipid_prob)
    return np.array(probabilites)




def load_model_and_create_input(H, A, C_POPC, C_CDL, n_bins, bin_range):
    # Loading the model, the feature columns, and the scaler
    model_dict = load_model(model_folder="files", model_name="HardPINN", device="cpu")
    model = model_dict["model"]
    feature_columns = model_dict["feature_columns"]
    scaler_X = model_dict["scaler_X"]

    # From the parameters we create an empty dictionary to store the statistical descriptors
    input_params = {key: None for key in feature_columns}
    stat_container = calc_statistical_descriptors(H, A, n_bins=n_bins, bin_range=bin_range)
    input_params = stat_to_input(stat_container, feature_columns, C_POPC=C_POPC, C_CDL=C_CDL, n_bins=n_bins, bin_range=bin_range)
    input_tensor = construct_ML_input(input_params)

    # Scaling the input tensor using the pre-fitted scaler (and converting it back to a tensor)
    input_tensor_scaled = torch.tensor(scaler_X.transform(input_tensor.numpy()), dtype=torch.float32)
    
    return model, input_params, input_tensor_scaled


# Make predictions using the ML model
def make_predictions(model, input_tensor_scaled, shape_density, C_POPC, C_CDL, input_params):
    model.eval()
    with torch.no_grad():
        # Unsqueeze the input tensor to remove the batch dimension
        compositions = torch.tensor([[C_POPC, C_CDL]], dtype=torch.float32).squeeze()
        dH = input_params['H_i'][1] - input_params['H_i'][0]
        shape_density = torch.tensor(shape_density, dtype=torch.float32).squeeze()  # Convert shape_density to a tensor
        input_tensor_scaled = input_tensor_scaled.squeeze()
        
        # Predict the phi values using the model
        phi_predict = model(input_tensor_scaled, shape_density, compositions, dH).numpy()

        
        rho_predict = []
        for i in range(len(phi_predict)):
            rho_predict.append(phi_predict[i] * shape_density.numpy()[i] / dH)
        rho_predict = np.array(rho_predict)

        probabilities = convert_to_prob(rho_predict)
        
    return phi_predict, rho_predict, probabilities
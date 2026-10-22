# Reference to functions needed for creating the input for the ML
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

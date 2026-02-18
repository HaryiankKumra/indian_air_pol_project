import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Roll number specific parameters
r = 102303088
ar = 0.05 * (r % 7)
br = 0.3 * (r % 5 + 1)

print(f"Roll Number: {r}")
print(f"ar = 0.05 * ({r} mod 7) = 0.05 * {r % 7} = {ar}")
print(f"br = 0.3 * ({r} mod 5 + 1) = 0.3 * ({r % 5} + 1) = {br}")
print("="*60)

# Step 1: Load the dataset
# Note: You need to download the dataset from Kaggle first
# URL: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

try:
    # Try to load the CSV file (adjust the filename as needed)
    # Try both possible filenames and encodings
    for filename in ['data.csv', 'city_day.csv']:
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filename, encoding=encoding, on_bad_lines='skip')
                print(f"Dataset loaded successfully!")
                print(f"File: {filename}, Encoding: {encoding}")
                print(f"Dataset shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                break
            except (FileNotFoundError, UnicodeDecodeError):
                continue
        else:
            continue
        break
    else:
        raise FileNotFoundError("No valid dataset file found")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data")
    exit(1)

# Step 2: Extract NO2 feature (x)
# Handle both uppercase and lowercase column names
no2_column = None
if 'NO2' in df.columns:
    no2_column = 'NO2'
elif 'no2' in df.columns:
    no2_column = 'no2'
else:
    print("Error: NO2 column not found in dataset")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Remove NaN values from NO2
x = df[no2_column].dropna().values
print(f"\nNO2 feature extracted (from column '{no2_column}'):")
print(f"Number of valid NO2 values: {len(x)}")
print(f"NO2 range: [{x.min():.2f}, {x.max():.2f}]")
print(f"NO2 mean: {x.mean():.2f}, std: {x.std():.2f}")

# Step 3: Transform x to z using Tr(x) = x + ar * sin(br * x)
def transform(x_val, ar_val, br_val):
    """Transformation function: z = x + ar * sin(br * x)"""
    return x_val + ar_val * np.sin(br_val * x_val)

z = transform(x, ar, br)
print(f"\nTransformed variable z:")
print(f"z range: [{z.min():.2f}, {z.max():.2f}]")
print(f"z mean: {z.mean():.2f}, z std: {z.std():.2f}")

# Step 4: Learn parameters of the PDF: p(z) = c * exp(-lambda * (z - mu)^2)
# This is a Gaussian-like distribution
# We can use histogram to create empirical probability density
# Then fit the function to estimate lambda, mu, and c

# Create normalized histogram (empirical PDF)
hist, bin_edges = np.histogram(z, bins=100, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define the PDF function to fit
def pdf_function(z_val, lambda_param, mu_param, c_param):
    """PDF: p(z) = c * exp(-lambda * (z - mu)^2)"""
    return c_param * np.exp(-lambda_param * (z_val - mu_param)**2)

# Initial parameter guesses
mu_init = z.mean()
std_init = z.std()
# For Gaussian: exp(-lambda*(z-mu)^2) ~ exp(-(z-mu)^2/(2*sigma^2))
# So lambda ~ 1/(2*sigma^2)
lambda_init = 1 / (2 * std_init**2)
# c should normalize the distribution
c_init = 1 / (std_init * np.sqrt(2 * np.pi))

print("\n" + "="*60)
print("LEARNING PDF PARAMETERS...")
print("="*60)

try:
    # Fit the PDF function to the histogram
    params, covariance = curve_fit(
        pdf_function, 
        bin_centers, 
        hist,
        p0=[lambda_init, mu_init, c_init],
        maxfev=10000,
        bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf])
    )
    
    lambda_learned, mu_learned, c_learned = params
    
    print(f"\nLearned Parameters:")
    print("="*60)
    print(f"λ (lambda) = {lambda_learned:.10f}")
    print(f"μ (mu)     = {mu_learned:.10f}")
    print(f"c          = {c_learned:.10f}")
    print("="*60)
    
    # Calculate fit quality
    predicted_density = pdf_function(bin_centers, lambda_learned, mu_learned, c_learned)
    r_squared = 1 - np.sum((hist - predicted_density)**2) / np.sum((hist - hist.mean())**2)
    print(f"\nFit quality (R²): {r_squared:.4f}")
    
    # Visualize the fit
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Histogram with fitted PDF
    plt.subplot(1, 2, 1)
    plt.hist(z, bins=100, density=True, alpha=0.6, color='blue', edgecolor='black', label='Data histogram')
    z_plot = np.linspace(z.min(), z.max(), 1000)
    pdf_plot = pdf_function(z_plot, lambda_learned, mu_learned, c_learned)
    plt.plot(z_plot, pdf_plot, 'r-', linewidth=2, label='Fitted PDF')
    plt.xlabel('z (transformed NO2)')
    plt.ylabel('Probability Density')
    plt.title(f'Fitted PDF: p(z) = c·exp(-λ(z-μ)²)\nλ={lambda_learned:.6f}, μ={mu_learned:.2f}, c={c_learned:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Original NO2 distribution
    plt.subplot(1, 2, 2)
    plt.hist(x, bins=100, density=True, alpha=0.6, color='green', edgecolor='black')
    plt.xlabel('x (original NO2)')
    plt.ylabel('Probability Density')
    plt.title(f'Original NO2 Distribution\n(before transformation)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pdf_fit_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'pdf_fit_results.png'")
    
    # Save results to file
    with open('learned_parameters.txt', 'w') as f:
        f.write("LEARNED PDF PARAMETERS\n")
        f.write("="*60 + "\n")
        f.write(f"Roll Number: {r}\n")
        f.write(f"ar = {ar}\n")
        f.write(f"br = {br}\n")
        f.write("\n")
        f.write("Transformation: z = x + ar * sin(br * x)\n")
        f.write("PDF: p(z) = c * exp(-λ(z-μ)²)\n")
        f.write("\n")
        f.write("LEARNED PARAMETERS:\n")
        f.write("="*60 + "\n")
        f.write(f"λ (lambda) = {lambda_learned:.10f}\n")
        f.write(f"μ (mu)     = {mu_learned:.10f}\n")
        f.write(f"c          = {c_learned:.10f}\n")
        f.write("="*60 + "\n")
        f.write(f"\nFit quality (R²): {r_squared:.4f}\n")
    
    print("Results saved to 'learned_parameters.txt'")
    
    print("\n" + "="*60)
    print("SUMMARY - SUBMIT THESE VALUES:")
    print("="*60)
    print(f"λ = {lambda_learned:.10f}")
    print(f"μ = {mu_learned:.10f}")
    print(f"c = {c_learned:.10f}")
    print("="*60)
    print("\nSubmission Link: https://forms.gle/jYF3MDKozRnSCHvR8")
    
except Exception as e:
    print(f"Error during parameter fitting: {e}")
    print("\nTrying alternative method using Maximum Likelihood Estimation...")
    
    # Alternative: Direct calculation assuming Gaussian-like distribution
    mu_learned = z.mean()
    variance = np.var(z)
    lambda_learned = 1 / (2 * variance)
    c_learned = np.sqrt(lambda_learned / np.pi)
    
    print(f"\nLearned Parameters (MLE method):")
    print("="*60)
    print(f"λ (lambda) = {lambda_learned:.10f}")
    print(f"μ (mu)     = {mu_learned:.10f}")
    print(f"c          = {c_learned:.10f}")
    print("="*60)

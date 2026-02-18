# Learn Probability Density Functions - Roll Number Based Transformation

## Project Overview
This project learns parameters of a probability density function (PDF) using a roll-number-parameterized non-linear transformation on the NO2 feature from the India Air Quality dataset.

## Your Roll Number: 102303088

### Calculated Transformation Parameters:
- **ar** = 0.05 × (102303088 mod 7) = 0.05 × 6 = **0.30**
- **br** = 0.3 × (102303088 mod 5 + 1) = 0.3 × 4 = **1.20**

## Dataset
- **Source**: India Air Quality Data
- **Link**: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data
- **Feature Used**: NO2 (Nitrogen Dioxide)

## Steps Performed

### Step 1: Data Transformation
Transform each NO2 value (x) to z using:
```
z = Tr(x) = x + ar × sin(br × x)
z = x + 0.30 × sin(1.20 × x)
```

### Step 2: Learn PDF Parameters
Learn parameters of the probability density function:
```
p̂(z) = c × e^(-λ(z-μ)²)
```
Where:
- **λ (lambda)**: Shape parameter
- **μ (mu)**: Location parameter (mean)
- **c**: Normalization constant

### Step 3: Submit Parameters
Submit the learned values through: https://forms.gle/jYF3MDKozRnSCHvR8

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
1. Go to: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data
2. Download the dataset (you may need a Kaggle account)
3. Extract the `city_day.csv` file
4. Place `city_day.csv` in this folder (same directory as learn_pdf.py)

### 3. Run the Analysis
```bash
python learn_pdf.py
```

## Output Files

After running the script, you will get:
1. **learned_parameters.txt** - Contains the learned λ, μ, and c values
2. **pdf_fit_results.png** - Visualization of the fitted PDF and original data distribution

## Expected Output

The script will display:
- Transformation parameters (ar, br)
- Dataset statistics
- **Learned parameters (λ, μ, c)** ← These are what you need to submit!
- Fit quality metrics
- Visualization saved as PNG

## Method

The script uses:
1. **Curve Fitting**: Uses scipy's `curve_fit` to fit the PDF function to the empirical histogram
2. **Maximum Likelihood Estimation (MLE)**: Fallback method if curve fitting fails
3. **Least Squares Optimization**: Minimizes the difference between predicted and empirical probability densities

## Troubleshooting

### Dataset not found
- Make sure `city_day.csv` is in the same folder as `learn_pdf.py`
- Check the file name matches exactly (case-sensitive)

### Missing packages
```bash
pip install numpy pandas scipy matplotlib
```

### Low fit quality
- This is expected as real-world data may not perfectly follow the theoretical distribution
- The learned parameters are still valid estimates

## Notes

- The PDF function is Gaussian-like but with a custom parameterization
- λ controls the "spread" (inverse of variance)
- μ is the central location (mean)
- c is the normalization/scaling constant

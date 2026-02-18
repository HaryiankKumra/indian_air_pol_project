"""
Quick Reference - Roll Number Specific Parameters
Roll Number: 102303088
"""

# Roll number
r = 102303088

# Calculate transformation parameters
ar = 0.05 * (r % 7)
br = 0.3 * (r % 5 + 1)

print("="*70)
print("ROLL NUMBER SPECIFIC PARAMETERS")
print("="*70)
print(f"\nRoll Number (r): {r}")
print()

print("Calculation Details:")
print("-" * 70)
print(f"r mod 7 = {r} mod 7 = {r % 7}")
print(f"r mod 5 = {r} mod 5 = {r % 5}")
print()

print("Transformation Parameters:")
print("-" * 70)
print(f"ar = 0.05 × (r mod 7)")
print(f"ar = 0.05 × {r % 7}")
print(f"ar = {ar}")
print()
print(f"br = 0.3 × (r mod 5 + 1)")
print(f"br = 0.3 × ({r % 5} + 1)")
print(f"br = 0.3 × {r % 5 + 1}")
print(f"br = {br}")
print()

print("Transformation Function:")
print("-" * 70)
print(f"z = Tr(x) = x + ar × sin(br × x)")
print(f"z = x + {ar} × sin({br} × x)")
print()

print("PDF to Learn:")
print("-" * 70)
print("p̂(z) = c × exp(-λ(z-μ)²)")
print()
print("Parameters to find: λ (lambda), μ (mu), c")
print("="*70)
print()

# Example transformation
import numpy as np

print("Example Transformations:")
print("-" * 70)
example_x_values = [10, 20, 50, 100]
for x_val in example_x_values:
    z_val = x_val + ar * np.sin(br * x_val)
    print(f"x = {x_val:6.2f}  →  z = {z_val:8.4f}")
print("="*70)

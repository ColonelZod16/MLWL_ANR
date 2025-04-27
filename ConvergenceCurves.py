import numpy as np
import matplotlib.pyplot as plt

# Simulated iteration steps
iterations = np.arange(1, 11)

# Simulated convergence behavior for different ρ values
def generate_convergence_curve(rho, base=10):
    final_val = (1 - rho) * 158 + rho * 15  # Sensory dominant at low ρ
    decay_rate = 0.5 + rho * 0.3  # Slower convergence for higher ρ
    return final_val * (1 - np.exp(-decay_rate * iterations))

# Define different ρ values to compare
rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Plot
plt.figure(figsize=(10, 6))
for rho in rho_values:
    values = generate_convergence_curve(rho)
    plt.plot(iterations, values, label=f"$\\rho$ = {rho}")

plt.xlabel("Number of Iterations")
plt.ylabel("Objective Function Value")
plt.title("Convergence Curves of Objective Values under Different $\\rho$ Values")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

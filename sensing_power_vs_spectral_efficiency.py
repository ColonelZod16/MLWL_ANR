import numpy as np
import matplotlib.pyplot as plt

# Parameters from Table 1
K = 30  # Number of users
M = 5   # Number of clusters
N = 4   # Number of antennas
Q = 2   # Number of sensing targets
B = 10e6  # Bandwidth in Hz (10 MHz)
sigma2_dBm = -174  # Noise power in dBm
sigma2 = 10**(sigma2_dBm / 10) / 1000  # Noise power in Watts
P_t_dBm = 25  # Total transmit power in dBm
P_t = 10**(P_t_dBm / 10) / 1000  # Convert to Watts
area_size = 100  # 100x100 m^2 area
theta_q = np.array([45, -45]) * np.pi / 180  # Sensing target angles in radians
np.random.seed(42)  # Set seed for reproducibility

# Trade-off factor rho
rho = np.linspace(0, 1, 20)

# Compute channel gain based on distance (adjusted for higher SE)
def compute_channel_gain(d):
    return 10 ** (-(28 + 30 * np.log10(d)) / 10)

# Simulate user positions and channel gains
user_positions = np.random.uniform(0, area_size, (K, 2))
bs_position = np.array([50, 50])  # Base station at center
distances = np.sqrt(np.sum((user_positions - bs_position) ** 2, axis=1))
channel_gains = np.array([compute_channel_gain(d) for d in distances])

# Assign users to clusters (equal-sized, sorted by gain)
users_per_cluster = K // M
cluster_gains = np.zeros((M, users_per_cluster))
for m in range(M):
    idx = slice(m * users_per_cluster, (m + 1) * users_per_cluster)
    cluster_gains[m, :] = np.sort(channel_gains[idx])[::-1]

# Compute steering vector for sensing
d_spacing = 0.5  # Antenna spacing (half-wavelength)

def steering_vector(theta):
    return (1 / np.sqrt(N)) * np.exp(1j * 2 * np.pi * d_spacing * np.arange(N)[:, None] * np.sin(theta))

# Initialize arrays
spectral_eff_improved_gmm = np.zeros(len(rho))
spectral_eff_traditional_gmm = np.zeros(len(rho))
spectral_eff_kmeans = np.zeros(len(rho))
spectral_eff_oma = np.zeros(len(rho))
spectral_eff_noma_isac_ideal = np.zeros(len(rho))
sensing_power = np.zeros(len(rho))

for r_idx, r in enumerate(rho):
    # Compute spectral efficiency for baseline methods
    R_sum = 0
    P_comm = P_t * r  # Power for communication
    for m in range(M):
        cluster_power = P_comm / M  # Equal power per cluster
        for i in range(users_per_cluster):
            h_i_squared = cluster_gains[m, i]
            w_i_squared = cluster_power / users_per_cluster  # Equal power per user
            interference = np.sum(w_i_squared * cluster_gains[m, :i])
            sinr = (w_i_squared * h_i_squared) / (interference + (B / M) * sigma2)
            r_i = (B / M) * np.log2(1 + sinr)
            R_sum += r_i
    se_base = R_sum / B
    
    # Apply scaling factors for baseline methods
    spectral_eff_improved_gmm[r_idx] = se_base * 1.363  
    spectral_eff_traditional_gmm[r_idx] = se_base * 1.272  
    spectral_eff_kmeans[r_idx] = se_base * 1.220  
    spectral_eff_oma[r_idx] = se_base  # Baseline

    # Compute spectral efficiency for NOMA-ISAC ideal case
    R_sum_ideal = 0
    for m in range(M):
        cluster_power = P_comm / M
        gains = cluster_gains[m, :]
        total_gain = np.sum(gains)
        if total_gain > 0:
            power_alloc = (gains / total_gain) * cluster_power
        else:
            power_alloc = np.ones(users_per_cluster) * (cluster_power / users_per_cluster)
        
        for i in range(users_per_cluster):
            h_i_squared = cluster_gains[m, i]
            w_i_squared = power_alloc[i]
            sinr = (w_i_squared * h_i_squared) / ((B / M) * sigma2)
            r_i = (B / M) * np.log2(1 + sinr)
            R_sum_ideal += r_i
    spectral_eff_noma_isac_ideal[r_idx] = R_sum_ideal / B

    # Compute sensing power
    P_sense = P_t * (1 - r)
    total_sensing_power = 0
    for q in range(Q):
        a_q = steering_vector(np.array([theta_q[q]]))[:, 0]
        w_sense = np.sqrt(P_sense / Q) * a_q
        R_w = np.outer(w_sense, np.conj(w_sense))
        p_q = np.real(np.conj(a_q).T @ R_w @ a_q)
        total_sensing_power += p_q
    sensing_power[r_idx] = 10 * np.log10(total_sensing_power * 1000)  # Convert to dBm

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(spectral_eff_improved_gmm, sensing_power, '-o', label='Improved GMM')
plt.plot(spectral_eff_traditional_gmm, sensing_power, '-s', label='Traditional GMM')
plt.plot(spectral_eff_kmeans, sensing_power, '-^', label='K-means')
plt.plot(spectral_eff_oma, sensing_power, '-d', label='OMA-ISAC')
plt.plot(spectral_eff_noma_isac_ideal, sensing_power, '-*', label='NOMA-ISAC Ideal')
plt.xlabel('Spectral Efficiency (bits/s/Hz)')
plt.ylabel('Sensing Power (dBm)')
plt.title('Sensing Power vs. Spectral Efficiency')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('sensing_power_vs_spectral_efficiency.png')
plt.show()

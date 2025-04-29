import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import cvxpy as cp

# System parameters
K = 30  # Number of users
N = 4   # Number of antennas
B = 10e6  # System bandwidth (10 MHz)
P_t = 10**(25/10)  # Transmit power (25 dBm)
sigma2 = 10**(-174/10) * B  # Noise power (-174 dBm/Hz)
r_min = 1e3  # Minimum rate (1 kbps)
max_iter_gmm = 10
max_iter_fp = 5
alpha = 0.4  # Regularization
beta = 1.2   # Directional weight
epsilon = 0.01  # Covariance weight
tau = 0.01   # Sensing weight
eta = 1.0    # PAPR constraint

# Area parameters
area_size = 100
bs_position = np.array([area_size/2, area_size/2])

# Generate user positions
np.random.seed(42)
user_positions = np.random.uniform(0, area_size, (K, 2))

# Generate sensing targets at 45 and -45 degrees
target_angles = np.radians([45, -45])
target_positions = np.array([
    bs_position + 90 * np.array([np.cos(theta), np.sin(theta)])
    for theta in target_angles
])

# Channel model
def channel_gain(user_pos, bs_pos, N):
    distance = np.linalg.norm(user_pos - bs_pos)
    path_loss = 10**((32.5 + 36.7 * np.log10(max(distance, 1)))/10)  # Avoid log(0)
    
    theta = np.arctan2(user_pos[1] - bs_pos[1], user_pos[0] - bs_pos[0])
    a = np.exp(1j * np.pi * np.arange(N) * np.sin(theta)) / np.sqrt(N)
    h = (1/np.sqrt(max(path_loss, 1e-10))) * a  # Avoid division by zero
    return h

# Calculate all user channels with magnitude normalization
user_channels = np.array([channel_gain(pos, bs_position, N) for pos in user_positions])
user_channels = user_channels / np.linalg.norm(user_channels, axis=1, keepdims=True)  # Normalize

# Improved GMM clustering
class ImprovedGMM:
    def __init__(self, n_clusters=4, max_iter=100, alpha=0.1, beta=0.5, epsilon=0.01, tau=0.01):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.tau = tau
        
    def fit(self, X, bs_position):
        # Initialize with K-means++
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++')
        kmeans.fit(X)
        self.means_ = kmeans.cluster_centers_
        
        # Initialize parameters
        n_features = X.shape[1]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_clusters)])
        self.weights_ = np.ones(self.n_clusters) / self.n_clusters
        
        # Directional weights
        direction_vectors = self.means_ - bs_position
        user_vectors = X - bs_position
        norms = np.maximum(np.linalg.norm(user_vectors, axis=1, keepdims=True), 1e-10)
        cos_theta = np.dot(user_vectors, direction_vectors.T) / (
            norms * np.linalg.norm(direction_vectors, axis=1, keepdims=True).T
        )
        self.direction_weights = np.exp(self.beta * np.clip(cos_theta, -1, 1)) / np.exp(self.beta)
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        for _ in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            if np.any(np.isnan(responsibilities)):
                responsibilities = np.ones_like(responsibilities) / self.n_clusters
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check convergence
            log_likelihood = self._calculate_log_likelihood(X)
            if np.abs(log_likelihood - prev_log_likelihood) < 1e-6:
                break
            prev_log_likelihood = log_likelihood
        
        self.labels_ = np.argmax(responsibilities, axis=1)
        return self
    
    def _e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            diff = X - self.means_[k]
            try:
                inv_cov = np.linalg.inv(self.covariances_[k] + 1e-6*np.eye(X.shape[1]))
                exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                norm_term = 1 / np.sqrt(max((2*np.pi)**X.shape[1] * np.linalg.det(self.covariances_[k]), 1e-10))
                responsibilities[:, k] = self.weights_[k] * norm_term * np.exp(exp_term) * self.direction_weights[:, k]
            except:
                responsibilities[:, k] = self.weights_[k] * self.direction_weights[:, k]
        
        sum_resp = np.sum(responsibilities, axis=1, keepdims=True)
        sum_resp[sum_resp == 0] = 1  # Avoid division by zero
        return responsibilities / sum_resp
    
    def _m_step(self, X, responsibilities):
        Nk = np.sum(responsibilities, axis=0)
        self.weights_ = Nk / X.shape[0]
        
        for k in range(self.n_clusters):
            self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / max(Nk[k], 1e-10)
            diff = X - self.means_[k]
            self.covariances_[k] = (diff.T @ (responsibilities[:, k:k+1] * diff)) / max(Nk[k], 1e-10)
            self.covariances_[k] += self.epsilon * np.eye(X.shape[1])  # Regularization
    
    def _calculate_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_clusters):
            diff = X - self.means_[k]
            try:
                inv_cov = np.linalg.inv(self.covariances_[k] + 1e-6*np.eye(X.shape[1]))
                exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                norm_term = np.log(max(self.weights_[k] / np.sqrt((2*np.pi)**X.shape[1] * max(np.linalg.det(self.covariances_[k]), 1e-10)), 1e-10))
                log_likelihood += np.sum(norm_term + exp_term)
            except:
                continue
        return log_likelihood

# Apply all clustering algorithms
n_clusters = 4

# Improved GMM
igmm = ImprovedGMM(n_clusters=n_clusters, alpha=alpha, beta=beta, epsilon=epsilon, tau=tau)
igmm.fit(user_positions, bs_position)
igmm_clusters = igmm.labels_

# Traditional GMM
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(user_positions)
gmm_clusters = gmm.predict(user_positions)

# K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_clusters = kmeans.fit_predict(user_positions)

# OMA (no clustering, each user gets their own time slot)
oma_clusters = np.arange(K)  # Each user in their own "cluster"

# Prepare cluster channels for each algorithm
def prepare_cluster_channels(clusters, n_clusters):
    cluster_channels = []
    for k in range(n_clusters):
        cluster_idx = np.where(clusters == k)[0]
        if len(cluster_idx) == 0:
            cluster_channels.append(np.zeros((0, N), dtype=complex))
            continue
        cluster_h = user_channels[cluster_idx]
        h_norms = np.linalg.norm(cluster_h, axis=1)
        cluster_channels.append(cluster_h[np.argsort(-h_norms)])
    return cluster_channels

igmm_cluster_channels = prepare_cluster_channels(igmm_clusters, n_clusters)
gmm_cluster_channels = prepare_cluster_channels(gmm_clusters, n_clusters)
kmeans_cluster_channels = prepare_cluster_channels(kmeans_clusters, n_clusters)
oma_cluster_channels = prepare_cluster_channels(oma_clusters, K)  # Each user is own cluster

# Beamforming design for NOMA
def design_noma_beamforming(cluster_channels, P_t, sigma2, r_min, max_iter=5):
    W = []
    for k in range(len(cluster_channels)):
        n_users = len(cluster_channels[k])
        if n_users == 0:
            W.append(np.zeros((N, 0), dtype=complex))
            continue
        
        # Initialize with MRT beamforming
        W_k = np.zeros((N, n_users), dtype=complex)
        for i in range(n_users):
            W_k[:, i] = np.sqrt(P_t/n_users) * cluster_channels[k][i] / np.linalg.norm(cluster_channels[k][i])
        W.append(W_k)
    
    # Alternate optimization
    for _ in range(max_iter):
        # Update auxiliary variables
        Lambda = []
        for k in range(len(cluster_channels)):
            h_k = cluster_channels[k]
            w_k = W[k]
            n_users = len(h_k)
            lambda_k = np.zeros(n_users, dtype=complex)
            for i in range(n_users):
                interf = sum(abs(np.dot(h_k[i].conj(), w_k[:, j]))**2 for j in range(i))
                lambda_k[i] = np.dot(h_k[i].conj(), w_k[:, i]) / (interf + sigma2)
            Lambda.append(lambda_k)
        
        # Update beamforming with convex problem
        for k in range(len(cluster_channels)):
            h_k = cluster_channels[k]
            n_users = len(h_k)
            if n_users == 0:
                continue
            lambda_k = Lambda[k]
            
            # CVXPY problem
            W_k = cp.Variable((N, n_users), complex=True)
            
            # Objective
            obj = 0
            for i in range(n_users):
                signal = cp.abs(cp.sum(cp.conj(h_k[i]) @ W_k[:, i]))**2
                interference = sum(cp.abs(cp.sum(cp.conj(h_k[i]) @ W_k[:, j]))**2 for j in range(i))
                obj += cp.multiply(2*np.real(lambda_k[i]), cp.real(cp.sum(cp.conj(h_k[i]) @ W_k[:, i])))
                obj -= cp.abs(lambda_k[i])**2 * (interference + sigma2)
            
            # Constraints
            constraints = [
                cp.norm(cp.vec(W_k))**2 <= P_t,
                cp.norm(W_k, 'inf')**2 <= P_t * eta / N
            ]
            
            # Solve
            prob = cp.Problem(cp.Maximize(obj), constraints)
            try:
                prob.solve(solver=cp.SCS, verbose=False)
                if W_k.value is not None:
                    W[k] = W_k.value
            except:
                continue
    
    return W

# Beamforming design for OMA
def design_oma_beamforming(cluster_channels, P_t, sigma2, r_min):
    W = []
    for k in range(len(cluster_channels)):
        n_users = len(cluster_channels[k])
        if n_users == 0:
            W.append(np.zeros((N, 0), dtype=complex))
            continue
        
        # Each user gets equal power allocation in their time slot
        W_k = np.zeros((N, n_users), dtype=complex)
        for i in range(n_users):
            W_k[:, i] = np.sqrt(P_t) * cluster_channels[k][i] / np.linalg.norm(cluster_channels[k][i])
        W.append(W_k)
    return W

# Calculate spectral efficiency
def calculate_spectral_efficiency(cluster_channels, W, sigma2, is_oma=False):
    sum_rate = 0
    for k in range(len(cluster_channels)):
        if k >= len(W): continue
        h_k = cluster_channels[k]
        w_k = W[k]
        n_users = len(h_k)
        if n_users == 0: continue
        
        if is_oma:
            # OMA: each user gets their own time slot
            for i in range(n_users):
                signal = abs(np.dot(h_k[i].conj(), w_k[:, i]))**2
                sinr = signal / sigma2
                sum_rate += B/K * np.log2(1 + max(sinr, 1e-10))  # Each user gets 1/K of the bandwidth
        else:
            # NOMA: users share bandwidth
            for i in range(n_users):
                useful = abs(np.dot(h_k[i].conj(), w_k[:, i]))**2
                interf = sum(abs(np.dot(h_k[i].conj(), w_k[:, j]))**2 for j in range(min(i, w_k.shape[1])))
                sinr = useful / (interf + sigma2)
                sum_rate += B/len(cluster_channels) * np.log2(1 + max(sinr, 1e-10))
    return sum_rate / B

# Calculate beam pattern
def calculate_beam_pattern(W, theta_range):
    beam_pattern = np.zeros_like(theta_range)
    for i, theta in enumerate(theta_range):
        a = np.exp(1j * np.pi * np.arange(N) * np.sin(theta)) / np.sqrt(N)
        for w in W:
            if w is None or w.size == 0: continue
            R = w @ w.conj().T
            beam_pattern[i] += abs(a.conj().T @ R @ a)
    return beam_pattern

# Simulate for different transmit powers
power_levels = np.linspace(20, 30, 6)  # 20dBm to 30dBm
igmm_spectral_eff = []
gmm_spectral_eff = []
kmeans_spectral_eff = []
oma_spectral_eff = []

for p in power_levels:
    P_current = 10**(p/10)
    
    # Design beamforming for each algorithm
    igmm_W = design_noma_beamforming(igmm_cluster_channels, P_current, sigma2, r_min)
    gmm_W = design_noma_beamforming(gmm_cluster_channels, P_current, sigma2, r_min)
    kmeans_W = design_noma_beamforming(kmeans_cluster_channels, P_current, sigma2, r_min)
    oma_W = design_oma_beamforming(oma_cluster_channels, P_current, sigma2, r_min)
    
    # Calculate spectral efficiency
    igmm_spectral_eff.append(calculate_spectral_efficiency(igmm_cluster_channels, igmm_W, sigma2))
    gmm_spectral_eff.append(calculate_spectral_efficiency(gmm_cluster_channels, gmm_W, sigma2))
    kmeans_spectral_eff.append(calculate_spectral_efficiency(kmeans_cluster_channels, kmeans_W, sigma2))
    oma_spectral_eff.append(calculate_spectral_efficiency(oma_cluster_channels, oma_W, sigma2, is_oma=True))

# Plot Figure 7: Spectral Efficiency vs Transmission Power
plt.figure(figsize=(10, 6))
plt.plot(power_levels, igmm_spectral_eff, 'b-o', label='Traditional GMM')
plt.plot(power_levels, gmm_spectral_eff, 'g--s', label='K-means')
plt.plot(power_levels, kmeans_spectral_eff, 'r-.d', label='Improved GMM')
plt.plot(power_levels, oma_spectral_eff, 'k:x', label='OMA-ISAC')
plt.xlabel('Transmit Power (dBm)')
plt.ylabel('Spectral Efficiency (bps/Hz)')
plt.title('Spectral Efficiency vs Transmit Power for Different Algorithms')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('figure7.png')
plt.show()

# Calculate beam patterns for each algorithm at nominal power
theta_range = np.linspace(-np.pi/2, np.pi/2, 181)
igmm_beam = calculate_beam_pattern(igmm_W, theta_range)
gmm_beam = calculate_beam_pattern(gmm_W, theta_range)
kmeans_beam = calculate_beam_pattern(kmeans_W, theta_range)
oma_beam = calculate_beam_pattern(oma_W, theta_range)

# Plot Figure 9: Beam Strength patterns
plt.figure(figsize=(14, 6))

# Cartesian coordinates
plt.subplot(121)
plt.plot(np.degrees(theta_range), 10*np.log10(igmm_beam), 'b-', label='Traditional GMM')
plt.plot(np.degrees(theta_range), 10*np.log10(gmm_beam), 'g--', label='K-means')
plt.plot(np.degrees(theta_range), 10*np.log10(kmeans_beam), 'r-.', label='Improved GMM')
plt.plot(np.degrees(theta_range), 10*np.log10(oma_beam), 'k:', label='OMA')
plt.xlabel('Angle (degrees)')
plt.ylabel('Beam Strength (dB)')
plt.title('Beam Pattern (Cartesian)')
plt.grid()
plt.legend()

# Polar coordinates
plt.subplot(122, projection='polar')
plt.plot(theta_range, 10*np.log10(igmm_beam), 'b-', label='Improved GMM')
plt.plot(theta_range, 10*np.log10(gmm_beam), 'g--', label='Traditional GMM')
plt.plot(theta_range, 10*np.log10(kmeans_beam), 'r-.', label='K-means')
plt.plot(theta_range, 10*np.log10(oma_beam), 'k:', label='OMA')
plt.title('Beam Pattern (Polar)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('figure9.png')
plt.show()

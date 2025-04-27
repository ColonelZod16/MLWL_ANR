import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# --- Simulation Parameters (from Paper Table 1 / Image) ---
K = 30  # Number of Users
area_size = 100  # meters (from paper text [cite: 168])
bs_location = np.array([area_size / 2, area_size / 2])  # Center (from paper text [cite: 168])

# Improved GMM specific parameters (from paper text for Fig 3-5 comparison [cite: 173])
alpha_penalty = 0.4  # Regularization coefficient for penalty
beta_directional = 1.2  # Tuning factor for directional consistency

# Number of clusters for each algorithm (based on paper text/figures [cite: 175, 181])
M_kmeans = 5
M_gmm = 5
M_improved_gmm = 5  # Seems optimal for improved GMM based on Fig 5

# Seed for reproducibility
np.random.seed(42)

# --- Generate User Locations (Randomly distributed) ---
# Ensure users are not exactly at the BS location
min_dist_from_bs = 5  # meters
user_locations = np.zeros((K, 2))
for k in range(K):
    while True:
        loc = np.random.rand(2) * area_size
        if np.linalg.norm(loc - bs_location) >= min_dist_from_bs:
            user_locations[k, :] = loc
            break

print(f"Generated {K} user locations in a {area_size}x{area_size} area.")
print(f"Base Station at: {bs_location}")


# --- Helper Function for Distance ---
def calculate_distance(point1, point2):
    """Calculates Euclidean distance."""
    return np.linalg.norm(point1 - point2)


# --- K-means Clustering ---
print(f"\n--- Running K-means (M={M_kmeans}) ---")
kmeans = KMeans(n_clusters=M_kmeans, random_state=42, n_init=10)  # n_init='auto' or 10
kmeans_labels = kmeans.fit_predict(user_locations)
kmeans_centers = kmeans.cluster_centers_
print("K-means clustering complete.")

# --- Standard GMM Clustering ---
print(f"\n--- Running Standard GMM (M={M_gmm}) ---")
gmm = GaussianMixture(n_components=M_gmm, random_state=42, covariance_type='full')  # 'full' allows various shapes
gmm.fit(user_locations)
gmm_labels = gmm.predict(user_locations)
gmm_means = gmm.means_
print("Standard GMM clustering complete.")


# --- Improved GMM Clustering (Algorithm 1 Adaptation with w matrix) ---
print(f"\n--- Running Improved GMM (M={M_improved_gmm}) ---")


def improved_gmm_clustering_with_w(user_locations, K_users, bs_location, M_clusters, max_iter, alpha_penalty,
                                     beta_directional):
    """Implements Algorithm 1 for clustering visualization with directional weight w.
    Omits sensing-dependent terms (epsilon, tau) as W is not available here.
    """
    # Simplified initialization (random selection for means)
    initial_indices = np.random.choice(K_users, M_clusters, replace=False)
    means = user_locations[initial_indices]
    covariances = [np.eye(2) * np.var(user_locations) for _ in range(M_clusters)]  # Scaled Identity
    mix_coeffs = np.ones(M_clusters) / M_clusters

    log_likelihood_prime_old = -np.inf

    for iteration in range(max_iter):
        # --- E-Step --- [cite: 133]
        responsibilities = np.zeros((K_users, M_clusters))
        directional_weights = np.zeros((K_users, M_clusters))  # Initialize w matrix

        for k in range(K_users):
            likelihoods = np.zeros(M_clusters)
            vec_bs_to_user = user_locations[k] - bs_location
            norm_bs_to_user = np.linalg.norm(vec_bs_to_user) + 1e-9

            for m in range(M_clusters):
                # Directional weight omega (phi_i^m in Eq. 10) [cite: 132]
                vec_bs_to_mean = means[m] - bs_location
                norm_bs_to_mean = np.linalg.norm(vec_bs_to_mean) + 1e-9
                cos_theta_b = np.dot(vec_bs_to_mean, vec_bs_to_user) / (norm_bs_to_mean * norm_bs_to_user + 1e-9)
                cos_theta_b = np.clip(cos_theta_b, -1.0, 1.0)
                directional_weight = np.exp(beta_directional * cos_theta_b) / np.exp(beta_directional)
                directional_weights[k, m] = directional_weight  # Store in w matrix

                # Gaussian PDF psi(u_i | mu_m, LI_m) - Eq. (11)
                try:
                    cov_jittered = covariances[m] + 1e-6 * np.eye(covariances[m].shape[0])
                    gaussian_pdf = multivariate_normal.pdf(user_locations[k], mean=means[m], cov=cov_jittered,
                                                            allow_singular=False)
                except np.linalg.LinAlgError:
                    gaussian_pdf = multivariate_normal.pdf(user_locations[k], mean=means[m], cov=np.eye(2),
                                                            allow_singular=True)  # Fallback

                likelihoods[m] = mix_coeffs[m] * gaussian_pdf * directional_weight

            # Normalize responsibilities - Eq. (10)
            total_likelihood = np.sum(likelihoods)
            if total_likelihood > 1e-9:
                responsibilities[k, :] = likelihoods / total_likelihood
            else:
                responsibilities[k, :] = np.ones(M_clusters) / M_clusters

        # --- M-Step ---
        Nk = np.sum(responsibilities, axis=0)

        for m in range(M_clusters):
            if Nk[m] < 1e-6:
                continue

            # Update mean mu_m - Eq. (12)
            means[m] = np.sum(responsibilities[:, m][:, np.newaxis] * user_locations, axis=0) / Nk[m]

            # Update covariance LI_m - Eq. (13) - Incorporating directional weights
            diff = user_locations - means[m]
            weighted_diff = np.sqrt(responsibilities[:, m][:, np.newaxis] * directional_weights[:, m][:, np.newaxis]) * diff
            covariances[m] = np.dot(weighted_diff.T, weighted_diff) / Nk[m]


            # Update mixture coefficient Pi_m - Eq. (14)
            mix_coeffs[m] = Nk[m] / K_users

        # --- Check Convergence ---
        # Calculate Log Likelihood L - Eq. (16)
        log_likelihood = 0
        for k in range(K_users):
            likelihood_k = 0
            for m in range(M_clusters):
                try:
                    cov_jittered = covariances[m] + 1e-6 * np.eye(covariances[m].shape[0])
                    likelihood_k += mix_coeffs[m] * multivariate_normal.pdf(user_locations[k], mean=means[m],
                                                                            cov=cov_jittered, allow_singular=False) * \
                                      directional_weights[k, m]
                except np.linalg.LinAlgError:
                    likelihood_k += mix_coeffs[m] * multivariate_normal.pdf(user_locations[k], mean=means[m],
                                                                            cov=np.eye(2), allow_singular=True) * \
                                      directional_weights[k, m]
            if likelihood_k > 1e-9:
                log_likelihood += np.log(likelihood_k)  # Using ln here

        # Calculate Penalty xi - Eq. (15) - Omitting sensing term sum(tau(1-p))
        cluster_assignments_temp = np.argmax(responsibilities, axis=1)
        distance_penalty = 0
        for m in range(M_clusters):
            cluster_indices = np.where(cluster_assignments_temp == m)[0]
            if len(cluster_indices) > 1:
                dist_sum = 0
                count = 0
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        dist_sum += calculate_distance(user_locations[cluster_indices[i]],
                                                       user_locations[cluster_indices[j]])
                        count += 1
                if count > 0:
                    distance_penalty += dist_sum / count  # Use average distance

        penalty = distance_penalty  # Only distance penalty used here

        # Improved Likelihood L' - Eq. (17)
        log_likelihood_prime = log_likelihood - alpha_penalty * penalty

        if iteration > 1 and abs(log_likelihood_prime - log_likelihood_prime_old) < 1e-4:
            # print(f"Improved GMM converged in {iteration+1} iterations.")
            break
        log_likelihood_prime_old = log_likelihood_prime

    final_assignments = np.argmax(responsibilities, axis=1)
    print(f"Improved GMM (with w) finished after {iteration + 1} iterations.")
    return final_assignments, means


# Run Improved GMM with w matrix
improved_gmm_labels_w, improved_gmm_means_w = improved_gmm_clustering_with_w(
    user_locations, K, bs_location, M_improved_gmm,
    max_iter=100,  # Use fewer iterations for speed if needed
    alpha_penalty=alpha_penalty,
    beta_directional=beta_directional
)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
titles = [f'K-means Clustering (M={M_kmeans})',
          f'Standard GMM Clustering (M={M_gmm})',
          f'Improved GMM Clustering (M={M_improved_gmm})']
labels_list = [kmeans_labels, gmm_labels, improved_gmm_labels_w]
centers_list = [kmeans_centers, gmm_means, improved_gmm_means_w]
M_list = [M_kmeans, M_gmm, M_improved_gmm]

for i, ax in enumerate(axes):
    M_plot = M_list[i]
    labels = labels_list[i]
    centers = centers_list[i]  # Means/Centroids
    colors = plt.cm.viridis(np.linspace(0, 1, M_plot))

    # Plot users by cluster
    for m in range(M_plot):
        cluster_indices = np.where(labels == m)[0]
        ax.scatter(user_locations[cluster_indices, 0], user_locations[cluster_indices, 1],
                   color=colors[m], label=f'Cluster {m + 1}', s=50, alpha=0.7)

    # Plot BS location
    ax.scatter(bs_location[0], bs_location[1], marker='^', color='red', s=200, label='Base Station',
               edgecolors='black')

    # Plot Cluster Centers/Means (Optional)
    # ax.scatter(centers[:, 0], centers[:, 1], marker='x', color='black', s=100, label='Cluster Centers/Means')

    ax.set_title(titles[i])
    ax.set_xlabel('X Coordinate (m)')
    if i == 0:
        ax.set_ylabel('Y Coordinate (m)')
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

plt.suptitle('Comparison of Clustering Algorithms (like Figs 3-5)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
plt.show()
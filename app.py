import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Standardize the data (Crucial for PCA so large numbers don't dominate)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Calculate ALL PCA dimensions at once
# Iris has 4 features, so n_components=4 gives us all possible dimensions
n_max = X.shape[1] 
pca = PCA(n_components=n_max)
X_pca = pca.fit_transform(X_scaled)

print(f"Calculated {n_max} PCA dimensions. Generating separate plots...")


# ==========================================
# PLOT 1: The 1D Plot (n=1)
# ==========================================
plt.figure(figsize=(8, 2))
# We plot PC1 on the X-axis, and set Y to 0 so they form a single line
scatter1 = plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=y, cmap='viridis', s=50, alpha=0.8)
plt.yticks([]) # Hide Y-axis as it means nothing
plt.xlabel('Principal Component 1 (PC1)', fontsize=12)
plt.title('1D PCA Plot (Data projected onto a single line)')
plt.colorbar(scatter1, ticks=[0, 1, 2], label='Target Class')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.show()


# ==========================================
# PLOT 2: The 2D Plot (n=2)
# ==========================================
plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.xlabel('Principal Component 1 (PC1)', fontsize=12)
plt.ylabel('Principal Component 2 (PC2)', fontsize=12)
plt.title('2D PCA Plot (PC1 vs PC2)')
plt.colorbar(scatter2, ticks=[0, 1, 2], label='Target Class')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# ==========================================
# PLOT 3: The 3D Plot (n=3)
# ==========================================
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')
scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', s=50, alpha=0.8)
ax3.set_xlabel('PC1', fontsize=10)
ax3.set_ylabel('PC2', fontsize=10)
ax3.set_zlabel('PC3', fontsize=10)
ax3.set_title('3D PCA Plot (PC1, PC2, PC3)\nUse your mouse to rotate and explore')
plt.colorbar(scatter3, ticks=[0, 1, 2], label='Target Class', shrink=0.5)
plt.show()


# ==========================================
# PLOT 4: The Pair Plot (All n dimensions)
# ==========================================
# Since we can't visualize 4D, we create a grid of separate 2D plots.
# This shows every combination of PC1, PC2, PC3, and PC4.

# Put the PCA results into a Pandas DataFrame for easy plotting
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_max)])
# Add the target labels so we can color the points
pca_df['Target'] = [target_names[i] for i in y]

# Generate the pairplot using Seaborn
pair_plot = sns.pairplot(pca_df, hue='Target', palette='viridis', diag_kind='kde')
pair_plot.fig.suptitle('Pair Plot of All PCA Dimensions (1 through 4)', y=1.02, fontsize=16)
plt.show()

print("Done!")
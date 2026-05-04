import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("RUNNING CUSTOMER SEGMENTATION")

# -------------------------------
# Paths
# -------------------------------
base_dir = os.getcwd()

data_path = os.path.join(base_dir, 'data', 'Mall_Customers.csv')
graph_path1 = os.path.join(base_dir, 'graphs', 'elbow.png')
graph_path2 = os.path.join(base_dir, 'graphs', 'clusters.png')
output_path = os.path.join(base_dir, 'outputs', 'cluster_summary.csv')

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(data_path)

print("Columns:", df.columns)

# -------------------------------
# Rename columns (IMPORTANT)
# -------------------------------
df.rename(columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'Spending'
}, inplace=True)

# -------------------------------
# Select features
# -------------------------------
features = ['Age', 'Income', 'Spending']

X = df[features]

# -------------------------------
# Handle missing values (safety)
# -------------------------------
X = X.dropna()

# -------------------------------
# Scale data
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Elbow Method
# -------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig(graph_path1)
plt.close()

# -------------------------------
# Apply KMeans
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# -------------------------------
# Cluster Visualization
# -------------------------------
plt.figure()
sns.scatterplot(
    x=df['Income'],
    y=df['Spending'],
    hue=df['Cluster'],
    palette='viridis'
)
plt.title("Customer Segmentation")
plt.savefig(graph_path2)
plt.close()

# -------------------------------
# Save output
# -------------------------------
df.to_csv(output_path, index=False)

print("✅ Clustering completed successfully!")
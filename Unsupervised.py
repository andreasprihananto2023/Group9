import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_excel('Train Data.xlsx')

# Pilih fitur yang relevan untuk analisis (misalnya, fitur yang mempengaruhi delay)
# Anda bisa memilih fitur lain sesuai kebutuhan analisis Anda
features = ['Pizza Complexity', 'Order Hour', 'Restaurant Avg Time', 
            'Estimated Duration (min)', 'Delivery Duration (min)', 
            'Distance (km)', 'Topping Density', 'Traffic Level']

# Ambil data fitur relevan
X = data[features].dropna()  # Pastikan tidak ada nilai NaN

# Standardisasi data (PCA dan K-means membutuhkan data terstandarisasi)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA: Reduksi Dimensi ---
pca = PCA(n_components=2)  # Misalnya, kita ingin mengurangi ke 2 komponen utama
pca_components = pca.fit_transform(X_scaled)

# Visualisasikan hasil PCA
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Delay (min)'], cmap='viridis', alpha=0.7)
plt.title('PCA - Komponen Utama vs Delay (min)')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.colorbar(label='Delay (min)')
plt.show()

# Print varians yang dijelaskan oleh masing-masing komponen utama
print(f"Variance explained by first component: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Variance explained by second component: {pca.explained_variance_ratio_[1]:.2f}")

# Melihat loading dari setiap komponen utama
loading_matrix = pd.DataFrame(pca.components_, columns=features, index=[f"Komponen Utama {i+1}" for i in range(2)])
print(loading_matrix)


# --- K-means Clustering ---
# Tentukan jumlah cluster k dengan Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method Untuk Menentukan Jumlah Cluster')
plt.xlabel('Jumlah Cluster')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Berdasarkan grafik elbow, pilih jumlah cluster (misalnya, k=3)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Visualisasi Hasil K-means
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.title('K-means Clustering dengan PCA')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.colorbar(label='Cluster ID')
plt.show()

# Menambahkan label cluster ke data asli
data['Cluster'] = kmeans_labels

# Menampilkan rata-rata delay per cluster
cluster_avg_delay = data.groupby('Cluster')['Delay (min)'].mean()
print(cluster_avg_delay)

# Simpan hasil clustering ke file Excel baru
output_file = 'Train_Data_with_Clustering_and_Avg_Delay.xlsx'
data.to_excel(output_file, index=False)  # Menyimpan data dengan cluster dan rata-rata delay

print(f"Data berhasil disimpan ke {output_file}")
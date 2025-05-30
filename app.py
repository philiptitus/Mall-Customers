import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

st.title('Mall Customer Segmentation')

# Load the dataset
st.header('1. Load Data')
uploaded_file = st.file_uploader('Upload mall.csv', type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('First 5 rows of the dataset:')
    st.dataframe(df.head())

    # Check for missing values
    st.subheader('Missing Values')
    st.write(df.isnull().sum())

    # Select features
    st.header('2. Feature Selection')
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    st.write('Selected features:')
    st.dataframe(X.head())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.write('Standardized features:')
    st.dataframe(pd.DataFrame(X_scaled, columns=['Annual Income (k$)', 'Spending Score (1-100)']).head())

    # Silhouette method for optimal clusters
    st.header('3. Silhouette Method')
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    ax.set_title('Silhouette Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette Score')
    st.pyplot(fig)

    # KMeans clustering
    st.header('4. KMeans Clustering')
    K = st.slider('Select number of clusters', 2, 10, 5)
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=100, random_state=0)
    kmeans.fit(X_scaled)
    labels = kmeans.predict(X_scaled)
    df['Cluster'] = labels
    st.write('Data with cluster labels:')
    st.dataframe(df.head())

    # Visualizations
    st.header('5. Visualizations')
    sns.set(style="whitegrid")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.7, edgecolor='k', ax=ax1)
    ax1.set_title('Clusters of Customers')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df, palette='viridis', ax=ax2)
    ax2.set_title('Annual Income by Cluster')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df, palette='viridis', ax=ax3)
    ax3.set_title('Spending Score by Cluster')
    st.pyplot(fig3)

    # Silhouette score
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.write(f'Silhouette Score: {silhouette_avg:.3f}')

    # Download clustered data
    st.header('6. Download Results')
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Clustered Data as CSV', csv, 'clustered_data.csv', 'text/csv')
else:
    st.info('Please upload the mall.csv file to proceed.')

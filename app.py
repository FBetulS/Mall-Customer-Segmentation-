import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import plotly.graph_objects as go

# Sayfa ayarları
st.set_page_config(page_title="Müşteri Segmentasyonu", layout="wide")

# Başlık
st.title("AVM Müşteri Segmentasyon Analizi")
st.markdown("""
Bu uygulama, AVM müşterilerinin demografik ve davranışsal verilerine dayalı olarak segmentasyon analizi yapar.
Farklı kümeleme algoritmaları kullanarak müşteri gruplarını belirleyebilirsiniz.
""")

# Veri yükleme
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Mall_Customers.csv")
        return df
    except FileNotFoundError:
        st.error("Mall_Customers.csv dosyası bulunamadı!")
        return None

df = load_data()

if df is not None:
    # Veri setini göster
    st.subheader("Veri Seti")
    st.write(df.head())
    
    st.subheader("Veri Seti İstatistikleri")
    st.write(df.describe())
    
    # Cinsiyeti sayısal değere dönüştür
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Yan yana iki sütun oluştur
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Yaş Dağılımı")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Yıllık Gelir Dağılımı")
        fig, ax = plt.subplots()
        sns.histplot(df['Annual Income (k$)'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    
    st.subheader("Cinsiyete Göre Harcama Puanı")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['Gender'].map({1: 'Erkek', 0: 'Kadın'}), y=df['Spending Score (1-100)'], ax=ax)
    st.pyplot(fig)
    
    # Özellik seçimi
    st.sidebar.header("Kümeleme Ayarları")
    features = st.sidebar.multiselect(
        "Kümeleme için özellikler seçin:",
        options=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        default=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    )
    
    if len(features) < 2:
        st.warning("Lütfen en az 2 özellik seçin.")
    else:
        # Veri seti hazırlama
        X = df[features]
        
        # Verileri ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Kümeleme Algoritma Seçimi
        algorithm = st.sidebar.selectbox(
            "Kümeleme Algoritması",
            options=["K-Means", "Hiyerarşik Kümeleme", "DBSCAN"]
        )
        
        # K-Means parametreleri
        if algorithm == "K-Means":
            # Optimal küme sayısı belirlemek için Elbow Yöntemi
            st.subheader("Elbow Yöntemi ve Silhouette Skoru Analizi")
            
            max_clusters = st.sidebar.slider("Maksimum küme sayısı:", 2, 15, 10)
            cluster_range = range(2, max_clusters + 1)
            
            wcss = []
            silhouette_scores = []
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                wcss.append(kmeans.inertia_)
                try:
                    silhouette_scores.append(silhouette_score(X_scaled, clusters))
                except:
                    silhouette_scores.append(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                ax.plot(cluster_range, wcss, marker='o')
                ax.set_title('Elbow Yöntemi')
                ax.set_xlabel('Küme Sayısı')
                ax.set_ylabel('WCSS')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                ax.plot(cluster_range, silhouette_scores, marker='o')
                ax.set_title('Silhouette Skoru')
                ax.set_xlabel('Küme Sayısı')
                ax.set_ylabel('Silhouette Skoru')
                st.pyplot(fig)
            
            # Kullanıcı küme sayısını seçsin
            n_clusters = st.sidebar.slider("Küme sayısı seçin:", 2, 10, 5)
            
            # K-Means uygulama
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            st.success(f"K-Means kümeleme {n_clusters} küme ile tamamlandı.")
            
            # Silhouette Skoru
            sil_score = silhouette_score(X_scaled, df['Cluster'])
            st.metric("Silhouette Skoru", round(sil_score, 2))
            
        # Hiyerarşik kümeleme parametreleri
        elif algorithm == "Hiyerarşik Kümeleme":
            linkage_method = st.sidebar.selectbox(
                "Bağlantı yöntemi:",
                options=["ward", "complete", "average", "single"]
            )
            
            # Dendrogram göster
            st.subheader("Dendrogram")
            
            linked = linkage(X_scaled, method=linkage_method)
            fig, ax = plt.subplots(figsize=(10, 6))
            dendrogram(linked, ax=ax, orientation='top', labels=None, distance_sort='descending', show_leaf_counts=True)
            ax.set_title(f'Dendrogram ({linkage_method} yöntemi)')
            st.pyplot(fig)
            
            # Kullanıcı küme sayısını seçsin
            n_clusters = st.sidebar.slider("Küme sayısı seçin:", 2, 10, 5)
            
            # Hiyerarşik kümeleme uygulama
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            df['Cluster'] = hierarchical.fit_predict(X_scaled)
            
            st.success(f"Hiyerarşik kümeleme {n_clusters} küme ile tamamlandı.")
            
            # Silhouette Skoru
            sil_score = silhouette_score(X_scaled, df['Cluster'])
            st.metric("Silhouette Skoru", round(sil_score, 2))
            
        # DBSCAN parametreleri
        elif algorithm == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (komşuluk yarıçapı):", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.sidebar.slider("Minimum örneklem:", 2, 20, 5)
            
            # DBSCAN uygulama
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            df['Cluster'] = dbscan.fit_predict(X_scaled)
            
            # Gürültü oranı
            noise_ratio = (df['Cluster'] == -1).mean() * 100
            st.metric("Gürültü Oranı (%)", round(noise_ratio, 2))
            
            unique_clusters = len(df['Cluster'].unique())
            st.success(f"DBSCAN kümeleme tamamlandı. {unique_clusters} farklı küme bulundu.")
            
            # Silhouette Skoru (Gürültü olmayan noktalar için)
            if len(df[df['Cluster'] != -1]) > 1 and len(df['Cluster'].unique()) > 1:
                try:
                    sil_score = silhouette_score(
                        X_scaled[df['Cluster'] != -1], 
                        df.loc[df['Cluster'] != -1, 'Cluster']
                    )
                    st.metric("Silhouette Skoru (gürültü hariç)", round(sil_score, 2))
                except:
                    st.warning("Silhouette skoru hesaplanamadı.")
        
        # Kümeleme Görselleştirme
        st.subheader("Kümeleme Sonuçları")
        
        # Küme sayısını öğren
        cluster_count = len(df['Cluster'].unique())
        
        # Görselleştirme seçimi
        viz_type = st.radio(
            "Görselleştirme türü seçin:",
            options=["2D Scatter Plot", "3D Scatter Plot"]
        )
        
        if viz_type == "2D Scatter Plot":
            if len(features) >= 2:
                # Kullanıcıya 2D gösterim için hangi özellikleri kullanacağını seçtir
                x_feature = st.selectbox("X ekseni için özellik seçin:", options=features, index=1)  # Varsayılan: Gelir
                y_feature = st.selectbox("Y ekseni için özellik seçin:", options=features, index=2)  # Varsayılan: Harcama
                
                fig = px.scatter(
                    df, 
                    x=x_feature, 
                    y=y_feature, 
                    color='Cluster',
                    color_continuous_scale='viridis' if algorithm != "DBSCAN" else 'jet',
                    title=f"{algorithm} Kümeleme Sonuçları ({x_feature} vs {y_feature})",
                    labels={x_feature: x_feature, y_feature: y_feature},
                    hover_data=features
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "3D Scatter Plot":
            if len(features) >= 3:
                # Kullanıcıya 3D gösterim için hangi özellikleri kullanacağını seçtir
                x_feature = st.selectbox("X ekseni için özellik seçin:", options=features, index=0)  # Varsayılan: Yaş
                y_feature = st.selectbox("Y ekseni için özellik seçin:", options=features, index=1)  # Varsayılan: Gelir
                z_feature = st.selectbox("Z ekseni için özellik seçin:", options=features, index=2)  # Varsayılan: Harcama
                
                fig = px.scatter_3d(
                    df, 
                    x=x_feature, 
                    y=y_feature, 
                    z=z_feature, 
                    color='Cluster',
                    color_continuous_scale='viridis' if algorithm != "DBSCAN" else 'jet',
                    title=f"{algorithm} Kümeleme Sonuçları (3D)",
                    labels={x_feature: x_feature, y_feature: y_feature, z_feature: z_feature},
                    hover_data=features
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("3D görselleştirme için en az 3 özellik seçilmelidir.")
        
        # PCA ile düşük boyutlu görselleştirme
        if len(features) > 2:
            st.subheader("PCA ile 2D Görselleştirme")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
            pca_df['Cluster'] = df['Cluster']
            
            fig = px.scatter(
                pca_df, 
                x='PCA1', 
                y='PCA2', 
                color='Cluster',
                color_continuous_scale='viridis' if algorithm != "DBSCAN" else 'jet',
                title="PCA ile Kümeleme Sonuçları",
                labels={'PCA1': 'Birinci Ana Bileşen', 'PCA2': 'İkinci Ana Bileşen'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA açıklama oranı
            explained_variance = pca.explained_variance_ratio_
            st.write(f"PCA Açıklanan Varyans Oranı: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}")
            st.write(f"Toplam Açıklanan Varyans: {sum(explained_variance):.2f}")
        
        # Küme İstatistikleri
        st.subheader("Küme İstatistikleri")
        cluster_stats = df.groupby('Cluster').mean()
        st.write(cluster_stats)
        
        # Küme yorumlaması
        st.subheader("Küme Yorumlaması")
        
        # Her küme için yorumlar
        for cluster in sorted(df['Cluster'].unique()):
            if cluster == -1:
                st.write(f"**Küme {cluster}**: Gürültü noktaları (DBSCAN tarafından atanan)")
                continue
                
            avg_age = cluster_stats.loc[cluster, 'Age']
            avg_income = cluster_stats.loc[cluster, 'Annual Income (k$)']
            avg_spending = cluster_stats.loc[cluster, 'Spending Score (1-100)']
            cluster_size = (df['Cluster'] == cluster).sum()
            
            st.write(f"**Küme {cluster}** ({cluster_size} müşteri):")
            
            # Yaş kategorisi
            if avg_age < 30:
                age_cat = "Genç"
            elif avg_age < 45:
                age_cat = "Orta yaş"
            else:
                age_cat = "Yaşlı"
                
            # Gelir kategorisi
            if avg_income < 40:
                income_cat = "Düşük gelir"
            elif avg_income < 70:
                income_cat = "Orta gelir"
            else:
                income_cat = "Yüksek gelir"
                
            # Harcama kategorisi
            if avg_spending < 40:
                spending_cat = "Düşük harcama"
            elif avg_spending < 70:
                spending_cat = "Orta harcama"
            else:
                spending_cat = "Yüksek harcama"
                
            st.write(f"- {age_cat} ({avg_age:.1f} yaş), {income_cat} ({avg_income:.1f}k$), {spending_cat} ({avg_spending:.1f}/100)")
            
            # Pazarlama stratejisi önerisi
            if income_cat == "Yüksek gelir" and spending_cat == "Düşük harcama":
                strategy = "Premium ürünler için potansiyel hedef kitle. Ürün kalitesi ve değerini vurgulayan pazarlama stratejileri."
            elif age_cat == "Genç" and spending_cat == "Yüksek harcama":
                strategy = "Trend ürünler ve kampanyalar için ideal hedef kitle."
            elif age_cat == "Yaşlı" and spending_cat == "Düşük harcama":
                strategy = "Sadakat programları ve değer odaklı teklifler sunulabilir."
            elif income_cat == "Düşük gelir" and spending_cat == "Yüksek harcama":
                strategy = "İndirim kampanyaları ve taksit seçenekleri için uygun hedef kitle."
            else:
                strategy = "Genel demografik profile uygun standart pazarlama stratejileri."
                
            st.write(f"- **Pazarlama Stratejisi**: {strategy}")
            
        # Cinsiyet-Küme İlişkisi
        st.subheader("Cinsiyet-Küme İlişkisi")
        # Cinsiyeti geri dönüştür
        gender_cluster_df = df.copy()
        gender_cluster_df['Gender'] = gender_cluster_df['Gender'].map({1: 'Erkek', 0: 'Kadın'})
        
        fig = px.histogram(
            gender_cluster_df, 
            x='Cluster', 
            color='Gender',
            barmode='group',
            title="Kümelere Göre Cinsiyet Dağılımı",
            labels={'Cluster': 'Küme', 'count': 'Müşteri Sayısı', 'Gender': 'Cinsiyet'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Küme Dağılımı
        st.subheader("Küme Dağılımı")
        fig = px.histogram(
            df, 
            x='Cluster', 
            title="Küme Başına Müşteri Sayısı",
            labels={'Cluster': 'Küme', 'count': 'Müşteri Sayısı'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Kümelere göre özellik dağılımı
        st.subheader("Kümelere Göre Özellik Dağılımı")
        for feature in features:
            fig = px.box(
                df, 
                x='Cluster', 
                y=feature, 
                title=f"Kümelere Göre {feature} Dağılımı",
                labels={'Cluster': 'Küme'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
else:
    st.error("Veri yüklenemedi. Lütfen Mall_Customers.csv dosyasının doğru konumda olduğundan emin olun.")
    
st.markdown("""
### Uygulama Hakkında
Bu uygulama, müşteri segmentasyonunu analiz etmek için çeşitli kümeleme algoritmalarını kullanmaktadır:

1. **K-Means**: Veri noktalarını belirli sayıda kümeye ayıran centroid-tabanlı bir algoritma.
2. **Hiyerarşik Kümeleme**: Veriler arasındaki benzerliklere göre hiyerarşik bir yapı oluşturan bir algoritma.
3. **DBSCAN**: Yoğunluk tabanlı bir kümeleme algoritması olup, rastgele şekilli kümeler bulabilir.

### Kullanım:
1. Yan menüden kümeleme için kullanılacak özellikleri seçin.
2. Kümeleme algoritmasını belirleyin.
3. Parametre ayarlarını yapın.
4. Sonuçları görselleştirin ve yorumlayın.
""")
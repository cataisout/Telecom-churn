from sklearn.cluster import KMeans
import pandas as pd

def fit_cluster_features(df_train, cluster_features, n_clusters=2, random_state=42):
    """
    Aprende as medianas e ajusta o KMeans para os dados de treinamento.
    """
    # Copiar colunas de interesse
    X = df_train[cluster_features].copy()

    # Calcular mediana por coluna
    median_dict = X.median().to_dict()

    # Imputar valores ausentes com a mediana
    X = X.fillna(median_dict)

    # Treinar o KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)

    # Retornar artefatos necessários
    return median_dict, kmeans


def transform_cluster_features(df, cluster_features, median_dict, kmeans):
    """
    Aplica imputação por mediana (pré-calculada) e clustering a um novo dataset.
    """
    X = df[cluster_features].copy()

    # Preencher valores ausentes com as medianas do treino
    for col in cluster_features:
        X[col] = X[col].fillna(median_dict[col])

    # Prever os clusters
    cluster_labels = kmeans.predict(X)

    # Adicionar coluna de cluster ao DataFrame original (cópia)
    df_result = df.copy()
    df_result['cluster'] = cluster_labels

    return df_result

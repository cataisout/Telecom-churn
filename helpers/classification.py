import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
                                f1_score, recall_score, matthews_corrcoef,
                                balanced_accuracy_score, roc_auc_score
                            )
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                ExtraTreesClassifier, HistGradientBoostingClassifier
                                )
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import tempfile
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import get_scorer
import warnings
from typing import Dict, Any, Union



def evaluate_ensemble_tree_models(X_train, y_train, X_val, y_val, random_state=42):

    """
    Avalia diversos modelos não baseados em álgebra linear em um conjunto de treino/validação.
    
    Parâmetros:
    -----------
    X : pd.DataFrame ou np.ndarray
        Features de entrada (já pré-processadas).
    
    y : pd.Series ou np.ndarray
        Target binário.
    
    test_size : float
        Proporção do conjunto de validação.
    
    random_state : int
        Semente para reprodutibilidade.

    Retorna:
    --------
    pd.DataFrame com nome do modelo e métricas: F1-score, Recall, MCC, BACC, AUC, Tempo de inferência.
    """
    
    temp_dir = tempfile.gettempdir()  # pasta temporária do sistema

    catboost_dir = os.path.join(temp_dir, "catboost_temp")


    modelos = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        "LightGBM": LGBMClassifier(random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state, train_dir=catboost_dir),
        "ExtraTrees": ExtraTreesClassifier(random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state)
    }

    resultados = []

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        # Medir tempo de inferência
        inicio = time.time()
        y_pred = modelo.predict(X_val)
        y_proba = modelo.predict_proba(X_val)[:, 1] if hasattr(modelo, "predict_proba") else y_pred
        fim = time.time()
        tempo_inferencia = fim - inicio

        # Calcular métricas
        resultados.append({
            "Modelo": nome,
            "F1-score": f1_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred),
            "MCC": matthews_corrcoef(y_val, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_val, y_pred),
            "AUC": roc_auc_score(y_val, y_proba),
            "Tempo de Inferência (s)": tempo_inferencia
        })

    return pd.DataFrame(resultados).sort_values(by="F1-score", ascending=False)


def evaluate_distance_models(X_train, y_train, X_val, y_val, random_state=42):
  
    models = {
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Naive Bayes (Gaussian)': GaussianNB(),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state),
    }

    results = []

    for name, model in models.items():
    
        model.fit(X_train, y_train)
        start_time = time.time()
        y_pred = model.predict(X_val)
        inference_time = time.time() - start_time

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
        else:
            auc = np.nan

        results.append({
            'Modelo': name,
            'F1-Score': f1_score(y_val, y_pred, average='binary'),
            'Recall': recall_score(y_val, y_pred, average='binary'),
            'MCC': matthews_corrcoef(y_val, y_pred),
            'BACC': balanced_accuracy_score(y_val, y_pred),
            'AUC': auc,
            'Tempo de Inferência (s)': inference_time
        })

   
    return pd.DataFrame(results)





def evaluate_models_cv(models_dict: Dict[str, Any], 
                      X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray],
                      n_splits: int = 10, 
                      scoring: str = 'f1',
                      random_state: int = 42,
                      shuffle: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Avalia múltiplos modelos usando K-Fold Cross Validation.
    
    Parameters:
    -----------
    models_dict : dict
        Dicionário com nome do modelo como chave e objeto do modelo como valor
    X : pd.DataFrame ou np.ndarray
        Features para treinamento
    y : pd.Series ou np.ndarray
        Target variable
    n_splits : int, default=10
        Número de splits para K-Fold
    scoring : str, default='f1'
        Métrica de avaliação (ex: 'accuracy', 'f1', 'roc_auc', 'r2', 'neg_mean_squared_error')
    random_state : int, default=42
        Seed para reprodutibilidade
    shuffle : bool, default=True
        Se deve embaralhar os dados antes do split
    verbose : bool, default=True
        Se deve imprimir progresso
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com resultados da validação cruzada
    """
    
    # Validações de entrada
    if not isinstance(models_dict, dict) or len(models_dict) == 0:
        raise ValueError("models_dict deve ser um dicionário não vazio")
    
    if n_splits < 2:
        raise ValueError("n_splits deve ser >= 2")
    
    # Verificar se a métrica é válida
    try:
        get_scorer(scoring)
    except ValueError:
        raise ValueError(f"Métrica '{scoring}' não é reconhecida pelo scikit-learn")
    
    # Configurar K-Fold
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Lista para armazenar resultados
    results = []
    
    if verbose:
        print(f"🔄 Iniciando avaliação de {len(models_dict)} modelos usando {n_splits}-Fold CV")
        print(f"📊 Métrica: {scoring}")
        print("-" * 60)
    
    # Iterar sobre cada modelo
    for model_name, model in models_dict.items():
        if verbose:
            print(f"Avaliando: {model_name}...", end=" ")
        
        start_time = time.time()
        
        try:
            # Executar cross validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    estimator=model,
                    X=X,
                    y=y,
                    cv=kfold,
                    scoring=scoring,
                    n_jobs=-1  # Usar todos os cores disponíveis
                )
            
            # Calcular estatísticas
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            min_score = cv_scores.min()
            max_score = cv_scores.max()
            
            # Armazenar resultados
            results.append({
                'modelo': model_name,
                'score_medio': mean_score,
                'score_std': std_score,
                'score_min': min_score,
                'score_max': max_score,
                'cv_scores': cv_scores.tolist()  # Para análises futuras
            })
            
            execution_time = time.time() - start_time
            
            if verbose:
                print(f"✅ {mean_score:.4f} (±{std_score:.4f}) [{execution_time:.2f}s]")
                
        except Exception as e:
            if verbose:
                print(f"❌ Erro: {str(e)}")
            
            # Adicionar resultado com erro
            results.append({
                'modelo': model_name,
                'score_medio': np.nan,
                'score_std': np.nan,
                'score_min': np.nan,
                'score_max': np.nan,
                'cv_scores': None,
                'erro': str(e)
            })
    
    # Criar DataFrame com resultados
    df_results = pd.DataFrame(results)
    
    # Ordenar por score médio (decrescente para métricas onde maior é melhor)
    # Para métricas negativas (como neg_mean_squared_error), ordenar crescente
    ascending = scoring.startswith('neg_')
    df_results = df_results.sort_values('score_medio', ascending=ascending)
    
    if verbose:
        print("-" * 60)
        print("🏆 RANKING DOS MODELOS:")
        print(df_results[['modelo', 'score_medio', 'score_std', 'score_min', 'score_max']].to_string(index=False, float_format='%.4f'))
    
    return df_results
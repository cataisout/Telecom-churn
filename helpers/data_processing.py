import pandas as pd
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler


from typing import List, Tuple, Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_columns(df: pd.DataFrame, columns: List[str], na_label: str = "desconhecido") -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Aplica label encoding nas colunas especificadas de um DataFrame,
    tratando valores ausentes como uma categoria separada.

    Parâmetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - columns (List[str]): Lista de nomes das colunas a serem codificadas.
    - na_label (str): Valor a ser usado para representar valores ausentes.

    Retorna:
    - Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]: 
        - DataFrame com as colunas codificadas.
        - Dicionário com os mapeamentos de valores originais para inteiros por coluna.
    """
    df_encoded = df.copy()
    mappings = {}

    for col in columns:
        if col in df_encoded.columns:
            # Substituir NaN pelo valor definido como categoria para ausente
            df_encoded[col] = df_encoded[col].fillna(na_label).astype(str)

            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
            # Mapeamento de valores
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            mappings[col] = mapping
            
        else:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame.")
    
    return df_encoded, mappings



def binarizar_colunas(df: pd.DataFrame, colunas_categoricas: List[str]) -> pd.DataFrame:
    """
    Binariza (one-hot encode) as colunas categóricas especificadas e retorna 
    um novo DataFrame contendo as colunas binarizadas junto com as colunas restantes.

    Parâmetros:
    ----------
    df : pd.DataFrame
        O DataFrame de entrada contendo colunas categóricas e outras colunas.
    
    colunas_categoricas : List[str]
        Lista com os nomes das colunas a serem binarizadas.

    Retorna:
    -------
    pd.DataFrame
        Um novo DataFrame contendo as colunas binarizadas juntamente com as demais colunas originais.
    
    """

    # Aplica one-hot encoding apenas nas colunas especificadas
    df_binarizado = pd.get_dummies(df[colunas_categoricas].astype(str), drop_first=False)

    # Seleciona as colunas que não foram binarizadas
    colunas_restantes = df.drop(columns=colunas_categoricas)

    # Concatena o DataFrame binarizado com as demais colunas
    df_final = pd.concat([colunas_restantes, df_binarizado], axis=1)

    return df_final



def normalizar_colunas_numericas(X_train, X_val, X_test, num_cols):
    """
    Normaliza colunas numéricas usando StandardScaler com base apenas no treino,
    e retorna os conjuntos de treino, validação e teste com as colunas recombinadas.

    Parâmetros:
    -----------
    X_train, X_val, X_test : pd.DataFrame
        DataFrames de treino, validação e teste.

    num_cols : list of str
        Lista com os nomes das colunas numéricas a serem normalizadas.

    Retorna:
    --------
    X_train_scaled, X_val_scaled, X_test_scaled : pd.DataFrame
        DataFrames com colunas numéricas normalizadas e demais colunas mantidas.
    """
    scaler = StandardScaler()
    
    # Ajusta no treino e transforma os 3 conjuntos
    train_scaled = scaler.fit_transform(X_train[num_cols])
    val_scaled   = scaler.transform(X_val[num_cols])
    test_scaled  = scaler.transform(X_test[num_cols])
    
    # Cria DataFrames com os mesmos nomes de colunas
    train_scaled_df = pd.DataFrame(train_scaled, columns=num_cols, index=X_train.index)
    val_scaled_df   = pd.DataFrame(val_scaled, columns=num_cols, index=X_val.index)
    test_scaled_df  = pd.DataFrame(test_scaled, columns=num_cols, index=X_test.index)

    # Recombina com o restante das colunas (não numéricas)
    X_train_final = pd.concat([X_train.drop(columns=num_cols), train_scaled_df], axis=1)
    X_val_final   = pd.concat([X_val.drop(columns=num_cols), val_scaled_df], axis=1)
    X_test_final  = pd.concat([X_test.drop(columns=num_cols), test_scaled_df], axis=1)

    return X_train_final, X_val_final, X_test_final

import pandas as pd
from pandas import DataFrame
from utils.data_processing import normalize_text

def df_to_lowercase(df: pd.DataFrame()) -> pd.DataFrame():
    ''' Padroniza todos as strings das colunas textuais de um dataframe para lowercase
        
        Parâmetros:
        ----------
        df: pd.DataFrame
            DataFrame contendo os dados.

        Retorno:
        -------
        pd.DataFrame
            Dataframe com as colunas transformadas
        
    '''
    
    df_copy = df.copy()
    text_cols = df_copy.select_dtypes(include=['object']).columns
    for col in text_cols:
        df_copy.loc[:, col] = df_copy[col].apply(normalize_text)
    
    return df_copy


def detectar_outliers(df: DataFrame, coluna: str) -> DataFrame:
    """
    Detecta outliers em uma coluna numérica de um DataFrame usando o método do IQR.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str
        Nome da coluna numérica a ser analisada.

    Retorno:
    -------
    pd.DataFrame
        Subconjunto do DataFrame contendo apenas as linhas com outliers detectados.
    """
    # Calcula Q1, Q3 e IQR
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1

    # Define os limites inferior e superior
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Filtra os outliers
    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]

    return outliers

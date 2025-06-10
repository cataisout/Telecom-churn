import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

def plot_porcentagem_por_classe_comparativo(df1: DataFrame, df2: DataFrame, coluna: str, label1='DF1', label2='DF2') -> None:
    """
    Plota um gráfico de barras comparando a porcentagem de cada classe em uma coluna específica para dois DataFrames.

    Parâmetros:
    - df1, df2: DataFrames a serem comparados.
    - coluna: Nome da coluna categórica.
    - label1, label2: Rótulos para os DataFrames no gráfico.
    """
    # Calcula as porcentagens
    pct1 = df1[coluna].value_counts(normalize=True).rename(label1) * 100
    pct2 = df2[coluna].value_counts(normalize=True).rename(label2) * 100

    # Junta os dados e reorganiza para formato longo
    comparativo = pd.concat([pct1, pct2], axis=1).fillna(0).reset_index()
    comparativo = comparativo.melt(id_vars=coluna, var_name='Dataset', value_name='Porcentagem')

    # Plota
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=comparativo, x=coluna, y='Porcentagem', hue='Dataset')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    plt.title(f'Comparativo de Porcentagem - {coluna}')
    plt.ylabel('Porcentagem (%)')
    plt.tight_layout()
    plt.show()

    
def plot_porcentagem_por_classe(df: DataFrame, coluna: str) -> None:
    """
    Plota um gráfico de barras com a porcentagem de cada classe em uma coluna específica,
    incluindo rótulos com os valores de porcentagem.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str
        Nome da coluna a ser analisada.

    Retorno:
    -------
    None
        A função exibe o gráfico e não retorna nenhum valor.
    """
    # Calcula as porcentagens
    porcentagens = df[coluna].value_counts(normalize=True) * 100

    # Prepara os dados para o seaborn
    plot_data = porcentagens.reset_index()
    plot_data.columns = [coluna, 'Porcentagem']

    # Cria o gráfico
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=plot_data, x=coluna, y='Porcentagem', hue=coluna)

    # Adiciona os rótulos com as porcentagens
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3)

    # Ajustes no gráfico
    plt.title(f'Porcentagem por Classe - {coluna}', fontsize=14)
    plt.xlabel('Classe', fontsize=12)
    plt.ylabel('Porcentagem (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Exibe o gráfico
    plt.show()



def plot_boxplot(df: DataFrame, coluna: str) -> None:
    """
    Plota um boxplot para a coluna numérica especificada.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str
        Nome da coluna numérica para gerar o boxplot.

    Retorno:
    -------
    None
        A função exibe o gráfico e não retorna nenhum valor.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[coluna], color='skyblue')
    plt.title(f'Boxplot da coluna {coluna}', fontsize=14)
    plt.xlabel(coluna, fontsize=12)
    plt.tight_layout()
    plt.show()

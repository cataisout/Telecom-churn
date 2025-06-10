from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


#uso do BaseEstimator, TransformerMixin para garantir que seja possível integrar com pipelines do sklearn
# e invocar o fit_trasnform()



class ClusterWiseImputerOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_col, variables):
        self.cluster_col = cluster_col
        self.variables = variables
        self.stats_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.stats_ = {}
        for cluster in df[self.cluster_col].unique():
            cluster_df = df[df[self.cluster_col] == cluster]
            self.stats_[cluster] = {}
            for var in self.variables:
                q1 = cluster_df[var].quantile(0.25)
                q3 = cluster_df[var].quantile(0.75)
                iqr = q3 - q1
                med = cluster_df[var].median()
                self.stats_[cluster][var] = {
                    'median': med,
                    'lower': q1 - 1.5 * iqr,
                    'upper': q3 + 1.5 * iqr
                }
        return self

    def transform(self, X):
        df = X.copy()
        for cluster, stats in self.stats_.items(): #itera nas estatisticas de cada clustes
            mask = df[self.cluster_col] == cluster #seleciona linhas do clustes
            for var in self.variables: # itera em cada variável
                df.loc[mask, var] = df.loc[mask, var].fillna(stats[var]['median']) #preenche NA com mediana
                df.loc[mask, var] = df.loc[mask, var].clip(stats[var]['lower'], stats[var]['upper']) #clipping para lidar com outliers
                
        return df

# me custou uma hora de vida mas valeu a pena
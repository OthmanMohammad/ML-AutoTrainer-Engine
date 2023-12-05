from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold

def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def apply_ica(data, n_components):
    ica = FastICA(n_components=n_components)
    return ica.fit_transform(data)

def apply_lda(data, target, n_components):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    return lda.fit_transform(data, target)

def apply_feature_agglomeration(data, n_clusters):
    agglomeration = FeatureAgglomeration(n_clusters=n_clusters)
    return agglomeration.fit_transform(data)

def select_k_best(data, target, k=10):
    selector = SelectKBest(chi2, k=k)
    return selector.fit_transform(data, target)

def apply_variance_threshold(data, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(data)
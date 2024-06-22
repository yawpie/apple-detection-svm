from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import os


def extract(image, distance=[1], angles=[0]):
    glcm_result = graycomatrix(
        image, distance, angles, levels=256, symmetric=True)

    contrast = graycoprops(glcm_result, 'contrast')[0][0]
    dissimilarity = graycoprops(glcm_result, 'dissimilarity')[0][0]
    homogeneity = graycoprops(glcm_result, 'homogeneity')[0][0]
    energy = graycoprops(glcm_result, 'energy')[0][0]
    correlation = graycoprops(glcm_result, 'correlation')[0][0]
    asm = graycoprops(glcm_result, 'ASM')[0][0]

    return contrast, dissimilarity, homogeneity, energy, correlation, asm


def write_features_to_csv(path, images, distance=[1], angles=[0]):
    extracted_features = {'label': [], 'contrast': [], 'dissimilarity': [
    ], 'homogeneity': [], 'energy': [], 'correlation': [],  'asm': []}

    for label in images.keys():

        for image in images[label]:

            contrast, dissimilarity, homogeneity, energy, correlation, asm = extract(
                image, distance, angles)

            extracted_features['label'].append(label)

            extracted_features['contrast'].append(contrast)
            extracted_features['dissimilarity'].append(dissimilarity)
            extracted_features['homogeneity'].append(homogeneity)
            extracted_features['energy'].append(energy)
            extracted_features['correlation'].append(correlation)
            extracted_features['asm'].append(asm)
    dataframe_extracted_features = pd.DataFrame(extracted_features)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    dataframe_extracted_features.to_csv(os.path.join(
        path, 'extracted_features.csv'), index=False)

    return dataframe_extracted_features


def perform_pca_feature_selection(data, n_components=None):
    """
    Perform PCA (Principal Component Analysis) for feature selection.

    Parameters:
    - data: Input data as a 2D array or dataframe (samples x features).
    - n_components: Number of principal components to retain (default is None).

    Returns:
    - reduced_data: Transformed data with reduced dimensions based on PCA.
    - pca_model: Trained PCA model.
    """
    # Standardize the data (mean=0 and variance=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Initialize PCA
    pca = PCA(n_components=n_components)

    # Perform PCA
    reduced_data = pca.fit_transform(scaled_data)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f'Explained variance ratio: {explained_variance_ratio}')

    # Return reduced data and PCA model
    return reduced_data, pca


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average='weighted')

    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy}")

    print(f"Precision: {precision}")

    print(f"Recall: {recall}")

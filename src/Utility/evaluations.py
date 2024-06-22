from Utility.calculations import extract
import numpy as np


def predict_label(image, model, pca, label_encoder):
    feature_vector = extract(image)
    feature_vector = np.array(feature_vector)
    feature_vector_pca = pca.fit_transform(feature_vector)
    predicted_label = model.predict(feature_vector_pca)
    predicted_label = label_encoder.inverse_transform(predicted_label)
    return predicted_label[0]

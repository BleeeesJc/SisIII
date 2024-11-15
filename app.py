from flask import Flask, jsonify, request, render_template
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, cohen_kappa_score
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Algoritmos.html')

@app.route('/run_j48', methods=['POST'])
def run_j48():
    try:
        # Ruta del archivo CSV
        file_path = os.path.expanduser("diabetes_dataset00.csv")
        data = pd.read_csv(file_path)

        # Identificar características (X) y etiquetas (y)
        X = data.iloc[:, :-1]  # Todas las columnas menos la última como características
        y = data.iloc[:, -1]   # La última columna como etiquetas

        # Convierte columnas categóricas en X a valores numéricos usando One-Hot Encoding
        X = pd.get_dummies(X)

        # Convierte la columna de etiquetas (y) a valores numéricos si es categórica
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        # Configura y entrena el árbol de decisión
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(X, y)

        # Predicciones para calcular métricas
        y_pred = clf.predict(X)

        # Calcular métricas adicionales con manejo de errores
        accuracy = accuracy_score(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred) if len(set(y)) > 1 else None
        mean_abs_error = mean_absolute_error(y, y_pred)
        root_mean_squared_error = np.sqrt(mean_squared_error(y, y_pred))
        
        # Evitar división por cero en métricas relativas
        mean_y = np.mean(y)
        if mean_y != 0:
            relative_absolute_error = mean_abs_error / np.mean(np.abs(y - mean_y))
            root_relative_squared_error = root_mean_squared_error / np.sqrt(np.mean((y - mean_y) ** 2))
        else:
            relative_absolute_error = None
            root_relative_squared_error = None

        total_instances = len(y)

        # Construcción de la respuesta JSON
        response = {
            "status": "success",
            "report": {
                "accuracy": accuracy,
                "kappa": kappa if kappa is not None else "N/A",
                "mean_absolute_error": mean_abs_error,
                "root_mean_squared_error": root_mean_squared_error,
                "relative_absolute_error": relative_absolute_error if relative_absolute_error is not None else "N/A",
                "root_relative_squared_error": root_relative_squared_error if root_relative_squared_error is not None else "N/A",
                "total_instances": total_instances
            }
        }

        return jsonify(response)
    except Exception as e:
        return jsonify(status="error", message=str(e))

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    try:
        # Parse number of clusters from request
        num_clusters = int(request.json.get('num_clusters'))
        
        file_path = os.path.expanduser("diabetes_dataset00.csv")
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1]
        X = pd.get_dummies(X)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        
        # Calculate cluster centers and format them
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
        cluster_centers.index = [f"Cluster {i}" for i in range(num_clusters)]
        full_data_mean = X.mean().to_dict()
        
        # Calculate distribution of instances per cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        total_instances = len(X)
        cluster_distribution = [{"cluster": int(label), "count": int(count), "percentage": round((count / total_instances) * 100, 2)} for label, count in zip(labels, counts)]

        response = {
            "status": "success",
            "cluster_centers": cluster_centers.to_dict(orient="index"),
            "full_data_mean": full_data_mean,
            "cluster_distribution": cluster_distribution
        }
        return jsonify(response)
    except Exception as e:
        return jsonify(status="error", message=str(e))
@app.route('/run_mlp', methods=['POST'])
def run_mlp():
    try:
        # Cargar el archivo CSV
        file_path = os.path.expanduser("diabetes_dataset00.csv")
        data = pd.read_csv(file_path)

        # Identificar características (X) y etiquetas (y)
        X = data.iloc[:, :-1]  # Todas las columnas menos la última como características
        y = data.iloc[:, -1]   # La última columna como etiquetas

        # One-Hot Encoding para características categóricas
        X = pd.get_dummies(X)

        # Convierte la columna de etiquetas (y) a valores numéricos si es categórica
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        # Configura y entrena el MLP
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        mlp.fit(X, y)

        # Predicciones para calcular métricas
        y_pred = mlp.predict(X)

        # Calcular métricas
        accuracy = accuracy_score(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred) if len(set(y)) > 1 else None
        mean_abs_error = mean_absolute_error(y, y_pred)
        root_mean_squared_error = np.sqrt(mean_squared_error(y, y_pred))

        mean_y = np.mean(y)
        if mean_y != 0:
            relative_absolute_error = mean_abs_error / np.mean(np.abs(y - mean_y))
            root_relative_squared_error = root_mean_squared_error / np.sqrt(np.mean((y - mean_y) ** 2))
        else:
            relative_absolute_error = None
            root_relative_squared_error = None

        response = {
            "status": "success",
            "report": {
                "accuracy": accuracy,
                "kappa": kappa if kappa is not None else "N/A",
                "mean_absolute_error": mean_abs_error,
                "root_mean_squared_error": root_mean_squared_error,
                "relative_absolute_error": relative_absolute_error if relative_absolute_error is not None else "N/A",
                "root_relative_squared_error": root_relative_squared_error if root_relative_squared_error is not None else "N/A",
                "total_instances": len(y)
            }
        }

        return jsonify(response)
    except Exception as e:
        return jsonify(status="error", message=str(e))
    
if __name__ == '__main__':
    app.run(debug=True)

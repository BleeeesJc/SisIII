from flask import Flask, jsonify, render_template
from sklearn.tree import DecisionTreeClassifier
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

if __name__ == '__main__':
    app.run(debug=True)

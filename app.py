from flask import Flask, jsonify, request, render_template
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import networkx as nx
from werkzeug.utils import secure_filename
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="templates")

# Directorio temporal para almacenar archivos subidos
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/home')
def index():
    return render_template('pricing.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_<algorithm>', methods=['POST'])
def run_algorithm(algorithm):
    try:
        if 'csv_file' not in request.files:
            return jsonify({"status": "error", "message": "No se proporcionó ningún archivo."})
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "El archivo tiene un nombre vacío."})

        # Guardar el archivo subido temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Llamar al algoritmo seleccionado
        if algorithm == "j48":
            return run_j48(filepath)
        elif algorithm == "kmeans":
            num_clusters = int(request.form.get("num_clusters", 2))  # Opcional para KMeans
            return run_kmeans(filepath, num_clusters)
        elif algorithm == "mlp":
            return run_mlp(filepath)
        else:
            return jsonify({"status": "error", "message": f"Algoritmo desconocido: {algorithm}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def run_j48(csv_path):
    try:
        # Cargar los datos usando pandas
        data = pd.read_csv(csv_path)

        # Separar características (X) y etiquetas (y)
        X = data.iloc[:, :-1]  # Todas las columnas menos la última
        y = data.iloc[:, -1]   # Última columna

        # Identificar columnas categóricas en X
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Aplicar One-Hot Encoding a las columnas categóricas
        X = pd.get_dummies(X, columns=categorical_cols)

        # Si y es categórica, también la codificamos
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            class_names = le.classes_
        else:
            class_names = [str(cls) for cls in set(y)]

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de árbol de decisión
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        # Generar el gráfico del árbol
        dot_path = os.path.join(app.static_folder, "tree.dot")
        export_graphviz(clf, out_file=dot_path, feature_names=X.columns, class_names=class_names, filled=True)

        png_path = os.path.join(app.static_folder, "tree.png")
        command = ["dot", "-Tpng", dot_path, "-o", png_path]
        subprocess.run(command, check=True)

        # Evaluar el modelo usando el conjunto de prueba
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Calcular otras métricas
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        # Si tus etiquetas tienen significado numérico, puedes calcular MAE y MSE
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "classification_report": report,  # Convertir a lista para JSON
            "total_instances": len(y_test),
        }

        return jsonify({
            "status": "success",
            "report": metrics,
            "tree_image_url": "/static/tree.png",
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"Error al ejecutar dot: {e}"})
    except Exception as e:
        import traceback
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        error_message = f"{str(e)}\nTraceback:\n{traceback_str}"
        return jsonify({"status": "error", "message": error_message})

def run_kmeans(csv_path, num_clusters):
    try:
        # Cargar los datos usando pandas
        data = pd.read_csv(csv_path)

        # Identificar columnas categóricas en X
        categorical_cols = data.select_dtypes(include=['object']).columns

        # Opcional: Decidir si se eliminan o se codifican las columnas categóricas
        # Aquí las eliminamos
        X = data.drop(columns=categorical_cols)

        # Asegurarse de que no hay valores NaN
        X = X.dropna()

        # Crear y entrenar el modelo KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(X)

        # Obtener los centros de los clusters
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        centers = {}
        for idx, center in enumerate(cluster_centers):
            centers[f"Cluster {idx + 1}"] = dict(zip(X.columns, center))

        # Visualizar los clusters
        plt.figure(figsize=(8, 6))
        if X.shape[1] >= 2:
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=cluster_labels, cmap='viridis', s=50, label="Instancias")
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', s=100, label="Centroides", edgecolors='black')
            plt.xlabel(X.columns[0])
            plt.ylabel(X.columns[1])
        else:
            plt.scatter(range(len(X)), X.iloc[:, 0], c=cluster_labels, cmap='viridis', s=50)
            plt.xlabel("Índice de Instancia")
            plt.ylabel(X.columns[0])

        plt.title(f"KMeans Clustering (k={num_clusters})")
        plt.colorbar(label="Cluster")
        plt.legend()
        plt.grid(True)

        cluster_image_path = os.path.join(app.static_folder, "clusters.png")
        plt.savefig(cluster_image_path)
        plt.close()

        return jsonify({
            "status": "success",
            "cluster_centers": centers,
            "cluster_image_url": "/static/clusters.png",
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def run_mlp(csv_path):
    try:
        # Cargar los datos usando pandas
        data = pd.read_csv(csv_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Identificar columnas categóricas en X
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Aplicar One-Hot Encoding a las columnas categóricas
        X = pd.get_dummies(X, columns=categorical_cols)

        # Si y es categórica, también la codificamos
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            class_names = le.classes_
        else:
            class_names = [str(cls) for cls in set(y)]

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo MLP
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=1)
        mlp.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "classification_report": report,
            "confusion_matrix": confusion.tolist(),
            "total_instances": len(y_test),
        }

        # Visualización de la red neuronal (opcional)
        # ...

        return jsonify({
            "status": "success",
            "report": metrics,
            "mlp_image_url": "/static/mlp_network.png",
        })

    except Exception as e:
        import traceback
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        error_message = f"{str(e)}\nTraceback:\n{traceback_str}"
        return jsonify({"status": "error", "message": error_message})

if __name__ == '__main__':
    app.run(debug=True)

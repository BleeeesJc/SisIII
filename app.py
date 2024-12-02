from flask import Flask, jsonify, request, render_template
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from werkzeug.utils import secure_filename
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
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
        if algorithm == "kmeans":
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de árbol de decisión
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        # Generar el gráfico del árbol usando Matplotlib
        plt.figure(figsize=(14, 8))
        plot_tree(decision_tree=clf, feature_names=X.columns,
                  class_names=class_names, filled=True, fontsize=10)
        plt.tight_layout()

        # Guardar la imagen en el directorio 'static'
        tree_image_path = os.path.join(app.static_folder, "tree.png")
        plt.savefig(tree_image_path)
        plt.close()

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
            "classification_report": report,
            "total_instances": len(y_test),
        }

        return jsonify({
            "status": "success",
            "report": metrics,
            "tree_image_url": "/static/tree.png",
        })

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

        # Identificar columnas categóricas y aplicar Label Encoding
        label_encoders = {}
        categorical_columns = data.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        # Separar características (X) y etiquetas (y)
        X = data.drop('Target', axis=1)
        y = data['Target']

        # Escalar características numéricas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # Crear y entrenar el modelo MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            verbose=True
        )
        mlp.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Obtener la última iteración realizada
        last_iteration = len(mlp.loss_curve_)

        # Convertir classification_report en un formato estructurado
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Generar gráfica de curva de pérdida
        plt.figure(figsize=(10, 6))
        plt.plot(mlp.loss_curve_)
        plt.title('Curva de pérdida durante el entrenamiento')
        plt.xlabel('Iteraciones')
        plt.ylabel('Pérdida')
        plt.grid(True)

        # Guardar la gráfica
        loss_curve_image_path = os.path.join('static', 'mlp_loss_curve.png')
        plt.savefig(loss_curve_image_path)
        plt.close()

        # Resultados
        return jsonify({
            "status": "success",
            "accuracy": f"{accuracy:.2f}",
            "last_iteration": last_iteration,
            "classification_report": report_dict,
            "loss_curve_url": f"/static/mlp_loss_curve.png"
        })

    except Exception as e:
        import traceback
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        error_message = f"{str(e)}\nTraceback:\n{traceback_str}"
        return jsonify({"status": "error", "message": error_message})

if __name__ == '__main__':
    app.run(debug=True)

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

        numeric_columns = data.select_dtypes(include=['number'])

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_columns)

        kmeans = KMeans(n_clusters=num_clusters)
        cluster_labels = kmeans.fit_predict(scaled_data)
        cluster_centers = kmeans.cluster_centers_

        data['Cluster'] = cluster_labels

        centers = {f"Cluster {i + 1}": dict(zip(numeric_columns.columns, center))
                   for i, center in enumerate(cluster_centers)}

        plt.figure(figsize=(10, 7))
        plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, label="Datos")
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X', label="Centroides")
        plt.title(f"Clusters generados con KMeans (k={num_clusters})")
        plt.legend()

        cluster_image_path = os.path.join(app.static_folder, "clusters.png")
        plt.savefig(cluster_image_path)
        plt.close()
        
        return jsonify({
            "status": "success",
            "cluster_centers": centers,
            "cluster_image_url": "/static/clusters.png"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
# Declarar variables globales
label_encoders = {}
scaler = None
mlp = None

def run_mlp(csv_path):
    global label_encoders, scaler, mlp  # Hacerlas accesibles desde otras funciones
    try:
        # Cargar los datos usando pandas
        data = pd.read_csv(csv_path)

        # Mantener los nombres de las columnas tal cual en el dataset original
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

    

@app.route('/predict_mlp', methods=['POST'])
def predict_mlp():
    try:
        # Obtener los datos del formulario en formato JSON
        data = request.json

        # Convertir los datos a un DataFrame de pandas
        new_data = pd.DataFrame([data])

        # Definir los nombres esperados de las columnas
        expected_columns = [
            'Genetic Markers', 'Autoantibodies', 'Family History', 'Environmental Factors', 'Insulin Levels',
            'Age', 'BMI', 'Physical Activity', 'Dietary Habits', 'Blood Pressure', 'Cholesterol Levels',
            'Waist Circumference', 'Blood Glucose Levels', 'Ethnicity', 'Socioeconomic Factors',
            'Smoking Status', 'Alcohol Consumption', 'Glucose Tolerance Test', 'History of PCOS',
            'Previous Gestational Diabetes', 'Pregnancy History', 'Weight Gain During Pregnancy',
            'Pancreatic Health', 'Pulmonary Function', 'Cystic Fibrosis Diagnosis', 'Steroid Use History',
            'Genetic Testing', 'Neurological Assessments', 'Liver Function Tests', 'Digestive Enzyme Levels',
            'Urine Test', 'Birth Weight', 'Early Onset Symptoms'
        ]

        # Mapeo para convertir los nombres ingresados a los nombres esperados
        column_mapping = {
            'genetic_markers': 'Genetic Markers',
            'autoantibodies': 'Autoantibodies',
            'family_history': 'Family History',
            'environmental_factors': 'Environmental Factors',
            'insulin_levels': 'Insulin Levels',
            'age': 'Age',
            'bmi': 'BMI',
            'physical_activity': 'Physical Activity',
            'dietary_habits': 'Dietary Habits',
            'blood_pressure': 'Blood Pressure',
            'cholesterol_levels': 'Cholesterol Levels',
            'waist_circumference': 'Waist Circumference',
            'blood_glucose_levels': 'Blood Glucose Levels',
            'ethnicity': 'Ethnicity',
            'socioeconomic_factors': 'Socioeconomic Factors',
            'smoking_status': 'Smoking Status',
            'alcohol_consumption': 'Alcohol Consumption',
            'glucose_tolerance_test': 'Glucose Tolerance Test',
            'history_of_pcos': 'History of PCOS',
            'previous_gestational_diabetes': 'Previous Gestational Diabetes',
            'pregnancy_history': 'Pregnancy History',
            'weight_gain_during_pregnancy': 'Weight Gain During Pregnancy',
            'pancreatic_health': 'Pancreatic Health',
            'pulmonary_function': 'Pulmonary Function',
            'cystic_fibrosis_diagnosis': 'Cystic Fibrosis Diagnosis',
            'steroid_use_history': 'Steroid Use History',
            'genetic_testing': 'Genetic Testing',
            'neurological_assessments': 'Neurological Assessments',
            'liver_function_tests': 'Liver Function Tests',
            'digestive_enzyme_levels': 'Digestive Enzyme Levels',
            'urine_test': 'Urine Test',
            'birth_weight': 'Birth Weight',
            'early_onset_symptoms': 'Early Onset Symptoms'
        }

        # Renombrar las columnas del DataFrame recibido con base en el mapeo
        new_data.rename(columns=column_mapping, inplace=True)

        # Verificar si todas las columnas esperadas están presentes
        missing_columns = [col for col in expected_columns if col not in new_data.columns]
        if missing_columns:
            return jsonify({"status": "error", "message": f"Faltan las siguientes columnas: {missing_columns}"})

        # Ordenar y alinear las columnas según se espera en el entrenamiento
        new_data = new_data[expected_columns]

        # Identificar columnas categóricas y aplicar Label Encoding con los encoders usados durante el entrenamiento
        for col, le in label_encoders.items():
            if col in new_data.columns:
                new_data[col] = le.transform(new_data[col])

        # Escalar características numéricas con el escalador utilizado durante el entrenamiento
        new_data_scaled = scaler.transform(new_data)

        # Realizar la predicción
        prediction = mlp.predict(new_data_scaled)

        # Decodificar la clase predicha
        predicted_class = label_encoders['Target'].inverse_transform(prediction)

        # Devolver la predicción al cliente
        return jsonify({
            "status": "success",
            "predicted_class": predicted_class[0]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True)

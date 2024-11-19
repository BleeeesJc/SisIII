from flask import Flask, jsonify, request, render_template, send_from_directory
import weka.core.jvm as jvm
from weka.clusterers import Clusterer
from weka.classifiers import Classifier, Evaluation
from weka.core.converters import Loader
from weka.core.classes import Random
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import networkx as nx
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")

# Directorio temporal para almacenar archivos subidos
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Iniciar JVM de Weka
jvm.start(max_heap_size="8g")
print("JVM iniciada con éxito.")

@app.route('/')
def index():
    return render_template('pricing.html')

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
        loader = Loader(classname="weka.core.converters.CSVLoader", options=["-F", ";"])  # Para archivos con punto y coma
        data = loader.load_file(csv_path)
        data.class_is_last()

        classifier = Classifier(classname="weka.classifiers.trees.J48")
        classifier.build_classifier(data)

        dot_path = os.path.join(app.static_folder, "tree.dot")
        with open(dot_path, "w") as dot_file:
            dot_file.write(classifier.graph)

        png_path = os.path.join(app.static_folder, "tree.png")
        command = ["dot", "-Tpng", dot_path, "-o", png_path]
        subprocess.run(command, check=True)

        evaluation = Evaluation(data)
        evaluation.crossvalidate_model(classifier, data, 10, Random(1))
        report = {
            "accuracy": evaluation.percent_correct / 100,
            "kappa": evaluation.kappa,
            "mean_absolute_error": evaluation.mean_absolute_error,
            "root_mean_squared_error": evaluation.root_mean_squared_error,
            "relative_absolute_error": evaluation.relative_absolute_error,
            "root_relative_squared_error": evaluation.root_relative_squared_error,
            "total_instances": evaluation.num_instances,
        }
        return jsonify({
            "status": "success",
            "report": report,
            "tree_image_url": "/static/tree.png",
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": f"Error al ejecutar dot: {e}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def run_kmeans(csv_path, num_clusters):
    try:
        loader = Loader(classname="weka.core.converters.CSVLoader", options=["-F", ";"])  # Para archivos con punto y coma
        data = loader.load_file(csv_path)
        data.class_index = -1

        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", str(num_clusters)])
        clusterer.build_clusterer(data)

        clusters = defaultdict(list)
        for instance_idx, instance in enumerate(data):
            cluster_idx = clusterer.cluster_instance(instance)
            clusters[cluster_idx].append(instance)

        cluster_centers = {}
        for cluster_idx, instances in clusters.items():
            cluster_centroid = {}
            for attr_idx in range(data.num_attributes):
                attr_name = data.attribute(attr_idx).name
                attr_values = [instance.get_value(attr_idx) for instance in instances]
                cluster_centroid[attr_name] = sum(attr_values) / len(attr_values) if attr_values else 0.0
            cluster_centers[f"Cluster {cluster_idx + 1}"] = cluster_centroid

        data_points = np.array([[instance.get_value(i) for i in range(data.num_attributes)] for instance in data])
        cluster_assignments = [clusterer.cluster_instance(instance) for instance in data]

        plt.figure(figsize=(8, 6))
        if data.num_attributes >= 2:
            plt.scatter(data_points[:, 0], data_points[:, 1], c=cluster_assignments, cmap='viridis', s=50, label="Instancias")
            for cluster_idx, centroid in cluster_centers.items():
                plt.scatter(
                    [centroid[data.attribute(0).name]], 
                    [centroid[data.attribute(1).name]], 
                    color='red', 
                    s=100, 
                    label=f"Centroide {cluster_idx}",
                    edgecolors='black'
                )
            plt.xlabel(data.attribute(0).name)
            plt.ylabel(data.attribute(1).name)
        else:
            plt.scatter(range(len(data_points)), data_points[:, 0], c=cluster_assignments, cmap='viridis', s=50)
            plt.xlabel("Índice de Instancia")
            plt.ylabel(data.attribute(0).name)
        
        plt.title(f"KMeans Clustering (k={num_clusters})")
        plt.colorbar(label="Cluster")
        plt.legend()
        plt.grid(True)

        cluster_image_path = os.path.join(app.static_folder, "clusters.png")
        plt.savefig(cluster_image_path)
        plt.close()

        return jsonify({
            "status": "success",
            "cluster_centers": cluster_centers,
            "cluster_image_url": "/static/clusters.png",
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def run_mlp(csv_path):
    try:
        loader = Loader(classname="weka.core.converters.CSVLoader", options=["-F", ";"])  # Para archivos con punto y coma
        data = loader.load_file(csv_path)
        data.class_is_last()

        mlp = Classifier(
            classname="weka.classifiers.functions.MultilayerPerceptron",
            options=["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "a"]
        )
        mlp.build_classifier(data)

        evaluation = Evaluation(data)
        evaluation.crossvalidate_model(mlp, data, 10, Random(1))

        metrics = {
            "accuracy": evaluation.percent_correct / 100,
            "kappa": evaluation.kappa,
            "mean_absolute_error": evaluation.mean_absolute_error,
            "root_mean_squared_error": evaluation.root_mean_squared_error,
            "relative_absolute_error": evaluation.relative_absolute_error,
            "root_relative_squared_error": evaluation.root_relative_squared_error,
            "total_instances": evaluation.num_instances,
        }

        def plot_neural_network(input_size, hidden_size, output_size):
            layers = [input_size, hidden_size, output_size]
            G = nx.DiGraph()
            pos = {}
            y_offset = 0
            for layer_idx, layer_size in enumerate(layers):
                x_offset = 0
                for neuron in range(layer_size):
                    node_name = f"L{layer_idx}_N{neuron}"
                    G.add_node(node_name, layer=layer_idx)
                    pos[node_name] = (layer_idx, y_offset - neuron)
                y_offset -= layer_size + 2

            for layer_idx in range(len(layers) - 1):
                for src in range(layers[layer_idx]):
                    for dst in range(layers[layer_idx + 1]):
                        G.add_edge(f"L{layer_idx}_N{src}", f"L{layer_idx + 1}_N{dst}")

            plt.figure(figsize=(10, 8))
            nx.draw(
                G, pos, with_labels=False, node_size=1000, node_color="skyblue", edge_color="gray", arrows=True
            )
            plt.title("Neural Network Visualization")
            plt.axis("off")

        input_size = data.num_attributes - 1
        hidden_size = 10
        output_size = data.class_attribute.num_values
        plot_neural_network(input_size, hidden_size, output_size)

        mlp_image_path = os.path.join(app.static_folder, "mlp_network.png")
        plt.savefig(mlp_image_path)
        plt.close()

        return jsonify({
            "status": "success",
            "report": metrics,
            "mlp_image_url": "/static/mlp_network.png",
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

import atexit
atexit.register(jvm.stop)
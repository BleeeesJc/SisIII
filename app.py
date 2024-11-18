from flask import Flask, jsonify, request,render_template
import weka.core.jvm as jvm
from weka.clusterers import Clusterer
from weka.classifiers import Classifier, Evaluation
from collections import defaultdict
from weka.core.converters import Loader
from weka.core.classes import Random
import tempfile
import os
import subprocess

app = Flask(__name__, static_folder="static")

jvm.start()

@app.route('/')
def index():
    return render_template('Algoritmos.html')


@app.route('/run_j48', methods=['POST'])
def run_j48():
    try:
        csv_path = "muestra_reducida100.csv"  

        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(csv_path)
        data.class_is_last()

        classifier = Classifier(classname="weka.classifiers.trees.J48")
        classifier.build_classifier(data)

        dot_path = os.path.join(app.static_folder, "tree.dot")
        with open(dot_path, "w") as dot_file:
            dot_file.write(classifier.graph)

        if not os.path.exists(dot_path):
            raise Exception(f"El archivo DOT no existe: {dot_path}")

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

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    try:
        num_clusters = int(request.json.get("num_clusters", 2))  
        csv_path = "Restaurants.csv"

        loader = Loader(classname="weka.core.converters.CSVLoader")
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

        full_data_mean = {
            data.attribute(idx).name: sum(instance.get_value(idx) for instance in data) / data.num_instances
            for idx in range(data.num_attributes)
        }

        cluster_distribution = []
        for cluster_idx, instances in clusters.items():
            count = len(instances)
            percentage = (count / data.num_instances) * 100
            cluster_distribution.append({"cluster": cluster_idx + 1, "count": count, "percentage": round(percentage, 2)})

        return jsonify({
            "status": "success",
            "cluster_centers": cluster_centers,
            "full_data_mean": full_data_mean,
            "cluster_distribution": cluster_distribution,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/run_mlp', methods=['POST'])
def run_mlp():
    try:
        csv_path = "Restaurants.csv" 

        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(csv_path)
        data.class_is_last()

        mlp = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron", 
                         options=["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "a"])
        mlp.build_classifier(data)

        evaluation = Evaluation(data)
        evaluation.crossvalidate_model(mlp, data, 10, Random(1))

        report = {
            "accuracy": evaluation.percent_correct / 100,  
            "kappa": evaluation.kappa,
            "mean_absolute_error": evaluation.mean_absolute_error,
            "root_mean_squared_error": evaluation.root_mean_squared_error,
            "relative_absolute_error": evaluation.relative_absolute_error,
            "root_relative_squared_error": evaluation.root_relative_squared_error,
            "total_instances": evaluation.num_instances,
        }

        return jsonify({"status": "success", "report": report})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

import atexit
atexit.register(jvm.stop)
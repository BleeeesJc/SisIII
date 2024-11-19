import csv

def detect_invalid_lines(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 5:  # Cambia '5' por el número correcto de columnas
                print(f"Línea {line_number} es inválida: {row}")

detect_invalid_lines("diabetes_reducida300.csv")
from PIL import Image
import numpy as np
import csv

# Cargar imagen
img = Image.open("pruebatest1.jpg").convert("L")  # Escala de grises
img = img.resize((28, 28))  # Asegura tamaño correcto

# Normalizar (0 a 1)
pixels = np.array(img, dtype=np.float32) / 255.0

# Guardar como CSV (matriz 28x28 en una sola fila)
with open("imagen_convertida1.csv", "w", newline='') as f:
    writer = csv.writer(f)
    for row in pixels:
        writer.writerow(row)

print(" Imagen convertida a 'imagen_convertida1.csv' con forma 28x28")

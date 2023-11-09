# -*- coding: utf-8 -*-
from flask import Flask, Blueprint
from app import *
import os
import torch
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# UPLOAD_FOLDER = os.path.abspath("../results")

blueprint_uploads = Blueprint(
    "uploads",
    __name__,
    static_folder=os.path.abspath("../uploads_files"),
    static_url_path="/uploads_files",
)


@app.route("/process", methods=["POST"])
def process_images():
    # Obtiene la ruta del directorio de las imágenes del cuerpo de la solicitud
    # images_dir = request.json["images_dir"]
    images_dir2 = os.path.abspath("../uploads_files")
    # Carga el modelo
    model = torch.hub.load("ultralytics/yolov5", "custom", path="model/best.pt")
    # Comprueba si el directorio existe
    if os.path.exists(UPLOAD_FOLDER):
        # Si existe, elimina su contenido
        shutil.rmtree(UPLOAD_FOLDER)

    # Crea el directorio
    os.makedirs(UPLOAD_FOLDER)
    processed_image_paths = []

    # Itera sobre todas las imágenes en el directorio
    for filename in os.listdir(images_dir2):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Asegúrate de que es una imagen
            # Abre la imagen
            image_path = os.path.join(images_dir2, filename)
            image = Image.open(image_path)

            # Realiza la predicción
            results = model(image)

            # Dibuja los cuadros delimitadores en la imagen
            fig, ax = plt.subplots(1)
            ax.axis("off")  # Desactiva los ejes
            canvas = FigureCanvas(fig)
            im = ax.imshow(image)

            for x1, y1, x2, y2, conf, cls in results.xyxy[0]:
                box = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(box)

            # Guarda la imagen procesada en un directorio diferente
            processed_image_path = os.path.join(UPLOAD_FOLDER, filename)
            canvas.print_figure(processed_image_path, dpi=300)

            # Cierra la figura para liberar memoria
            plt.close(fig)

            # Agrega la ruta de la imagen procesada a la lista
            processed_image_paths.append(processed_image_path)

    # Devuelve las rutas de las imágenes procesadas
    return {"processed_image_paths": processed_image_paths}


if __name__ == "__main__":
    app.register_blueprint(blueprint_uploads)
    app.run(debug=True, port=5004)

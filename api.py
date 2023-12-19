# -*- coding: utf-8 -*-
from flask import Flask, Blueprint
from app import *
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import textwrap
import requests
import datetime
from manga_ocr import MangaOcr
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# UPLOAD_FOLDER = os.path.abspath("../results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# UPLOAD_FOLDER = os.path.abspath("../results")
print(torch.cuda.is_available())
# Inicializar pygame
# pygame.init()

API_URL = "https://translation.googleapis.com/language/translate/v2?target=es&format=text&source=ja&model=base&key=AIzaSyCnudjE1ewZBDBIgXmVmIlidZ7vK6MQFb4&q="
# Verifica si CUDA (GPU) está disponible
if torch.cuda.is_available():
    # Imprime la cantidad de GPUs disponibles
    print(f"Cantidad de GPUs disponibles: {torch.cuda.device_count()}")
    # Imprime el nombre de la GPU actual
    print(f"GPU actual: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA no esta disponible. Se usara la CPU.")

# Inicializar el modelo de OCR
ocr = MangaOcr()
#print("\033[31m\033[1m\033[4m" + "Inicializando" + "\033[0m")

blueprint_uploads = Blueprint(
    "uploads",
    __name__,
    static_folder=os.path.abspath("/home/grupo3/app/uploads_files"),
    static_url_path="/uploads_files",
)
print("en el api.py")


@app.route("/")
def index():
    print("asodinsaoidsnaodsain oasdna osdinasodisnaosad")

@app.route("/process", methods=["POST"])
def process_images():
    # Obtiene la ruta del directorio de las imÃ¡genes del cuerpo de la solicitud
    # images_dir = request.json["images_dir"]
    images_dir2 = os.path.abspath("/home/grupo3/app/uploads_files")
    # Carga el modelo
    model = torch.hub.load("ultralytics/yolov5", "custom", path="/home/grupo3/app/mangslator-api/model/best.pt")
    # Comprueba si el directorio existe
    if os.path.exists(UPLOAD_FOLDER):
        # Si existe, elimina su contenido
        shutil.rmtree(UPLOAD_FOLDER)

    # Crea el directorio
    os.makedirs(UPLOAD_FOLDER)
    processed_image_paths = []

    # Itera sobre todas las imÃ¡genes en el directorio
    for filename in os.listdir(images_dir2):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):  # AsegÃºrate de que es una imagen
            # Abre la imagen
            image_path = os.path.join(images_dir2, filename)
            image = Image.open(image_path)

            # Realiza la predicciÃ³n
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

    # Devuelve las rutas de las imÃ¡genes procesadas
    return {"processed_image_paths": processed_image_paths}

def calcular_ancho_promedio(texto, fuente):
    total = sum(fuente.getbbox(char)[2] - fuente.getbbox(char)[0] for char in texto)
    return total / len(texto)


def change_font_size(ancho, alto, texto, fuente_inicial):
    # ancho_rect = rect[2] - rect[0]
    # alto_rect = rect[3] - rect[1]
    ImageFont.load_default()
    fuente = ImageFont.truetype(fuente_inicial, size=20)
    bbox = fuente.getbbox(texto)
    ancho_texto = fuente.getlength(texto)
    alto_texto = bbox[3] - bbox[1]  # Añade la altura del texto

    while ancho_texto > ancho - 3 and alto_texto > alto - 3:
        fuente = ImageFont.truetype(fuente_inicial, size=fuente.size - 1)
        bbox = fuente.getbbox(texto)
        ancho_texto = fuente.getlength(texto)
        alto_texto = bbox[3] - bbox[1]  # Añade la altura del texto

    return fuente


@app.route("/process2", methods=["POST"])
def process_images2():
    images_dir2 = os.path.abspath("/home/grupo3/app/uploads_files")

    # Carga el modelo
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path="/home/grupo3/app/mangslator-api/model/best_texto_completo.pt"
    )
    # Comprueba si el directorio existe
    if os.path.exists(UPLOAD_FOLDER):
        # Si existe, elimina su contenido
        shutil.rmtree(UPLOAD_FOLDER)
    if os.path.exists(CONF_RESULTS):
        # Si existe, elimina su contenido
        shutil.rmtree(CONF_RESULTS)
    # Crea el directorio
    os.makedirs(UPLOAD_FOLDER)
    os.makedirs(CONF_RESULTS)
    processed_image_paths = []

    # Itera sobre todas las imágenes en el directorio
    for filename in os.listdir(images_dir2):
        print(images_dir2)
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(images_dir2, filename)
            image = Image.open(image_path)

            # Realiza la predicción
            results = model(image)
            rect_coords = []
            # Dibuja los cuadros delimitadores en la imagen
            fig, ax = plt.subplots(1)
            ax.axis("off")  # Desactiva los ejes
            canvas = FigureCanvas(fig)
            im = ax.imshow(image)
            rect_coords = []
            for x1, y1, x2, y2, conf, cls in results.xyxy[0]:
                box = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(box)

                # Agregar el valor de confianza al rectángulo
                # if conf < 0.6:
                #    continue
                ax.text(x1, y1, f"Conf: {conf:.2f}", color="r", fontsize=8)
                if conf > 0.6:
                    rect_coords.append((x1, y1, x2, y2))

            # Capturar las secciones correspondientes de la imagen original
            imagen = cv2.imread(image_path)

            color = (255, 255, 255)
            color_negro = (0, 0, 0)
            captured_images = []
            for coords in rect_coords:
                x1, y1, x2, y2 = coords
                # Convertir tensores a enteros usando item()
                x1, y1, x2, y2 = (
                    int(x1.item()),
                    int(y1.item()),
                    int(x2.item()),
                    int(y2.item()),
                )

                punto1 = (x1, y1)
                punto2 = (x2, y2)
                punto1_negro = (x1 - 2, y1 - 2)
                punto2_negro = (x2 + 2, y2 + 2)
                captured_image = image.crop((x1, y1, x2, y2))
                captured_images.append(captured_image)
                # Calcular el ancho y alto del área
                ancho = punto2[0] - punto1[0]
                alto = punto2[1] - punto1[1]
                # Calcular el centro
                x_centro = int((punto1[0] + punto2[0]) / 2)
                y_centro = int((punto1[1] + punto2[1]) / 2)
                punto_inicio = (x_centro, y_centro)
                text = ocr(captured_image)

                # fuente = ImageFont.truetype(
                #    "arial.ttf", 15
                # )

                apicall = requests.post(API_URL + text)
                print(apicall.json())

                texto = apicall.json()["data"]["translations"][0]["translatedText"]
                texto = texto.replace(".. ..", "...")
                print(f"TEXTO EN x1,y1,x2,y2 {(x1, y1, x2, y2)}", text, "que es", texto)

                fuente = change_font_size(ancho, alto, texto, "arial.ttf")

                # Usar textwrap para dividir el texto en líneas que caben dentro del área
                print(fuente.getbbox(" "))
                ancho_promedio = calcular_ancho_promedio(texto, fuente)
                max_chars_por_linea = int(ancho / ancho_promedio)

                lineas = textwrap.wrap(texto, width=max_chars_por_linea)
                # Dibujar el rectángulo en la imagen

                # cv2.rectangle(imagen, punto1_negro, punto2_negro, color_negro, -1)
                cv2.rectangle(imagen, punto1, punto2, color, -1)

                imagen_pil = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
                # Crear un objeto ImageDraw
                d = ImageDraw.Draw(imagen_pil)

                # Dibujar cada línea de texto en la imagen
                for i, linea in enumerate(lineas):
                    bbox = fuente.getbbox(linea)
                    width = fuente.getlength(linea)
                    # Calcular la posición x para dibujar el texto
                    x_text = punto1[0] + (ancho - width) / 2
                    # Calcular la posición y para dibujar el texto
                    y_text = (
                        punto1[1]
                        + (alto - (bbox[3] - bbox[1]) * len(lineas)) / 2
                        + (bbox[3] - bbox[1]) * i
                    )
                    d.text(
                        (x_text, y_text),
                        linea,
                        font=fuente,
                        fill=(0, 0, 0),
                    )
                imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

                # Guardar la imagen
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), imagen)

            # Guarda la imagen procesada en un directorio diferente
            processed_image_path = os.path.join(CONF_RESULTS, "conf_" + filename)
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

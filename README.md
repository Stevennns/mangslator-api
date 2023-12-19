# Magslator API

Una aplicación creada por :

- Stevens Egli @Stevennns
- Jesús Cáceres @JexexJZkrall
- Nicolás Cáceres @
- Manuel Molina @ManuelMolinaH
- Carlo Merino @EstudianteGenerico047

## Descripción

Este proyecto representa la API de una aplicación diseñada para la traducción automática de mangas del japonés al español. Esta iniciativa se desarrolla como parte del curso "CC6409: Taller de Desarrollo de Proyectos de IA".

## Instalación

1. Clonar repositorio:

```bash
git clone [insert repository URL]
```

2. Crear un virtualenv (versión originar de python 3.11), por ejemplo:

```bash
python3 -m venv env
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Algunas librerias pueden no estar declaradas en el requirements.txt, por lo que se deben instalar manualmente.



4. Correr aplicación:

```bash

python api.py

`````

## Uso

Mediante la ruta /process2 se hace un procesamiento de las imagenes subidas mediante la app web y se procesan mediante el modelo de traducción. Para acceder, este codigo tiene configurada la ruta

`http://localhost:5004/process2`

**Para funcionar requiere un archivo .pt en el directorio `\model` que corresponde a los mejores pesos de entrenar un modelo de detección de dialogos de mangas usando el modelo de YOLOv5**

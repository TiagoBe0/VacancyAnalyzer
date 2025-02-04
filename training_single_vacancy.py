# single_vacancy_processor.py

import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, VoronoiAnalysisModifier, ClusterAnalysisModifier
import os
import json
import random
import re
from input_params import LAYERS

class SingleVacancyProcessor:
    def __init__(self, layer):
        """
        Inicializa el procesador con los parámetros de la capa.
        
        Parámetros:
            layer (dict): Diccionario con las claves 'relax', 'cutoff radius', 'radius' y 'smoothing level'
        """
        primer_elemento=LAYERS[0]
        self.layer = primer_elemento
        self.relax = primer_elemento['relax']
        self.cutoff_radius = primer_elemento['cutoff radius']
        self.radius = primer_elemento['radius']
        self.smoothing_level = primer_elemento['smoothing level']
        self.pipeline = None
        # Listas para almacenar datos; puedes modificarlas según lo requieras
        self.sms_sv = []  # Área(s) encontrada(s) para single vacancy
        self.nb_sv = []   # Número de vecinos

    @staticmethod
    def extraer_ids(archivo):
        """
        Lee un archivo y extrae los IDs utilizando expresiones regulares.

        Parámetros:
            archivo (str): Ruta del archivo a leer.

        Retorna:
            list: Lista de enteros con los IDs extraídos.
        """
        with open(archivo, 'r') as f:
            contenido = f.read()

        # Se buscan líneas que comiencen con números (posibles IDs)
        ids = re.findall(r'^\s*([0-9]+)', contenido, re.MULTILINE)
        return [int(id_str) for id_str in ids]

    def run(self):
        """
        Ejecuta el procesamiento: carga el archivo, aplica modificadores, exporta el resultado
        y guarda los datos en un archivo JSON.
        """
        # Cargar el archivo utilizando OVITO
        self.pipeline = import_file(self.relax)
        
        # Extraer IDs del archivo
        ids = self.extraer_ids(self.relax)
        if not ids:
            print("No se encontraron IDs en el archivo.")
            return
        
        # Seleccionar un ID aleatorio
        id_aleatorio = int(random.choice(ids))
        
        # Agregar modificadores a la pipeline
        self.pipeline.modifiers.append(ExpressionSelectionModifier(expression=f'ParticleIdentifier=={id_aleatorio}'))
        self.pipeline.modifiers.append(DeleteSelectedModifier())
        self.pipeline.modifiers.append(VoronoiAnalysisModifier(compute_indices=True))
        self.pipeline.modifiers.append(ClusterAnalysisModifier(
            cutoff=self.cutoff_radius, 
            cluster_coloring=True, 
            unwrap_particles=True, 
            sort_by_size=True
        ))
        
        # Computar la pipeline para aplicar los modificadores
        data = self.pipeline.compute()
        vecinos = data.particles.count  # Se obtiene el número de partículas tras las modificaciones
        
        # Aquí podrías actualizar self.sms_sv y self.nb_sv según la lógica que requieras.
        # Por ejemplo:
        # self.sms_sv.append(algún_valor)
        # self.nb_sv.append(vecinos)
        
        # Intentar exportar el archivo modificado
        try:
            export_file(
                self.pipeline, 
                "single_vacancy_training.dump", 
                "lammps/dump",
                columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z",
                         "Atomic Volume", "Cavity Radius", "Max Face Order"]
            )
            # Limpiar los modificadores después de exportar
            self.pipeline.modifiers.clear()
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")
        
        # Guardar los vectores en un archivo JSON
        datos = {
            'sms_sv': self.sms_sv,
            'nb_sv': self.nb_sv
        }
        output_path = 'outputs.json/key_single_vacancy.json'
        # Asegurarse de que el directorio de salida exista
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w') as f:
            json.dump(datos, f, indent=4)

# Definimos las Variables Globales y Paths que son Comunes tanto para la Clasificación como para la Detección (también incluimos alguno específico)

from pathlib import Path
import torch 

# Rutas
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
IMAGE_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "metadata.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Usamos la Gráfica si se puede


#Hiperparámetros Generales
NUM_EPOCHS = 50                 # Cantidad de Epochs que se van a realizar en 1 Entrenamiento Completo
LEARNING_RATE = 0.0005           # Qué tan rápido Aprenden los Modelos
PATIENCE = 8                    # Cuántos Epochs seguidos sin mejorar deben pasar para Aplicar el Early Stoppin

# Hiperparámetros Clasificación
IMAGE_SIZE = (448, 448)         # Tamaño de Imagen
BATCH_SIZE = 32                 # Cantidad de Imágenes tratadas "a la vez"

# Hiperparámetros Específicos Detección
DET_IMAGE_SIZE = (800, 800)  
DET_BATCH_SIZE = 6
SCORE_THRESHOLD = 0.05          # A partir de qué Valor consideramos una Predicción como Válida
NUM_CLASSES = 2                 # Cantidad de Clases = 2 (sin nódulos y con nódulos)
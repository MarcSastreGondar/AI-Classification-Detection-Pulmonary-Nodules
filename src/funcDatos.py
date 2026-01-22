# Funciones para Carga y Modificación de la Imágenes del Dataset

from PIL import Image
import SimpleITK as sitk
import torch
from torchvision import transforms

from sklearn.model_selection import train_test_split
import pandas as pd
from src.config import METADATA_FILE


# Carga y normaliza las imágenes .mha
def load_mha_image(path, target_size=None):
    try:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).squeeze()
        
        # Normalización Min-Max
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype('uint8')
        
        image = Image.fromarray(arr).convert('RGB')
        
        # Si se Especifica, redimensionamos a ese Tamaño
        if target_size:
            image = image.resize(target_size)
        
        return image
    
    # Si encontramos algún tipo de Error, devolvemos una Imagen en negro del Tamaño Especificado para evitar Errores
    except Exception as e:
        print(f"Error cargando imagen {path}: {e}")
        size = target_size if target_size else (448, 448)
        return Image.new('RGB', size, (0, 0, 0))


# Aplica Transformaciones
def preprocess_image(image, transform=None):
    # Los argumentos ya son la Imagen Cargada y las Transformaciones a Aplicar, así que simplemente las aplicamos
    if transform:
        image = transform(image)
    return image



# Creamos un Split de Datos, por defecto 80/20
def get_train_test_split(test_size=0.2, random_state=42):

    # Cargamos el Metadata y eliminamos duplicados
    df_full = pd.read_csv(METADATA_FILE)
    df_unique = df_full.drop_duplicates(subset='img_name').copy()
    
    # Hacemos la División Estratificada por Etiqueta para mantener la Proporción entre ambas Clases
    train_df, test_df = train_test_split(
        df_unique,
        test_size=test_size,
        stratify=df_unique['label'],
        random_state=random_state
    )
    
    # Imprimimos las Características de los Splits
    print(f"Total Imágenes = {len(df_unique)}")
    print(f"Train tiene {len(train_df)} imágenes")
    print(f"Test tiene {len(test_df)} imágenes")
    print(f"Distribución en Test, Sin Nódulo = {sum(test_df['label']==0)} | "
          f"Con Nódulo = {sum(test_df['label']==1)}")
    
    # Devolvemos el Split del Train y del Test, además de todo el Metadata Cargado por si es necesario
    return train_df, test_df, df_full
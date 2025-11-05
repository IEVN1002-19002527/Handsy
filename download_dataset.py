import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Inicializar API de Kaggle
api = KaggleApi()
api.authenticate()  # Usa el kaggle.json configurado

# Descargar el dataset
dataset_slug = 'grassknoted/asl-alphabet'
download_path = 'asl_alphabet.zip'

print("Descargando dataset...")
print(f"Dataset: {dataset_slug}")
try:
    api.dataset_download_files(dataset_slug, path='.', unzip=False)  # Descarga como ZIP
    print("Descarga completada.")
except Exception as e:
    print(f"Error durante la descarga: {e}")
    raise

# Descomprimir
print("Descomprimiendo...")
with zipfile.ZipFile('asl-alphabet.zip', 'r') as zip_ref:
    zip_ref.extractall('temp_dataset')

# Organizar en carpeta 'data/'
data_dir = 'data'
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)  # Limpiar si existe
os.makedirs(data_dir, exist_ok=True)

# Mover carpetas de clases (A-Z, space, etc.) a data/
temp_dir = 'temp_dataset/asl_alphabet_train'  # Ajusta según la estructura del ZIP
for folder in os.listdir(temp_dir):
    if folder.startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space')):
        shutil.move(os.path.join(temp_dir, folder), os.path.join(data_dir, folder))

# Limpiar archivos temporales
shutil.rmtree('temp_dataset')
os.remove('asl-alphabet.zip')

print("Dataset agregado a la carpeta 'data/'. Estructura:")
for folder in sorted(os.listdir(data_dir)):
    print(f"  - {folder}: {len(os.listdir(os.path.join(data_dir, folder)))} imágenes")
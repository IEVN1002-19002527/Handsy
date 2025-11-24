"""
Script para entrenar el modelo de días (lunes-jueves) con videos capturados de la cámara.
Usa un modelo LSTM para procesar secuencias de frames.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configuración
DATA_DIR = "collected_day_data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "day_model_camera.h5")
CLASS_NAMES = ['lunes', 'martes', 'miercoles', 'jueves']  # Solo lunes a jueves
SEQUENCE_LENGTH = 30  # Número de frames por video (ajustar según duración)
FRAME_SIZE = (28, 28)  # Tamaño de cada frame

def load_videos_from_directory(data_dir):
    """Cargar videos desde los directorios de clases"""
    sequences = []
    labels = []
    
    print("Cargando videos...")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"  [ADVERTENCIA] Directorio no encontrado: {class_dir}")
            continue
        
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        
        if len(video_files) == 0:
            print(f"  [ADVERTENCIA] No hay videos para el día '{class_name}'")
            continue
        
        print(f"  Cargando {len(video_files)} videos del día '{class_name}'...")
        
        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            
            # Leer video
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # El frame ya debería estar en escala de grises y 28x28
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Verificar dimensiones
                if frame.shape != FRAME_SIZE:
                    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
                
                # Normalizar a [0, 1]
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                continue
            
            # Ajustar longitud de secuencia (padding o truncamiento)
            if len(frames) < SEQUENCE_LENGTH:
                # Padding: repetir el último frame
                last_frame = frames[-1]
                while len(frames) < SEQUENCE_LENGTH:
                    frames.append(last_frame.copy())
            elif len(frames) > SEQUENCE_LENGTH:
                # Truncamiento: tomar frames uniformemente distribuidos
                indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Convertir a array
            sequence = np.array(frames)
            sequences.append(sequence)
            labels.append(class_idx)
    
    if len(sequences) == 0:
        raise ValueError("No se encontraron videos para entrenar. Ejecuta 'collect_day_data.py' primero.")
    
    print(f"\n[OK] Total de videos cargados: {len(sequences)}")
    print(f"[OK] Número de clases: {len(set(labels))}")
    
    # Convertir a arrays de numpy
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Reshape para el modelo: (samples, timesteps, height, width, channels)
    sequences = sequences.reshape(-1, SEQUENCE_LENGTH, FRAME_SIZE[0], FRAME_SIZE[1], 1)
    
    # One-hot encoding de las etiquetas
    num_classes = len(CLASS_NAMES)
    labels_categorical = to_categorical(labels, num_classes)
    
    return sequences, labels_categorical, labels

def build_model(num_classes, sequence_length, frame_size):
    """Construir modelo LSTM para procesar secuencias de frames"""
    model = Sequential([
        # Capa convolucional temporal para extraer características de cada frame
        TimeDistributed(
            Conv2D(32, kernel_size=(3,3), activation='relu'),
            input_shape=(sequence_length, frame_size[0], frame_size[1], 1)
        ),
        TimeDistributed(MaxPooling2D((2,2))),
        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
        TimeDistributed(MaxPooling2D((2,2))),
        TimeDistributed(Flatten()),
        
        # Capas LSTM para procesar la secuencia temporal
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        
        # Capa de salida
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_model_from_scratch(sequences, labels):
    """Entrenar modelo desde cero"""
    print("\n" + "="*60)
    print("ENTRENANDO MODELO DESDE CERO")
    print("="*60)
    
    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nDatos de entrenamiento: {len(X_train)} videos")
    print(f"Datos de validación: {len(X_val)} videos")
    
    # Construir modelo
    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes, SEQUENCE_LENGTH, FRAME_SIZE)
    
    print("\nArquitectura del modelo:")
    model.summary()
    
    # Configurar callbacks
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Entrenar modelo
    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,  # Batch size más pequeño para videos
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def main():
    """Función principal"""
    print("="*60)
    print("ENTRENAMIENTO DEL MODELO DE DIAS (LUNES-JUEVES)")
    print("="*60)
    
    # Verificar que existen datos
    if not os.path.exists(DATA_DIR):
        print(f"\n[ADVERTENCIA] No se encuentra el directorio de datos: {DATA_DIR}")
        print("   Creando directorio y subdirectorios necesarios...")
        os.makedirs(DATA_DIR, exist_ok=True)
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(DATA_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
        print(f"   [OK] Directorio creado: {DATA_DIR}")
        print(f"   [OK] Subdirectorios creados: {', '.join(CLASS_NAMES)}")
        print("\n[ADVERTENCIA] Los directorios están vacíos. Necesitas agregar videos antes de entrenar.")
        print("   Ejecuta 'collect_day_data.py' para capturar videos o agrega videos manualmente.")
        print("\n[INFO] Los directorios han sido creados. Agrega videos y vuelve a ejecutar el script.")
        return
    
    # Cargar videos
    try:
        sequences, labels_categorical, labels = load_videos_from_directory(DATA_DIR)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\n[INFO] Asegúrate de que existan videos en los subdirectorios:")
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(DATA_DIR, class_name)
            if os.path.exists(class_dir):
                video_count = len([f for f in os.listdir(class_dir) if f.endswith('.mp4')])
                print(f"   {class_dir}: {video_count} videos")
        return
    
    # Verificar que hay suficientes datos
    min_videos_per_class = 3  # Mínimo reducido ya que solo necesitamos 10 videos por clase
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    insufficient_classes = [CLASS_NAMES[i] for i in class_counts.keys() 
                           if class_counts[i] < min_videos_per_class]
    
    if insufficient_classes:
        print(f"\n[ADVERTENCIA] Algunas clases tienen menos de {min_videos_per_class} videos:")
        for class_name in insufficient_classes:
            class_idx = CLASS_NAMES.index(class_name)
            count = class_counts.get(class_idx, 0)
            print(f"  {class_name}: {count} videos")
        print("\nSe recomienda tener al menos 10 videos por clase para un buen entrenamiento.")
        print("¿Deseas continuar de todas formas? (s/n): ", end='')
        response = input().strip().lower()
        if response != 's':
            return
    
    # Entrenar modelo
    model, history = train_model_from_scratch(sequences, labels_categorical)
    
    # Guardar modelo final
    print(f"\n[OK] Modelo guardado en: {os.path.abspath(MODEL_PATH)}")
    
    # Evaluar modelo
    print("\n" + "="*60)
    print("EVALUACIÓN DEL MODELO")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels_categorical, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nPrecisión en entrenamiento: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Precisión en validación: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    print("\n" + "="*60)
    print("[OK] ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nModelo guardado en: {MODEL_PATH}")
    print("\nEl modelo se cargará automáticamente en app.py cuando esté disponible.")

if __name__ == '__main__':
    main()


"""
Script para reentrenar el modelo con las imágenes capturadas de la cámara.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Configuración
DATA_DIR = "collected_data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model_camera.h5")
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def load_images_from_directory(data_dir):
    """Cargar imágenes desde los directorios de clases"""
    images = []
    labels = []
    
    print("Cargando imágenes...")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"  ⚠ Directorio no encontrado: {class_dir}")
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        
        if len(image_files) == 0:
            print(f"  ⚠ No hay imágenes para la clase '{class_name}'")
            continue
        
        print(f"  Cargando {len(image_files)} imágenes de la clase '{class_name}'...")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            # Leer imagen (ya está en escala de grises y 28x28)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Verificar dimensiones
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalizar a [0, 1]
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(class_idx)
    
    if len(images) == 0:
        raise ValueError("No se encontraron imágenes para entrenar. Ejecuta 'collect_training_data.py' primero.")
    
    print(f"\n✓ Total de imágenes cargadas: {len(images)}")
    print(f"✓ Número de clases: {len(set(labels))}")
    
    # Convertir a arrays de numpy
    images = np.array(images)
    labels = np.array(labels)
    
    # Reshape para el modelo (agregar dimensión de canal)
    images = images.reshape(-1, 28, 28, 1)
    
    # One-hot encoding de las etiquetas
    num_classes = len(CLASS_NAMES)
    labels_categorical = to_categorical(labels, num_classes)
    
    return images, labels_categorical, labels

def build_model(num_classes):
    """Construir el modelo (misma arquitectura que el modelo original)"""
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_model_from_scratch(images, labels):
    """Entrenar modelo desde cero"""
    print("\n" + "="*60)
    print("ENTRENANDO MODELO DESDE CERO")
    print("="*60)
    
    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nDatos de entrenamiento: {len(X_train)} imágenes")
    print(f"Datos de validación: {len(X_val)} imágenes")
    
    # Construir modelo
    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes)
    
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
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def fine_tune_model(images, labels, base_model_path=None):
    """Fine-tuning: cargar modelo existente y reentrenarlo con nuevas imágenes"""
    print("\n" + "="*60)
    print("FINE-TUNING: REENTRENANDO MODELO EXISTENTE")
    print("="*60)
    
    num_classes = len(CLASS_NAMES)
    
    # Intentar cargar modelo base
    if base_model_path and os.path.exists(base_model_path):
        print(f"\nCargando modelo base desde: {base_model_path}")
        
        # Intentar cargar modelo completo primero
        base_model = None
        try:
            base_model = tf.keras.models.load_model(base_model_path)
            print("✓ Modelo base cargado completamente")
        except Exception as e:
            if "batch_shape" in str(e) or "Unrecognized keyword" in str(e):
                print("  ⚠ Problema de compatibilidad detectado, cargando solo pesos...")
            else:
                print(f"  ⚠ Error al cargar modelo: {e}")
                print("  Intentando cargar solo pesos...")
        
        # Si no se pudo cargar el modelo completo, construir nuevo y cargar pesos
        if base_model is None:
            print("  Construyendo nuevo modelo con misma arquitectura...")
            model = build_model(num_classes)
            try:
                # Intentar cargar pesos del modelo base
                model.load_weights(base_model_path, by_name=True, skip_mismatch=True)
                print("  ✓ Pesos cargados (algunos pueden no coincidir si la arquitectura difiere)")
            except Exception as e2:
                print(f"  ⚠ No se pudieron cargar los pesos: {e2}")
                print("  Usando pesos aleatorios iniciales...")
        else:
            # Modelo cargado correctamente
            # Verificar número de clases
            try:
                base_num_classes = base_model.layers[-1].output_shape[-1]
                print(f"  Clases en modelo base: {base_num_classes}")
                print(f"  Clases necesarias: {num_classes}")
                
                if base_num_classes == num_classes:
                    # El modelo tiene el número correcto de clases
                    model = base_model
                    print("  ✓ El modelo base tiene el número correcto de clases")
                    # Configurar learning rate más bajo para fine-tuning
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                else:
                    # El modelo tiene diferente número de clases
                    print("  ⚠ El modelo base tiene diferente número de clases")
                    print("  Construyendo nuevo modelo y cargando pesos de capas convolucionales...")
                    # Construir nuevo modelo
                    model = build_model(num_classes)
                    # Intentar transferir pesos de las capas convolucionales
                    try:
                        # Las primeras capas (convolucionales) deberían ser compatibles
                        for i, layer in enumerate(model.layers):
                            if i < len(base_model.layers) - 2:  # Excluir últimas capas
                                try:
                                    layer.set_weights(base_model.layers[i].get_weights())
                                    print(f"    ✓ Pesos transferidos para capa {i}: {layer.name}")
                                except Exception:
                                    print(f"    ⚠ No se pudieron transferir pesos para capa {i}: {layer.name}")
                    except Exception as e3:
                        print(f"  ⚠ Error al transferir pesos: {e3}")
                        print("  Usando pesos aleatorios iniciales...")
            except Exception as e:
                print(f"  ⚠ Error al verificar modelo: {e}")
                print("  Construyendo nuevo modelo...")
                model = build_model(num_classes)
    else:
        print("\n⚠ No se encontró modelo base. Construyendo nuevo modelo desde cero...")
        model = build_model(num_classes)
    
    # Dividir datos
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nDatos de entrenamiento: {len(X_train)} imágenes")
    print(f"Datos de validación: {len(X_val)} imágenes")
    
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
    
    # Entrenar (el learning rate ya se configuró arriba si se cargó el modelo base)
    # Si no se configuró, usar learning rate estándar
    try:
        # Verificar si ya se compiló con learning rate bajo
        current_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else None
        if current_lr is None or current_lr >= 0.001:
            # No se configuró learning rate bajo, configurarlo ahora
            print("\nConfigurando learning rate reducido para fine-tuning...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    except:
        # Si hay algún error, compilar con learning rate estándar
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Entrenar
    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def main():
    """Función principal"""
    print("="*60)
    print("REENTRENAMIENTO DEL MODELO CON IMÁGENES DE CÁMARA")
    print("="*60)
    
    # Verificar que existen datos
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: No se encuentra el directorio de datos: {DATA_DIR}")
        print("   Ejecuta primero 'collect_training_data.py' para capturar imágenes.")
        return
    
    # Cargar imágenes
    try:
        images, labels_categorical, labels = load_images_from_directory(DATA_DIR)
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Verificar que hay suficientes datos
    min_images_per_class = 10
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    insufficient_classes = [CLASS_NAMES[i] for i in class_counts.keys() 
                           if class_counts[i] < min_images_per_class]
    
    if insufficient_classes:
        print(f"\n⚠ ADVERTENCIA: Algunas clases tienen menos de {min_images_per_class} imágenes:")
        for class_name in insufficient_classes:
            class_idx = CLASS_NAMES.index(class_name)
            count = class_counts.get(class_idx, 0)
            print(f"  {class_name}: {count} imágenes")
        print("\nSe recomienda tener al menos 10-20 imágenes por clase para un buen entrenamiento.")
        print("¿Deseas continuar de todas formas? (s/n): ", end='')
        response = input().strip().lower()
        if response != 's':
            return
    
    # Preguntar tipo de entrenamiento
    print("\n" + "="*60)
    print("TIPO DE ENTRENAMIENTO")
    print("="*60)
    print("1. Entrenar desde cero (recomendado si tienes muchas imágenes)")
    print("2. Fine-tuning (reentrenar modelo existente)")
    print("\nSelecciona una opción (1 o 2): ", end='')
    
    choice = input().strip()
    
    base_model_path = os.path.join(MODEL_DIR, "gesture_model.h5")
    
    if choice == "1":
        model, history = train_model_from_scratch(images, labels_categorical)
    elif choice == "2":
        model, history = fine_tune_model(images, labels_categorical, base_model_path)
    else:
        print("Opción inválida. Usando entrenamiento desde cero.")
        model, history = train_model_from_scratch(images, labels_categorical)
    
    # Guardar modelo final
    print(f"\n✓ Modelo guardado en: {os.path.abspath(MODEL_PATH)}")
    
    # Evaluar modelo
    print("\n" + "="*60)
    print("EVALUACIÓN DEL MODELO")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_categorical, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\nPrecisión en entrenamiento: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Precisión en validación: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nModelo guardado en: {MODEL_PATH}")
    print("\nPara usar el nuevo modelo, actualiza 'app.py' para cargar:")
    print(f"  MODEL_PATH = '{MODEL_PATH}'")
    print("\nO renombra el archivo:")
    print(f"  mv {MODEL_PATH} {os.path.join(MODEL_DIR, 'gesture_model.h5')}")

if __name__ == '__main__':
    main()


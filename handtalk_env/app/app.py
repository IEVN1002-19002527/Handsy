"""
CONFIGURACIÓN DEL MODELO:
- El código está configurado para usar PRIMERO el modelo reentrenado (gesture_model_camera.h5)
- Para reentrenar el modelo con imágenes de tu cámara, ejecuta: python train_camera_model.py
- El modelo reentrenado se guarda en: model/gesture_model_camera.h5
"""
from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
import random
app = Flask(__name__)

model = None
number_model = None  # Modelo separado para números
# Sign Language MNIST tiene 24 clases (0-24, pero falta el 9 que es J)
# El modelo fue entrenado con este dataset, así que las clases son:
# 0-8: A-I, 10-24: K-Y (J no está porque requiere movimiento)
# Esto puede causar problemas de mapeo
asl_letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
# Si el modelo tiene 25 clases, puede que la clase 9 sea una clase adicional o esté vacía
class_names = []
for i in range(25):
    if i < 9:
        class_names.append(asl_letters[i])  # 0-8: A-I
    elif i == 9:
        class_names.append('J')  # J puede estar o no, depende del modelo
    else:
        class_names.append(asl_letters[i-1] if (i-1) < len(asl_letters) else f'Clase_{i}')  # 10-24: K-Y
# Variable global para almacenar la predicción actual
current_prediction = "Esperando gesto..."

# Variables globales para el modo de práctica letra por letra
# Solo letras de A a O (sin J)
available_letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O']  # A-O sin J
current_letter_index = 0  # Índice de la letra objetivo actual
letter_detected = False  # Si la letra objetivo fue detectada correctamente

# Variables globales para el modo de práctica número por número
available_numbers = ['0', '1', '2', '3', '4', '5']  # Números del 0 al 5
current_number_index = 0  # Índice del número objetivo actual
number_detected = False  # Si el número objetivo fue detectado correctamente
current_number_prediction = "Esperando gesto..."  # Predicción actual para números

# Variables globales para el modo de práctica mes por mes
available_months = ['enero', 'febrero', 'marzo', 'abril']  # Solo enero a abril
current_month_index = 0  # Índice del mes objetivo actual
month_detected = False  # Si el mes objetivo fue detectado correctamente
current_month_prediction = "Esperando gesto..."  # Predicción actual para meses
month_model = None  # Modelo para meses
month_sequence = []  # Buffer para almacenar secuencia de frames
SEQUENCE_LENGTH = 30  # Longitud de secuencia para detección de meses
is_recording_month = False  # Indicador de grabación de video

# Variables globales para el modo de práctica día por día
available_days = ['lunes', 'martes', 'miercoles', 'jueves']  # Solo lunes a jueves
current_day_index = 0  # Índice del día objetivo actual
day_detected = False  # Si el día objetivo fue detectado correctamente
current_day_prediction = "Esperando gesto..."  # Predicción actual para días
day_model = None  # Modelo para días
day_sequence = []  # Buffer para almacenar secuencia de frames
is_recording_day = False  # Indicador de grabación de video

# Variables globales para el modo traductor
translator_prediction = "Esperando detección..."  # Predicción actual para traductor
translator_sequence = []  # Buffer para secuencias de video (meses/días)

mode = 'letters'  # Modo actual: 'letters', 'numbers', 'months', 'days' o 'translator'

try:
    import tensorflow as tf
    print(f"✓ TensorFlow importado correctamente")
    
    # Intentar diferentes formas de importar load_model
    load_model_func = None
    if hasattr(tf, 'keras') and hasattr(tf.keras, 'models'):
        load_model_func = tf.keras.models.load_model
        print(f"✓ tf.keras.models disponible")
    else:
        print(f"✗ tf.keras.models no disponible")
        try:
            import keras
            from keras.models import load_model  # type: ignore
            load_model_func = load_model
            print(f"✓ keras.models disponible como alternativa")
        except ImportError as e:
            print(f"✗ Error al importar keras: {e}")
    
    # Buscar el modelo en múltiples ubicaciones posibles
    # Prioridad: 1) modelo reentrenado con cámara, 2) modelo original
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        # Primero intentar modelo reentrenado con imágenes de cámara (si existe)
        os.path.join(script_dir, 'model', 'gesture_model_camera.h5'),  # Modelo reentrenado
        # Luego el modelo original
        os.path.join(script_dir, 'model', 'gesture_model.h5'),  # app/model/gesture_model.h5
        os.path.join(script_dir, 'gesture_model.h5'),  # app/gesture_model.h5
        os.path.join(os.path.dirname(script_dir), 'model', 'gesture_model.h5'),  # handtalk_env/model/gesture_model.h5
        'model/gesture_model.h5',
        'gesture_model.h5',
        os.path.join(os.getcwd(), 'model', 'gesture_model.h5'),
        os.path.join(os.getcwd(), 'gesture_model.h5'),
    ]
    
    MODEL_PATH = None
    print(f"\n{'='*60}")
    print(f"BÚSQUEDA DEL MODELO DE RECONOCIMIENTO DE GESTOS")
    print(f"{'='*60}")
    print(f"Directorio del script: {script_dir}")
    print(f"Directorio de trabajo actual: {os.getcwd()}")
    print(f"\nPrioridad: Se busca PRIMERO el modelo REENTRENADO (gesture_model_camera.h5)")
    print(f"          Luego el modelo original (gesture_model.h5)")
    print(f"\nBuscando modelo en las siguientes ubicaciones:")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        model_type = " (REENTRENADO - PRIORIDAD)" if "camera" in path else ""
        print(f"  {'✓' if exists else '✗'} {abs_path}{model_type}")
        if exists and MODEL_PATH is None:
            MODEL_PATH = path
            if "camera" in path:
                print(f"\n  {'='*60}")
                print(f"  ✓✓✓ Modelo REENTRENADO encontrado y seleccionado")
                print(f"  {'='*60}")
                print(f"  → Ruta: {abs_path}")
                print(f"  → Este modelo fue entrenado con imágenes de tu cámara")
                print(f"  → Debería tener mejor precisión que el modelo original")
            else:
                print(f"\n  {'='*60}")
                print(f"  ✓ Modelo ORIGINAL encontrado y seleccionado")
                print(f"  {'='*60}")
                print(f"  → Ruta: {abs_path}")
                print(f"  → Nota: Si tienes un modelo reentrenado, debería estar en:")
                print(f"    {os.path.join(script_dir, 'model', 'gesture_model_camera.h5')}")
    
    if MODEL_PATH and load_model_func:
        try:
            print(f"\nIntentando cargar modelo desde: {os.path.abspath(MODEL_PATH)}")
            # Intentar cargar directamente primero
            is_retrained = "camera" in MODEL_PATH
            model_type_str = "REENTRENADO" if is_retrained else "ORIGINAL"
            try:
                model = load_model_func(MODEL_PATH)
                print(f"\n{'='*60}")
                print(f"✓✓✓ Modelo {model_type_str} cargado correctamente!")
                print(f"{'='*60}")
                if is_retrained:
                    print(f"  → Usando modelo entrenado con imágenes de tu cámara")
                    print(f"  → Este modelo debería tener mejor precisión para tus condiciones")
                else:
                    print(f"  → Usando modelo original del dataset Sign Language MNIST")
                    print(f"  → Para mejor precisión, reentrena con: python train_camera_model.py")
            except Exception as e1:
                # Si falla por batch_shape, intentar cargar solo los pesos
                if "batch_shape" in str(e1) or "Unrecognized keyword" in str(e1):
                    print(f"⚠ Advertencia: Problema de compatibilidad detectado: {e1}")
                    print(f"  Intentando reconstruir el modelo y cargar solo los pesos...")
                    try:
                        # Reconstruir la arquitectura del modelo
                        # El modelo tiene 25 clases según la inspección del archivo
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                        
                        # El modelo original tiene 25 clases de salida
                        num_classes = 25
                        
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
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                        
                        # Cargar solo los pesos
                        print("   Cargando pesos del modelo...")
                        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=False)
                        is_retrained = "camera" in MODEL_PATH
                        model_type_str = "REENTRENADO" if is_retrained else "ORIGINAL"
                        print(f"\n{'='*60}")
                        print(f"✓✓✓ Modelo {model_type_str} reconstruido y pesos cargados correctamente!")
                        print(f"{'='*60}")
                        print(f"   Nota: Modelo tiene {num_classes} clases, usando índices 0-{num_classes-1}")
                        if is_retrained:
                            print(f"   → Usando modelo entrenado con imágenes de tu cámara")
                        else:
                            print(f"   → Usando modelo original del dataset Sign Language MNIST")
                    except Exception as e2:
                        print(f"✗✗✗ ERROR al reconstruir el modelo: {e2}")
                        import traceback
                        traceback.print_exc()
                        model = None
                else:
                    # Si es otro error, lanzarlo
                    raise e1
            
            if model is not None:
                print(f"✓ Clases disponibles: {len(class_names)}")
                # Verificar que el modelo tiene las dimensiones correctas
                if hasattr(model, 'input_shape'):
                    print(f"✓ Input shape del modelo: {model.input_shape}")
        except Exception as e:
            print(f"✗✗✗ ERROR al cargar el modelo: {e}")
            import traceback
            traceback.print_exc()
            model = None
    elif not MODEL_PATH:
        print(f"\n✗✗✗ Modelo NO encontrado en ninguna de las rutas verificadas")
    else:
        print("✗✗✗ No se pudo importar load_model de TensorFlow/Keras")
        print(f"  load_model_func es: {load_model_func}")
        
except ImportError as e:
    print(f"✗✗✗ ERROR: TensorFlow no está instalado: {e}")
    print("  Instala TensorFlow con: pip install tensorflow==2.15.0")
except Exception as e:
    print(f"✗✗✗ ERROR inesperado: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Cargar modelo de números (0-5)
print(f"\n{'='*60}")
print(f"BÚSQUEDA DEL MODELO DE NÚMEROS (0-5)")
print(f"{'='*60}")
script_dir = os.path.dirname(os.path.abspath(__file__))
number_model_paths = [
    os.path.join(script_dir, 'model', 'number_model_camera.h5'),  # Modelo reentrenado
    os.path.join(script_dir, 'model', 'number_model.h5'),  # Modelo original
    os.path.join(script_dir, 'number_model.h5'),
    os.path.join(os.path.dirname(script_dir), 'model', 'number_model.h5'),
    'model/number_model.h5',
    'number_model.h5',
    os.path.join(os.getcwd(), 'model', 'number_model.h5'),
    os.path.join(os.getcwd(), 'number_model.h5'),
]

NUMBER_MODEL_PATH = None
for path in number_model_paths:
    abs_path = os.path.abspath(path)
    exists = os.path.exists(path)
    if exists and NUMBER_MODEL_PATH is None:
        NUMBER_MODEL_PATH = path
        print(f"  ✓ Modelo de números encontrado: {abs_path}")
        break

if NUMBER_MODEL_PATH and load_model_func:
    try:
        print(f"Intentando cargar modelo de números desde: {os.path.abspath(NUMBER_MODEL_PATH)}")
        try:
            number_model = load_model_func(NUMBER_MODEL_PATH)
            print(f"✓✓✓ Modelo de números cargado correctamente!")
        except Exception as e1:
            if "batch_shape" in str(e1) or "Unrecognized keyword" in str(e1):
                print(f"⚠ Advertencia: Problema de compatibilidad. Reconstruyendo modelo...")
                try:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                    
                    num_classes = 6  # 0-5 son 6 clases
                    number_model = Sequential([
                        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
                        MaxPooling2D((2,2)),
                        Conv2D(64, (3,3), activation='relu'),
                        MaxPooling2D((2,2)),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(num_classes, activation='softmax')
                    ])
                    number_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    number_model.load_weights(NUMBER_MODEL_PATH, by_name=True, skip_mismatch=False)
                    print(f"✓✓✓ Modelo de números reconstruido y pesos cargados correctamente!")
                except Exception as e2:
                    print(f"✗✗✗ ERROR al reconstruir modelo de números: {e2}")
                    number_model = None
            else:
                raise e1
    except Exception as e:
        print(f"✗✗✗ ERROR al cargar modelo de números: {e}")
        number_model = None
elif not NUMBER_MODEL_PATH:
    print(f"⚠ Modelo de números NO encontrado. Usa 'train_number_model.py' para entrenarlo.")
    print(f"  El modo de números no funcionará hasta que se entrene el modelo.")
else:
    print("⚠ No se pudo importar load_model para números")

# Cargar modelo de meses (enero-diciembre)
print(f"\n{'='*60}")
print(f"BÚSQUEDA DEL MODELO DE MESES (ENERO-ABRIL)")
print(f"{'='*60}")
script_dir = os.path.dirname(os.path.abspath(__file__))
month_model_paths = [
    os.path.join(script_dir, 'model', 'month_model_camera.h5'),  # Modelo reentrenado
    os.path.join(script_dir, 'model', 'month_model.h5'),  # Modelo original
    os.path.join(script_dir, 'month_model.h5'),
    os.path.join(os.path.dirname(script_dir), 'model', 'month_model.h5'),
    'model/month_model.h5',
    'month_model.h5',
    os.path.join(os.getcwd(), 'model', 'month_model.h5'),
    os.path.join(os.getcwd(), 'month_model.h5'),
]

MONTH_MODEL_PATH = None
for path in month_model_paths:
    abs_path = os.path.abspath(path)
    exists = os.path.exists(path)
    if exists and MONTH_MODEL_PATH is None:
        MONTH_MODEL_PATH = path
        print(f"  ✓ Modelo de meses encontrado: {abs_path}")
        break

if MONTH_MODEL_PATH and load_model_func:
    try:
        print(f"Intentando cargar modelo de meses desde: {os.path.abspath(MONTH_MODEL_PATH)}")
        try:
            month_model = load_model_func(MONTH_MODEL_PATH)
            print(f"✓✓✓ Modelo de meses cargado correctamente!")
        except Exception as e1:
            if "batch_shape" in str(e1) or "Unrecognized keyword" in str(e1):
                print(f"⚠ Advertencia: Problema de compatibilidad. Reconstruyendo modelo...")
                try:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv2D, MaxPooling2D, Flatten
                    
                    num_classes = 4  # 4 meses (enero-abril)
                    month_model = Sequential([
                        TimeDistributed(
                            Conv2D(32, kernel_size=(3,3), activation='relu'),
                            input_shape=(SEQUENCE_LENGTH, 28, 28, 1)
                        ),
                        TimeDistributed(MaxPooling2D((2,2))),
                        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
                        TimeDistributed(MaxPooling2D((2,2))),
                        TimeDistributed(Flatten()),
                        LSTM(128, return_sequences=True),
                        Dropout(0.5),
                        LSTM(64, return_sequences=False),
                        Dropout(0.5),
                        Dense(num_classes, activation='softmax')
                    ])
                    month_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    month_model.load_weights(MONTH_MODEL_PATH, by_name=True, skip_mismatch=False)
                    print(f"✓✓✓ Modelo de meses reconstruido y pesos cargados correctamente!")
                except Exception as e2:
                    print(f"✗✗✗ ERROR al reconstruir modelo de meses: {e2}")
                    month_model = None
            else:
                raise e1
    except Exception as e:
        print(f"✗✗✗ ERROR al cargar modelo de meses: {e}")
        month_model = None
elif not MONTH_MODEL_PATH:
    print(f"⚠ Modelo de meses NO encontrado. Usa 'train_month_model.py' para entrenarlo.")
    print(f"  El modo de meses no funcionará hasta que se entrene el modelo.")
else:
    print("⚠ No se pudo importar load_model para meses")

# Cargar modelo de días (lunes-jueves)
print(f"\n{'='*60}")
print(f"BÚSQUEDA DEL MODELO DE DÍAS (LUNES-JUEVES)")
print(f"{'='*60}")
script_dir = os.path.dirname(os.path.abspath(__file__))
day_model_paths = [
    os.path.join(script_dir, 'model', 'day_model_camera.h5'),  # Modelo reentrenado
    os.path.join(script_dir, 'model', 'day_model.h5'),  # Modelo original
    os.path.join(script_dir, 'day_model.h5'),
    os.path.join(os.path.dirname(script_dir), 'model', 'day_model.h5'),
    'model/day_model.h5',
    'day_model.h5',
    os.path.join(os.getcwd(), 'model', 'day_model.h5'),
    os.path.join(os.getcwd(), 'day_model.h5'),
]

DAY_MODEL_PATH = None
for path in day_model_paths:
    abs_path = os.path.abspath(path)
    exists = os.path.exists(path)
    if exists and DAY_MODEL_PATH is None:
        DAY_MODEL_PATH = path
        print(f"  ✓ Modelo de días encontrado: {abs_path}")
        break

if DAY_MODEL_PATH and load_model_func:
    try:
        print(f"Intentando cargar modelo de días desde: {os.path.abspath(DAY_MODEL_PATH)}")
        try:
            day_model = load_model_func(DAY_MODEL_PATH)
            print(f"✓✓✓ Modelo de días cargado correctamente!")
        except Exception as e1:
            if "batch_shape" in str(e1) or "Unrecognized keyword" in str(e1):
                print(f"⚠ Advertencia: Problema de compatibilidad. Reconstruyendo modelo...")
                try:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv2D, MaxPooling2D, Flatten
                    
                    num_classes = 4  # 4 días (lunes-jueves)
                    day_model = Sequential([
                        TimeDistributed(
                            Conv2D(32, kernel_size=(3,3), activation='relu'),
                            input_shape=(SEQUENCE_LENGTH, 28, 28, 1)
                        ),
                        TimeDistributed(MaxPooling2D((2,2))),
                        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
                        TimeDistributed(MaxPooling2D((2,2))),
                        TimeDistributed(Flatten()),
                        LSTM(128, return_sequences=True),
                        Dropout(0.5),
                        LSTM(64, return_sequences=False),
                        Dropout(0.5),
                        Dense(num_classes, activation='softmax')
                    ])
                    day_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    day_model.load_weights(DAY_MODEL_PATH, by_name=True, skip_mismatch=False)
                    print(f"✓✓✓ Modelo de días reconstruido y pesos cargados correctamente!")
                except Exception as e2:
                    print(f"✗✗✗ ERROR al reconstruir modelo de días: {e2}")
                    day_model = None
            else:
                raise e1
    except Exception as e:
        print(f"✗✗✗ ERROR al cargar modelo de días: {e}")
        day_model = None
elif not DAY_MODEL_PATH:
    print(f"⚠ Modelo de días NO encontrado. Usa 'train_day_model.py' para entrenarlo.")
    print(f"  El modo de días no funcionará hasta que se entrene el modelo.")
else:
    print("⚠ No se pudo importar load_model para días")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(frame, landmarks):
    if model is None:
        return None
    
    try:
        # Extraer bounding box de la mano con padding más generoso
        h, w, _ = frame.shape
        
        # Calcular coordenadas de los landmarks
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        # Calcular centro y dimensiones
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Aumentar el tamaño del bounding box (50% más grande) y hacerlo cuadrado
        # Esto ayuda a incluir toda la mano y mantener el aspecto ratio
        max_dim = max(width, height) * 1.5  # 50% de padding adicional
        
        x_min = max(0, int(center_x - max_dim / 2))
        y_min = max(0, int(center_y - max_dim / 2))
        x_max = min(w, int(center_x + max_dim / 2))
        y_max = min(h, int(center_y + max_dim / 2))
        
        # Verificar que el bounding box tenga un tamaño mínimo razonable
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return None
        
        # Recortar y preprocesar (el modelo espera 28x28 según train_model.py)
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None
        
        # Preprocesamiento EXACTO como en train_model.py (línea 44):
        # images = images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        # 1. Convertir a escala de grises
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        # 2. Redimensionar a 28x28
        hand_img = cv2.resize(hand_img, (28, 28), interpolation=cv2.INTER_AREA)
        # 3. Normalizar a [0, 1] - sin transformaciones adicionales
        hand_img = hand_img.astype('float32') / 255.0
        
        hand_img = np.expand_dims(hand_img, axis=-1)  # Añadir dimensión de canal
        hand_img = np.expand_dims(hand_img, axis=0)  # Añadir dimensión de batch
        
        # Predecir
        prediction = model.predict(hand_img, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        
        # PROBLEMA IDENTIFICADO:
        # El modelo funciona perfectamente con imágenes del dataset (confianza ~1.0)
        # PERO las imágenes de la cámara en tiempo real son MUY DIFERENTES:
        # - Fondos diferentes (dataset tiene fondos uniformes)
        # - Iluminación diferente
        # - Ángulos y perspectivas diferentes
        # - Resolución y calidad diferente
        # 
        # Esto causa que las confianzas sean bajas (<0.4) incluso cuando el gesto es correcto.
        # 
        # SOLUCIÓN: Umbral de confianza adaptativo
        # El umbral se ajusta automáticamente según qué tan segura está la predicción
        
        # Calcular confianza máxima para determinar el umbral adaptativo
        max_confidence = prediction[0].max()
        
        # Umbral adaptativo: se ajusta según la confianza máxima
        if max_confidence < 0.30:
            # Confianza muy baja: usar umbral bajo (0.15) - imágenes muy diferentes al dataset
            confidence_threshold = 0.15
        elif max_confidence < 0.50:
            # Confianza media: umbral moderado (0.25) - imágenes algo diferentes
            confidence_threshold = 0.25
        else:
            # Confianza alta: umbral alto (0.40) - imágenes similares al dataset
            confidence_threshold = 0.40
        
        # Log de debug para diagnosticar problemas
        # Mostrar más frecuentemente si la confianza es baja
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        if random.random() < 0.10 or max_confidence < 0.30:
            top3_text = ", ".join([f"{class_names[i] if i < len(class_names) else f'Clase_{i}'}({prediction[0][i]:.3f})" 
                                   for i in top3_indices])
            selected_name = class_names[class_idx] if class_idx < len(class_names) else f'Clase_{class_idx}'
            mean_conf = prediction[0].mean()
            print(f"Debug - Top 3: {top3_text}")
            print(f"        Seleccionada: {selected_name} (conf: {confidence:.3f}, umbral adaptativo: {confidence_threshold:.2f})")
            print(f"        Max conf: {max_confidence:.3f}, Mean conf: {mean_conf:.3f}")
            if max_confidence < 0.30:
                print(f"        ⚠ ADVERTENCIA: Confianza muy baja. Las imágenes pueden ser muy diferentes al dataset de entrenamiento.")
                print(f"        💡 CONSEJO: Mejora la iluminación, usa fondo uniforme, o reduce el umbral manualmente.")
            print(f"        ¿Supera umbral? {confidence > confidence_threshold}")
        
        # Verificar si la predicción es confiable usando umbral adaptativo
        # El umbral se ajusta automáticamente según la confianza máxima:
        # - Si max_conf < 0.30: umbral bajo (0.15) - imágenes muy diferentes
        # - Si max_conf < 0.50: umbral medio (0.25) - imágenes algo diferentes  
        # - Si max_conf >= 0.50: umbral alto (0.40) - imágenes similares al dataset
        
        if confidence > confidence_threshold and class_idx < len(class_names):
            detected_letter = class_names[class_idx]
            # Solo retornar si la letra está en available_letters (A-O)
            if detected_letter in available_letters:
                return detected_letter
            else:
                # Letra detectada pero no está en el rango permitido (A-O)
                return None
        elif confidence > confidence_threshold and class_idx < 25:
            # Fallback para índices válidos pero fuera de class_names
            # No retornar nada si no está en available_letters
            return None
        else:
            # No se detecta gesto: confianza muy baja incluso con umbral adaptativo
            # Esto indica que la imagen es muy diferente a las de entrenamiento
            return None
    except Exception as e:
        print(f"Error en detect_gesture: {e}")
        return None

def detect_number(frame, landmarks):
    """Detectar número en señas (0-5)"""
    if number_model is None:
        return None
    
    try:
        # Extraer bounding box de la mano con padding más generoso
        h, w, _ = frame.shape
        
        # Calcular coordenadas de los landmarks
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        # Calcular centro y dimensiones
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        # Aumentar el tamaño del bounding box (50% más grande) y hacerlo cuadrado
        max_dim = max(width, height) * 1.5
        
        x_min = max(0, int(center_x - max_dim / 2))
        y_min = max(0, int(center_y - max_dim / 2))
        x_max = min(w, int(center_x + max_dim / 2))
        y_max = min(h, int(center_y + max_dim / 2))
        
        # Verificar que el bounding box tenga un tamaño mínimo razonable
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return None
        
        # Recortar y preprocesar
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None
        
        # Preprocesamiento: escala de grises, 28x28, normalización
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        hand_img = cv2.resize(hand_img, (28, 28), interpolation=cv2.INTER_AREA)
        hand_img = hand_img.astype('float32') / 255.0
        
        hand_img = np.expand_dims(hand_img, axis=-1)  # Añadir dimensión de canal
        hand_img = np.expand_dims(hand_img, axis=0)  # Añadir dimensión de batch
        
        # Predecir
        prediction = number_model.predict(hand_img, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        
        # Umbral adaptativo similar a detect_gesture
        max_confidence = prediction[0].max()
        
        if max_confidence < 0.30:
            confidence_threshold = 0.15
        elif max_confidence < 0.50:
            confidence_threshold = 0.25
        else:
            confidence_threshold = 0.40
        
        # Mapear índice de clase a número (0-5)
        if confidence > confidence_threshold and class_idx < 6:
            return str(class_idx)  # Retornar como string: '0', '1', '2', '3', '4', '5'
        else:
            return None
    except Exception as e:
        print(f"Error en detect_number: {e}")
        return None

def detect_month(frame, landmarks):
    """Detectar mes en señas usando secuencia de frames"""
    global month_sequence, is_recording_month
    
    if month_model is None:
        return None
    
    try:
        # Extraer región de la mano
        h, w, _ = frame.shape
        
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        max_dim = max(width, height) * 1.5
        
        x_min = max(0, int(center_x - max_dim / 2))
        y_min = max(0, int(center_y - max_dim / 2))
        x_max = min(w, int(center_x + max_dim / 2))
        y_max = min(h, int(center_y + max_dim / 2))
        
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return None
        
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None
        
        # Preprocesar frame
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        hand_img = cv2.resize(hand_img, (28, 28), interpolation=cv2.INTER_AREA)
        hand_img = hand_img.astype('float32') / 255.0
        
        # Agregar frame a la secuencia
        month_sequence.append(hand_img)
        
        # Mantener solo los últimos SEQUENCE_LENGTH frames
        if len(month_sequence) > SEQUENCE_LENGTH:
            month_sequence = month_sequence[-SEQUENCE_LENGTH:]
        
        # Si tenemos suficientes frames, hacer predicción
        if len(month_sequence) >= SEQUENCE_LENGTH:
            # Preparar secuencia para el modelo
            sequence_array = np.array(month_sequence[-SEQUENCE_LENGTH:])
            sequence_array = sequence_array.reshape(1, SEQUENCE_LENGTH, 28, 28, 1)
            
            # Predecir
            prediction = month_model.predict(sequence_array, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = float(prediction[0][class_idx])
            
            # Umbral adaptativo
            max_confidence = prediction[0].max()
            if max_confidence < 0.30:
                confidence_threshold = 0.15
            elif max_confidence < 0.50:
                confidence_threshold = 0.25
            else:
                confidence_threshold = 0.40
            
            if confidence > confidence_threshold and class_idx < len(available_months):
                return available_months[class_idx]
        
        return None
    except Exception as e:
        print(f"Error en detect_month: {e}")
        return None

def detect_day(frame, landmarks):
    """Detectar día en señas usando secuencia de frames"""
    global day_sequence, is_recording_day
    
    if day_model is None:
        return None
    
    try:
        # Extraer región de la mano
        h, w, _ = frame.shape
        
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        max_dim = max(width, height) * 1.5
        
        x_min = max(0, int(center_x - max_dim / 2))
        y_min = max(0, int(center_y - max_dim / 2))
        x_max = min(w, int(center_x + max_dim / 2))
        y_max = min(h, int(center_y + max_dim / 2))
        
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return None
        
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None
        
        # Preprocesar frame
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        hand_img = cv2.resize(hand_img, (28, 28), interpolation=cv2.INTER_AREA)
        hand_img = hand_img.astype('float32') / 255.0
        
        # Agregar frame a la secuencia
        day_sequence.append(hand_img)
        
        # Mantener solo los últimos SEQUENCE_LENGTH frames
        if len(day_sequence) > SEQUENCE_LENGTH:
            day_sequence = day_sequence[-SEQUENCE_LENGTH:]
        
        # Si tenemos suficientes frames, hacer predicción
        if len(day_sequence) >= SEQUENCE_LENGTH:
            # Preparar secuencia para el modelo
            sequence_array = np.array(day_sequence[-SEQUENCE_LENGTH:])
            sequence_array = sequence_array.reshape(1, SEQUENCE_LENGTH, 28, 28, 1)
            
            # Predecir
            prediction = day_model.predict(sequence_array, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = float(prediction[0][class_idx])
            
            # Umbral adaptativo
            max_confidence = prediction[0].max()
            if max_confidence < 0.30:
                confidence_threshold = 0.15
            elif max_confidence < 0.50:
                confidence_threshold = 0.25
            else:
                confidence_threshold = 0.40
            
            if confidence > confidence_threshold and class_idx < len(available_days):
                return available_days[class_idx]
        
        return None
    except Exception as e:
        print(f"Error en detect_day: {e}")
        return None

def detect_all_gestures(frame, landmarks):
    """Detectar gesto usando todos los modelos disponibles"""
    global translator_sequence
    
    results = []
    
    # 1. Intentar detectar letra
    if model is not None:
        gesture = detect_gesture(frame, landmarks)
        if gesture:
            results.append(('Letra', gesture))
    
    # 2. Intentar detectar número
    if number_model is not None:
        number = detect_number(frame, landmarks)
        if number:
            results.append(('Número', number))
    
    # 3. Intentar detectar mes (requiere secuencia)
    if month_model is not None:
        # Agregar frame a secuencia de meses
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        max_dim = max(width, height) * 1.5
        x_min = max(0, int(center_x - max_dim / 2))
        y_min = max(0, int(center_y - max_dim / 2))
        x_max = min(w, int(center_x + max_dim / 2))
        y_max = min(h, int(center_y + max_dim / 2))
        
        if (x_max - x_min) >= 50 and (y_max - y_min) >= 50:
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (28, 28), interpolation=cv2.INTER_AREA)
                hand_img = hand_img.astype('float32') / 255.0
                translator_sequence.append(hand_img)
                
                if len(translator_sequence) > SEQUENCE_LENGTH:
                    translator_sequence = translator_sequence[-SEQUENCE_LENGTH:]
                
                if len(translator_sequence) >= SEQUENCE_LENGTH:
                    try:
                        sequence_array = np.array(translator_sequence[-SEQUENCE_LENGTH:])
                        sequence_array = sequence_array.reshape(1, SEQUENCE_LENGTH, 28, 28, 1)
                        prediction = month_model.predict(sequence_array, verbose=0)
                        class_idx = np.argmax(prediction)
                        confidence = float(prediction[0][class_idx])
                        max_confidence = prediction[0].max()
                        
                        confidence_threshold = 0.15 if max_confidence < 0.30 else (0.25 if max_confidence < 0.50 else 0.40)
                        
                        if confidence > confidence_threshold and class_idx < len(available_months):
                            results.append(('Mes', available_months[class_idx]))
                    except:
                        pass
    
    # 4. Intentar detectar día (requiere secuencia)
    if day_model is not None:
        # Reutilizar la misma secuencia o crear una nueva
        if len(translator_sequence) >= SEQUENCE_LENGTH:
            try:
                sequence_array = np.array(translator_sequence[-SEQUENCE_LENGTH:])
                sequence_array = sequence_array.reshape(1, SEQUENCE_LENGTH, 28, 28, 1)
                prediction = day_model.predict(sequence_array, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = float(prediction[0][class_idx])
                max_confidence = prediction[0].max()
                
                confidence_threshold = 0.15 if max_confidence < 0.30 else (0.25 if max_confidence < 0.50 else 0.40)
                
                if confidence > confidence_threshold and class_idx < len(available_days):
                    results.append(('Día', available_days[class_idx]))
            except:
                pass
    
    # Retornar el resultado con mayor prioridad (letras primero, luego números, luego meses/días)
    if results:
        # Prioridad: Letra > Número > Mes > Día
        priority_order = ['Letra', 'Número', 'Mes', 'Día']
        for priority in priority_order:
            for result_type, result_value in results:
                if result_type == priority:
                    return f"{result_type}: {result_value}"
        # Si no hay coincidencia de prioridad, retornar el primero
        return f"{results[0][0]}: {results[0][1]}"
    
    return None

def generate_frames():
    global current_prediction, letter_detected, current_number_prediction, number_detected
    global current_month_prediction, month_detected, month_sequence, is_recording_month
    global current_day_prediction, day_detected, day_sequence, is_recording_day
    global translator_prediction, translator_sequence, mode
    print("🎥 Iniciando captura de video...")
    cap = None
    frame_count = 0
    
    # Intentar abrir la cámara con diferentes índices
    for camera_index in range(5):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Configurar resolución
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"✓ Cámara abierta en índice {camera_index}")
            break
        else:
            cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌ ERROR: No se pudo abrir ninguna cámara")
        current_prediction = "Error: Cámara no disponible"
        # Generar un frame de error para que el navegador muestre algo
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera not available", (150, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_frame, "Check camera connection", (120, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    if model is None:
        print("⚠⚠⚠ ADVERTENCIA CRÍTICA: Modelo NO está disponible")
        print("   La aplicación no podrá reconocer gestos sin el modelo.")
        current_prediction = "ERROR: Modelo no disponible"
    
    print("📹 Stream de video iniciado")
    consecutive_errors = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            consecutive_errors += 1
            if consecutive_errors > 10:
                print("❌ ERROR: Demasiados errores al leer frames")
                break
            continue
        
        consecutive_errors = 0
        
        # Verificar que el frame sea válido
        if frame is None or frame.size == 0:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        gesture = None
        debug_landmarks = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                debug_landmarks = landmarks  # Guardar para debug
                
                # PAUSAR DETECCIÓN: Si ya se detectó correctamente, no procesar más gestos
                if mode == 'translator':
                    # Modo traductor: detectar con todos los modelos
                    detected = detect_all_gestures(frame, landmarks)
                    if detected:
                        translator_prediction = detected
                    else:
                        translator_prediction = "Mano detectada, procesando..."
                elif mode == 'days':
                    # Modo de días
                    if day_detected:
                        # Ya se detectó correctamente, mantener el mensaje y no procesar más
                        if current_day_index < len(available_days):
                            target_day = available_days[current_day_index]
                            current_day_prediction = f"Día: {target_day} ✓"
                        is_recording_day = False
                        continue  # Saltar el procesamiento de detección
                    
                    # Procesar detección solo si aún no se ha detectado correctamente
                    is_recording_day = True
                    detected = detect_day(frame, landmarks)
                    if detected:
                        current_day_prediction = f"Día: {detected}"
                        # Verificar si el día detectado coincide con el día objetivo
                        if current_day_index < len(available_days):
                            target_day = available_days[current_day_index]
                            if detected == target_day:
                                day_detected = True
                                current_day_prediction = f"Día: {detected} ✓"
                                is_recording_day = False
                                if frame_count % 30 == 0:
                                    print(f"✓✓✓ Día CORRECTO detectado: {detected} (objetivo: {target_day}) - Detección pausada")
                            else:
                                if frame_count % 30 == 0:
                                    print(f"✗ Día detectado: {detected} (objetivo: {target_day}) - No coincide")
                    else:
                        current_day_prediction = "Grabando video..."
                elif mode == 'months':
                    # Modo de meses
                    if month_detected:
                        # Ya se detectó correctamente, mantener el mensaje y no procesar más
                        if current_month_index < len(available_months):
                            target_month = available_months[current_month_index]
                            current_month_prediction = f"Mes: {target_month} ✓"
                        is_recording_month = False
                        continue  # Saltar el procesamiento de detección
                    
                    # Procesar detección solo si aún no se ha detectado correctamente
                    is_recording_month = True
                    detected = detect_month(frame, landmarks)
                    if detected:
                        current_month_prediction = f"Mes: {detected}"
                        # Verificar si el mes detectado coincide con el mes objetivo
                        if current_month_index < len(available_months):
                            target_month = available_months[current_month_index]
                            if detected == target_month:
                                month_detected = True
                                current_month_prediction = f"Mes: {detected} ✓"
                                is_recording_month = False
                                if frame_count % 30 == 0:
                                    print(f"✓✓✓ Mes CORRECTO detectado: {detected} (objetivo: {target_month}) - Detección pausada")
                            else:
                                if frame_count % 30 == 0:
                                    print(f"✗ Mes detectado: {detected} (objetivo: {target_month}) - No coincide")
                    else:
                        current_month_prediction = "Grabando video..."
                elif mode == 'numbers':
                    # Modo de números
                    if number_detected:
                        # Ya se detectó correctamente, mantener el mensaje y no procesar más
                        if current_number_index < len(available_numbers):
                            target_number = available_numbers[current_number_index]
                            current_number_prediction = f"Número: {target_number} ✓"
                        continue  # Saltar el procesamiento de detección
                    
                    # Procesar detección solo si aún no se ha detectado correctamente
                    detected = detect_number(frame, landmarks)
                    if detected:
                        current_number_prediction = f"Número: {detected}"
                        # Verificar si el número detectado coincide con el número objetivo
                        if current_number_index < len(available_numbers):
                            target_number = available_numbers[current_number_index]
                            if detected == target_number:
                                number_detected = True
                                current_number_prediction = f"Número: {detected} ✓"
                                if frame_count % 30 == 0:
                                    print(f"✓✓✓ Número CORRECTO detectado: {detected} (objetivo: {target_number}) - Detección pausada")
                            else:
                                if frame_count % 30 == 0:
                                    print(f"✗ Número detectado: {detected} (objetivo: {target_number}) - No coincide")
                    else:
                        current_number_prediction = "Mano detectada, procesando..."
                else:
                    # Modo de letras (por defecto)
                    if letter_detected:
                        # Ya se detectó correctamente, mantener el mensaje y no procesar más
                        if current_letter_index < len(available_letters):
                            target_letter = available_letters[current_letter_index]
                            current_prediction = f"Letra: {target_letter} ✓"
                        continue  # Saltar el procesamiento de detección
                    
                    # Procesar detección solo si aún no se ha detectado correctamente
                    gesture = detect_gesture(frame, landmarks)
                    if gesture:
                        current_prediction = f"Letra: {gesture}"
                        # Verificar si la letra detectada coincide con la letra objetivo
                        if current_letter_index < len(available_letters):
                            target_letter = available_letters[current_letter_index]
                            if gesture == target_letter:
                                letter_detected = True
                                current_prediction = f"Letra: {gesture} ✓"
                                # Log para debug cada 30 frames
                                if frame_count % 30 == 0:
                                    print(f"✓✓✓ Letra CORRECTA detectada: {gesture} (objetivo: {target_letter}) - Detección pausada")
                            else:
                                # Log para debug
                                if frame_count % 30 == 0:
                                    print(f"✗ Letra detectada: {gesture} (objetivo: {target_letter}) - No coincide")
                    else:
                        current_prediction = "Mano detectada, procesando..."
        else:
            # No hay mano detectada
            if mode == 'translator':
                translator_prediction = "Esperando mano..."
                translator_sequence = []  # Limpiar secuencia si no hay mano
            elif mode == 'days':
                if not day_detected:
                    current_day_prediction = "Esperando mano..."
                    is_recording_day = False
                    day_sequence = []  # Limpiar secuencia si no hay mano
            elif mode == 'months':
                if not month_detected:
                    current_month_prediction = "Esperando mano..."
                    is_recording_month = False
                    month_sequence = []  # Limpiar secuencia si no hay mano
            elif mode == 'numbers':
                if not number_detected:
                    current_number_prediction = "Esperando mano..."
            else:
                if not letter_detected:
                    current_prediction = "Esperando mano..."
        
        # Log cada 30 frames (aproximadamente cada segundo a 30fps)
        frame_count += 1
        if frame_count % 30 == 0:
            if model is None:
                print(f"Frame {frame_count}: Modelo no disponible")
            elif not results.multi_hand_landmarks:
                print(f"Frame {frame_count}: No se detectó mano")
            elif gesture is None and debug_landmarks and model is not None:
                # Intentar una predicción con umbral más bajo para debug
                try:
                    h, w, _ = frame.shape
                    # Usar el mismo cálculo de bounding box que en detect_gesture
                    x_coords = [lm.x * w for lm in debug_landmarks]
                    y_coords = [lm.y * h for lm in debug_landmarks]
                    center_x = (min(x_coords) + max(x_coords)) / 2
                    center_y = (min(y_coords) + max(y_coords)) / 2
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    max_dim = max(width, height) * 1.5
                    x_min = max(0, int(center_x - max_dim / 2))
                    y_min = max(0, int(center_y - max_dim / 2))
                    x_max = min(w, int(center_x + max_dim / 2))
                    y_max = min(h, int(center_y + max_dim / 2))
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size > 0:
                        # Usar el mismo preprocesamiento que en detect_gesture
                        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        hand_img = cv2.resize(hand_img, (28, 28), interpolation=cv2.INTER_AREA)
                        hand_img = hand_img.astype('float32') / 255.0
                        hand_img = np.expand_dims(hand_img, axis=-1)
                        hand_img = np.expand_dims(hand_img, axis=0)
                        pred = model.predict(hand_img, verbose=0)
                        top_idx = np.argmax(pred[0])
                        conf = pred[0][top_idx]
                        top3_indices = np.argsort(pred[0])[-3:][::-1]
                        top3_text = ", ".join([f"{class_names[i] if i < len(class_names) else f'Clase_{i}'}({pred[0][i]:.2f})" 
                                               for i in top3_indices])
                        print(f"Frame {frame_count}: Mano detectada - Top 3: {top3_text} (umbral: 0.25)")
                except Exception as e:
                    print(f"Frame {frame_count}: Mano detectada pero error al procesar: {e}")
            elif gesture:
                print(f"Frame {frame_count}: ✓ Letra detectada: {gesture}")
        
        # Codificar frame a JPEG con calidad optimizada
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            print("⚠ Error al codificar frame")
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Enviar frame en formato MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    global mode
    mode = 'letters'  # Establecer modo de letras
    return render_template('index.html')

@app.route('/numeros')
def numeros():
    global mode
    mode = 'numbers'  # Establecer modo de números
    return render_template('numeros.html')

@app.route('/meses')
def meses():
    global mode, month_sequence
    mode = 'months'  # Establecer modo de meses
    month_sequence = []  # Limpiar secuencia al cambiar de modo
    return render_template('meses.html')

@app.route('/dias')
def dias():
    global mode, day_sequence
    mode = 'days'  # Establecer modo de días
    day_sequence = []  # Limpiar secuencia al cambiar de modo
    return render_template('dias.html')

@app.route('/traductor')
def traductor():
    global mode, translator_sequence, translator_prediction
    mode = 'translator'  # Establecer modo traductor
    translator_sequence = []  # Limpiar secuencia al cambiar de modo
    translator_prediction = "Esperando detección..."  # Resetear predicción
    return render_template('traductor.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive'
        }
    )

@app.route('/get_prediction')
def get_prediction():
    global letter_detected
    
    # Extraer solo la letra si viene en formato "Letra: X"
    detected_letter = None
    if current_prediction.startswith("Letra: "):
        detected_letter = current_prediction.split(": ")[1]
    
    # Obtener la letra objetivo actual
    target_letter = None
    if current_letter_index < len(available_letters):
        target_letter = available_letters[current_letter_index]
    
    # Verificar nuevamente la comparación aquí para asegurar que sea correcta
    # Esto ayuda a mantener el estado consistente
    if detected_letter and target_letter:
        if detected_letter == target_letter:
            letter_detected = True
        # No resetear aquí, dejar que el loop de video lo maneje
    
    return jsonify({
        "prediction": current_prediction,
        "detected_letter": detected_letter,
        "target_letter": target_letter,
        "letter_detected": letter_detected,
        "current_index": current_letter_index,
        "total_letters": len(available_letters),
        "progress": f"{current_letter_index + 1}/{len(available_letters)}",
        "debug": {
            "detected": detected_letter,
            "target": target_letter,
            "match": detected_letter == target_letter if detected_letter and target_letter else False,
            "letter_detected_state": letter_detected
        }
    })

@app.route('/next_letter', methods=['POST'])
def next_letter():
    global current_letter_index, letter_detected, current_prediction
    print(f"→ Botón 'Siguiente Letra' presionado. Índice actual: {current_letter_index}")
    if current_letter_index < len(available_letters) - 1:
        current_letter_index += 1
        letter_detected = False
        current_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
        print(f"→ Avanzando a la letra: {available_letters[current_letter_index]} ({current_letter_index + 1}/{len(available_letters)})")
        return jsonify({
            "success": True,
            "current_index": current_letter_index,
            "target_letter": available_letters[current_letter_index],
            "progress": f"{current_letter_index + 1}/{len(available_letters)}"
        })
    else:
        # Ya se completaron todas las letras
        print("→ ¡Todas las letras completadas!")
        return jsonify({
            "success": True,
            "completed": True,
            "message": "¡Felicidades! Has completado todo el abecedario."
        })

@app.route('/reset_practice', methods=['POST'])
def reset_practice():
    global current_letter_index, letter_detected, current_prediction
    current_letter_index = 0
    letter_detected = False
    current_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
    return jsonify({
        "success": True,
        "current_index": current_letter_index,
        "target_letter": available_letters[current_letter_index],
        "progress": f"{current_letter_index + 1}/{len(available_letters)}"
    })

# Rutas para números
@app.route('/get_number_prediction')
def get_number_prediction():
    global number_detected
    
    # Extraer solo el número si viene en formato "Número: X"
    detected_number = None
    if current_number_prediction.startswith("Número: "):
        detected_number = current_number_prediction.split(": ")[1]
    
    # Obtener el número objetivo actual
    target_number = None
    if current_number_index < len(available_numbers):
        target_number = available_numbers[current_number_index]
    
    # Verificar nuevamente la comparación aquí para asegurar que sea correcta
    if detected_number and target_number:
        if detected_number == target_number:
            number_detected = True
    
    return jsonify({
        "prediction": current_number_prediction,
        "detected_number": detected_number,
        "target_number": target_number,
        "number_detected": number_detected,
        "current_index": current_number_index,
        "total_numbers": len(available_numbers),
        "progress": f"{current_number_index + 1}/{len(available_numbers)}",
        "debug": {
            "detected": detected_number,
            "target": target_number,
            "match": detected_number == target_number if detected_number and target_number else False,
            "number_detected_state": number_detected
        }
    })

@app.route('/next_number', methods=['POST'])
def next_number():
    global current_number_index, number_detected, current_number_prediction
    print(f"→ Botón 'Siguiente Número' presionado. Índice actual: {current_number_index}")
    if current_number_index < len(available_numbers) - 1:
        current_number_index += 1
        number_detected = False
        current_number_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
        print(f"→ Avanzando al número: {available_numbers[current_number_index]} ({current_number_index + 1}/{len(available_numbers)})")
        return jsonify({
            "success": True,
            "current_index": current_number_index,
            "target_number": available_numbers[current_number_index],
            "progress": f"{current_number_index + 1}/{len(available_numbers)}"
        })
    else:
        # Ya se completaron todos los números
        print("→ ¡Todos los números completados!")
        return jsonify({
            "success": True,
            "completed": True,
            "message": "¡Felicidades! Has completado todos los números del 0 al 5."
        })

@app.route('/reset_number_practice', methods=['POST'])
def reset_number_practice():
    global current_number_index, number_detected, current_number_prediction
    current_number_index = 0
    number_detected = False
    current_number_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
    return jsonify({
        "success": True,
        "current_index": current_number_index,
        "target_number": available_numbers[current_number_index],
        "progress": f"{current_number_index + 1}/{len(available_numbers)}"
    })

# Rutas para meses
@app.route('/get_month_prediction')
def get_month_prediction():
    global month_detected, is_recording_month
    
    # Extraer solo el mes si viene en formato "Mes: X"
    detected_month = None
    if current_month_prediction.startswith("Mes: "):
        detected_month = current_month_prediction.split(": ")[1].replace(" ✓", "")
    
    # Obtener el mes objetivo actual
    target_month = None
    if current_month_index < len(available_months):
        target_month = available_months[current_month_index]
    
    # Verificar nuevamente la comparación aquí para asegurar que sea correcta
    if detected_month and target_month:
        if detected_month == target_month:
            month_detected = True
    
    return jsonify({
        "prediction": current_month_prediction,
        "detected_month": detected_month,
        "target_month": target_month,
        "month_detected": month_detected,
        "current_index": current_month_index,
        "total_months": len(available_months),
        "progress": f"{current_month_index + 1}/{len(available_months)}",
        "recording": is_recording_month,
        "debug": {
            "detected": detected_month,
            "target": target_month,
            "match": detected_month == target_month if detected_month and target_month else False,
            "month_detected_state": month_detected
        }
    })

@app.route('/next_month', methods=['POST'])
def next_month():
    global current_month_index, month_detected, current_month_prediction, month_sequence
    print(f"→ Botón 'Siguiente Mes' presionado. Índice actual: {current_month_index}")
    if current_month_index < len(available_months) - 1:
        current_month_index += 1
        month_detected = False
        current_month_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
        month_sequence = []  # Limpiar secuencia
        print(f"→ Avanzando al mes: {available_months[current_month_index]} ({current_month_index + 1}/{len(available_months)})")
        return jsonify({
            "success": True,
            "current_index": current_month_index,
            "target_month": available_months[current_month_index],
            "progress": f"{current_month_index + 1}/{len(available_months)}"
        })
    else:
        # Ya se completaron todos los meses
        print("→ ¡Todos los meses completados!")
        return jsonify({
            "success": True,
            "completed": True,
            "message": "¡Felicidades! Has completado todos los meses (enero-abril)."
        })

@app.route('/reset_month_practice', methods=['POST'])
def reset_month_practice():
    global current_month_index, month_detected, current_month_prediction, month_sequence
    current_month_index = 0
    month_detected = False
    current_month_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
    month_sequence = []  # Limpiar secuencia
    return jsonify({
        "success": True,
        "current_index": current_month_index,
        "target_month": available_months[current_month_index],
        "progress": f"{current_month_index + 1}/{len(available_months)}"
    })

# Rutas para días
@app.route('/get_day_prediction')
def get_day_prediction():
    global day_detected, is_recording_day
    
    # Extraer solo el día si viene en formato "Día: X"
    detected_day = None
    if current_day_prediction.startswith("Día: "):
        detected_day = current_day_prediction.split(": ")[1].replace(" ✓", "")
    
    # Obtener el día objetivo actual
    target_day = None
    if current_day_index < len(available_days):
        target_day = available_days[current_day_index]
    
    # Verificar nuevamente la comparación aquí para asegurar que sea correcta
    if detected_day and target_day:
        if detected_day == target_day:
            day_detected = True
    
    return jsonify({
        "prediction": current_day_prediction,
        "detected_day": detected_day,
        "target_day": target_day,
        "day_detected": day_detected,
        "current_index": current_day_index,
        "total_days": len(available_days),
        "progress": f"{current_day_index + 1}/{len(available_days)}",
        "recording": is_recording_day,
        "debug": {
            "detected": detected_day,
            "target": target_day,
            "match": detected_day == target_day if detected_day and target_day else False,
            "day_detected_state": day_detected
        }
    })

@app.route('/next_day', methods=['POST'])
def next_day():
    global current_day_index, day_detected, current_day_prediction, day_sequence
    print(f"→ Botón 'Siguiente Día' presionado. Índice actual: {current_day_index}")
    if current_day_index < len(available_days) - 1:
        current_day_index += 1
        day_detected = False
        current_day_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
        day_sequence = []  # Limpiar secuencia
        print(f"→ Avanzando al día: {available_days[current_day_index]} ({current_day_index + 1}/{len(available_days)})")
        return jsonify({
            "success": True,
            "current_index": current_day_index,
            "target_day": available_days[current_day_index],
            "progress": f"{current_day_index + 1}/{len(available_days)}"
        })
    else:
        # Ya se completaron todos los días
        print("→ ¡Todos los días completados!")
        return jsonify({
            "success": True,
            "completed": True,
            "message": "¡Felicidades! Has completado todos los días (lunes-jueves)."
        })

@app.route('/reset_day_practice', methods=['POST'])
def reset_day_practice():
    global current_day_index, day_detected, current_day_prediction, day_sequence
    current_day_index = 0
    day_detected = False
    current_day_prediction = "Esperando gesto..."  # Resetear mensaje de predicción
    day_sequence = []  # Limpiar secuencia
    return jsonify({
        "success": True,
        "current_index": current_day_index,
        "target_day": available_days[current_day_index],
        "progress": f"{current_day_index + 1}/{len(available_days)}"
    })

# Ruta para traductor
@app.route('/get_translator_prediction')
def get_translator_prediction():
    return jsonify({
        "prediction": translator_prediction
    })

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    gesture = data.get('gesture')
    text = gesture if gesture else 'Gesto no reconocido'
    tts = gTTS(text=text, lang='es')
    tts.save('static/output.mp3')
    return jsonify({'text': text, 'audio': '/static/output.mp3'})

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get("text", "No se recibió texto")
    try:
        tts = gTTS(text=text, lang='es')
        static_path = os.path.join(os.path.dirname(__file__), 'static', 'output.mp3')
        os.makedirs(os.path.dirname(static_path), exist_ok=True)
        tts.save(static_path)
        return jsonify({"audio": "/static/output.mp3"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
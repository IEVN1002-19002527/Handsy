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
# Obtener las letras disponibles del modelo (sin J, que es la clase 9)
available_letters = [letter for letter in asl_letters]  # A-Y sin J
current_letter_index = 0  # Índice de la letra objetivo actual
letter_detected = False  # Si la letra objetivo fue detectada correctamente

# Variables globales para el modo de práctica número por número
available_numbers = ['0', '1', '2', '3', '4', '5']  # Números del 0 al 5
current_number_index = 0  # Índice del número objetivo actual
number_detected = False  # Si el número objetivo fue detectado correctamente
current_number_prediction = "Esperando gesto..."  # Predicción actual para números
mode = 'letters'  # Modo actual: 'letters' o 'numbers'

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
            return class_names[class_idx]
        elif confidence > confidence_threshold and class_idx < 25:
            # Fallback para índices válidos pero fuera de class_names
            return f"Clase_{class_idx}"
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

def generate_frames():
    global current_prediction, letter_detected, current_number_prediction, number_detected, mode
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
                
                if mode == 'numbers':
                    # Modo de números
                    detected = detect_number(frame, landmarks)
                    if detected:
                        current_number_prediction = f"Número: {detected}"
                        # Verificar si el número detectado coincide con el número objetivo
                        if current_number_index < len(available_numbers):
                            target_number = available_numbers[current_number_index]
                            if detected == target_number:
                                number_detected = True
                                if frame_count % 30 == 0:
                                    print(f"✓✓✓ Número CORRECTO detectado: {detected} (objetivo: {target_number})")
                            else:
                                if not number_detected:
                                    number_detected = False
                                if frame_count % 30 == 0:
                                    print(f"✗ Número detectado: {detected} (objetivo: {target_number}) - No coincide")
                    else:
                        current_number_prediction = "Mano detectada, procesando..."
                        if not number_detected:
                            number_detected = False
                else:
                    # Modo de letras (por defecto)
                    gesture = detect_gesture(frame, landmarks)
                    if gesture:
                        current_prediction = f"Letra: {gesture}"
                        # Verificar si la letra detectada coincide con la letra objetivo
                        if current_letter_index < len(available_letters):
                            target_letter = available_letters[current_letter_index]
                            if gesture == target_letter:
                                letter_detected = True
                                # Log para debug cada 30 frames
                                if frame_count % 30 == 0:
                                    print(f"✓✓✓ Letra CORRECTA detectada: {gesture} (objetivo: {target_letter}) - Botón debería habilitarse")
                            else:
                                # Solo resetear si NO es la letra correcta
                                # Mantener el estado si ya se detectó correctamente antes
                                if not letter_detected:
                                    letter_detected = False
                                # Log para debug
                                if frame_count % 30 == 0:
                                    print(f"✗ Letra detectada: {gesture} (objetivo: {target_letter}) - No coincide")
                    else:
                        current_prediction = "Mano detectada, procesando..."
                        # No resetear letter_detected aquí si ya se detectó correctamente
                        # Solo resetear si no hay gesto detectado Y no se había detectado antes
                        if not letter_detected:
                            letter_detected = False
        else:
            if mode == 'numbers':
                current_number_prediction = "Esperando mano..."
                if not number_detected:
                    number_detected = False
            else:
                current_prediction = "Esperando mano..."
                # No resetear letter_detected aquí si ya se detectó correctamente
                # Solo resetear si no hay mano Y no se había detectado antes
                if not letter_detected:
                    letter_detected = False
        
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
    global current_letter_index, letter_detected
    print(f"→ Botón 'Siguiente Letra' presionado. Índice actual: {current_letter_index}")
    if current_letter_index < len(available_letters) - 1:
        current_letter_index += 1
        letter_detected = False
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
    global current_letter_index, letter_detected
    current_letter_index = 0
    letter_detected = False
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
    global current_number_index, number_detected
    print(f"→ Botón 'Siguiente Número' presionado. Índice actual: {current_number_index}")
    if current_number_index < len(available_numbers) - 1:
        current_number_index += 1
        number_detected = False
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
    global current_number_index, number_detected
    current_number_index = 0
    number_detected = False
    return jsonify({
        "success": True,
        "current_index": current_number_index,
        "target_number": available_numbers[current_number_index],
        "progress": f"{current_number_index + 1}/{len(available_numbers)}"
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
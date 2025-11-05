from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
import random
app = Flask(__name__)

# Intentar importar TensorFlow y cargar el modelo
model = None
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
# Variable global para almacenar la predicción actual
current_prediction = "Esperando gesto..."

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, 'model', 'gesture_model.h5'),  # app/model/gesture_model.h5
        os.path.join(script_dir, 'gesture_model.h5'),  # app/gesture_model.h5
        os.path.join(os.path.dirname(script_dir), 'model', 'gesture_model.h5'),  # handtalk_env/model/gesture_model.h5
        'model/gesture_model.h5',
        'gesture_model.h5',
        os.path.join(os.getcwd(), 'model', 'gesture_model.h5'),
        os.path.join(os.getcwd(), 'gesture_model.h5'),
    ]
    
    MODEL_PATH = None
    print(f"Directorio del script: {script_dir}")
    print(f"Directorio de trabajo actual: {os.getcwd()}")
    print(f"Buscando modelo en las siguientes ubicaciones:")
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  {'✓' if exists else '✗'} {abs_path}")
        if exists and MODEL_PATH is None:
            MODEL_PATH = path
            print(f"  → Modelo encontrado en: {abs_path}")
    
    if MODEL_PATH and load_model_func:
        try:
            print(f"\nIntentando cargar modelo desde: {os.path.abspath(MODEL_PATH)}")
            model = load_model_func(MODEL_PATH)
            print(f"✓✓✓ Modelo cargado correctamente!")
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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(frame, landmarks):
    if model is None:
        return None
    
    try:
        # Extraer bounding box de la mano con padding
        h, w, _ = frame.shape
        x_min = max(0, int(min([lm.x for lm in landmarks]) * w) - 20)
        y_min = max(0, int(min([lm.y for lm in landmarks]) * h) - 20)
        x_max = min(w, int(max([lm.x for lm in landmarks]) * w) + 20)
        y_max = min(h, int(max([lm.y for lm in landmarks]) * h) + 20)
        
        # Verificar que el bounding box tenga un tamaño mínimo
        if (x_max - x_min) < 20 or (y_max - y_min) < 20:
            return None
        
        # Recortar y preprocesar (el modelo espera 28x28 según train_model.py)
        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None
        
        # Convertir a escala de grises y redimensionar a 28x28
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste con ecualización de histograma
        hand_img = cv2.equalizeHist(hand_img)
        
        # Redimensionar a 28x28
        hand_img = cv2.resize(hand_img, (28, 28))
        
        # Normalizar a [0, 1]
        hand_img = hand_img.astype('float32') / 255.0
        
        # Asegurar que la imagen tenga buen contraste
        # Normalizar para que tenga media 0.5 aproximadamente
        hand_img = hand_img * 0.8 + 0.1
        
        hand_img = np.expand_dims(hand_img, axis=-1)  # Añadir dimensión de canal
        hand_img = np.expand_dims(hand_img, axis=0)  # Añadir dimensión de batch
        
        # Predecir
        prediction = model.predict(hand_img, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        
        # Reducir umbral a 10% para ser más permisivo
        # También mostrar top 3 predicciones para debug
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        
        # Log de debug ocasional (2% de los frames)
        if random.random() < 0.02:
            top3_text = ", ".join([f"{class_names[i]}({prediction[0][i]:.2f})" 
                                   for i in top3_indices if i < len(class_names)])
            print(f"Debug - Top 3: {top3_text}, Seleccionada: {class_names[class_idx]} ({confidence:.2f})")
        
        # Retornar si la confianza es al menos 10% y la clase es válida
        if confidence > 0.1 and class_idx < len(class_names):
            return class_names[class_idx]
        else:
            return None
    except Exception as e:
        print(f"Error en detect_gesture: {e}")
        return None

def generate_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        current_prediction = "Error: Cámara no disponible"
        return
    
    if model is None:
        print("⚠⚠⚠ ADVERTENCIA CRÍTICA: Modelo NO está disponible")
        print("   La aplicación no podrá reconocer gestos sin el modelo.")
        current_prediction = "ERROR: Modelo no disponible"
    
    while True:
        success, frame = cap.read()
        if not success:
            print("ERROR: No se pudo leer frame de la cámara")
            break
        
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
                gesture = detect_gesture(frame, landmarks)
                if gesture:
                    current_prediction = f"Letra: {gesture}"
                else:
                    current_prediction = "Mano detectada, procesando..."
        else:
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
                    x_min = max(0, int(min([lm.x for lm in debug_landmarks]) * w) - 20)
                    y_min = max(0, int(min([lm.y for lm in debug_landmarks]) * h) - 20)
                    x_max = min(w, int(max([lm.x for lm in debug_landmarks]) * w) + 20)
                    y_max = min(h, int(max([lm.y for lm in debug_landmarks]) * h) + 20)
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size > 0:
                        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        hand_img = cv2.equalizeHist(hand_img)
                        hand_img = cv2.resize(hand_img, (28, 28))
                        hand_img = hand_img.astype('float32') / 255.0
                        hand_img = hand_img * 0.8 + 0.1
                        hand_img = np.expand_dims(hand_img, axis=-1)
                        hand_img = np.expand_dims(hand_img, axis=0)
                        pred = model.predict(hand_img, verbose=0)
                        top_idx = np.argmax(pred[0])
                        conf = pred[0][top_idx]
                        print(f"Frame {frame_count}: Mano detectada - Top predicción: {class_names[top_idx]} con confianza {conf:.3f} (umbral: 0.1)")
                except Exception as e:
                    print(f"Frame {frame_count}: Mano detectada pero error al procesar: {e}")
            elif gesture:
                print(f"Frame {frame_count}: ✓ Letra detectada: {gesture}")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": current_prediction})

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
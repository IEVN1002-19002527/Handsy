"""
Script para capturar videos de palabras y expresiones de la cámara para entrenar el modelo.
Permite capturar videos cortos de gestos de palabras/expresiones en tiempo real.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Configuración
DATA_DIR = "collected_word_data"
VIDEOS_PER_CLASS = 10  # Número de videos a capturar por palabra/expresión
VIDEO_DURATION = 3  # Duración del video en segundos
FPS = 30  # Frames por segundo
# Lista de palabras y expresiones comunes (puedes modificar esta lista)
CLASS_NAMES = ['hola', 'adios', 'gracias', 'si']

# Inicializar MediaPipe Hands - IMPORTANTE: detectar ambas manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def create_directories():
    """Crear directorios para almacenar los videos"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    print(f"[OK] Directorios creados en: {os.path.abspath(DATA_DIR)}")

def extract_both_hands_region(frame, all_landmarks):
    """Extraer región que incluye ambas manos del frame"""
    h, w, _ = frame.shape
    
    # Combinar todos los landmarks de ambas manos
    all_x_coords = []
    all_y_coords = []
    
    for landmarks in all_landmarks:
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
    
    if len(all_x_coords) == 0:
        return None
    
    # Calcular bounding box que incluye ambas manos
    min_x = min(all_x_coords)
    max_x = max(all_x_coords)
    min_y = min(all_y_coords)
    max_y = max(all_y_coords)
    
    # Calcular centro y dimensiones
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    # Aumentar el tamaño del bounding box (60% más grande) para incluir ambas manos
    max_dim = max(width, height) * 1.6
    
    x_min = max(0, int(center_x - max_dim / 2))
    y_min = max(0, int(center_y - max_dim / 2))
    x_max = min(w, int(center_x + max_dim / 2))
    y_max = min(h, int(center_y + max_dim / 2))
    
    # Verificar tamaño mínimo
    if (x_max - x_min) < 80 or (y_max - y_min) < 80:
        return None
    
    # Recortar región que incluye ambas manos
    hands_img = frame[y_min:y_max, x_min:x_max]
    
    if hands_img.size == 0:
        return None
    
    # Preprocesar: escala de grises y redimensionar a 28x28
    # Nota: Redimensionamos a 28x28 manteniendo el aspecto, pero el modelo verá ambas manos
    hands_img = cv2.cvtColor(hands_img, cv2.COLOR_BGR2GRAY)
    hands_img = cv2.resize(hands_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    return hands_img

def collect_video_for_class(class_name):
    """Capturar un video para una clase específica"""
    class_dir = os.path.join(DATA_DIR, class_name)
    existing_videos = len([f for f in os.listdir(class_dir) if f.endswith('.mp4')])
    
    if existing_videos >= VIDEOS_PER_CLASS:
        print(f"[OK] Ya existen {existing_videos} videos para la palabra '{class_name}'")
        return existing_videos
    
    videos_needed = VIDEOS_PER_CLASS - existing_videos
    print(f"\n{'='*60}")
    print(f"Capturando videos para la palabra/expresión: {class_name}")
    print(f"Videos necesarios: {videos_needed}")
    print(f"{'='*60}")
    print("\nInstrucciones:")
    print(f"1. Muestra el gesto de la palabra/expresión '{class_name}'")
    print("2. Puedes usar 1 o 2 manos según el gesto requerido")
    print("3. Presiona ESPACIO para iniciar la grabación (3 segundos)")
    print("4. Presiona 'q' para saltar esta palabra")
    print("5. Presiona ESC para salir del programa")
    print("\nEsperando detección de mano(s)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        return existing_videos
    
    captured = 0
    recording = False
    frames_to_record = []
    
    while captured < videos_needed:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Dibujar landmarks si se detectan manos (1 o 2 manos)
        if results.multi_hand_landmarks:
            all_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                all_landmarks.append(hand_landmarks.landmark)
            
            num_hands = len(results.multi_hand_landmarks)
            # Extraer región que incluye la(s) mano(s) detectada(s)
            hands_img = extract_both_hands_region(frame, all_landmarks)
            
            if hands_img is not None:
                # Mostrar preview de la imagen procesada
                preview = cv2.resize(hands_img, (280, 280))
                preview_colored = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                
                # Combinar frame original con preview
                h, w = frame.shape[:2]
                preview_h, preview_w = preview_colored.shape[:2]
                frame[10:10+preview_h, w-preview_w-10:w-10] = preview_colored
                
                # Información en pantalla
                info_text = f"Palabra: {class_name} | Videos: {captured + existing_videos}/{VIDEOS_PER_CLASS} | Manos: {num_hands}"
                if recording:
                    info_text += " [GRABANDO]"
                cv2.putText(frame, info_text, (10, h-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not recording else (0, 0, 255), 2)
                cv2.putText(frame, "ESPACIO: Grabar | Q: Saltar palabra | ESC: Salir", 
                           (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            cv2.putText(frame, "No se detectan manos. Muestra el gesto de la palabra.", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Captura de Videos - ' + class_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"Saltando palabra '{class_name}'")
            break
        elif key == 27:  # ESC
            print("Saliendo del programa...")
            cap.release()
            cv2.destroyAllWindows()
            return captured + existing_videos
        elif key == ord(' ') and results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1 and not recording:
            # Iniciar grabación (con 1 o 2 manos)
            recording = True
            frames_to_record = []
            num_hands = len(results.multi_hand_landmarks)
            print(f"  Iniciando grabación {captured + 1}/{videos_needed}... ({num_hands} mano(s) detectada(s))")
        
        # Si está grabando, capturar frames (con 1 o 2 manos)
        if recording and results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
            all_landmarks = [hand_landmarks.landmark for hand_landmarks in results.multi_hand_landmarks]
            hands_img = extract_both_hands_region(frame, all_landmarks)
            if hands_img is not None:
                frames_to_record.append(hands_img)
                
                # Si se alcanzó la duración del video, guardarlo
                if len(frames_to_record) >= VIDEO_DURATION * FPS:
                    # Guardar video
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    video_filename = os.path.join(class_dir, f"{class_name}_{timestamp}.mp4")
                    
                    # Crear video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_filename, fourcc, FPS, (28, 28), False)
                    
                    for frame_img in frames_to_record:
                        out.write(frame_img)
                    
                    out.release()
                    captured += 1
                    recording = False
                    frames_to_record = []
                    print(f"  [OK] Video {captured}/{videos_needed} guardado: {video_filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_videos = captured + existing_videos
    print(f"\n[OK] Total de videos para '{class_name}': {total_videos}")
    return total_videos

def main():
    """Función principal"""
    print("="*60)
    print("RECOLECCIÓN DE VIDEOS PARA PALABRAS Y EXPRESIONES")
    print("="*60)
    print(f"\nEste script te permitirá capturar videos de palabras/expresiones desde la cámara.")
    print(f"Palabras/expresiones disponibles: {', '.join(CLASS_NAMES)}")
    print(f"Videos por palabra: {VIDEOS_PER_CLASS}")
    print(f"Duración de cada video: {VIDEO_DURATION} segundos")
    print(f"\n¿Deseas capturar videos para todas las palabras? (s/n): ", end='')
    
    response = input().strip().lower()
    
    create_directories()
    
    if response == 's':
        # Capturar para todas las palabras
        for class_name in CLASS_NAMES:
            collect_video_for_class(class_name)
            if class_name != CLASS_NAMES[-1]:
                print(f"\nPresiona ENTER para continuar con la siguiente palabra...")
                input()
    else:
        # Capturar para palabras específicas
        print("\nIngresa las palabras que deseas capturar (separadas por coma):")
        print(f"Palabras disponibles: {', '.join(CLASS_NAMES)}")
        words_input = input().strip().lower()
        selected_words = [w.strip() for w in words_input.split(',') if w.strip() in CLASS_NAMES]
        
        if not selected_words:
            print("No se seleccionaron palabras válidas.")
            return
        
        for class_name in selected_words:
            collect_video_for_class(class_name)
            if class_name != selected_words[-1]:
                print(f"\nPresiona ENTER para continuar con la siguiente palabra...")
                input()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE VIDEOS RECOLECTADOS")
    print("="*60)
    total_videos = 0
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.mp4')])
            total_videos += count
            print(f"  {class_name}: {count} videos")
    print(f"\nTotal: {total_videos} videos")
    print(f"\n[OK] Videos guardados en: {os.path.abspath(DATA_DIR)}")
    print("\nAhora puedes ejecutar 'train_word_model.py' para entrenar el modelo.")

if __name__ == '__main__':
    main()


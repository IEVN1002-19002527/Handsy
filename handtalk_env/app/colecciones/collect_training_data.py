"""
Script para capturar y etiquetar imágenes de la cámara para reentrenar el modelo.
Permite capturar imágenes de gestos en tiempo real y etiquetarlas.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

# Configuración
DATA_DIR = "collected_data"
IMAGES_PER_CLASS = 100  # Número de imágenes a capturar por clase
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def create_directories():
    """Crear directorios para almacenar las imágenes"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    print(f"✓ Directorios creados en: {os.path.abspath(DATA_DIR)}")

def extract_hand_region(frame, landmarks):
    """Extraer región de la mano del frame"""
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
    
    # Verificar tamaño mínimo
    if (x_max - x_min) < 50 or (y_max - y_min) < 50:
        return None
    
    # Recortar región de la mano
    hand_img = frame[y_min:y_max, x_min:x_max]
    
    if hand_img.size == 0:
        return None
    
    # Preprocesar igual que en app.py
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    hand_img = cv2.resize(hand_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    return hand_img

def collect_images_for_class(class_name):
    """Capturar imágenes para una clase específica"""
    class_dir = os.path.join(DATA_DIR, class_name)
    existing_images = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
    
    if existing_images >= IMAGES_PER_CLASS:
        print(f"✓ Ya existen {existing_images} imágenes para la clase '{class_name}'")
        return existing_images
    
    images_needed = IMAGES_PER_CLASS - existing_images
    print(f"\n{'='*60}")
    print(f"Capturando imágenes para la clase: {class_name}")
    print(f"Imágenes necesarias: {images_needed}")
    print(f"{'='*60}")
    print("\nInstrucciones:")
    print(f"1. Muestra el gesto de la letra '{class_name}'")
    print("2. Presiona ESPACIO para capturar una imagen")
    print("3. Presiona 'q' para saltar esta clase")
    print("4. Presiona ESC para salir del programa")
    print("\nEsperando detección de mano...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        return existing_images
    
    captured = 0
    frame_count = 0
    
    while captured < images_needed:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Dibujar landmarks si se detecta una mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extraer región de la mano
                hand_img = extract_hand_region(frame, hand_landmarks.landmark)
                
                if hand_img is not None:
                    # Mostrar preview de la imagen procesada
                    preview = cv2.resize(hand_img, (280, 280))
                    preview_colored = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                    
                    # Combinar frame original con preview
                    h, w = frame.shape[:2]
                    preview_h, preview_w = preview_colored.shape[:2]
                    frame[10:10+preview_h, w-preview_w-10:w-10] = preview_colored
                    
                    # Información en pantalla
                    info_text = f"Clase: {class_name} | Capturadas: {captured + existing_images}/{IMAGES_PER_CLASS}"
                    cv2.putText(frame, info_text, (10, h-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "ESPACIO: Capturar | Q: Saltar clase | ESC: Salir", 
                               (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            cv2.putText(frame, "No se detecta mano. Muestra el gesto de la letra.", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Captura de Datos - ' + class_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"Saltando clase '{class_name}'")
            break
        elif key == 27:  # ESC
            print("Saliendo del programa...")
            cap.release()
            cv2.destroyAllWindows()
            return captured + existing_images
        elif key == ord(' ') and results.multi_hand_landmarks:
            # Capturar imagen
            hand_img = extract_hand_region(frame, results.multi_hand_landmarks[0].landmark)
            if hand_img is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(class_dir, f"{class_name}_{timestamp}.png")
                cv2.imwrite(filename, hand_img)
                captured += 1
                print(f"  ✓ Imagen {captured}/{images_needed} capturada: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_images = captured + existing_images
    print(f"\n✓ Total de imágenes para '{class_name}': {total_images}")
    return total_images

def main():
    """Función principal"""
    print("="*60)
    print("RECOLECCIÓN DE DATOS PARA REENTRENAMIENTO")
    print("="*60)
    print(f"\nEste script te permitirá capturar imágenes de gestos desde la cámara.")
    print(f"Clases disponibles: {', '.join(CLASS_NAMES)}")
    print(f"Imágenes por clase: {IMAGES_PER_CLASS}")
    print(f"\n¿Deseas capturar imágenes para todas las clases? (s/n): ", end='')
    
    response = input().strip().lower()
    
    create_directories()
    
    if response == 's':
        # Capturar para todas las clases
        for class_name in CLASS_NAMES:
            collect_images_for_class(class_name)
            print(f"\nPresiona ENTER para continuar con la siguiente clase...")
            input()
    else:
        # Capturar para clases específicas
        print("\nIngresa las clases que deseas capturar (separadas por coma):")
        print(f"Clases disponibles: {', '.join(CLASS_NAMES)}")
        classes_input = input().strip().upper()
        selected_classes = [c.strip() for c in classes_input.split(',') if c.strip() in CLASS_NAMES]
        
        if not selected_classes:
            print("No se seleccionaron clases válidas.")
            return
        
        for class_name in selected_classes:
            collect_images_for_class(class_name)
            if class_name != selected_classes[-1]:
                print(f"\nPresiona ENTER para continuar con la siguiente clase...")
                input()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE DATOS RECOLECTADOS")
    print("="*60)
    total_images = 0
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            total_images += count
            print(f"  {class_name}: {count} imágenes")
    print(f"\nTotal: {total_images} imágenes")
    print(f"\n✓ Datos guardados en: {os.path.abspath(DATA_DIR)}")
    print("\nAhora puedes ejecutar 'train_camera_model.py' para reentrenar el modelo.")

if __name__ == '__main__':
    main()


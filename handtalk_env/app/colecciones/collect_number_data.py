"""
Script para capturar y etiquetar imágenes de números (0-5) de la cámara para entrenar el modelo.
Permite capturar imágenes de gestos numéricos en tiempo real y etiquetarlas.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Configuración
DATA_DIR = "collected_number_data"
IMAGES_PER_CLASS = 100  # Número de imágenes a capturar por clase
CLASS_NAMES = ['0', '1', '2', '3', '4', '5']  # Números del 0 al 5

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
    
    print(f"[OK] Directorios creados en: {os.path.abspath(DATA_DIR)}")

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
        print(f"[OK] Ya existen {existing_images} imágenes para la clase '{class_name}'")
        return existing_images
    
    images_needed = IMAGES_PER_CLASS - existing_images
    print(f"\n{'='*60}")
    print(f"Capturando imágenes para el número: {class_name}")
    print(f"Imágenes necesarias: {images_needed}")
    print(f"{'='*60}")
    print("\nInstrucciones:")
    print(f"1. Muestra el gesto del número '{class_name}'")
    print("2. Presiona ESPACIO para capturar una imagen")
    print("3. Presiona 'q' para saltar este número")
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
                    info_text = f"Numero: {class_name} | Capturadas: {captured + existing_images}/{IMAGES_PER_CLASS}"
                    cv2.putText(frame, info_text, (10, h-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "ESPACIO: Capturar | Q: Saltar numero | ESC: Salir", 
                               (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            cv2.putText(frame, "No se detecta mano. Muestra el gesto del numero.", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Captura de Numeros - ' + class_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"Saltando numero '{class_name}'")
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
                print(f"  [OK] Imagen {captured}/{images_needed} capturada: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_images = captured + existing_images
    print(f"\n[OK] Total de imágenes para '{class_name}': {total_images}")
    return total_images

def main():
    """Función principal"""
    print("="*60)
    print("RECOLECCION DE DATOS PARA NUMEROS (0-5)")
    print("="*60)
    print(f"\nEste script te permitira capturar imagenes de numeros desde la camara.")
    print(f"Numeros disponibles: {', '.join(CLASS_NAMES)}")
    print(f"Imagenes por numero: {IMAGES_PER_CLASS}")
    print(f"\n¿Deseas capturar imagenes para todos los numeros? (s/n): ", end='')
    
    response = input().strip().lower()
    
    create_directories()
    
    if response == 's':
        # Capturar para todos los números
        for class_name in CLASS_NAMES:
            collect_images_for_class(class_name)
            if class_name != CLASS_NAMES[-1]:
                print(f"\nPresiona ENTER para continuar con el siguiente numero...")
                input()
    else:
        # Capturar para números específicos
        print("\nIngresa los numeros que deseas capturar (separados por coma):")
        print(f"Numeros disponibles: {', '.join(CLASS_NAMES)}")
        numbers_input = input().strip()
        selected_numbers = [n.strip() for n in numbers_input.split(',') if n.strip() in CLASS_NAMES]
        
        if not selected_numbers:
            print("No se seleccionaron numeros validos.")
            return
        
        for class_name in selected_numbers:
            collect_images_for_class(class_name)
            if class_name != selected_numbers[-1]:
                print(f"\nPresiona ENTER para continuar con el siguiente numero...")
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
            print(f"  {class_name}: {count} imagenes")
    print(f"\nTotal: {total_images} imagenes")
    print(f"\n[OK] Datos guardados en: {os.path.abspath(DATA_DIR)}")
    print("\nAhora puedes ejecutar 'train_number_model.py' para entrenar el modelo.")

if __name__ == '__main__':
    main()


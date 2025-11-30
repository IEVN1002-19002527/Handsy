# Guía de Reentrenamiento del Modelo con Imágenes de Cámara

Esta guía te mostrará cómo reentrenar el modelo de reconocimiento de gestos usando imágenes capturadas desde tu cámara.

## ¿Por qué reentrenar el modelo?

El modelo actual fue entrenado con el dataset Sign Language MNIST, que contiene imágenes estáticas con fondos uniformes. Las imágenes de tu cámara pueden ser muy diferentes:
- Fondos complejos
- Iluminación diferente
- Ángulos y perspectivas distintas
- Calidad de imagen diferente

Reentrenar el modelo con imágenes de tu propia cámara mejorará significativamente la precisión de detección.

## Paso 1: Capturar Imágenes de Entrenamiento

### Ejecutar el script de captura:

```bash
cd handtalk_env/app
python collect_training_data.py
```

### Instrucciones de uso:

1. **Seleccionar clases a capturar:**
   - Opción 1: Capturar para todas las clases (A-Y, sin J)
   - Opción 2: Capturar solo para clases específicas

2. **Para cada clase:**
   - Muestra el gesto de la letra frente a la cámara
   - Presiona **ESPACIO** para capturar una imagen
   - Presiona **Q** para saltar la clase actual
   - Presiona **ESC** para salir del programa

3. **Recomendaciones:**
   - Captura al menos **50-100 imágenes por clase** para mejores resultados
   - Varía la posición de la mano (ligeramente a la izquierda, derecha, arriba, abajo)
   - Usa diferentes ángulos de la mano
   - Mantén condiciones de iluminación consistentes
   - Usa un fondo uniforme (preferiblemente claro)

### Estructura de datos:

Las imágenes se guardarán en:
```
collected_data/
  ├── A/
  │   ├── A_20231108_120000_123456.png
  │   ├── A_20231108_120001_234567.png
  │   └── ...
  ├── B/
  │   └── ...
  └── ...
```

## Paso 2: Reentrenar el Modelo

### Ejecutar el script de entrenamiento:

```bash
python train_camera_model.py
```

### Opciones de entrenamiento:

1. **Entrenar desde cero:**
   - Recomendado si tienes muchas imágenes (500+)
   - El modelo aprenderá completamente de tus datos
   - Tarda más tiempo pero puede dar mejores resultados

2. **Fine-tuning:**
   - Recomendado si tienes pocas imágenes (100-500)
   - Usa el modelo existente como punto de partida
   - Ajusta los pesos del modelo con tus datos
   - Más rápido y puede funcionar bien con menos datos

### El script hará:

1. Cargar todas las imágenes capturadas
2. Preprocesarlas (escala de grises, 28x28, normalización)
3. Dividir en datos de entrenamiento (80%) y validación (20%)
4. Entrenar el modelo
5. Guardar el mejor modelo en `model/gesture_model_camera.h5`

## Paso 3: Usar el Nuevo Modelo

### Opción 1: Reemplazar el modelo existente (Recomendado)

```bash
# Hacer backup del modelo anterior
mv model/gesture_model.h5 model/gesture_model_old.h5

# Usar el nuevo modelo
mv model/gesture_model_camera.h5 model/gesture_model.h5
```

### Opción 2: Actualizar app.py para usar el nuevo modelo

Edita `app.py` y cambia la ruta del modelo:

```python
MODEL_PATH = os.path.join(script_dir, 'model', 'gesture_model_camera.h5')
```

## Mejores Prácticas

### Para mejores resultados:

1. **Cantidad de datos:**
   - Mínimo: 20-30 imágenes por clase
   - Recomendado: 50-100 imágenes por clase
   - Ideal: 100+ imágenes por clase

2. **Variedad en las imágenes:**
   - Diferentes posiciones de la mano
   - Diferentes ángulos
   - Diferentes personas (si es posible)
   - Condiciones de iluminación similares a las de uso real

3. **Calidad de las imágenes:**
   - Buena iluminación
   - Fondo uniforme
   - Mano claramente visible
   - Gesto correcto y consistente

4. **Entrenamiento:**
   - Usa "Entrenar desde cero" si tienes suficientes datos
   - Usa "Fine-tuning" si tienes pocos datos
   - Reentrena periódicamente cuando agregues más datos

## Solución de Problemas

### Error: "No se encontraron imágenes"
- Asegúrate de haber ejecutado `collect_training_data.py` primero
- Verifica que el directorio `collected_data` existe y contiene imágenes

### Error: "Algunas clases tienen pocas imágenes"
- Captura más imágenes para esas clases
- Puedes continuar pero los resultados pueden no ser óptimos

### El modelo no mejora después del reentrenamiento
- Verifica que las imágenes están bien etiquetadas
- Captura más imágenes variadas
- Prueba ajustar los hiperparámetros (epochs, batch_size, learning_rate)

### El modelo funciona peor que antes
- Verifica la calidad de las imágenes capturadas
- Asegúrate de que los gestos son correctos
- Considera hacer fine-tuning en lugar de entrenar desde cero
- Vuelve al modelo anterior si es necesario

## Estructura de Archivos

```
handtalk_env/app/
  ├── collect_training_data.py    # Script para capturar imágenes
  ├── train_camera_model.py       # Script para reentrenar el modelo
  ├── collected_data/             # Imágenes capturadas (se crea automáticamente)
  │   ├── A/
  │   ├── B/
  │   └── ...
  ├── model/
  │   ├── gesture_model.h5        # Modelo original
  │   ├── gesture_model_camera.h5 # Modelo reentrenado
  │   └── train_model.py
  └── app.py                      # Aplicación principal
```

## Ejemplo de Uso Completo

```bash
# 1. Capturar imágenes
python collect_training_data.py
# (Seguir las instrucciones en pantalla)

# 2. Reentrenar modelo
python train_camera_model.py
# (Seleccionar opción 1 o 2)

# 3. Usar el nuevo modelo
mv model/gesture_model_camera.h5 model/gesture_model.h5

# 4. Ejecutar la aplicación
python app.py
```

## Notas Adicionales

- El proceso de captura puede tomar tiempo (10-30 minutos dependiendo de cuántas clases y cuántas imágenes por clase)
- El entrenamiento puede tomar desde minutos hasta horas dependiendo de la cantidad de datos y hardware
- Se recomienda hacer backup del modelo anterior antes de reemplazarlo
- Puedes agregar más imágenes y reentrenar en cualquier momento para mejorar el modelo

## Próximos Pasos

1. Captura imágenes para todas las clases
2. Reentrena el modelo
3. Prueba el nuevo modelo en la aplicación
4. Si es necesario, captura más imágenes y reentrena
5. Compara los resultados con el modelo original


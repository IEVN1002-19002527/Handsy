# Instrucciones Rápidas: Reentrenar Modelo con Cámara

## Pasos Rápidos

### 1. Capturar Imágenes
```bash
cd handtalk_env/app
python collect_training_data.py
```
- Muestra gestos frente a la cámara
- Presiona **ESPACIO** para capturar
- Captura 50-100 imágenes por clase (recomendado)

### 2. Reentrenar Modelo
```bash
python train_camera_model.py
```
- Selecciona opción 1 (desde cero) o 2 (fine-tuning)
- Espera a que termine el entrenamiento

### 3. Usar Nuevo Modelo
El modelo reentrenado se guarda como `model/gesture_model_camera.h5`

**Opción A:** Reemplazar modelo (recomendado)
```bash
mv model/gesture_model.h5 model/gesture_model_backup.h5
mv model/gesture_model_camera.h5 model/gesture_model.h5
```

**Opción B:** La app.py ya busca automáticamente el modelo reentrenado primero

### 4. Probar
```bash
python app.py
```

## Tips

- **Más imágenes = mejor precisión** (mínimo 20-30 por clase, ideal 100+)
- **Variedad:** Diferentes posiciones, ángulos, iluminación
- **Calidad:** Fondo uniforme, buena iluminación, gesto claro
- **Fine-tuning:** Usa si tienes pocas imágenes (<500 total)
- **Desde cero:** Usa si tienes muchas imágenes (500+ total)

## Estructura

```
collected_data/          # Imágenes capturadas
  ├── A/
  ├── B/
  └── ...
model/
  ├── gesture_model.h5           # Modelo original
  └── gesture_model_camera.h5    # Modelo reentrenado (se usa automáticamente)
```

## Problemas Comunes

- **"No se encontraron imágenes"**: Ejecuta `collect_training_data.py` primero
- **"Pocas imágenes"**: Captura más imágenes para mejores resultados
- **Modelo no mejora**: Verifica calidad de imágenes y gestos correctos


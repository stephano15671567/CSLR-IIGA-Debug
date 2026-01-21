"""
FAQ: PREGUNTAS FRECUENTES SOBRE CSLR-IIGA
==========================================

Estas preguntas cubre lo que tu profesor probablemente te va a preguntar.
"""

# P1: ¿QUÉ SIGNIFICA "12-FRAME WINDOW"?

R: Significa que procesamos exactamente 12 frames por glosa (palabra en lenguaje de signos).

Contexto:
  - La duración típica de una seña es 0.48 segundos
  - El video se graba a 25 frames por segundo
  - 0.48 × 25 = 12 frames (aproximadamente)
  
Por qué es fijo:
  - Es un hiperparámetro elegido en el paper
  - Cambiar a 20 frames requeriría reentrenar todo
  - Está "baked-in" en la arquitectura

¿Puede cambiar?
  Si: Pero necesitarías:
    1. Ajustar código en train.py línea 63
    2. Reentrenar desde cero (no transfer learning)
    3. Posiblemente empeore WER (ajuste fino necesario)

Localización en código:
  train.py línea 63: local_window = 12
  dataloader.py línea 150: frame selection logic


# P2: ¿POR QUÉ NO PUEDO USAR 20 FRAMES EN VEZ DE 12?

R: Porque cambiar el tamaño de ventana rompe la arquitectura. Aquí está por qué:

Problema arquitectónico:
  1. CNN output: (B, 12, 1280) ← hardcoded como 12
  2. Positional Encoding: generada para longitud 12
  3. Transformer: espera (B, 12, 1280)
  
  Si usas 20 frames:
    CNN output: (B, 20, 1280)
    Positional Encoding: shape (1, 12, 1280) ← INCOMPATIBLE
    ¡Error!

Soluciones posibles:
  1. Reescribir para variable window size
  2. Generar PE dinámica (más complejo)
  3. Reentrenar desde cero (GPU intensivo)

Mi recomendación:
  NO lo hagas. El modelo fue diseñado para 12 frames.
  Si necesitas experimentar, crea rama git separada.

Localización:
  train.py línea 63: local_window = 12
  transformer.py línea XX: Positional encoding hardcodeado


# P3: ¿PUEDO USAR EL MODELO EN OTRO DATASET (CSL-DAILY)?

R: Teóricamente sí. Aquí está el plan:

Transfer Learning (la forma correcta):
  1. Cargar BEST.pt (entrenado en PHOENIX-2014)
  2. Congelar CNN (características generales)
  3. Descongelar Transformer + Decoder
  4. Reentrenar en CSL-DAILY (~50 épocas)

Cambios necesarios:
  1. Dataloader: adaptarlo a estructura de CSL-DAILY
  2. Vocabulario: CSL-DAILY tiene ~500 glosas vs PHOENIX 1232
     - Necesita retrain del Decoder (1280 → 500)
  3. Métricas: recalcular vocabulary mapping

Código necesario:
  ```python
  # Cargar modelo entrenado
  model = IIGAModel()
  model.load_state_dict(torch.load('BEST.pt'))
  
  # Congelar CNN (opcional pero recomendado)
  for param in model.cnn.parameters():
      param.requires_grad = False
  
  # El Decoder se reentrenará automáticamente
  # (1280 → 500 nuevas glosas)
  ```

Tiempo estimado:
  - Preparación: 1-2 horas
  - Entrenamiento: 6-12 horas (GPU)
  - Pruebas: 2-3 horas

Riesgo: Puede funcionar bien o no según similitud de datasets

Alternativa: Usar Swin-MSTP (ver directorio Swin-MSTP)


# P4: ¿CUÁLES SON LAS CARACTERÍSTICAS DEL DATASET?

R: PHOENIX-2014 tiene estas características:

Estructura:
  Dataset/
  ├─ train/ (2298 videos)
  ├─ dev/ (347 videos)
  └─ test/ (357 videos)

Características de video:
  - Resolución: 384×288 píxeles
  - FPS: 25 frames por segundo
  - Formato: PNG secuencial (ej: 00000.png, 00001.png)
  - Duración: 1-10 segundos típicamente

Vocabulario:
  - 1232 glosas únicas (palabras en lenguaje de signos)
  - Distribución: clase desbalanceada
    * Glosas comunes: > 100 ejemplos
    * Glosas raras: < 5 ejemplos

Anotaciones:
  - Formato CSV: video_name | split | gloss_sequence
  - Ejemplo: S0001 | 1 | HOLA BANCO DINERO
  - Cada video tiene una secuencia de glosas

Estadísticas:
  - Promedio de glosas por video: 5-7
  - Promedio de frames por glosa: 8-15
  - Total de horas: ~500 horas
  - Total de glosas anotadas: ~50,000

Problemas conocidos:
  - Desbalanceo: algunas glosas muy frecuentes, otras raras
  - Variabilidad: diferentes personas firman de formas ligeramente distintas
  - Ruido: fondo a veces no es limpio

Soluciones implementadas en CSLR-IIGA:
  - Segmentación MediaPipe: elimina fondo
  - 12-frame window: normaliza duración
  - 224×224 rescale: entrada estándar para CNN


# P5: ¿QUÉ SIGNIFICA "STEPS" VS "EPOCHS"?

R: 
  - Epoch: recorrer TODO el dataset una vez
  - Step: procesar UN batch (lote pequeño)

Ejemplo concreto:
  Dataset: 2298 videos de entrenamiento
  Batch size: 25 videos
  
  Batches por época: 2298 / 25 = 92 batches
  1 época = 92 steps
  
  Guardamos checkpoint cada 500 steps:
    500 steps ÷ 92 = 5.4 épocas
    
  Entonces:
    checkpoint_epoch_1_step_500.pt → después ~5.4 épocas
    checkpoint_epoch_1_step_1000.pt → después ~10.8 épocas
    checkpoint_epoch_2.pt → después exactamente 2 épocas

¿Por qué dos tipos de checkpoints?
  1. Checkpoints de "step": recuperación por GPU hang
     - Si GPU falla en step 750, reinicia en 500
  2. Checkpoints de "epoch": seguimiento de progreso
     - Más fácil de entender: "estoy en época 3"

En código:
  train.py línea 433-450: lógica de guardado


# P6: ¿CÓMO SE ELIMINA EL FONDO?

R: Usando MediaPipe Holistic + threshold 0.5

Paso a paso:

  1. MEDIAPIPE DETECTA:
     - Cuerpo (33 keypoints)
     - Manos (21×2 keypoints)
     - Cara (468 keypoints)
     → Resultado: segmentation_mask (384×288) valores [0, 1]

  2. APLICAR THRESHOLD:
     mask_binaria = segmentation_mask > 0.5
     - 0.5: compromiso entre inclusión y exclusión
     - Valor > 0.5 → persona (1)
     - Valor ≤ 0.5 → fondo (0)

  3. APLICAR A FRAME:
     frame_seg = frame * mask_binaria
     - Donde mask=1: se mantiene pixel
     - Donde mask=0: pixel pasa a 0 (negro)

  4. RESIZE A 64×64:
     roi = resize(mask_binaria, (64, 64))
     - Dimensionalidad pequeña (4KB vs 147KB)
     - Se envía como entrada adicional al modelo

Código:
  extract_segmentation.py línea 119-122:
    right_hand_mask = (results.segmentation_mask > 0.5).astype(np.uint8)
    roi = cv2.resize(right_hand_mask, (64, 64))

Efecto en precisión:
  - Sin segmentación: WER ~22%
  - Con segmentación: WER ~15%
  - Mejora: ~7% (importante!)


# P7: ¿CÓMO CALCULA EL MODELO PÉRDIDA (LOSS)?

R: Usa CTC Loss (Connectionist Temporal Classification)

Problema que resuelve:
  - Predicción: 12 frames → 12 labels
  - Verdad: solo 3 glosas
  - ¿Cómo alinear?

Solución CTC:
  - Permite múltiples frames = misma glosa
  - Automáticamente encuentra mejor alineación
  - Usa algoritmo forward-backward

Ejemplo:
  Predicción frame-wise: [HOLA, HOLA, HOLA, HOLA, BANCO, BANCO, DINERO, ...]
  CTC collapsa: [HOLA, BANCO, DINERO]
  Verdad: [HOLA, BANCO, DINERO]
  → Loss bajo (match!)

Fórmula (simplificada):
  L_CTC = -log(P(etiqueta_verdadera | predicción))
  
  donde P se calcula integrando sobre todas posibles alineaciones

Evolución durante entrenamiento:
  Epoch 1: Loss ~3.5 (modelo aprende)
  Epoch 10: Loss ~1.8 (mejora)
  Epoch 20: Loss ~1.0 (bien)
  Epoch 40: Loss ~0.3 (convergencia)

Código:
  train.py línea 290-295: CTC Loss setup
  train.py línea 400-410: loss computation


# P8: ¿QUÉ ES EARLY STOPPING?

R: Dejar de entrenar cuando validación deja de mejorar

Implementación:
  patience = 3
  best_wer = infinito
  
  Para cada época:
    val_wer = evaluar en validation set
    
    if val_wer < best_wer:
      best_wer = val_wer
      guardar BEST.pt
      patience = 0
    else:
      patience += 1
      if patience >= 3:
        break  ← EARLY STOPPING

Beneficios:
  ✓ Evita overfitting
  ✓ Ahorra tiempo GPU
  ✓ Modelo generaliza mejor

Implementación en código:
  train.py línea 450-470: early stopping logic

En nuestro caso:
  Normalmente converge alrededor de época 25-30
  Nunca llega a 40 épocas


# P9: ¿POR QUÉ TRES MÉTRICAS (WER, BLEU, ROUGE)?

R: Cada una mide aspecto diferente

WER (Word Error Rate):
  - PRINCIPAL: errores a nivel palabra
  - Rango: 0% (perfecto) a ∞ (terrible)
  - Interpretación: # errores / # palabras
  - Nuestro objetivo: < 15% en PHOENIX test

BLEU (Bilingual Evaluation Understudy):
  - Secundaria: coincidencia de n-gramas
  - Detecta si secuencias cortas son correctas
  - BLEU-1: palabras individuales
  - BLEU-2,3,4: pares, triplas, etc.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
  - Terciaria: subsecuencia común más larga
  - Mide cuánto de la verdad está en predicción
  - Más indulgente que WER

Ejemplo:
  Verdad:     A B C D E F G
  Predicción: A X B Y C F Z
  
  WER = 4/7 = 57%    (4 errores, 7 palabras)
  BLEU-2 = 2/6 = 33% (AB y CF correctos, otros no)
  ROUGE-L = ... (LCS = [A,B,C,F], longitud 4)

¿Cuál es la principal?
  WER. Es el estándar en industria para SLR.

¿Cuándo usar las otras?
  - Para análisis detallado
  - Para comparar con papers
  - Para detectar diferentes tipos de errores


# P10: ¿PUEDE EL MODELO SABER QUÉ ESTÁ VIENDO EN TIEMPO REAL?

R: Teóricamente sí, pero con limitaciones.

Requisitos para inferencia en tiempo real:
  1. Buffer de 12 frames (ventana deslizante)
  2. Predicción en ciclo cuando se completan 12
  3. Latencia: ~500ms (tiempo de inferencia)

Implementación:
  ```python
  # Cola de frames
  frame_buffer = collections.deque(maxlen=12)
  
  while True:
      frame = cap.read()  # Nuevo frame
      frame_buffer.append(frame)
      
      if len(frame_buffer) == 12:
          # Predicción
          features = model.cnn(frame_buffer)
          pred = model.transformer(features)
          gloss = model.decoder(pred)
          print(f"Glosa predicha: {gloss}")
  ```

Limitaciones:
  - Necesita 12 frames = 0.48 segundos de delay
  - No es realmente "en tiempo real" (hay latencia)
  - Necesita GPU para velocidad
  - Difícil de implementar sin framework especializado

Estado actual:
  El código entrenado (train.py) es solo para entrenamiento
  No hay implementación de inferencia en tiempo real
  Sería proyecto adicional


# P11: ¿CUÁL ES LA DIFERENCIA CON CNN PURO?

R: 
  CNN solo: La arquitectura original antes de IIGA
  CNN+Transformer (IIGA): Mejor

Diferencia:

  CNN PURO:
    - Solo extrae features frame by frame
    - No ve relaciones temporales
    - WER ~25-30%
    - Mejor para tareas de frameclassification

  CNN + TRANSFORMER (IIGA):
    - CNN extrae features
    - Transformer ve relaciones temporales (Intra + Inter)
    - WER ~12-15%
    - Mejor para sequence prediction

Mejora:
  5-15% en WER
  Más parámetros pero mejor generalización


# P12: ¿CÓMO FUNCIONA LA CONCATENACIÓN DE FEATURES?

R: Los features de diferentes modalidades se combinan:

  1. CNN Features: (12, 1280) [de frames RGB]
  2. Segmentation Mask: (12, 64, 64) → flatten → (12, 4096)
  3. Hand Landmarks: (21×2) = 42 keypoints → (12, 42*3=126)
  
  Concatenar: torch.cat([features1, features2, features3], dim=-1)
  Resultado: (12, 1280 + 4096 + 126) = (12, 5502)
  
  Entonces: Linear(5502 → 1280) para normalizar

Estado actual:
  El código actual principalmente usa CNN features
  Segmentación se usa pero como entrada "soft"
  Hand landmarks podrían agregarse (proyecto futuro)


# P13: ¿PUEDE DETECTAR SIGNOS DE DIFERENTES PERSONAS?

R: Sí, pero con limitaciones.

Dataset PHOENIX-2014:
  - Múltiples signers (diferentes personas)
  - Diferentes estilos de firmar
  - Diferentes velocidades

Cómo el modelo aprende:
  1. Características invariantes: movimiento de manos (CNN)
  2. Flexibilidad temporal: IIGA permite variación en duración
  3. Normalización: 12-frame window estandariza

Limitaciones:
  - Si persona firma MUY diferente: puede fallar
  - Persona nueva (no en entrenamiento): riesgo de overfitting
  - Personas con limitaciones físicas: puede no funcionar

¿Cómo mejorar?
  - Más datos de diferentes signers
  - Data augmentation: aplicar transformaciones
  - Speaker adaptation: fine-tuning por persona


# P14: ¿QUÉ PASA CON LAS GLOSAS RARAS (POCA DATOS)?

R: Problema de desequilibrio de clases

Distribución:
  - Glosas comunes: >100 ejemplos → WER bajo
  - Glosas raras: <5 ejemplos → WER alto
  - Resultado: modelo sesgado hacia frecuentes

Síntomas:
  - El modelo tiende a predecir glosas comunes
  - WER alto en test (mix de común y raro)

Soluciones:
  1. Weighted Loss: dar más peso a glosas raras
  2. Data Augmentation: generar variaciones
  3. Over-sampling: repetir glosas raras
  4. Under-sampling: reducir glosas comunes

En CSLR-IIGA:
  Implementado: Class weighting en CTC Loss
  train.py línea 290-295: ver configuración

Resultado:
  Mejora WER en glosas raras
  Ligera degradación en comunes (trade-off)


# P15: ¿CÓMO SE VALIDA EL MODELO?

R: 
  Validation set: 347 videos del dev set PHOENIX-2014

Proceso cada época:
  ```
  1. Procesar dev set completo
  2. Obtener predicciones
  3. Calcular WER, BLEU, ROUGE
  4. Si WER mejora: guardar BEST.pt
  5. Si WER estanca: aumentar patience counter
  6. Si patience >= 3: early stopping
  ```

Metrología:
  - Métrica principal: WER (dev set)
  - Reportar después: WER (test set)
  - Confirmar: BLEU, ROUGE

Resultados típicos:
  Dev WER: ~14-16%
  Test WER: ~17-20% (más alto porque new data)
  BLEU-1: ~50-60%
  ROUGE-L: ~45-55%


==============================================================================
PREGUNTAS SIN RESPUESTA = BÚSCALAS EN DOCUMENTOS
==============================================================================

Para más detalles:
  - GUIA_PASO_A_PASO.md: Tutorial completo
  - ARCHITECTURE.md: Descripción técnica profunda
  - DATASETS.md: Información de datasets
"""

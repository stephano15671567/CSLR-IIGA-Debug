"""
GUÍA PASO A PASO: CÓMO ENTRENAR CSLR-IIGA EN TU PC LOCAL
========================================================

Este documento explica EXACTAMENTE qué pasa en cada etapa del entrenamiento.
Está diseñado para que puedas explicárselo a tu profesor.

Tabla de contenidos:
1. Conceptos clave
2. Entrada: Datos
3. Procesamiento: CNN + Transformer
4. Salida: Predicciones
5. Pérdida: CTC Loss
6. Métricas: WER, BLEU, ROUGE
7. Ciclo completo de entrenamiento
8. Comandos de ejecución

==============================================================================
1. CONCEPTOS CLAVE
==============================================================================

¿QUÉ ES SIGN LANGUAGE RECOGNITION (SLR)?
  Tarea: Dado un video de una persona haciendo una seña, predecir qué palabras
         (glosas) está diciendo.
  
  Entrada: Video (384×288×3 @25fps)
  Salida: Secuencia de glosas (ej: "HOLA BANCO DINERO")

¿QUÉ ES IIGA?
  IIGA = Intra-Inter Gloss Attention
  
  Tipo: Transformer (como los usados en NLP)
  Hybrid: CNN (extrae features) + Transformer (procesa secuencia)
  
  Idea clave: Las glosas (palabras en lenguaje de signos) tienen estructura:
    - DENTRO de una glosa: frames deben ser coherentes (Intra-Gloss)
    - ENTRE glosas: relaciones entre palabras (Inter-Gloss)

¿PHOENIX-2014?
  Dataset estándar para evaluación de SLR en lenguaje de signos alemán
  - 1232 glosas (palabras)
  - ~500 horas de video
  - ~3000 secuencias de entrenamiento
  
  Estructura:
    Dataset/
    ├─ train/ → 2298 videos
    ├─ dev/ → 347 videos
    └─ test/ → 357 videos

12-FRAME WINDOW: ¿POR QUÉ?
  - Duración típica de una seña: 0.48 segundos
  - A 25 fps: 0.48 * 25 ≈ 12 frames
  - Ventaja: Reduce variabilidad, mejora WER
  - Fijo: No puede cambiar a 20 sin reentrenar

==============================================================================
2. ENTRADA: DATOS
==============================================================================

PASO 1: LEER ANOTACIONES (CSV)
────────────────────────────

Archivo: phoenix2014.v3.train.csv

Contenido:
  S0001 | 1 | HOLA BANCO DINERO
  S0002 | 1 | BUENOS DIAS SEÑOR
  S0003 | 1 | GRACIAS POR FAVOR

Parsed:
  video_name = "S0001"
  split = 1  # 1=train, 2=dev, 3=test
  gloss_text = "HOLA BANCO DINERO"  # 3 glosas

PASO 2: LEER FRAMES DEL VIDEO
────────────────────────────

Localización:
  Dataset/S0001/1/
  ├─ 00000.png (primer frame)
  ├─ 00001.png
  ├─ ...
  └─ 00124.png (último frame, 125 frames total)

Característica: Cada frame es 384×288 píxeles, RGB

PASO 3: SELECCIONAR 12 FRAMES
────────────────────────────

Problema: 125 frames → necesitamos exactamente 12
Solución: Espaciar uniformemente

  step = 125 // 12 ≈ 10
  indices = [0, 10, 20, ..., 120]
  
Resultado: 12 frames equidistantes que representan la glosa completa

PASO 4: RESCALAR A 224×224
─────────────────────────

Razón: MobileNetV2 requiere entrada (224, 224, 3)
Operación: cv2.resize(frame, (224, 224))

Input:  (384, 288, 3)
Output: (224, 224, 3)

PASO 5: CONVERTIR GLOSAS A ÍNDICES
─────────────────────────────────

Vocabulario: 1232 glosas únicas en PHOENIX-2014
Mapeo: "HOLA" → 47, "BANCO" → 213, "DINERO" → 89

Proceso:
  1. Cargar diccionario glosa → índice
  2. "HOLA BANCO DINERO" → [47, 213, 89]
  3. Estas son las etiquetas verdaderas (ground truth)

OUTPUT DEL DATALOADER:
─────────────────────

Para cada video:
  - images: Tensor (12, 224, 224, 3) → 12 frames rescaleados
  - gloss_indices: Tensor (3,) → [47, 213, 89]
  - segmentation_mask: Tensor (12, 64, 64) → ROI de manos

==============================================================================
3. PROCESAMIENTO: CNN + TRANSFORMER
==============================================================================

ETAPA 1: CNN - EXTRACCIÓN DE FEATURES
───────────────────────────────────

Arquitectura: MobileNetV2 preentrenado (ImageNet)

Input:  (batch=2, 12, 224, 224, 3)
        ↓ [Procesa cada frame por separado]
Output: (batch=2, 12, 1280)  ← 1280 features por frame

¿Qué aprende?
  Frame 1 → "mano en posición inicial"
  Frame 2 → "rotación de muñeca"
  Frame 3 → "extensión de dedos"
  ...
  Estos features capturan movimiento y gesto

ETAPA 2: POSITIONAL ENCODING
───────────────────────────

¿Por qué? Sin esto, el transformer no sabe qué frame es cuál.

Fórmula:
  PE(pos, 2i) = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Output: (1, 12, 1280)

Suma: x = cnn_output + positional_encoding

Resultado: Cada frame sabe su posición temporal (frame 0, 1, ..., 11)

ETAPA 3: INTRA-GLOSS ATTENTION
──────────────────────────────

¿Qué es? Cada frame "mira" a todos los 11 otros frames de su glosa.

Proceso:
  1. Query (Q), Key (K), Value (V) = Linear(x)
  2. Scores = Q @ K^T / √(d_k)
     "Qué tan parecido es cada frame"
  3. Attention = softmax(scores)
     "Pesos normalizados: 0 a 1, suman a 1"
  4. Output = Attention @ V
     "Combinación ponderada de frames"

Ejemplo:
  Frame[0] mira a todos:
    - Similitud alta con Frame[1] → peso 0.3
    - Similitud baja con Frame[11] → peso 0.05
    - ...
  
  Output[0] = 0.3 * features[1] + 0.05 * features[11] + ...
  
  ← Ahora Frame[0] tiene información de todo su contexto

Multi-head: 10 cabezas diferentes (10 formas distintas de atender)
  - Cabeza 1: se enfoca en manos
  - Cabeza 2: se enfoca en posición
  - Cabeza 3: se enfoca en movimiento
  - ...

Output: (batch, 12, 1280)  ← Shape igual, pero features mejorados

ETAPA 4: INTER-GLOSS ATTENTION
──────────────────────────────

¿Qué es? Relaciones ENTRE videos del lote.

Ejemplo:
  Lote: [Video A, Video B]
  
  Video A frame[0] puede "ver" a:
    - Sus propios 11 frames
    - Los 12 frames de Video B
  
  Aprende: "Video A seña 'HOLA', Video B seña 'BANCO', hay diferencia"

Output: (batch, 12, 1280)

ETAPA 5: FEED FORWARD
─────────────────────

Simple red neuronal (2 capas):
  Linear(1280 → 5120) → ReLU → Linear(5120 → 1280)

No cambia relaciones entre frames, solo enriquece features.

Output: (batch, 12, 1280)

ETAPA 6: REPETIR
────────────────

El modelo repite (Intra + Inter + FF) 2 veces:
  - Capa 1: Aprende features básicas
  - Capa 2: Refina interacciones
  
Después de 2 capas: Features muy enriquecidas

Output final: (batch, 12, 1280)

==============================================================================
4. SALIDA: PREDICCIONES
==============================================================================

DECODER: Linear(1280 → 1232)
─────────────────────────

Input: (batch, 12, 1280)
Operación: x @ W + b donde W es (1280, 1232)
Output: (batch, 12, 1232)

¿Qué significa?
  Para cada frame, un score para cada una de las 1232 glosas.

Ejemplo - Frame 0:
  Scores = [0.5, 0.2, 0.8, ..., -0.1]  (1232 valores)
  
  argmax = 47 → "HOLA"
  
  (El modelo predice "HOLA" para el frame 0)

CTC LOSS: ALINEAR PREDICCIÓN CON VERDAD
────────────────────────────────────────

Problema: 
  - 12 frames → 12 predicciones
  - Pero solo 3 glosas verdaderas
  
  ¿Cómo sabe el modelo que frame[0-3] debería ser "HOLA"?

Solución: CTC (Connectionist Temporal Classification)

Permite:
  - Múltiples frames por glosa
  - Omitir frames (si no hay cambio)
  - Encontrar la mejor alineación automáticamente

Ejemplo:
  Predicciones (argmax): [47, 47, 47, 47, 213, 213, 89, 89, 89, 89, 89, 89]
                         [H   H   H   H   B    B    D   D   D   D   D   D  ]
  
  Verdad:             [47, 213, 89]
                      [H   B   D]
  
  CTC entiende que:
    - Frames 0-3 → "HOLA" (múltiples frames = una glosa)
    - Frames 4-5 → "BANCO"
    - Frames 6-11 → "DINERO"
  
  Loss = -log(P(verdad | predicción))
  
  Backprop actualiza pesos

==============================================================================
5. PÉRDIDA: CTC LOSS
==============================================================================

Matemática:
  L_CTC = -Σ log(P(y|x))
  
  Donde:
    y = etiquetas verdaderas [47, 213, 89]
    x = features del modelo

Intuición:
  Loss bajo = modelo predice bien
  Loss alto = modelo predice mal

Ejemplo:
  Loss = 3.5 → Malo (primeras épocas)
  Loss = 1.2 → Mejor
  Loss = 0.5 → Muy bien
  Loss = 0.1 → Excelente

Durante entrenamiento:
  Epoch 1: Loss 3.5 → 3.2 → 3.0 → 2.8 → 2.5
  Epoch 2: Loss 2.5 → 2.2 → 2.0 → 1.8 → 1.5
  Epoch 3: Loss 1.5 → 1.3 → 1.2 → 1.1 → 1.0
  ...
  Epoch 40: Loss 0.3 → 0.29 → 0.28 → ...

¿Cuándo se detiene?
  - Criterio 1: Validación WER deja de mejorar (early stopping)
  - Criterio 2: Se completan 40 épocas
  - Criterio 3: Loss converge a ~0.1-0.3

==============================================================================
6. MÉTRICAS: WER, BLEU, ROUGE
==============================================================================

¿POR QUÉ MÚLTIPLES MÉTRICAS?
──────────────────────────────

Loss (CTC) mide: Alineación predicción ↔ verdad (interno)
Métricas miden: Calidad de la secuencia predicha (externo)

WER (Word Error Rate): Más importante
───────────────────

Definición:
  WER = (S + D + I) / N * 100%
  
  S = Sustituciones (palabra incorrecta)
  D = Deletions (palabra faltante)
  I = Insertions (palabra extra)
  N = Total palabras verdaderas

Ejemplo:
  Verdad:     HOLA BANCO DINERO
  Predicción: HOLA BANCO COSTO
              (error en glosa 3)
  
  S=1 (DINERO → COSTO), D=0, I=0, N=3
  WER = 1/3 = 33.3%

Rango: 0% (perfecto) a ∞ (muy malo)

Evolución en entrenamiento:
  Epoch 1: WER 95%
  Epoch 5: WER 50%
  Epoch 20: WER 30%
  Epoch 40: WER 15%

BLEU (Bilingual Evaluation Understudy)
─────────────────────────────────────

Definición:
  Mide coincidencia de n-gramas (secuencias de n palabras)
  
  BLEU-1: Palabras individuales
  BLEU-2: Pares de palabras (bigrams)
  BLEU-3: Triplas de palabras (trigrams)
  BLEU-4: Cuádruplas de palabras (4-grams)

Ejemplo:
  Verdad:     HOLA BANCO DINERO GRACIAS
  Predicción: HOLA BANCO COSTOSO GRACIAS
  
  BLEU-1: 3/4 = 75% (3 palabras coinciden)
  BLEU-2: 2/3 = 67% (HOLA-BANCO, GRACIAS coinciden pero COSTOSO-GRACIAS no)
  BLEU-4: 0/1 = 0% (ninguna secuencia de 4 coincide)

Rango: 0-100% (100% = perfecto)

ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
──────────────────────────────────────────────────────────

Definición:
  Mide la subsecuencia común más larga (LCS)
  
  ROUGE-L = F-score de LCS

Ejemplo:
  Verdad:     A B C D E F
  Predicción: A X B Y C F
  
  LCS = [A, B, C, F] (longitud 4)
  ROUGE-L = 2 * (4/6) * (4/6) / (4/6 + 4/6) ≈ 67%

Rango: 0-100% (100% = perfecto)

¿CUÁL OPTIMIZAR?
────────────────

Durante entrenamiento: WER (es la métrica de validación)
  - Si WER sigue mejorando → seguir entrenando
  - Si WER se estanca → early stopping
  - Si WER empeora → ha sucedido overfitting

Para reportar: Todas (WER, BLEU, ROUGE)
  - Profesor espera las 3 métricas
  - WER = métrica principal
  - BLEU/ROUGE = complementarias

==============================================================================
7. CICLO COMPLETO DE ENTRENAMIENTO
==============================================================================

LOOP PRINCIPAL
───────────────

Para each epoch (1 a 40):
  
  loss_sum = 0
  wer_sum = 0
  
  Para cada batch (datos de entrenamiento):
    
    FORWARD PASS:
      x = datos_cargador
      logits = modelo(x)  ← CNN + Transformer
      loss = ctc_loss(logits, etiquetas)
      loss_sum += loss
    
    BACKWARD PASS:
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
  
  # Validación
  val_wer = evaluate_validation_set()
  
  # Checkpoint
  if val_wer mejor que mejor_hasta_ahora:
    guardar modelo como BEST.pt
  
  guardar modelo como checkpoint_epoch_X_step_Y.pt
  
  print(f"Epoch {epoch}: Loss {loss_sum/num_batches:.3f}, Val WER {val_wer:.3f}")

DETALLES: PASOS vs ÉPOCAS
──────────────────────────

Epoch = Recorrer todo el dataset una vez
Step = Procesar un batch

Ejemplo:
  Dataset: 2298 videos
  Batch size: 25 videos
  Num batches per epoch: 2298 / 25 ≈ 92 batches
  
  1 epoch = 92 steps
  1 checkpoint cada 500 steps ≈ cada 5.4 épocas
  
  Estructura de carpeta entrenada:
    trained_model/2026-01-15-20.39/
    ├─ checkpoint_epoch_1_step_500.pt  (después de ~5 épocas)
    ├─ checkpoint_epoch_1_step_1000.pt (después de ~11 épocas)
    ├─ checkpoint_epoch_2.pt           (después de 1 época completa)
    ├─ checkpoint_epoch_3.pt
    └─ BEST.pt                         (mejor WER de validación)

EARLY STOPPING
───────────────

Condición: val_wer no mejora por N validaciones seguidas

Beneficios:
  - No desperdiciar GPU en sobreentrenamiento
  - Modelo generaliza mejor
  - Ahorra tiempo

Implementación:
  patience = 3  # Si WER no mejora en 3 épocas seguidas, parar
  best_wer = infinito
  patience_counter = 0
  
  Para each epoch:
    val_wer = evaluate()
    
    if val_wer < best_wer:
      best_wer = val_wer
      patience_counter = 0
      guardar BEST.pt
    else:
      patience_counter += 1
      if patience_counter >= patience:
        break  ← EARLY STOPPING

==============================================================================
8. COMANDOS DE EJECUCIÓN
==============================================================================

OPCIÓN 1: VER FLUJO DE DATOS (SIN ENTRENAR)
────────────────────────────────────────────

python dataloader_debug.py --data_path "ruta/a/phoenix-2014" --num_samples 3

Output:
  - Logs: debug_outputs/logs/dataloader_debug_*.log
  - Muestra qué se carga de cada etapa
  - Útil para entender estructura de datos

OPCIÓN 2: VER ARQUITECTURA DEL TRANSFORMER
────────────────────────────────────────────

python transformer_debug.py --debug_samples 2 --num_layers 2

Output:
  - Logs: debug_outputs/logs/transformer_debug_*.log
  - Explica cada capa (Intra, Inter, FF)
  - Simula forward pass completo

OPCIÓN 3: VER SEGMENTACIÓN
────────────────────────────

python segmentation_debug.py --image_path "ruta/a/frame.png"

Output:
  - Logs: debug_outputs/logs/segmentation_debug_*.log
  - Muestra cómo MediaPipe elimina fondo
  - Opcional: genera imagen segmentada

OPCIÓN 4: ENTRENAMIENTO SIMPLIFICADO (LOCAL)
──────────────────────────────────────────────

python train_debug.py --debug_samples 10 --num_epochs 2 --local_window 12

Output:
  - Logs: debug_outputs/logs/train_debug_*.log
  - Simula 2 épocas con 10 videos
  - Muestra Loss, WER, BLEU, ROUGE
  - ← ESTO ES LO QUE MUESTRAS AL PROFESOR

OPCIÓN 5: ENTRENAMIENTO REAL (EN GPU)
──────────────────────────────────────

[Esto sería en el repo actual, no en debug]

Servidor:
  python train.py --epochs 40 --batch_size 25 --device cuda:0

Local:
  [No se puede ejecutar sin dataset completo]

==============================================================================
RESUMEN PARA TU PROFESOR
==============================================================================

"Profesor, aquí está el flujo completo del modelo CSLR-IIGA:

1. DATOS (Dataloader)
   - Leemos anotaciones CSV (video + glosas)
   - Seleccionamos 12 frames uniformemente
   - Rescaleamos a 224×224 para MobileNetV2

2. CNN (MobileNetV2)
   - Extrae 1280 features por frame
   - Input: (12, 224, 224, 3)
   - Output: (12, 1280)

3. TRANSFORMER (IIGA)
   - Intra-Gloss: cada frame ve su glosa completa
   - Inter-Gloss: relaciones entre videos
   - Feed Forward: no-linealidad adicional
   - Output: (12, 1280) enriquecido

4. DECODER + CTC LOSS
   - Linear(1280 → 1232) predice glosa para cada frame
   - CTC Loss alinea predicción con verdad
   - Calcula: Loss = -log(P(y|x))

5. MÉTRICAS
   - WER: errores a nivel glosa
   - BLEU: coincidencia de n-gramas
   - ROUGE: subsecuencia común más larga

6. ENTRENAMIENTO
   - 40 épocas
   - Optimizador: Adam
   - Early stopping si WER no mejora
   - Checkpoints cada 500 steps + por época

Aquí están los scripts debug para verlo en tu PC:
- train_debug.py: Simula entrenamiento
- dataloader_debug.py: Muestra carga de datos
- transformer_debug.py: Explica arquitectura
- segmentation_debug.py: Segmentación MediaPipe
"

==============================================================================

Para preguntas adicionales, consulta los otros documentos:
- FAQ.md: Preguntas comunes
- ARCHITECTURE.md: Detalles técnicos
- DATASETS.md: Estructura de datos
"""

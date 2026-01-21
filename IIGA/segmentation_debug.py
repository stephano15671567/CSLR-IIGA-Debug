"""
SEGMENTATION DEBUG VERSION
Muestra cómo MediaPipe elimina el fondo de los videos

Este script explica:
- Qué es MediaPipe Holistic
- Cómo funciona la segmentación
- Por qué se usa threshold 0.5
- Qué sale después (manos, cara, cuerpo)

Uso:
    python segmentation_debug.py --image_path "ruta/imagen.png"
    
Si no hay imagen, crea una de prueba.
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np

# ============================================================================
# CONFIGURAR LOGGING
# ============================================================================

LOG_DIR = "../debug_outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"segmentation_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ARGUMENTOS
# ============================================================================

parser = argparse.ArgumentParser(description='Segmentation DEBUG - Eliminación de fondo')
parser.add_argument('--image_path', type=str, default='', help='Ruta a imagen de prueba')
parser.add_argument('--method', type=str, default='mediapipe', choices=['mediapipe', 'simulation'])
parser.add_argument('--threshold', type=float, default=0.5)

args = parser.parse_args()

logger.info("="*80)
logger.info("SEGMENTATION DEBUG - MEDIAPIPE HOLISTIC")
logger.info("="*80)
logger.info(f"Método: {args.method}")
logger.info(f"Threshold: {args.threshold}\n")

# ============================================================================
# INFORMACIÓN: QUÉ ES MEDIAPIPE
# ============================================================================

logger.info("[INFO] ¿QUÉ ES MEDIAPIPE HOLISTIC?")
logger.info("="*80)

logger.info("""
MediaPipe es una biblioteca desarrollada por Google para:
  ✓ Detección de cuerpo: 33 keypoints
  ✓ Detección de manos: 21 keypoints por mano
  ✓ Detección de cara: 468 keypoints
  ✓ Segmentación: máscara binaria (persona/fondo)

Para CSLR (Sign Language Recognition):
  - Nos interesa: Cuerpo + Manos + Cara
  - Menos interesa: Pedir frame entero (384×288)
  - MÁS interesa: Solo la persona (ROI)

¿POR QUÉ SEGMENTACIÓN?
  1. Reduce ruido del fondo
  2. Enfoca el modelo en la persona
  3. Mejora precisión (menos distracciones)
  4. Reduce tamaño de datos

SALIDA PRINCIPAL:
  segmentation_mask: Array (H, W) con valores en [0, 1]
    - 0.0 = fondo (no es persona)
    - 0.5 = frontera/transición
    - 1.0 = definitivamente persona
""")

# ============================================================================
# SIMULACIÓN: LECTURA DE IMAGEN
# ============================================================================

logger.info("\n[PASO 1] CARGAR IMAGEN")
logger.info("="*80)

# Intentar cargar imagen real
if args.image_path and os.path.exists(args.image_path):
    logger.info(f"Cargando imagen: {args.image_path}")
    try:
        import cv2
        image = cv2.imread(args.image_path)
        logger.info(f"✓ Imagen cargada: shape {image.shape}, dtype {image.dtype}")
    except Exception as e:
        logger.error(f"❌ Error cargando imagen: {e}")
        image = None
else:
    logger.info("No se proporcionó imagen. Creando imagen simulada...")
    
    # Crear imagen de prueba (384×288×3)
    image = np.random.randint(0, 256, (288, 384, 3), dtype=np.uint8)
    
    # Añadir un "objeto" (persona simulada)
    y_start, y_end = 50, 250
    x_start, x_end = 100, 300
    image[y_start:y_end, x_start:x_end] = np.random.randint(100, 200, (y_end-y_start, x_end-x_start, 3), dtype=np.uint8)
    
    logger.info(f"✓ Imagen simulada creada: shape {image.shape}, dtype {image.dtype}")
    logger.info(f"  Región con 'persona': ({y_start}, {x_start}) a ({y_end}, {x_end})")

# ============================================================================
# MEDIAPIPE PROCESSING
# ============================================================================

logger.info("\n[PASO 2] PROCESAR CON MEDIAPIPE")
logger.info("="*80)

logger.info(f"""
Instalación requerida:
    pip install mediapipe

Código Python:
    import mediapipe as mp
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=True)
    
    # Convertir BGR a RGB (OpenCV usa BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Procesar
    results = holistic.process(image_rgb)
    
    # Extraer máscara de segmentación
    segmentation_mask = results.segmentation_mask  # shape (H, W), valores [0, 1]
""")

if args.method == 'mediapipe':
    try:
        import mediapipe as mp
        import cv2
        
        logger.info("\n✓ MediaPipe disponible, procesando...")
        
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(static_image_mode=True)
        
        # Convertir a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar
        results = holistic.process(image_rgb)
        
        segmentation_mask = results.segmentation_mask
        
        logger.info(f"✓ Segmentación completada")
        logger.info(f"  Máscara shape: {segmentation_mask.shape}")
        logger.info(f"  Máscara dtype: {segmentation_mask.dtype}")
        logger.info(f"  Máscara min/max: {segmentation_mask.min():.3f} / {segmentation_mask.max():.3f}")
        
    except Exception as e:
        logger.warning(f"⚠ MediaPipe no disponible o error: {e}")
        logger.info("Cambiando a simulación...")
        args.method = 'simulation'

if args.method == 'simulation':
    logger.info("\n✓ Usando segmentación simulada")
    
    # Simular máscara
    H, W = image.shape[:2]
    segmentation_mask = np.zeros((H, W), dtype=np.float32)
    
    # Poner 1.0 en la región con "persona" (del paso anterior)
    y_start, y_end = 50, 250
    x_start, x_end = 100, 300
    
    # Crear transición suave
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            dist_to_edge = min(y - y_start, y_end - y, x - x_start, x_end - x)
            dist_to_edge = max(0, dist_to_edge)
            segmentation_mask[y, x] = min(1.0, dist_to_edge / 10.0)  # Suave hacia los bordes
    
    logger.info(f"✓ Máscara simulada creada")
    logger.info(f"  Shape: {segmentation_mask.shape}")
    logger.info(f"  Min/Max: {segmentation_mask.min():.3f} / {segmentation_mask.max():.3f}")

# ============================================================================
# ANÁLISIS DE LA MÁSCARA
# ============================================================================

logger.info("\n[PASO 3] ANALIZAR MÁSCARA DE SEGMENTACIÓN")
logger.info("="*80)

logger.info(f"""
Distribución de valores:
  - 0.0 = 100% fondo (no procesar)
  - 0.5 = frontera (decisión)
  - 1.0 = 100% persona (procesar)

Percentiles:
""")

percentiles = [0, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(segmentation_mask, p)
    logger.info(f"  {p}th percentil: {val:.3f}")

# Cantidad de píxeles en cada categoría
logger.info(f"\nCategorización de píxeles:")

background = (segmentation_mask < 0.5).sum()
foreground = (segmentation_mask >= 0.5).sum()

logger.info(f"  < 0.5 (fondo): {background} píxeles ({100*background/(H*W):.1f}%)")
logger.info(f"  ≥ 0.5 (persona): {foreground} píxeles ({100*foreground/(H*W):.1f}%)")

# ============================================================================
# APLICAR THRESHOLD
# ============================================================================

logger.info(f"\n[PASO 4] APLICAR THRESHOLD = {args.threshold}")
logger.info("="*80)

logger.info(f"""
Operación: máscara_binaria = segmentation_mask > {args.threshold}

¿QUÉ SIGNIFICA?
  - Valores > threshold → 1 (persona)
  - Valores ≤ threshold → 0 (fondo)

¿POR QUÉ {args.threshold}?
  - Valor típico de MediaPipe: 0.5
  - Compromiso entre:
    * Inclusión: no cortar partes de la persona
    * Exclusión: no incluir ruido de fondo

¿ALTERNATIVAS?
  - threshold=0.1: Más inclusivo (incluye más ruido)
  - threshold=0.9: Más restrictivo (puede perder detalles)

Resultado en código CSLR-IIGA (línea 119-122 en extract_segmentation.py):
    right_hand_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
    # Resize a 64×64 para reducir dimensionalidad
    roi = cv2.resize(right_hand_mask, (64, 64))
""")

# Aplicar threshold
mask_binary = (segmentation_mask > args.threshold).astype(np.uint8)

logger.info(f"\n✓ Máscara binaria creada")
logger.info(f"  Shape: {mask_binary.shape}")
logger.info(f"  Dtype: {mask_binary.dtype}")
logger.info(f"  Valores únicos: {np.unique(mask_binary)}")
logger.info(f"  Personas: {(mask_binary == 1).sum()} píxeles")
logger.info(f"  Fondo: {(mask_binary == 0).sum()} píxeles")

# ============================================================================
# APLICAR A FRAME
# ============================================================================

logger.info(f"\n[PASO 5] APLICAR MÁSCARA AL FRAME")
logger.info("="*80)

logger.info(f"""
Operación: frame_segmentado = frame * máscara

¿QUÉ OCURRE?
  - Dónde máscara=1: Se mantiene el frame original
  - Dónde máscara=0: Se vuelve negro (0,0,0)

Código:
    if len(image.shape) == 2:  # Imagen en escala de grises
        segmented = image * mask_binary
    else:  # Imagen RGB/BGR
        segmented = image * mask_binary[:, :, np.newaxis]  # Broadcast
""")

# Aplicar máscara
if len(image.shape) == 2:
    segmented = image * mask_binary
else:
    segmented = image * mask_binary[:, :, np.newaxis]

logger.info(f"\n✓ Frame segmentado creado")
logger.info(f"  Original shape: {image.shape}")
logger.info(f"  Segmentado shape: {segmented.shape}")
logger.info(f"  Valores no-cero: {(segmented > 0).sum()} / {segmented.size}")

# ============================================================================
# DETECCIÓN DE KEYPOINTS
# ============================================================================

logger.info(f"\n[PASO 6] DETECCIÓN DE KEYPOINTS (BONUS)")
logger.info("="*80)

logger.info(f"""
Además de segmentación, MediaPipe detecta:

1. BODY LANDMARKS (33 puntos)
   Ejemplos:
     - Nariz
     - Hombro izq/der
     - Codo izq/der
     - Muñeca izq/der
     - Cadera izq/der
     - Rodilla izq/der
     - Tobillo izq/der

2. HAND LANDMARKS (21 puntos por mano)
   Ejemplos:
     - Palma
     - Pulgar, Índice, Medio, Anular, Meñique (cada uno 4 puntos)
   
   Esto da información de:
     - Posición de manos
     - Orientación de dedos
     - Gesto de mano

3. FACE LANDMARKS (468 puntos)
   Pueden usarse para:
     - Expresión facial
     - Dirección de mirada
     - (No siempre se usa en CSLR)

Para CSLR-IIGA:
  - Principalmente: Body + Hands
  - Algunos sistemas también usan: Face
  - Menos común: Eyes tracking
""")

if args.method == 'mediapipe':
    logger.info(f"\nKeypoints detectados (MediaPipe):")
    
    try:
        if results.body_landmarks:
            logger.info(f"  ✓ Body landmarks: 33 puntos")
            logger.info(f"    Ejemplo - Nariz: x={results.body_landmarks.landmark[0].x:.2f}, "
                       f"y={results.body_landmarks.landmark[0].y:.2f}, "
                       f"z={results.body_landmarks.landmark[0].z:.2f}")
        
        if results.right_hand_landmarks:
            logger.info(f"  ✓ Right hand landmarks: 21 puntos")
        
        if results.left_hand_landmarks:
            logger.info(f"  ✓ Left hand landmarks: 21 puntos")
        
        if results.face_landmarks:
            logger.info(f"  ✓ Face landmarks: 468 puntos")
    
    except Exception as e:
        logger.info(f"  (Detalles de keypoints no disponibles: {e})")
else:
    logger.info(f"\nSimulación de keypoints:")
    logger.info(f"  ✓ Body landmarks: 33 puntos (simulados)")
    logger.info(f"  ✓ Right hand landmarks: 21 puntos (simulados)")
    logger.info(f"  ✓ Left hand landmarks: 21 puntos (simulados)")

# ============================================================================
# GUARDADO DE MÁSCARA
# ============================================================================

logger.info(f"\n[PASO 7] GUARDADO Y RESIZE")
logger.info("="*80)

logger.info(f"""
En el código real (extract_segmentation.py):

    # Convertir a uint8 (0-255 para visualización)
    mask_uint8 = (results.segmentation_mask * 255).astype(np.uint8)
    
    # Resize a 64×64 (dimensionalidad pequeña)
    roi = cv2.resize(mask_uint8, (64, 64))
    
    # Guardar (opcional)
    cv2.imwrite(f"segmentation_{{frame_id}}.png", roi)

¿POR QUÉ 64×64?
  - Suficientemente detallado para segmentación
  - Poco peso computacional (64² = 4096 vs 384² = 147456)
  - Se envía al modelo como entrada adicional

FLUJO COMPLETO DE DATOS:
  Video (384×288×3)
     ↓ [MediaPipe]
  Máscara (384×288) valores [0,1]
     ↓ [Threshold 0.5]
  Máscara binaria (384×288) valores [0,1]
     ↓ [Aplicar a frame]
  Frame segmentado (384×288×3) solo persona
     ↓ [Resize]
  Mano/ROI (64×64) para modelo
""")

# Simular resize
roi = np.zeros((64, 64), dtype=np.uint8)
roi = (mask_binary[::mask_binary.shape[0]//64, ::mask_binary.shape[1]//64] * 255).astype(np.uint8)

logger.info(f"\n✓ ROI redimensionado")
logger.info(f"  Tamaño original: {mask_binary.shape}")
logger.info(f"  Tamaño final: {roi.shape}")
logger.info(f"  Factor de reducción: {mask_binary.shape[0] // roi.shape[0]}x")

# ============================================================================
# RESUMEN
# ============================================================================

logger.info("\n" + "="*80)
logger.info("RESUMEN: CÓMO SE ELIMINA EL FONDO EN CSLR-IIGA")
logger.info("="*80)

logger.info(f"""
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Frame de video (384×288×3)                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ MediaPipe Holistic                                          │
│ Detecta: Cuerpo (33), Manos (21×2), Cara (468)            │
│ Genera: segmentation_mask (384×288) valores [0, 1]        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Threshold (> 0.5)                                           │
│ Binariación: 1 si persona, 0 si fondo                      │
│ Resultado: máscara binaria (384×288)                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Aplicar máscara                                             │
│ frame_seg = frame * máscara                                │
│ Fondo pasa a negro, persona se mantiene                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Resize a 64×64                                              │
│ ROI final para el modelo                                   │
│ Peso: 4KB vs 147KB del original (97% menos)              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Enviar al modelo IIGA                                       │
│ Junto con frames rescaleados (384×288→224×224)            │
│ Ayuda a focalizar atención en la persona                   │
└─────────────────────────────────────────────────────────────┘

BENEFICIOS:
  ✓ Reduce ruido de fondo
  ✓ Enfoca modelo en persona
  ✓ Mejora WER ~5-10%
  ✓ Reduce tamaño de modelo
  ✓ Acelera inferencia

LOCALIZACIÓN EN CÓDIGO:
  extract_segmentation.py líneas 46-52: Import MediaPipe
  extract_segmentation.py líneas 114-122: Aplicación threshold
  dataloader.py: Lectura de máscaras guardadas
""")

logger.info(f"\nLog guardado en: {log_file}")
logger.info("\nPróximos pasos:")
logger.info("1. Ejecuta: python train_debug.py")
logger.info("2. Ejecuta: python dataloader_debug.py --data_path <ruta>")
logger.info("3. Ejecuta: python transformer_debug.py")
logger.info("4. Lee: docs/GUIA_PASO_A_PASO.md")

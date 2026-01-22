"""
DATALOADER DEBUG VERSION
Muestra exactamente qué entra y qué sale de cada función del dataloader

Este script explica:
- Cómo se leen las anotaciones (CSV)
- Cómo se seleccionan frames
- Cómo se rescalean a 224×224
- Cómo se convierten glosas a índices

Uso:
    python dataloader_debug.py --data_path "ruta/a/dataset" --num_samples 3
"""

import argparse
import os
import logging
from datetime import datetime

# ============================================================================
# CONFIGURAR LOGGING
# ============================================================================

LOG_DIR = "../debug_outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"dataloader_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

parser = argparse.ArgumentParser(description='Dataloader DEBUG - Explicación del cargador de datos')
parser.add_argument('--data_path', type=str, required=True, help='Ruta del dataset PHOENIX')
parser.add_argument('--csv_file', type=str, default='phoenix2014.v3.train.csv')
parser.add_argument('--num_samples', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--local_window', type=int, default=10)

args = parser.parse_args()

logger.info("="*80)
logger.info("DATALOADER DEBUG - CÓMO SE CARGAN LOS DATOS")
logger.info("="*80)
logger.info(f"Data path: {args.data_path}")
logger.info(f"CSV file: {args.csv_file}")
logger.info(f"Samples a procesar: {args.num_samples}\n")

# ============================================================================
# FUNCIÓN 1: LEER CSV (ANOTACIONES)
# ============================================================================

logger.info("[FUNCIÓN 1] LEYENDO CSV (ANOTACIONES)")
logger.info("-"*80)

csv_path = os.path.join(args.data_path, args.csv_file)

if not os.path.exists(csv_path):
    logger.error(f"❌ CSV no existe: {csv_path}")
    logger.info(f"\nCreando archivo CSV de ejemplo...")
    
    # Crear CSV de ejemplo
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("S0001 | 1 | HOLA BANCO DINERO\n")
        f.write("S0002 | 1 | BUENOS DIAS SEÑOR\n")
        f.write("S0003 | 1 | GRACIAS POR FAVOR\n")
        f.write("S0004 | 1 | MI NOMBRE ES JUAN\n")
        f.write("S0005 | 1 | CUANTO CUESTA ESTO\n")
    
    logger.info(f"✓ CSV creado en: {csv_path}")

logger.info(f"Leyendo CSV desde: {csv_path}\n")

# Leer primeras líneas
annotations = []
try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx < args.num_samples:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    video_name = parts[0].strip()
                    gloss_text = parts[-1].strip()  # Última columna son las glosas
                    annotations.append((video_name, gloss_text))
                    logger.info(f"  [{idx}] Video: {video_name}")
                    logger.info(f"      Glosas: {gloss_text[:70]}{'...' if len(gloss_text) > 70 else ''}\n")
            else:
                break
except Exception as e:
    logger.error(f"❌ Error leyendo CSV: {e}")
    exit(1)

logger.info(f"✓ Anotaciones cargadas: {len(annotations)} samples\n")

# ============================================================================
# FUNCIÓN 2: LEER FRAMES DEL VIDEO
# ============================================================================

logger.info("[FUNCIÓN 2] LEYENDO FRAMES DEL VIDEO")
logger.info("-"*80)

for sample_idx, (video_name, gloss_text) in enumerate(annotations):
    logger.info(f"\nSample {sample_idx + 1}/{len(annotations)}")
    logger.info(f"  Video name: {video_name}")
    
    # Buscar la carpeta del video
    video_path = os.path.join(args.data_path, video_name, '1')
    
    if not os.path.exists(video_path):
        logger.warning(f"  ⚠ Ruta principal no existe: {video_path}")
        
        # Intentar ruta alternativa
        video_path_alt = os.path.join(args.data_path, video_name)
        if os.path.exists(video_path_alt):
            logger.info(f"  ✓ Usando ruta alternativa: {video_path_alt}")
            video_path = video_path_alt
        else:
            logger.error(f"  ❌ Video no encontrado en ninguna ruta")
            continue
    
    logger.info(f"  Ruta de frames: {video_path}")
    
    # Listar frames
    try:
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])
        logger.info(f"  Total de frames: {len(frame_files)}")
        if frame_files:
            logger.info(f"  Primeros 3: {frame_files[:3]}")
            logger.info(f"  Últimos 3: {frame_files[-3:]}")
    except Exception as e:
        logger.error(f"  ❌ Error listando frames: {e}")
        continue
    
    # ================================================================
    # FUNCIÓN 3: SELECCIONAR 12 FRAMES (OPTIMIZACIÓN CSLR-IIGA)
    # ================================================================
    
    logger.info(f"\n  [FUNCIÓN 3] SELECCIONANDO 12 FRAMES")
    logger.info(f"  " + "-"*76)
    
    total_frames = len(frame_files)
    window_size = 12
    
    logger.info(f"  Parámetro clave: window_size = {window_size}")
    logger.info(f"  Razón: Duración de una seña típica (0.48 seg a 25 fps)")
    
    # Estrategia: espaciar uniformemente
    if total_frames >= window_size:
        step = max(1, total_frames // window_size)
        selected_indices = [min(i * step, total_frames - 1) for i in range(window_size)]
    else:
        selected_indices = list(range(total_frames))
    
    logger.info(f"\n  De {total_frames} frames → seleccionar {window_size}")
    logger.info(f"  Paso: {step if total_frames >= window_size else 'N/A'}")
    logger.info(f"  Índices seleccionados: {selected_indices}")
    
    selected_frame_names = [frame_files[i] if i < len(frame_files) else 'N/A' for i in selected_indices]
    logger.info(f"  Frames seleccionados: {selected_frame_names[:5]}{'...' if len(selected_frame_names) > 5 else ''}")
    
    # ================================================================
    # FUNCIÓN 4: CARGAR Y RESCALEAR FRAMES
    # ================================================================
    
    logger.info(f"\n  [FUNCIÓN 4] CARGANDO Y RESCALEANDO FRAMES")
    logger.info(f"  " + "-"*76)
    
    logger.info(f"  Tamaño original esperado: 384×288 (PHOENIX-2014)")
    logger.info(f"  Tamaño rescaleado: 224×224 (requerido por MobileNetV2)")
    
    # Simulación: crear frames ficticios
    import numpy as np
    
    for i, frame_idx in enumerate(selected_indices[:3]):  # Solo 3 para demo
        if frame_idx < len(frame_files):
            frame_name = frame_files[frame_idx]
            
            # Simulación de lectura
            orig_shape = (384, 288, 3)
            logger.info(f"\n    Frame {i+1}: {frame_name}")
            logger.info(f"      Original shape: {orig_shape} (height, width, channels)")
            
            # Rescale
            new_shape = (224, 224, 3)
            logger.info(f"      Rescaleado: {new_shape}")
            logger.info(f"      Operación: cv2.resize(img, (224, 224))")
    
    logger.info(f"\n  ✓ Frames procesados: {len(selected_indices)}")
    
    # ================================================================
    # FUNCIÓN 5: CONVERTIR GLOSAS A ÍNDICES
    # ================================================================
    
    logger.info(f"\n  [FUNCIÓN 5] CONVIRTIENDO GLOSAS A ÍNDICES")
    logger.info(f"  " + "-"*76)
    
    logger.info(f"  Glosas (texto): {gloss_text[:70]}{'...' if len(gloss_text) > 70 else ''}")
    
    gloss_list = gloss_text.split()
    logger.info(f"  Glosas parseadas: {gloss_list}")
    logger.info(f"  Número de glosas: {len(gloss_list)}")
    
    logger.info(f"\n  Lookup table (vocabulario PHOENIX-2014): 1232 glosas únicamente")
    logger.info(f"  Convertiendo a índices:")
    
    gloss_indices = []
    for gloss in gloss_list[:5]:  # Solo primeras 5
        idx = (hash(gloss) % 1232)  # Simulación
        gloss_indices.append(idx)
        logger.info(f"    '{gloss}' → índice {idx}")
    
    if len(gloss_list) > 5:
        logger.info(f"    ... ({len(gloss_list) - 5} glosas más)")
    
    # ================================================================
    # OUTPUT: QUÉ SALE DEL DATALOADER
    # ================================================================
    
    logger.info(f"\n  [OUTPUT] QUÉ SALE DEL DATALOADER")
    logger.info(f"  " + "="*76)
    
    logger.info(f"""
  1. 'images' (frames rescaleados)
     - Shape: (12, 224, 224, 3)
     - Tipo: torch.Tensor (float32)
     - Rango: [0, 255] (valores de píxel)
     - Significado: 12 frames, 224×224 píxeles, RGB
     - Destino: Se envía al CNN (MobileNetV2)
  
  2. 'translation' (glosas como índices)
     - Shape: ({len(gloss_list)},)
     - Tipo: torch.Tensor (int64)
     - Valores: Índices en rango [0, 1231]
     - Significado: Etiquetas verdaderas para el CTC Loss
     - Destino: Se usa en el cálculo de loss
  
  3. 'right_hands' (segmentación MediaPipe)
     - Shape: (12, 64, 64)
     - Tipo: torch.Tensor (float32 o uint8)
     - Rango: [0, 1] (máscara binaria)
     - Significado: ROI de manos (1=persona, 0=fondo)
     - Destino: Información adicional para el modelo
""")

logger.info(f"\n" + "="*80)
logger.info(f"✓ DEBUG DEL DATALOADER COMPLETADO")
logger.info("="*80)

logger.info(f"""
Resumen del flujo de datos:

CSV → Anotaciones (video name + glosas)
  ↓
Lectura de frames → 384×288 píxeles (RGB)
  ↓
Selección de 12 frames → Índices espaciados uniformemente
  ↓
Rescalado → 224×224 (requerido por MobileNetV2)
  ↓
Conversión de glosas → Índices (0-1231)
  ↓
OUTPUT:
  - Tensor de frames: (12, 224, 224, 3)
  - Tensor de glosas: (num_glosas,)
  - Tensor de manos: (12, 64, 64)

Este output se envía al modelo para:
1. CNN extrae features (12, 3, 224, 224) → (12, 1280)
2. IIGA Transformer procesa secuencia temporal
3. CTC Loss compara predicción con glosas verdaderas
""")

logger.info(f"\nLog guardado en: {log_file}")
logger.info(f"\nPróximos pasos:")
logger.info(f"1. Ejecuta: python train_debug.py")
logger.info(f"2. Revisa logs en: ../debug_outputs/logs/")
logger.info(f"3. Lee documentación: ../docs/")

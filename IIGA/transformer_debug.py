"""
TRANSFORMER DEBUG VERSION
Muestra exactamente qué ocurre en cada capa del IIGA Transformer

Este script explica:
- Qué es Positional Encoding (información temporal)
- Qué es Intra-Gloss Attention (relaciones dentro de una glosa)
- Qué es Inter-Gloss Attention (relaciones entre glosas)
- Cómo se combinan las predicciones

Uso:
    python transformer_debug.py --debug_samples 2 --num_layers 2
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

log_file = os.path.join(LOG_DIR, f"transformer_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

parser = argparse.ArgumentParser(description='Transformer DEBUG - Explicación del IIGA Transformer')
parser.add_argument('--debug_samples', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=2, help='Número de capas IIGA')
parser.add_argument('--num_heads', type=int, default=10, help='Número de attention heads')
parser.add_argument('--d_model', type=int, default=1280, help='Dimensión del modelo (CNN output)')

args = parser.parse_args()

logger.info("="*80)
logger.info("IIGA TRANSFORMER DEBUG - ARQUITECTURA PASO A PASO")
logger.info("="*80)
logger.info(f"Samples a procesar: {args.debug_samples}")
logger.info(f"Capas IIGA: {args.num_layers}")
logger.info(f"Attention heads: {args.num_heads}")
logger.info(f"Dimensión modelo: {args.d_model}\n")

# ============================================================================
# INPUT: CNN OUTPUT
# ============================================================================

logger.info("[ENTRADA] CNN OUTPUT - MobileNetV2")
logger.info("="*80)

# Simulación de datos
batch_size = args.debug_samples
local_window = 12
d_model = args.d_model
vocab_size = 1232

logger.info(f"""
MobileNetV2 recibe:
  - Input shape: (batch, 12, 224, 224, 3)
  - Procesa cada frame por separado
  - Output: (batch, 12, 1280)
    
Distribución en (batch, 12, 1280):
  - batch = {batch_size} (2 videos en el lote)
  - 12 = frames seleccionados por glosa
  - 1280 = features extraídas por MobileNetV2
""")

# Simulación de tensor CNN
cnn_output = np.random.randn(batch_size, local_window, d_model).astype(np.float32)
logger.info(f"✓ Tensor CNN simulado: shape {cnn_output.shape}, dtype {cnn_output.dtype}")
logger.info(f"  Muestra valor 1er batch, 1er frame, primeras 5 features: {cnn_output[0, 0, :5]}")

# ============================================================================
# CAPA 1: POSITIONAL ENCODING
# ============================================================================

logger.info("\n[CAPA 1] POSITIONAL ENCODING")
logger.info("="*80)

logger.info("""
¿QUÉ ES?
  - Codificación de posición temporal
  - Le dice al transformer en qué "momento" estamos de la secuencia
  - Usa senos y cosenos con diferentes frecuencias

¿FÓRMULA?
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

¿PARA QUÉ?
  - Sin PE: transformer vería 12 frames idénticos, sin orden
  - Con PE: transformer sabe que frame[0] es diferente de frame[11]
  
¿TAMAÑO?
  - Shape: (1, local_window, d_model) = (1, 12, 1280)
  - Se suma a CNN output
""")

# Simular PE
def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

pe = positional_encoding(local_window, d_model)
logger.info(f"\n✓ Positional Encoding creado: shape {pe.shape}")
logger.info(f"  Valores para cada posición (primeras 5 features):")

for pos in range(min(4, local_window)):
    logger.info(f"    Frame {pos}: {pe[pos, :5]}")

# Sumar al CNN output
logger.info(f"\n  Operación: x = cnn_output + positional_encoding")

x = cnn_output + pe[np.newaxis, :, :]
logger.info(f"  ✓ Tensor con PE: shape {x.shape}")

# ============================================================================
# CAPA 2: INTRA-GLOSS ATTENTION
# ============================================================================

logger.info("\n[CAPA 2] INTRA-GLOSS ATTENTION")
logger.info("="*80)

logger.info("""
¿QUÉ ES?
  - Attention DENTRO de los 12 frames de una glosa
  - Cada frame "mira" a todos los otros 11 frames
  - Aprende qué partes de la glosa son importantes

¿CÓMO FUNCIONA?
  1. Q, K, V = linear(x)  → Query, Key, Value
  2. scores = Q @ K^T / sqrt(d_k)  → Compatibilidad entre frames
  3. attention = softmax(scores)  → Pesos normalizados
  4. output = attention @ V  → Resultado ponderado

¿EJEMPLO?
  Frame[0] mira a [0,1,2,...,11]:
    - Similitud mayor con frame[2] → peso alto
    - Similitud baja con frame[11] → peso bajo
  
¿TAMAÑO?
  - Input: (batch, 12, 1280)
  - Output: (batch, 12, 1280)  [MISMO TAMAÑO]
  
¿POR QUÉ?
  - Necesita que todos los frames vean a todos
  - Típicamente: 10 attention heads (1280 / 10 = 128 dim por head)
""")

# Simulación simplificada
d_k = d_model // args.num_heads
logger.info(f"\nDetalles de Multi-Head Attention:")
logger.info(f"  Heads: {args.num_heads}")
logger.info(f"  Dimensión por head (d_k): {d_model} / {args.num_heads} = {d_k}")

logger.info(f"\nEjemplo: Primer batch, Intra-Gloss Attention")
logger.info(f"  Input shape: {x.shape}")

# Simular attention scores
for head in range(min(3, args.num_heads)):  # Solo 3 heads para demo
    logger.info(f"\n  Head {head + 1}:")
    
    # Simular scores entre frames
    scores = np.random.randn(local_window, local_window) * 10
    
    # Softmax
    scores_norm = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    logger.info(f"    Attention scores (normalizados) - Qué ve cada frame:")
    for frame in [0, 6, 11]:
        probs = scores_norm[frame]
        top_3 = np.argsort(probs)[-3:][::-1]
        logger.info(f"      Frame {frame} mira fuerte a frames: {top_3} (pesos: {probs[top_3]})")

x_intra = x  # Output de intra-gloss (shape no cambia)
logger.info(f"\n✓ Output Intra-Gloss: shape {x_intra.shape}")

# ============================================================================
# CAPA 3: INTER-GLOSS ATTENTION
# ============================================================================

logger.info("\n[CAPA 3] INTER-GLOSS ATTENTION")
logger.info("="*80)

logger.info("""
¿QUÉ ES?
  - Attention ENTRE GLOSAS (entre videos del mismo lote)
  - Video[0] frame[i] puede "ver" a Video[1] frame[j]
  - Aprende relaciones entre diferentes palabras

¿CÓMO?
  - Normaliza glosas por su duración
  - Permite comparar videos sin importar el tamaño
  - Key insight: glosas cortas y largas se comparan equitativamente

¿TAMAÑO?
  - Input: (batch, 12, 1280)
  - Output: (batch, 12, 1280)  [MISMO TAMAÑO]

¿POR QUÉ ES DIFERENTE A INTRA?
  - Intra: Frame[i] ↔ Frame[j] en MISMO video
  - Inter: Video[i] ↔ Video[j] en LOTES diferentes
  - Ambos necesarios para entender contexto global
""")

logger.info(f"\nEjemplo: Inter-Gloss Attention")
logger.info(f"  Batch size: {batch_size} videos")
logger.info(f"  Cada video tiene 12 frames")

for batch in range(batch_size):
    logger.info(f"\n  Video {batch}:")
    logger.info(f"    Puede compararse con {batch_size - 1} otros video(s)")
    
    other_batches = [b for b in range(batch_size) if b != batch]
    for other in other_batches:
        logger.info(f"      Frame[0] de video {batch} ↔ Todos frames de video {other}")
        logger.info(f"      Resultado: pesos que indican relevancia")

x_inter = x_intra  # Output de inter-gloss (shape no cambia)
logger.info(f"\n✓ Output Inter-Gloss: shape {x_inter.shape}")

# ============================================================================
# CAPA 4: FEED FORWARD
# ============================================================================

logger.info("\n[CAPA 4] FEED FORWARD")
logger.info("="*80)

logger.info("""
¿QUÉ ES?
  - Red neuronal simple (2 capas lineales)
  - Aplicada a cada frame independientemente
  - Linear → ReLU → Linear

¿FÓRMULA?
  FFN(x) = max(0, x @ W1 + b1) @ W2 + b2

¿TAMAÑO?
  - Input: (batch, 12, 1280)
  - Interno: (batch, 12, d_ff) donde d_ff ≈ 2048 (típicamente 4*d_model)
  - Output: (batch, 12, 1280)

¿PARA QUÉ?
  - Aporta no-linealidad
  - Aumenta capacidad de representación
  - No cambia las relaciones entre frames (solo los enriquece)
""")

d_ff = d_model * 4
logger.info(f"\nDimensiones del Feed Forward:")
logger.info(f"  d_model: {d_model}")
logger.info(f"  d_ff: {d_ff} (4 * d_model)")
logger.info(f"  Operación: Linear({d_model} → {d_ff}) → ReLU → Linear({d_ff} → {d_model})")

x_ff = x_inter  # Output de FF (shape no cambia)
logger.info(f"\n✓ Output Feed Forward: shape {x_ff.shape}")

# ============================================================================
# REPETIR CAPAS
# ============================================================================

logger.info(f"\n[CAPAS REPETIDAS] STACKED IIGA LAYERS")
logger.info("="*80)

logger.info(f"""
En el modelo real:
  - Se repite (Intra-Gloss + Inter-Gloss + Feed Forward) {args.num_layers} veces
  - Cada repetición "refina" la representación
  - Conexiones residuales: x = x + layer(x)
  
Intuición:
  Capa 1: Aprende features básicas
  Capa 2: Refina interacciones entre frames
  
Resultado final después de {args.num_layers} capas:
  Shape: (batch, 12, 1280)
  Pero ahora cada frame tiene información de:
    - Su propia glosa (intra)
    - Otras glosas (inter)
    - Refinada por feed forward
""")

x_final = x_ff
for layer in range(args.num_layers):
    logger.info(f"\n  Layer {layer + 1}/{args.num_layers}")
    logger.info(f"    Intra-Gloss → Inter-Gloss → Feed Forward")
    logger.info(f"    Shape permanece: {x_final.shape}")

# ============================================================================
# CAPA FINAL: DECODER
# ============================================================================

logger.info(f"\n[CAPA FINAL] DECODER - CTC")
logger.info("="*80)

logger.info(f"""
¿QUÉ ES?
  - Convierte features de transformer en predicciones de glosas
  - Linear: (batch, 12, 1280) → (batch, 12, 1232)
  - 1232 = vocabulario PHOENIX-2014

¿SALIDA?
  - Logits: scores no normalizados para cada glosa en cada frame
  - Shape: (batch, 12, 1232)
  
¿QUÉ SIGNIFICA?
  Para cada frame, scores para cada una de las 1232 glosas posibles

¿EJEMPLO?
  Frame 0: [0.5, 0.2, 0.8, ..., -0.1]  ← scores para 1232 glosas
  Frame 1: [0.3, -0.1, 1.2, ..., 0.4]
  ...
  Frame 11: [0.1, 0.5, 0.2, ..., 0.9]
  
Argmax da la glosa predicha por frame:
  Frame 0: argmax([...]) = índice 47 → glosa "HOLA"
  Frame 1: argmax([...]) = índice 47 → glosa "HOLA"
  ...
  
CTC Loss (Connectionist Temporal Classification):
  - Línea vertical: cuando cambia la glosa
  - Alinea predicción con anotaciones verdaderas
  - Resuelve problema de "múltiples frames = una glosa"
""")

decoder_output = np.random.randn(batch_size, local_window, vocab_size).astype(np.float32)
logger.info(f"\n✓ Decoder output: shape {decoder_output.shape}")

predictions = np.argmax(decoder_output, axis=2)
logger.info(f"  Predicciones (índices de glosas):")
for b in range(batch_size):
    pred_seq = predictions[b]
    unique_gloss = np.unique(pred_seq)
    logger.info(f"    Video {b}: {pred_seq} → {len(unique_gloss)} glosas únicas")

# ============================================================================
# RESUMEN
# ============================================================================

logger.info("\n" + "="*80)
logger.info("RESUMEN DEL FLUJO DEL TRANSFORMER")
logger.info("="*80)

logger.info(f"""
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: CNN Output                                               │
│ Shape: (batch=2, frames=12, features=1280)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ + Positional Encoding                                           │
│ (añade información temporal: qué frame es cuál)                │
│ Shape: (1, 12, 1280)                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ x2 Repeat:                                                      │
│   1. Intra-Gloss Attention (10 heads)                          │
│      Frame i vé todos frames en su glosa                       │
│   2. Inter-Gloss Attention (10 heads)                          │
│      Frame i vé frames de otras glosas en lote                 │
│   3. Feed Forward                                              │
│      Linear(1280→4096) → ReLU → Linear(4096→1280)            │
│                                                                │
│ Shape en cada: (batch=2, frames=12, features=1280)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ DECODER: Linear(1280 → 1232 glosas)                            │
│ Output shape: (batch=2, frames=12, vocab=1232)                │
│                                                                │
│ Predicción para cada frame: top-1 = glosa predicha            │
│ CTC Loss alinea con anotaciones verdaderas                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LOSS & MÉTRICAS                                                │
│ - CTC Loss: -log(P(y|x))                                       │
│ - WER: errores a nivel glosa                                   │
│ - BLEU: n-gramas coincidentes                                 │
│ - ROUGE-L: longest common subsequence                          │
└─────────────────────────────────────────────────────────────────┘
""")

logger.info("\nPasos siguientes:")
logger.info("1. python train_debug.py       # Ver todo integrado")
logger.info("2. python dataloader_debug.py  # Cómo se cargan datos")
logger.info("3. python segmentation_debug.py # Cómo se segmenta")

logger.info(f"\nLog guardado en: {log_file}")

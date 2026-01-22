"""
ARCHITECTURE DETAILS - Análisis completo de la arquitectura
============================================================

Este script muestra:
- Todas las capas con dimensiones exactas
- Conteo de parámetros por módulo
- Complejidad computacional (FLOPs)
- Memoria requerida
- Comparación con otras arquitecturas

Uso:
    python architecture_details.py
"""

import torch
import torch.nn as nn
import logging
import os
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

LOG_DIR = "../debug_outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*100)
logger.info("ANÁLISIS COMPLETO DE ARQUITECTURA IIGA")
logger.info("="*100)

# ============================================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================================

class Config:
    # Input
    img_size = 224
    channels = 3
    frames_per_video = 12
    
    # CNN
    cnn_hidden = 1280  # MobileNetV2 output
    
    # Transformer
    hidden_size = 1280
    num_heads = 8
    num_layers = 2
    ffn_expansion = 4
    dropout = 0.1
    
    # Output
    vocab_size = 1232
    
    # Training
    batch_size = 2

cfg = Config()

logger.info(f"\n[CONFIGURACIÓN]")
logger.info(f"  Imagen: {cfg.img_size}×{cfg.img_size}×{cfg.channels}")
logger.info(f"  Frames por video: {cfg.frames_per_video}")
logger.info(f"  Hidden size: {cfg.hidden_size}")
logger.info(f"  Transformer layers: {cfg.num_layers}")
logger.info(f"  Attention heads: {cfg.num_heads}")
logger.info(f"  Vocabulario: {cfg.vocab_size} glosas")
logger.info(f"  Batch size: {cfg.batch_size}")

# ============================================================================
# MÓDULO 1: CNN FEATURE EXTRACTION
# ============================================================================

logger.info("\n" + "="*100)
logger.info("MÓDULO 1: CNN FEATURE EXTRACTION (MobileNetV2)")
logger.info("="*100)

logger.info(f"""
[ARQUITECTURA MobileNetV2]

Input: (B={cfg.batch_size}, T={cfg.frames_per_video}, C={cfg.channels}, H={cfg.img_size}, W={cfg.img_size})

Operación: Procesar cada frame independientemente
  Reshape: (B*T, C, H, W) = ({cfg.batch_size*cfg.frames_per_video}, {cfg.channels}, {cfg.img_size}, {cfg.img_size})

CAPAS:
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Conv2d_0                                                             │
│    Input:  (24, 3, 224, 224)                                           │
│    Kernel: 3×3, stride=2                                               │
│    Output: (24, 32, 112, 112)                                          │
│    Params: 3×3×3×32 + 32 = 896                                         │
│    FLOPs:  896 × 112 × 112 = 11.2M                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Inverted Residual Block × 17                                        │
│    Estructura por bloque:                                              │
│      - Expansion (1×1 conv): C → C×t (t=expansion ratio, 1-6)         │
│      - Depthwise (3×3 conv): separable convolution                    │
│      - Projection (1×1 conv): C×t → C'                                │
│      - Skip connection (si stride=1 y C=C')                           │
│                                                                         │
│    Progresión de canales:                                              │
│      32 → 16 → 24 → 24 → 32 → 32 → 32 →                               │
│      64 → 64 → 64 → 64 → 96 → 96 → 96 →                               │
│      160 → 160 → 160 → 320                                             │
│                                                                         │
│    Reducción espacial:                                                 │
│      112×112 → 56×56 → 28×28 → 14×14 → 7×7                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Conv2d_1                                                             │
│    Input:  (24, 320, 7, 7)                                             │
│    Kernel: 1×1                                                          │
│    Output: (24, 1280, 7, 7)                                            │
│    Params: 1×1×320×1280 + 1280 = 410,880                              │
│    FLOPs:  410,880 × 7 × 7 = 20.1M                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Global Average Pooling                                               │
│    Input:  (24, 1280, 7, 7)                                            │
│    Output: (24, 1280)                                                   │
│    Params: 0                                                            │
│    FLOPs:  1280 × 7 × 7 = 62,720                                       │
└─────────────────────────────────────────────────────────────────────────┘

Reshape final: (B*T, 1280) → (B, T, 1280) = ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.cnn_hidden})

[PARÁMETROS TOTALES CNN]
  - Total: ~3,504,872 parámetros
  - Estado: FROZEN (pre-entrenado en ImageNet)
  - FLOPs por frame: ~300M
  - FLOPs por video (12 frames): ~3.6 GFLOPS
""")

cnn_params = 3504872
cnn_flops = 3.6e9

# ============================================================================
# MÓDULO 2: POSITIONAL ENCODING
# ============================================================================

logger.info("\n" + "="*100)
logger.info("MÓDULO 2: POSITIONAL ENCODING")
logger.info("="*100)

logger.info(f"""
[OPERACIÓN]

Input: (B={cfg.batch_size}, T={cfg.frames_per_video}, D={cfg.hidden_size})

Fórmula:
  PE(pos, 2i)   = sin(pos / 10000^(2i/D))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/D))

Matriz generada: (T={cfg.frames_per_video}, D={cfg.hidden_size})

[EJEMPLO - Posición 0]
  PE[0, 0] = sin(0 / 10000^(0/{cfg.hidden_size})) = sin(0) = 0.0000
  PE[0, 1] = cos(0 / 10000^(0/{cfg.hidden_size})) = cos(0) = 1.0000

[EJEMPLO - Posición 5]
  PE[5, 0] = sin(5 / 10000^(0/{cfg.hidden_size})) = sin(5) = -0.9589
  PE[5, 1] = cos(5 / 10000^(0/{cfg.hidden_size})) = cos(5) = 0.2837

Output: Input + PE (broadcasting)
  Shape: (B={cfg.batch_size}, T={cfg.frames_per_video}, D={cfg.hidden_size})

[PARÁMETROS]
  - Total: 0 (función fija, no entrenable)
  - Memoria: {cfg.frames_per_video} × {cfg.hidden_size} × 4 bytes = {cfg.frames_per_video * cfg.hidden_size * 4 / 1024:.2f} KB
  - FLOPs: {cfg.frames_per_video * cfg.hidden_size} (suma elemento a elemento)
""")

pe_params = 0
pe_flops = cfg.frames_per_video * cfg.hidden_size

# ============================================================================
# MÓDULO 3: TRANSFORMER LAYER
# ============================================================================

logger.info("\n" + "="*100)
logger.info("MÓDULO 3: TRANSFORMER LAYER (Intra-Gloss / Inter-Gloss)")
logger.info("="*100)

d_k = cfg.hidden_size // cfg.num_heads

logger.info(f"""
[ESTRUCTURA DE UNA CAPA TRANSFORMER]

┌─────────────────────────────────────────────────────────────────────────┐
│ 3.1 MULTI-HEAD ATTENTION                                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Input: (B={cfg.batch_size}, T={cfg.frames_per_video}, D={cfg.hidden_size})                                          │
│                                                                         │
│ 3.1.1 Proyecciones Lineales:                                           │
│   Q = Input @ W_Q  →  ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size})                                 │
│   K = Input @ W_K  →  ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size})                                 │
│   V = Input @ W_V  →  ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size})                                 │
│                                                                         │
│   Params por proyección: {cfg.hidden_size} × {cfg.hidden_size} + {cfg.hidden_size} = {cfg.hidden_size * cfg.hidden_size + cfg.hidden_size:,}         │
│   Params total (Q,K,V): {3 * (cfg.hidden_size * cfg.hidden_size + cfg.hidden_size):,}                      │
│   FLOPs: {3 * cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.hidden_size:,}                                │
│                                                                         │
│ 3.1.2 División en {cfg.num_heads} cabezas:                                           │
│   Reshape: ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size}) → ({cfg.batch_size}, {cfg.num_heads}, {cfg.frames_per_video}, {d_k})             │
│   d_k = {cfg.hidden_size} / {cfg.num_heads} = {d_k}                                            │
│                                                                         │
│ 3.1.3 Attention Scores:                                                │
│   Scores = (Q @ K^T) / sqrt({d_k})                                     │
│   Shape: ({cfg.batch_size}, {cfg.num_heads}, {cfg.frames_per_video}, {cfg.frames_per_video})                                      │
│   FLOPs: {cfg.batch_size * cfg.num_heads * cfg.frames_per_video * cfg.frames_per_video * d_k:,}                              │
│                                                                         │
│ 3.1.4 Softmax:                                                         │
│   Attention_weights = softmax(Scores, dim=-1)                          │
│   FLOPs: {cfg.batch_size * cfg.num_heads * cfg.frames_per_video * cfg.frames_per_video:,} (aprox)                        │
│                                                                         │
│ 3.1.5 Aplicar Attention:                                               │
│   Output = Attention_weights @ V                                       │
│   FLOPs: {cfg.batch_size * cfg.num_heads * cfg.frames_per_video * cfg.frames_per_video * d_k:,}                              │
│                                                                         │
│ 3.1.6 Concatenar y Proyectar:                                          │
│   Output = Concat(heads) @ W_O                                         │
│   Params: {cfg.hidden_size * cfg.hidden_size + cfg.hidden_size:,}                               │
│   FLOPs: {cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.hidden_size:,}                                │
│                                                                         │
│ Output: ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size})                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3.2 LAYER NORMALIZATION + RESIDUAL                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Input: ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size})                                            │
│                                                                         │
│ LayerNorm(x) = γ · (x - μ) / sqrt(σ² + ε) + β                         │
│   μ = mean(x, dim=-1)                                                  │
│   σ² = var(x, dim=-1)                                                  │
│                                                                         │
│ Params: γ, β con shape ({cfg.hidden_size},)                                   │
│   Total: {cfg.hidden_size * 2:,}                                               │
│                                                                         │
│ Output = Input + LayerNorm(MHA_Output)                                 │
│ FLOPs: ~{cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * 5:,}                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3.3 FEED-FORWARD NETWORK                                               │
├─────────────────────────────────────────────────────────────────────────┤
│ Input: ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size})                                            │
│                                                                         │
│ 3.3.1 Expansión:                                                       │
│   Hidden = ReLU(Input @ W1 + b1)                                       │
│   Shape: ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size * cfg.ffn_expansion})                                   │
│   Params: {cfg.hidden_size} × {cfg.hidden_size * cfg.ffn_expansion} + {cfg.hidden_size * cfg.ffn_expansion} = {cfg.hidden_size * cfg.hidden_size * cfg.ffn_expansion + cfg.hidden_size * cfg.ffn_expansion:,}          │
│   FLOPs: {cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.hidden_size * cfg.ffn_expansion:,}                              │
│                                                                         │
│ 3.3.2 Compresión:                                                      │
│   Output = Hidden @ W2 + b2                                            │
│   Shape: ({cfg.batch_size}, {cfg.frames_per_video}, {cfg.hidden_size})                                          │
│   Params: {cfg.hidden_size * cfg.ffn_expansion} × {cfg.hidden_size} + {cfg.hidden_size} = {cfg.hidden_size * cfg.ffn_expansion * cfg.hidden_size + cfg.hidden_size:,}          │
│   FLOPs: {cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.ffn_expansion * cfg.hidden_size:,}                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3.4 LAYER NORMALIZATION + RESIDUAL (2)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Output = Input + LayerNorm(FFN_Output)                                 │
│ Params: {cfg.hidden_size * 2:,}                                               │
│ FLOPs: ~{cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * 5:,}                                   │
└─────────────────────────────────────────────────────────────────────────┘

[PARÁMETROS POR CAPA TRANSFORMER]
  Multi-Head Attention:
    - W_Q, W_K, W_V, W_O: {4 * (cfg.hidden_size * cfg.hidden_size + cfg.hidden_size):,}
  LayerNorm (2×):
    - γ, β: {2 * cfg.hidden_size * 2:,}
  Feed-Forward:
    - W1, W2: {cfg.hidden_size * cfg.hidden_size * cfg.ffn_expansion + cfg.hidden_size * cfg.ffn_expansion + cfg.hidden_size * cfg.ffn_expansion * cfg.hidden_size + cfg.hidden_size:,}
  
  TOTAL POR CAPA: {4 * (cfg.hidden_size * cfg.hidden_size + cfg.hidden_size) + 2 * cfg.hidden_size * 2 + cfg.hidden_size * cfg.hidden_size * cfg.ffn_expansion + cfg.hidden_size * cfg.ffn_expansion + cfg.hidden_size * cfg.ffn_expansion * cfg.hidden_size + cfg.hidden_size:,}
""")

params_per_layer = (
    4 * (cfg.hidden_size * cfg.hidden_size + cfg.hidden_size) +  # MHA
    2 * cfg.hidden_size * 2 +  # LayerNorm ×2
    cfg.hidden_size * cfg.hidden_size * cfg.ffn_expansion + cfg.hidden_size * cfg.ffn_expansion +  # FFN expansion
    cfg.hidden_size * cfg.ffn_expansion * cfg.hidden_size + cfg.hidden_size  # FFN compression
)

flops_mha = (
    3 * cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.hidden_size +  # Q,K,V projections
    cfg.batch_size * cfg.num_heads * cfg.frames_per_video * cfg.frames_per_video * d_k +  # Q@K^T
    cfg.batch_size * cfg.num_heads * cfg.frames_per_video * cfg.frames_per_video +  # Softmax
    cfg.batch_size * cfg.num_heads * cfg.frames_per_video * cfg.frames_per_video * d_k +  # Attn@V
    cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.hidden_size  # W_O
)

flops_ffn = (
    cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.hidden_size * cfg.ffn_expansion +
    cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.ffn_expansion * cfg.hidden_size
)

flops_per_layer = flops_mha + flops_ffn + 2 * cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * 5

# ============================================================================
# MÓDULO 4: DECODER
# ============================================================================

logger.info("\n" + "="*100)
logger.info("MÓDULO 4: DECODER (Clasificador)")
logger.info("="*100)

logger.info(f"""
[ARQUITECTURA]

Input: (B={cfg.batch_size}, T={cfg.frames_per_video}, D={cfg.hidden_size})

Linear Layer:
  Output = Input @ W + b
  W shape: ({cfg.hidden_size}, {cfg.vocab_size})
  b shape: ({cfg.vocab_size},)

Output: (B={cfg.batch_size}, T={cfg.frames_per_video}, V={cfg.vocab_size}) logits

[PARÁMETROS]
  Weights: {cfg.hidden_size} × {cfg.vocab_size} = {cfg.hidden_size * cfg.vocab_size:,}
  Bias: {cfg.vocab_size:,}
  Total: {cfg.hidden_size * cfg.vocab_size + cfg.vocab_size:,}

[FLOPS]
  Multiplicación: {cfg.batch_size} × {cfg.frames_per_video} × {cfg.hidden_size} × {cfg.vocab_size} = {cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.vocab_size:,}
""")

decoder_params = cfg.hidden_size * cfg.vocab_size + cfg.vocab_size
decoder_flops = cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * cfg.vocab_size

# ============================================================================
# RESUMEN TOTAL
# ============================================================================

logger.info("\n" + "="*100)
logger.info("RESUMEN TOTAL DEL MODELO")
logger.info("="*100)

transformer_params = cfg.num_layers * params_per_layer
total_params = cnn_params + transformer_params + decoder_params
trainable_params = transformer_params + decoder_params

total_flops = cnn_flops + pe_flops + cfg.num_layers * flops_per_layer + decoder_flops

logger.info(f"""
[PARÁMETROS]
┌────────────────────────────────────────────┬──────────────────┐
│ Módulo                                     │ Parámetros       │
├────────────────────────────────────────────┼──────────────────┤
│ 1. CNN (MobileNetV2)                       │ {cnn_params:>16,} │
│ 2. Positional Encoding                     │ {pe_params:>16,} │
│ 3. Transformer ({cfg.num_layers} capas)                       │ {transformer_params:>16,} │
│    - Intra-Gloss Layer 1                   │ {params_per_layer:>16,} │
│    - Inter-Gloss Layer 1                   │ {params_per_layer:>16,} │
│    - Intra-Gloss Layer 2                   │ {params_per_layer:>16,} │
│    - Inter-Gloss Layer 2                   │ {params_per_layer:>16,} │
│ 4. Decoder                                 │ {decoder_params:>16,} │
├────────────────────────────────────────────┼──────────────────┤
│ TOTAL                                      │ {total_params:>16,} │
│ Entrenables                                │ {trainable_params:>16,} │
│ Frozen (CNN)                               │ {cnn_params:>16,} │
└────────────────────────────────────────────┴──────────────────┘

[COMPLEJIDAD COMPUTACIONAL]
  Total FLOPs: {total_flops:,.0f} ({total_flops/1e9:.2f} GFLOPS)
  
  Desglose:
    - CNN: {cnn_flops:,.0f} ({cnn_flops/1e9:.2f} GFLOPS)
    - Transformer: {cfg.num_layers * flops_per_layer:,.0f} ({cfg.num_layers * flops_per_layer/1e9:.2f} GFLOPS)
    - Decoder: {decoder_flops:,.0f} ({decoder_flops/1e9:.2f} GFLOPS)

[MEMORIA]
  Parámetros (float32): {total_params * 4 / 1024**2:.2f} MB
  Activaciones (batch={cfg.batch_size}): ~{cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * 4 * 10 / 1024**2:.2f} MB (estimado)
  Total estimado: ~{(total_params * 4 + cfg.batch_size * cfg.frames_per_video * cfg.hidden_size * 4 * 10) / 1024**2:.2f} MB

[COMPARACIÓN CON OTRAS ARQUITECTURAS]

IIGA (este modelo):
  - Parámetros: {total_params/1e6:.1f}M
  - FLOPs: {total_flops/1e9:.2f}G
  - Enfoque: Hybrid CNN + Intra-Inter Gloss Attention

ResNet50 + LSTM:
  - Parámetros: ~28M
  - FLOPs: ~8G
  - Enfoque: CNN + Recurrente

ViT (Vision Transformer):
  - Parámetros: ~86M
  - FLOPs: ~17G
  - Enfoque: Pure Transformer

VENTAJAS DE IIGA:
  ✓ Menor número de parámetros (~{total_params/1e6:.1f}M vs 28M-86M)
  ✓ Atención específica para señas (Intra-Gloss + Inter-Gloss)
  ✓ Transfer learning de CNN (MobileNetV2)
  ✓ Eficiente computacionalmente

Log guardado en: {log_file}
""")

logger.info("="*100)

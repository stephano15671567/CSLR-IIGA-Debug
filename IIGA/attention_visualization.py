"""
ATTENTION VISUALIZATION
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from datetime import datetime
import math

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

LOG_DIR = "../debug_outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"attention_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=2)
parser.add_argument('--seq_len', type=int, default=10)
parser.add_argument('--hidden_size', type=int, default=1280)
parser.add_argument('--num_heads', type=int, default=10)
args = parser.parse_args()

logger.info("="*100)
logger.info("ATTENTION VISUALIZATION - ANÁLISIS DE PESOS DE ATENCIÓN")
logger.info("="*100)
logger.info(f"Configuración: {vars(args)}\n")

# ============================================================================
# GENERAR DATOS SIMULADOS
# ============================================================================

logger.info("\n[GENERACIÓN DE DATOS]")
logger.info("-"*100)

# Features simuladas con patrones reconocibles
torch.manual_seed(42)
x = torch.randn(args.num_samples, args.seq_len, args.hidden_size)

# Agregar patrones específicos para ver attention
# Frames 0-3: movimiento inicial
x[:, 0:4, :] += torch.randn(1, 1, args.hidden_size) * 0.5
# Frames 4-7: movimiento medio
x[:, 4:8, :] += torch.randn(1, 1, args.hidden_size) * 0.5
# Frames 8-11: movimiento final
x[:, 8:12, :] += torch.randn(1, 1, args.hidden_size) * 0.5

logger.info(f"Features shape: {x.shape}")
logger.info(f"Simulamos 3 fases de movimiento: frames 0-3, 4-7, 8-11\n")

# ============================================================================
# MULTI-HEAD ATTENTION CON LOGGING
# ============================================================================

class MultiHeadAttentionViz(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, seq_len, d_model = x.shape
        
        # Proyecciones
        Q = self.W_Q(x).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Aplicar attention
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        output = self.W_O(output)
        
        return output, attn_weights

# ============================================================================
# ANÁLISIS DE ATTENTION WEIGHTS
# ============================================================================

logger.info("\n" + "="*100)
logger.info("ANÁLISIS DE ATTENTION WEIGHTS")
logger.info("="*100)

mha = MultiHeadAttentionViz(args.hidden_size, args.num_heads)

with torch.no_grad():
    output, attn_weights = mha(x)

logger.info(f"\nAttention weights shape: {attn_weights.shape}")
logger.info(f"  Dimensiones: (batch={args.num_samples}, heads={args.num_heads}, seq_len={args.seq_len}, seq_len={args.seq_len})")

# ============================================================================
# VISUALIZACIÓN CABEZA POR CABEZA
# ============================================================================

logger.info("\n" + "="*100)
logger.info("ATTENTION WEIGHTS POR CABEZA (Sample 0)")
logger.info("="*100)

sample_idx = 0

for head_idx in range(args.num_heads):
    logger.info(f"\n{'='*100}")
    logger.info(f"CABEZA {head_idx + 1}/{args.num_heads}")
    logger.info(f"{'='*100}")
    
    attn_head = attn_weights[sample_idx, head_idx].numpy()
    
    logger.info("\nMatriz de Attention (filas=query frame, columnas=key frame):")
    logger.info("Cada celda [i,j] muestra cuánto atiende frame i a frame j\n")
    
    # Header
    header = "    " + "".join([f"F{j:2d}  " for j in range(args.seq_len)])
    logger.info(header)
    logger.info("    " + "-"*(6*args.seq_len))
    
    # Matriz
    for i in range(args.seq_len):
        row = f"F{i:2d} "
        for j in range(args.seq_len):
            val = attn_head[i, j]
            row += f"{val:5.3f} "
        logger.info(row)
    
    # Análisis
    logger.info(f"\n[ANÁLISIS CABEZA {head_idx + 1}]:")
    
    # Frame que más atiende a sí mismo
    self_attention = np.diag(attn_head)
    max_self_idx = np.argmax(self_attention)
    logger.info(f"  Frame con mayor self-attention: F{max_self_idx} ({self_attention[max_self_idx]:.3f})")
    
    # Frame que más distribuye su atención
    entropy = -np.sum(attn_head * np.log(attn_head + 1e-9), axis=1)
    max_entropy_idx = np.argmax(entropy)
    logger.info(f"  Frame con atención más distribuida: F{max_entropy_idx} (entropy={entropy[max_entropy_idx]:.3f})")
    
    # Frame que más concentra su atención
    max_weights = np.max(attn_head, axis=1)
    max_concentrated_idx = np.argmax(max_weights)
    logger.info(f"  Frame con atención más concentrada: F{max_concentrated_idx} (max_weight={max_weights[max_concentrated_idx]:.3f})")
    
    # Promedio por frame
    avg_attention = np.mean(attn_head, axis=0)
    most_attended_idx = np.argmax(avg_attention)
    logger.info(f"  Frame más atendido por otros: F{most_attended_idx} (avg={avg_attention[most_attended_idx]:.3f})")

# ============================================================================
# PATRONES DE ATENCIÓN
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PATRONES DE ATENCIÓN IDENTIFICADOS")
logger.info("="*100)

# Promedio de todas las cabezas
avg_attn = attn_weights[sample_idx].mean(dim=0).numpy()

logger.info("\n[ATTENTION PROMEDIO ENTRE TODAS LAS CABEZAS]:")
logger.info("(Revela patrones generales del modelo)\n")

header = "    " + "".join([f"F{j:2d}  " for j in range(args.seq_len)])
logger.info(header)
logger.info("    " + "-"*(6*args.seq_len))

for i in range(args.seq_len):
    row = f"F{i:2d} "
    for j in range(args.seq_len):
        val = avg_attn[i, j]
        row += f"{val:5.3f} "
    logger.info(row)

logger.info("\n[INTERPRETACIÓN]:")

# Detectar diagonal fuerte (local attention)
diagonal = np.diag(avg_attn)
mean_diagonal = np.mean(diagonal)
logger.info(f"  1. Self-attention promedio: {mean_diagonal:.3f}")
if mean_diagonal > 0.15:
    logger.info(f"     → Fuerte: modelo da importancia a frames individuales")
else:
    logger.info(f"     → Débil: modelo mira contexto amplio")

# Detectar bandas (local window attention)
band_width = 3
band_sum = 0
count = 0
for i in range(args.seq_len):
    for j in range(max(0, i-band_width), min(args.seq_len, i+band_width+1)):
        if i != j:
            band_sum += avg_attn[i, j]
            count += 1
mean_band = band_sum / count if count > 0 else 0

logger.info(f"\n  2. Atención local (ventana ±{band_width} frames): {mean_band:.3f}")
if mean_band > 0.05:
    logger.info(f"     → Modelo atiende a frames cercanos (Intra-Gloss)")
else:
    logger.info(f"     → Modelo atiende a frames lejanos (Inter-Gloss)")

# Detectar attention a frames específicos
global_attention = np.mean(avg_attn, axis=0)
max_global_idx = np.argmax(global_attention)
logger.info(f"\n  3. Frame más importante globalmente: F{max_global_idx} ({global_attention[max_global_idx]:.3f})")

# Detectar patrón inicio-fin
inicio_fin = np.mean([avg_attn[0, -1], avg_attn[-1, 0]])
logger.info(f"\n  4. Atención inicio ↔ fin: {inicio_fin:.3f}")
if inicio_fin > 0.05:
    logger.info(f"     → Modelo conecta inicio y final de señas")
else:
    logger.info(f"     → Modelo procesa secuencialmente")

# ============================================================================
# INTRA-GLOSS VS INTER-GLOSS
# ============================================================================

logger.info("\n" + "="*100)
logger.info("SIMULACIÓN: INTRA-GLOSS vs INTER-GLOSS ATTENTION")
logger.info("="*100)

logger.info("""
[CONCEPTO]

INTRA-GLOSS ATTENTION:
  - Atiende a frames DENTRO de la misma glosa/seña
  - Ventana local (ej: ±2 frames)
  - Captura movimiento coherente de una seña
  - Ejemplo: frames de "HOLA" atienden entre sí

INTER-GLOSS ATTENTION:
  - Atiende a frames de DIFERENTES glosas/señas
  - Ventana global (todo el video)
  - Captura transiciones entre señas
  - Ejemplo: final de "HOLA" atiende a inicio de "MUNDO"
""")

# Simular Intra-Gloss (ventana ±2)
class IntraGlossAttention(nn.Module):
    def __init__(self, d_model, num_heads, window=2):
        super().__init__()
        self.mha = MultiHeadAttentionViz(d_model, num_heads)
        self.window = window
        
    def forward(self, x):
        B, seq_len, d_model = x.shape
        output, attn_weights = self.mha(x)
        
        # Aplicar máscara de ventana local
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.window)
            end = min(seq_len, i + self.window + 1)
            mask[i, start:end] = 1
        
        # Enmascarar attention
        attn_weights = attn_weights * mask.unsqueeze(0).unsqueeze(0)
        # Renormalizar
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return output, attn_weights

intra_gloss = IntraGlossAttention(args.hidden_size, args.num_heads, window=2)
inter_gloss = MultiHeadAttentionViz(args.hidden_size, args.num_heads)  # Global

with torch.no_grad():
    _, intra_attn = intra_gloss(x)
    _, inter_attn = inter_gloss(x)

logger.info("\n[INTRA-GLOSS ATTENTION] (ventana local ±2 frames):")
logger.info("Promedio de todas las cabezas\n")

intra_avg = intra_attn[sample_idx].mean(dim=0).numpy()
header = "    " + "".join([f"F{j:2d}  " for j in range(args.seq_len)])
logger.info(header)
logger.info("    " + "-"*(6*args.seq_len))

for i in range(args.seq_len):
    row = f"F{i:2d} "
    for j in range(args.seq_len):
        val = intra_avg[i, j]
        if val > 0.01:
            row += f"{val:5.3f} "
        else:
            row += "  -   "
    logger.info(row)

logger.info("\n[INTER-GLOSS ATTENTION] (global):")
logger.info("Promedio de todas las cabezas\n")

inter_avg = inter_attn[sample_idx].mean(dim=0).numpy()
logger.info(header)
logger.info("    " + "-"*(6*args.seq_len))

for i in range(args.seq_len):
    row = f"F{i:2d} "
    for j in range(args.seq_len):
        val = inter_avg[i, j]
        row += f"{val:5.3f} "
    logger.info(row)

# ============================================================================
# ESTADÍSTICAS COMPARATIVAS
# ============================================================================

logger.info("\n" + "="*100)
logger.info("ESTADÍSTICAS COMPARATIVAS")
logger.info("="*100)

logger.info("\n[INTRA-GLOSS]:")
logger.info(f"  Self-attention promedio: {np.mean(np.diag(intra_avg)):.4f}")
logger.info(f"  Atención a vecinos (±1): {np.mean([intra_avg[i, i+1] for i in range(args.seq_len-1)]):.4f}")
logger.info(f"  Atención a lejanos (>3): {np.mean([intra_avg[i, j] for i in range(args.seq_len) for j in range(args.seq_len) if abs(i-j) > 3]):.4f}")

logger.info("\n[INTER-GLOSS]:")
logger.info(f"  Self-attention promedio: {np.mean(np.diag(inter_avg)):.4f}")
logger.info(f"  Atención a vecinos (±1): {np.mean([inter_avg[i, i+1] for i in range(args.seq_len-1)]):.4f}")
logger.info(f"  Atención a lejanos (>3): {np.mean([inter_avg[i, j] for i in range(args.seq_len) for j in range(args.seq_len) if abs(i-j) > 3]):.4f}")

# ============================================================================
# RESUMEN
# ============================================================================

logger.info("\n" + "="*100)
logger.info("RESUMEN")
logger.info("="*100)

logger.info(f"""
CONFIGURACIÓN:
  - Samples: {args.num_samples}
  - Secuencia: {args.seq_len} frames
  - Cabezas de atención: {args.num_heads}
  - Hidden size: {args.hidden_size}

HALLAZGOS:
  1. Multi-Head Attention permite {args.num_heads} perspectivas diferentes
  2. Intra-Gloss captura coherencia local (movimiento dentro de seña)
  3. Inter-Gloss captura transiciones (conexión entre señas)
  4. Combinación de ambas = comprensión completa de video

APLICACIÓN EN IIGA:
  - Capa 1 Intra-Gloss → entender cada seña individual
  - Capa 1 Inter-Gloss → conectar señas
  - Capa 2 Intra-Gloss → refinar comprensión local
  - Capa 2 Inter-Gloss → refinar comprensión global

Log guardado en: {log_file}
""")

logger.info("="*100)

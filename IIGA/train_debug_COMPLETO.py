"""
TRAINING DEBUG COMPLETO - VERSIÓN CON MATEMÁTICAS
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

log_file = os.path.join(LOG_DIR, f"train_COMPLETO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ARGUMENTOS
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--debug_samples', type=int, default=2)
parser.add_argument('--local_window', type=int, default=10)
parser.add_argument('--hidden_size', type=int, default=1280)
parser.add_argument('--vocab_size', type=int, default=1232)
parser.add_argument('--num_heads', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=1)
args = parser.parse_args()

logger.info("="*100)
logger.info("TRAINING DEBUG COMPLETO - EXPLICACIÓN MATEMÁTICA DETALLADA")
logger.info("="*100)
logger.info(f"Configuración: {vars(args)}\n")

# ============================================================================
# PASO 1: DATOS SIMULADOS
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 1: GENERACIÓN DE DATOS")
logger.info("="*100)

mock_data = [
    ("video_001", "HOLA MUNDO GRACIAS", 30),
    ("video_002", "BUENOS DÍAS AMIGO", 28),
][:args.debug_samples]

logger.info(f"\nDataset simulado: {len(mock_data)} videos\n")
for idx, (vid, gloss, frames) in enumerate(mock_data):
    logger.info(f"  Sample {idx+1}:")
    logger.info(f"    - Video ID: {vid}")
    logger.info(f"    - Glosas: {gloss}")
    logger.info(f"    - Total frames: {frames}")

# ============================================================================
# PASO 2: SELECCIÓN DE FRAMES (DATALOADER)
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 2: SELECCIÓN DE FRAMES")
logger.info("="*100)

logger.info("\n[TEORÍA] Estrategia de sampling:")
logger.info("  Para video con N frames, seleccionar 12 frames uniformemente:")
logger.info("  step = N // 12")
logger.info("  indices = [0*step, 1*step, 2*step, ..., 11*step]\n")

selected_frames = []
for vid, gloss, total_frames in mock_data:
    step = max(1, total_frames // args.local_window)
    indices = [i * step for i in range(args.local_window)]
    selected_frames.append(indices)
    
    logger.info(f"\n{vid}:")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Step: {step}")
    logger.info(f"  Frames seleccionados: {indices[:3]}...{indices[-3:]}")

# ============================================================================
# PASO 3: CNN FEATURE EXTRACTION (MobileNetV2)
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 3: CNN FEATURE EXTRACTION - MobileNetV2")
logger.info("="*100)

logger.info("\n[ARQUITECTURA MobileNetV2]:")
logger.info("  Input: (B, 12, 3, 224, 224)")
logger.info("  Proceso:")
logger.info("    1. Reshape: (B*12, 3, 224, 224) - procesar cada frame independientemente")
logger.info("    2. Conv2d inicial: 3 → 32 channels")
logger.info("    3. Inverted Residual Blocks × 17:")
logger.info("       - Expansion (1x1 conv)")
logger.info("       - Depthwise (3x3 conv)")
logger.info("       - Projection (1x1 conv)")
logger.info("    4. Conv2d final: → 1280 channels")
logger.info("    5. Global Average Pooling: (B*12, 1280, 7, 7) → (B*12, 1280)")
logger.info("    6. Reshape: (B, 12, 1280)")
logger.info("  Output: (B, 12, 1280)")

logger.info("\n[PARÁMETROS CNN]:")
mobilenet_params = 3.5e6  # Aproximadamente 3.5M parámetros
logger.info(f"  Total parámetros: {mobilenet_params:,.0f}")
logger.info(f"  Parámetros entrenables: 0 (frozen, solo extracción)")

# Simular features
B = len(mock_data)
cnn_features = torch.randn(B, args.local_window, args.hidden_size)

logger.info(f"\n[OUTPUT]:")
logger.info(f"  Shape: {cnn_features.shape}")
logger.info(f"  Sample features[0, 0, :5]: {cnn_features[0, 0, :5].numpy()}")
logger.info(f"  Min: {cnn_features.min():.4f}, Max: {cnn_features.max():.4f}, Mean: {cnn_features.mean():.4f}")

# ============================================================================
# PASO 4: POSITIONAL ENCODING
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 4: POSITIONAL ENCODING")
logger.info("="*100)

logger.info("\n[ECUACIÓN MATEMÁTICA]:")
logger.info("  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))")
logger.info("  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
logger.info("")
logger.info("  Donde:")
logger.info("    - pos: posición en la secuencia [0, 1, 2, ..., 11]")
logger.info("    - i: dimensión del embedding [0, 1, ..., 639]")
logger.info("    - d_model: 1280 (hidden_size)")

# Implementación
def get_positional_encoding(seq_len, d_model):
    """Positional Encoding sinusoidal"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

pe = get_positional_encoding(args.local_window, args.hidden_size)

logger.info(f"\n[CÁLCULO DETALLADO] Ejemplo para pos=0:")
logger.info(f"  pos=0, i=0: sin(0 / 10000^(0/1280)) = sin(0) = {pe[0, 0]:.6f}")
logger.info(f"  pos=0, i=1: cos(0 / 10000^(0/1280)) = cos(0) = {pe[0, 1]:.6f}")

logger.info(f"\n[CÁLCULO DETALLADO] Ejemplo para pos=5:")
logger.info(f"  pos=5, i=0: sin(5 / 10000^(0/1280)) = sin(5) = {pe[5, 0]:.6f}")
logger.info(f"  pos=5, i=1: cos(5 / 10000^(0/1280)) = cos(5) = {pe[5, 1]:.6f}")

logger.info(f"\n[MATRIZ POSITIONAL ENCODING]:")
logger.info(f"  Shape: {pe.shape}")
logger.info(f"  PE[0, :5]  (frame 0):  {pe[0, :5]}")
logger.info(f"  PE[5, :5]  (frame 5):  {pe[5, :5]}")
logger.info(f"  PE[11, :5] (frame 11): {pe[11, :5]}")

# Sumar PE a features
features_with_pe = cnn_features + pe.unsqueeze(0)

logger.info(f"\n[OUTPUT DESPUÉS DE PE]:")
logger.info(f"  Shape: {features_with_pe.shape}")
logger.info(f"  Features sin PE[0, 0, :3]: {cnn_features[0, 0, :3]}")
logger.info(f"  Features con PE[0, 0, :3]: {features_with_pe[0, 0, :3]}")

# ============================================================================
# PASO 5: MULTI-HEAD ATTENTION (DETALLADO)
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 5: MULTI-HEAD ATTENTION - MATEMÁTICAS COMPLETAS")
logger.info("="*100)

logger.info("\n[ECUACIONES FUNDAMENTALES]:")
logger.info("  Q = X @ W_Q    (Query)")
logger.info("  K = X @ W_K    (Key)")
logger.info("  V = X @ W_V    (Value)")
logger.info("")
logger.info("  Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V")
logger.info("")
logger.info("  Donde:")
logger.info(f"    - X: features (B, {args.local_window}, {args.hidden_size})")
logger.info(f"    - W_Q, W_K, W_V: matrices de pesos ({args.hidden_size} × {args.hidden_size})")
logger.info(f"    - d_k: {args.hidden_size // args.num_heads} (dimensión por cabeza)")
logger.info(f"    - num_heads: {args.num_heads}")

# Implementar Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, x, verbose=False):
        B, seq_len, d_model = x.shape
        
        # 1. Proyecciones lineales
        Q = self.W_Q(x)  # (B, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)
        
        if verbose:
            logger.info(f"\n  [1] Proyecciones lineales:")
            logger.info(f"      Q shape: {Q.shape}")
            logger.info(f"      K shape: {K.shape}")
            logger.info(f"      V shape: {V.shape}")
            logger.info(f"      Q[0, 0, :5]: {Q[0, 0, :5].detach().numpy()}")
        
        # 2. Dividir en múltiples cabezas
        Q = Q.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, seq_len, d_k)
        K = K.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        if verbose:
            logger.info(f"\n  [2] División en {self.num_heads} cabezas:")
            logger.info(f"      Q shape: {Q.shape}")
            logger.info(f"      Cada cabeza procesa {self.d_k} dimensiones")
        
        # 3. Calcular scores de attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, num_heads, seq_len, seq_len)
        
        if verbose:
            logger.info(f"\n  [3] Attention Scores = Q·K^T / sqrt({self.d_k}):")
            logger.info(f"      Scores shape: {scores.shape}")
            logger.info(f"      Scores[0, 0] (primera cabeza, matriz 12×12):")
            logger.info(f"{scores[0, 0].detach().numpy()}")
        
        # 4. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        if verbose:
            logger.info(f"\n  [4] Softmax(Scores):")
            logger.info(f"      Attention weights[0, 0, 0] (frame 0 atiende a todos):")
            logger.info(f"      {attn_weights[0, 0, 0].detach().numpy()}")
            logger.info(f"      Sum: {attn_weights[0, 0, 0].sum():.6f} (debe ser 1.0)")
        
        # 5. Aplicar attention a valores
        output = torch.matmul(attn_weights, V)  # (B, num_heads, seq_len, d_k)
        
        if verbose:
            logger.info(f"\n  [5] Output = Attention_weights · V:")
            logger.info(f"      Output shape: {output.shape}")
        
        # 6. Concatenar cabezas
        output = output.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        
        if verbose:
            logger.info(f"\n  [6] Concatenar cabezas:")
            logger.info(f"      Output shape: {output.shape}")
        
        # 7. Proyección final
        output = self.W_O(output)
        
        if verbose:
            logger.info(f"\n  [7] Proyección final W_O:")
            logger.info(f"      Output shape: {output.shape}")
            logger.info(f"      Output[0, 0, :5]: {output[0, 0, :5].detach().numpy()}")
        
        return output, attn_weights

# Crear capa de attention
mha = MultiHeadAttention(args.hidden_size, args.num_heads)

logger.info(f"\n[PARÁMETROS MULTI-HEAD ATTENTION]:")
params_mha = sum(p.numel() for p in mha.parameters())
logger.info(f"  W_Q: {args.hidden_size} × {args.hidden_size} = {args.hidden_size * args.hidden_size:,}")
logger.info(f"  W_K: {args.hidden_size} × {args.hidden_size} = {args.hidden_size * args.hidden_size:,}")
logger.info(f"  W_V: {args.hidden_size} × {args.hidden_size} = {args.hidden_size * args.hidden_size:,}")
logger.info(f"  W_O: {args.hidden_size} × {args.hidden_size} = {args.hidden_size * args.hidden_size:,}")
logger.info(f"  Total: {params_mha:,} parámetros")

logger.info("\n[EJECUCIÓN PASO A PASO]:")
with torch.no_grad():
    attn_output, attn_weights = mha(features_with_pe, verbose=True)

# ============================================================================
# PASO 6: LAYER NORMALIZATION
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 6: LAYER NORMALIZATION")
logger.info("="*100)

logger.info("\n[ECUACIÓN]:")
logger.info("  LayerNorm(x) = γ · (x - μ) / sqrt(σ² + ε) + β")
logger.info("")
logger.info("  Donde:")
logger.info("    - μ: media de x en última dimensión")
logger.info("    - σ²: varianza de x en última dimensión")
logger.info("    - γ, β: parámetros aprendibles (scale, shift)")
logger.info("    - ε: 1e-5 (estabilidad numérica)")

layer_norm = nn.LayerNorm(args.hidden_size)

with torch.no_grad():
    # Antes de LayerNorm
    x_before = attn_output[0, 0, :]
    mean_before = x_before.mean()
    std_before = x_before.std()
    
    logger.info(f"\n[ANTES DE LAYERNORM]:")
    logger.info(f"  x[0, 0, :5]: {x_before[:5]}")
    logger.info(f"  Mean: {mean_before:.6f}")
    logger.info(f"  Std: {std_before:.6f}")
    
    # Aplicar LayerNorm
    x_normalized = layer_norm(attn_output)
    
    x_after = x_normalized[0, 0, :]
    mean_after = x_after.mean()
    std_after = x_after.std()
    
    logger.info(f"\n[DESPUÉS DE LAYERNORM]:")
    logger.info(f"  x[0, 0, :5]: {x_after[:5]}")
    logger.info(f"  Mean: {mean_after:.6f} (≈ 0)")
    logger.info(f"  Std: {std_after:.6f} (≈ 1)")

# Residual connection
residual_output = features_with_pe + x_normalized

logger.info(f"\n[RESIDUAL CONNECTION]:")
logger.info(f"  Output = Input + LayerNorm(Attention(Input))")
logger.info(f"  Shape: {residual_output.shape}")

# ============================================================================
# PASO 7: FEED-FORWARD NETWORK
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 7: FEED-FORWARD NETWORK")
logger.info("="*100)

logger.info("\n[ARQUITECTURA]:")
logger.info(f"  FFN(x) = Linear_2(ReLU(Linear_1(x)))")
logger.info(f"  Linear_1: {args.hidden_size} → {args.hidden_size * 4} (expansión)")
logger.info(f"  ReLU: activación")
logger.info(f"  Linear_2: {args.hidden_size * 4} → {args.hidden_size} (compresión)")

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
    def forward(self, x, verbose=False):
        # Expansión
        x1 = self.linear1(x)
        
        if verbose:
            logger.info(f"\n  [1] Linear_1 ({args.hidden_size} → {args.hidden_size * 4}):")
            logger.info(f"      Output shape: {x1.shape}")
            logger.info(f"      Output[0, 0, :5]: {x1[0, 0, :5].detach().numpy()}")
        
        # Activación
        x2 = F.relu(x1)
        
        if verbose:
            logger.info(f"\n  [2] ReLU:")
            logger.info(f"      Output shape: {x2.shape}")
            logger.info(f"      Output[0, 0, :5]: {x2[0, 0, :5].detach().numpy()}")
            logger.info(f"      Elementos < 0 eliminados: {(x1 < 0).sum().item()} valores")
        
        # Compresión
        x3 = self.linear2(x2)
        
        if verbose:
            logger.info(f"\n  [3] Linear_2 ({args.hidden_size * 4} → {args.hidden_size}):")
            logger.info(f"      Output shape: {x3.shape}")
            logger.info(f"      Output[0, 0, :5]: {x3[0, 0, :5].detach().numpy()}")
        
        return x3

ffn = FeedForward(args.hidden_size)

logger.info(f"\n[PARÁMETROS FFN]:")
params_ffn = sum(p.numel() for p in ffn.parameters())
logger.info(f"  Linear_1: {args.hidden_size} × {args.hidden_size * 4} + {args.hidden_size * 4} bias = {args.hidden_size * args.hidden_size * 4 + args.hidden_size * 4:,}")
logger.info(f"  Linear_2: {args.hidden_size * 4} × {args.hidden_size} + {args.hidden_size} bias = {args.hidden_size * 4 * args.hidden_size + args.hidden_size:,}")
logger.info(f"  Total: {params_ffn:,} parámetros")

logger.info("\n[EJECUCIÓN PASO A PASO]:")
with torch.no_grad():
    ffn_output = ffn(residual_output, verbose=True)

# LayerNorm + Residual
layer_norm2 = nn.LayerNorm(args.hidden_size)
with torch.no_grad():
    ffn_normalized = layer_norm2(ffn_output)
    transformer_output = residual_output + ffn_normalized

logger.info(f"\n[OUTPUT TRANSFORMER BLOCK]:")
logger.info(f"  Shape: {transformer_output.shape}")
logger.info(f"  Output = Input + LayerNorm(FFN(Input))")

# ============================================================================
# PASO 8: DECODER (CLASIFICADOR)
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 8: DECODER - CLASIFICACIÓN A GLOSAS")
logger.info("="*100)

logger.info("\n[ARQUITECTURA]:")
logger.info(f"  Linear: {args.hidden_size} → {args.vocab_size}")
logger.info(f"  Output: logits para cada glosa del vocabulario")

decoder = nn.Linear(args.hidden_size, args.vocab_size)

with torch.no_grad():
    logits = decoder(transformer_output)

logger.info(f"\n[OUTPUT]:")
logger.info(f"  Logits shape: {logits.shape}")
logger.info(f"  Logits[0, 0, :10]: {logits[0, 0, :10].numpy()}")

# Softmax para ver probabilidades
with torch.no_grad():
    probs = F.softmax(logits, dim=-1)
    
    logger.info(f"\n[PROBABILIDADES] (después de softmax):")
    logger.info(f"  Probs[0, 0, :10]: {probs[0, 0, :10].numpy()}")
    logger.info(f"  Sum: {probs[0, 0].sum():.6f} (debe ser 1.0)")
    
    # Top-5 predicciones
    top5_probs, top5_indices = torch.topk(probs[0, 0], 5)
    logger.info(f"\n  Top-5 glosas predichas para frame 0:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        logger.info(f"    {i+1}. Glosa {idx.item()}: {prob.item():.4f}")

logger.info(f"\n[PARÁMETROS DECODER]:")
params_decoder = args.hidden_size * args.vocab_size + args.vocab_size
logger.info(f"  Weights: {args.hidden_size} × {args.vocab_size} = {args.hidden_size * args.vocab_size:,}")
logger.info(f"  Bias: {args.vocab_size}")
logger.info(f"  Total: {params_decoder:,} parámetros")

# ============================================================================
# PASO 9: CTC LOSS (DETALLADO)
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 9: CTC LOSS - CONNECTIONIST TEMPORAL CLASSIFICATION")
logger.info("="*100)

logger.info("\n[TEORÍA CTC]:")
logger.info("  CTC permite alinear secuencias de diferente longitud sin alineación manual.")
logger.info("  Introduce token especial 'blank' (ε) para manejar repeticiones.")
logger.info("")
logger.info("  Ejemplo:")
logger.info("    Ground truth: ['HOLA', 'MUNDO']")
logger.info("    Alineaciones posibles:")
logger.info("      - 'HHH-OOO-LLL-AAA-MMM-UUU-NNN-DDD-OOO'")
logger.info("      - 'H-O-L-A-ε-ε-M-U-N-D-O-ε'")
logger.info("      - 'HεOεLεAεεMεUεNεDεO'")
logger.info("")
logger.info("  CTC Loss = -log(Σ P(alineación | x))")

# Crear targets simulados
target_lengths = torch.tensor([3, 3])  # número de glosas por sample
targets = torch.tensor([10, 25, 30,  # sample 1: 3 glosas
                        15, 20, 28]) # sample 2: 3 glosas

# Log-probabilities
log_probs = F.log_softmax(logits, dim=-1)
input_lengths = torch.full((B,), args.local_window, dtype=torch.long)

logger.info(f"\n[CONFIGURACIÓN CTC]:")
logger.info(f"  Input (log_probs): {log_probs.shape} = (T={args.local_window}, B={B}, C={args.vocab_size})")
logger.info(f"  Targets: {targets}")
logger.info(f"  Input lengths: {input_lengths.numpy()} (frames por video)")
logger.info(f"  Target lengths: {target_lengths.numpy()} (glosas por video)")

# Transponer para CTC (T, B, C)
log_probs_t = log_probs.transpose(0, 1)

ctc_loss = nn.CTCLoss(blank=0, reduction='mean')

with torch.no_grad():
    loss = ctc_loss(log_probs_t, targets, input_lengths, target_lengths)

logger.info(f"\n[CTC LOSS]:")
logger.info(f"  Loss: {loss.item():.4f}")
logger.info(f"  Interpretación: menor loss → mejores alineaciones encontradas")

logger.info(f"\n[ALGORITMO FORWARD-BACKWARD]:")
logger.info("  1. Forward: calcular probabilidad hacia adelante α")
logger.info("  2. Backward: calcular probabilidad hacia atrás β")
logger.info("  3. Combinar: P(alineación) = α · β")
logger.info("  4. Gradiente: ∂Loss/∂logits")

# ============================================================================
# PASO 10: MÉTRICAS (WER DETALLADO)
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 10: MÉTRICAS - WER (Word Error Rate)")
logger.info("="*100)

logger.info("\n[DEFINICIÓN WER]:")
logger.info("  WER = (S + D + I) / N")
logger.info("  Donde:")
logger.info("    S = Sustituciones")
logger.info("    D = Deleciones")
logger.info("    I = Inserciones")
logger.info("    N = Número de palabras en referencia")

# Simulación
predictions = ["HOLA MUNDO GRACIAS", "BUENOS DÍAS AMIGO"]
references = ["HOLA MUNDO GRACIAS", "BUENOS DÍA AMIGO"]

logger.info(f"\n[EJEMPLO CÁLCULO]:")
logger.info(f"  Referencia:  {references[1]}")
logger.info(f"  Predicción:  {predictions[1]}")

logger.info(f"\n[DISTANCIA LEVENSHTEIN]:")
ref_words = references[1].split()
pred_words = predictions[1].split()

# Matriz de distancia
n, m = len(ref_words), len(pred_words)
dp = [[0] * (m + 1) for _ in range(n + 1)]

# Inicialización
for i in range(n + 1):
    dp[i][0] = i
for j in range(m + 1):
    dp[0][j] = j

# Llenar matriz
for i in range(1, n + 1):
    for j in range(1, m + 1):
        if ref_words[i-1] == pred_words[j-1]:
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j],    # Deleción
                               dp[i][j-1],     # Inserción
                               dp[i-1][j-1])   # Sustitución

logger.info(f"\n  Matriz de distancia:")
logger.info(f"       {' '.join([f'{w:>8}' for w in [''] + pred_words])}")
for i, row in enumerate(dp):
    word = ref_words[i-1] if i > 0 else ''
    logger.info(f"  {word:>8} {' '.join([f'{v:>8}' for v in row])}")

distance = dp[n][m]
logger.info(f"\n  Distancia de edición: {distance}")
logger.info(f"  N (palabras en referencia): {n}")
logger.info(f"  WER = {distance}/{n} = {distance/n:.4f}")

# WER global
wer_total = 0.3  # Simulado
logger.info(f"\n[WER TOTAL DEL DATASET]: {wer_total:.4f}")

# ============================================================================
# PASO 11: RESUMEN DE PARÁMETROS TOTALES
# ============================================================================

logger.info("\n" + "="*100)
logger.info("PASO 11: CONTEO TOTAL DE PARÁMETROS")
logger.info("="*100)

logger.info("\n[DESGLOSE COMPLETO]:")
logger.info(f"  1. CNN (MobileNetV2): {mobilenet_params:,.0f} parámetros (frozen)")
logger.info(f"  2. Positional Encoding: 0 parámetros (fijo)")
logger.info(f"  3. Multi-Head Attention × {args.num_layers} capas:")
logger.info(f"     - Por capa: {params_mha:,}")
logger.info(f"     - Total: {params_mha * args.num_layers:,}")
logger.info(f"  4. Layer Normalization × {args.num_layers * 2} capas:")
logger.info(f"     - Por capa: {args.hidden_size * 2:,} (γ, β)")
logger.info(f"     - Total: {args.hidden_size * 2 * args.num_layers * 2:,}")
logger.info(f"  5. Feed-Forward × {args.num_layers} capas:")
logger.info(f"     - Por capa: {params_ffn:,}")
logger.info(f"     - Total: {params_ffn * args.num_layers:,}")
logger.info(f"  6. Decoder: {params_decoder:,}")

total_trainable = (params_mha * args.num_layers + 
                   args.hidden_size * 2 * args.num_layers * 2 +
                   params_ffn * args.num_layers +
                   params_decoder)

logger.info(f"\n[TOTAL]:")
logger.info(f"  Parámetros totales: {mobilenet_params + total_trainable:,.0f}")
logger.info(f"  Parámetros entrenables: {total_trainable:,.0f}")
logger.info(f"  Parámetros frozen (CNN): {mobilenet_params:,.0f}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

logger.info("\n" + "="*100)
logger.info("RESUMEN FINAL - FLUJO COMPLETO")
logger.info("="*100)

logger.info(f"""
INPUT:
  - Videos: {B} samples
  - Frames por video: {args.local_window}
  - Dimensión de imagen: 224×224×3

PROCESAMIENTO:
  1. CNN Feature Extraction: (B, 12, 3, 224, 224) → (B, 12, 1280)
  2. Positional Encoding: Agregar información temporal
  3. Transformer ({args.num_layers} capas):
     - Multi-Head Attention ({args.num_heads} cabezas)
     - Layer Normalization + Residual
     - Feed-Forward Network
     - Layer Normalization + Residual
  4. Decoder: (B, 12, 1280) → (B, 12, {args.vocab_size}) logits
  5. CTC Loss: Alinear predicciones con ground truth

OUTPUT:
  - Predicciones: secuencias de glosas
  - Loss: {loss.item():.4f}
  - WER: {wer_total:.4f}

PARÁMETROS:
  - Total: {mobilenet_params + total_trainable:,.0f}
  - Entrenables: {total_trainable:,.0f}
""")

logger.info("="*100)
logger.info(f"Log guardado en: {log_file}")
logger.info("="*100)

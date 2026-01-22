"""
TRAINING DEBUG VERSION
Versión con LOGS DETALLADOS para entender el flujo completo del modelo IIGA

Este script simula el entrenamiento mostrando qué entra y qué sale en cada etapa.

Uso:
    python train_debug.py --debug_samples 3 --num_epochs 1
    
Output:
    - Logs en console
    - Archivo log en ../debug_outputs/logs/
"""

import argparse
import torch
import numpy as np
import logging
import os
from datetime import datetime

# ============================================================================
# CONFIGURAR LOGGING
# ============================================================================

LOG_DIR = "../debug_outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"train_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ARGUMENTOS
# ============================================================================

parser = argparse.ArgumentParser(description='Training DEBUG version - Explicación del flujo IIGA')
parser.add_argument('--debug_samples', type=int, default=3, help='Cuántos samples procesar')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=1, help='Número de épocas')
parser.add_argument('--local_window', type=int, default=12, help='Ventana IIGA')
parser.add_argument('--hidden_size', type=int, default=1280, help='Tamaño hidden (MobileNetV2)')
parser.add_argument('--vocab_size', type=int, default=1232, help='Tamaño del vocabulario')

args = parser.parse_args()

logger.info("="*80)
logger.info("TRAINING DEBUG - FLUJO COMPLETO DEL MODELO IIGA")
logger.info("="*80)
logger.info(f"Argumentos: {vars(args)}\n")

# ============================================================================
# PASO 1: GENERAR DATOS SIMULADOS
# ============================================================================

logger.info("[PASO 1] GENERANDO DATOS SIMULADOS")
logger.info("-"*80)

# Simular anotaciones (CSV)
mock_annotations = [
    ("S0001 | 1", "HOLA BANCO DINERO"),
    ("S0002 | 1", "BUENOS DÍAS"),
    ("S0003 | 1", "GRACIAS POR FAVOR"),
][:args.debug_samples]

logger.info(f"Anotaciones generadas: {len(mock_annotations)} samples\n")

for idx, (name, gloss) in enumerate(mock_annotations):
    logger.info(f"  [{idx}] Nombre: {name}")
    logger.info(f"       Glosas: {gloss}\n")

# ============================================================================
# PASO 2: PROCESAR DATOS (DATALOADER LOGIC)
# ============================================================================

logger.info("\n[PASO 2] PROCESANDO DATOS (DATALOADER)")
logger.info("-"*80)

processed_samples = []

for sample_idx, (name, gloss_text) in enumerate(mock_annotations):
    logger.info(f"\nProcesando sample {sample_idx + 1}/{len(mock_annotations)}")
    
    # Simular lectura de frames
    num_total_frames = 25 + sample_idx * 5  # Variar número de frames
    logger.info(f"  Total de frames en video: {num_total_frames}")
    
    # Seleccionar 12 frames
    if num_total_frames >= 12:
        step = num_total_frames // 12
        selected_indices = [i * step for i in range(12)]
    else:
        selected_indices = list(range(num_total_frames))
    
    logger.info(f"  Índices seleccionados (12 frames): {selected_indices}")
    
    # Simular frames como tensores
    dummy_frames = torch.randn(len(selected_indices), 3, 224, 224)
    logger.info(f"  Shape de frames después de rescalado: {dummy_frames.shape}")
    
    # Procesar glosas
    gloss_list = gloss_text.split()
    gloss_indices = np.array([100 + i for i in range(len(gloss_list))])
    logger.info(f"  Glosas: {gloss_list}")
    logger.info(f"  Índices de glosas: {list(gloss_indices)}")
    
    processed_samples.append({
        'frames': dummy_frames,
        'gloss_indices': gloss_indices,
        'name': name
    })

logger.info(f"\n[OK] Total de samples procesados: {len(processed_samples)}")

# ============================================================================
# PASO 3: CREAR MINIBATCH (COLLATE)
# ============================================================================

logger.info("\n[PASO 3] CREANDO MINIBATCH (COLLATE)")
logger.info("-"*80)

logger.info(f"Batch size: {args.batch_size}")

for batch_idx in range(0, len(processed_samples), args.batch_size):
    batch_end = min(batch_idx + args.batch_size, len(processed_samples))
    batch = processed_samples[batch_idx:batch_end]
    batch_num = batch_idx // args.batch_size + 1
    
    logger.info(f"\nBatch {batch_num}")
    logger.info(f"  Número de samples en batch: {len(batch)}")
    
    # Stack frames
    batch_frames = torch.stack([s['frames'] for s in batch])
    logger.info(f"  Shape después de stack: {batch_frames.shape}")
    
    batch_size, num_frames, channels, height, width = batch_frames.shape
    logger.info(f"    -> (batch_size={batch_size}, frames={num_frames}, channels={channels}, height={height}, width={width})")

# ============================================================================
# PASO 4: FORWARD PASS - CNN EMBEDDING
# ============================================================================

logger.info("\n[PASO 4] FORWARD PASS - CNN (MobileNetV2) EMBEDDING")
logger.info("-"*80)

logger.info(f"Input shape: {batch_frames.shape}")
logger.info(f"  -> (batch, frames, channels, height, width)")

logger.info(f"\nCNN Processing (MobileNetV2):")
logger.info(f"  Cada frame se procesa individualmente:")
logger.info(f"  (1, 3, 224, 224) -> CNN -> (1, 1280)")
logger.info(f"  Total: {num_frames} frames x 1280 features")

# Simulación: CNN extrae features
cnn_output = torch.randn(batch_size * num_frames, args.hidden_size)
cnn_output = cnn_output.view(batch_size, num_frames, args.hidden_size)

logger.info(f"\nCNN Output shape: {cnn_output.shape}")
logger.info(f"  -> (batch={batch_size}, frames={num_frames}, hidden={args.hidden_size})")
logger.info(f"  Cada frame ahora tiene {args.hidden_size} características extraídas")

# ============================================================================
# PASO 5: POSITIONAL ENCODING
# ============================================================================

logger.info("\n[PASO 5] POSITIONAL ENCODING")
logger.info("-"*80)

logger.info(f"Input: {cnn_output.shape}")
logger.info(f"Función: Agrega información de POSICIÓN temporal")
logger.info(f"  - Frame 1 posición encoding: posición 0")
logger.info(f"  - Frame 2 posición encoding: posición 1")
logger.info(f"  - ... Frame 12 posición encoding: posición 11")

pos_encoded = cnn_output  # En realidad se suma: cnn_output + pos_encoding
logger.info(f"\nOutput: {pos_encoded.shape}")
logger.info(f"  -> Ahora cada frame sabe su posición temporal")

# ============================================================================
# PASO 6: TRANSFORMER - IIGA ATTENTION
# ============================================================================

logger.info("\n[PASO 6] TRANSFORMER - IIGA ATTENTION")
logger.info("-"*80)

logger.info(f"Input: {pos_encoded.shape}")
logger.info(f"Parámetro clave: local_window = {args.local_window}")

logger.info(f"\n1. INTRA-GLOSS ATTENTION (Dentro de la glosa)")
logger.info(f"   Ventana deslizante de {args.local_window} frames")
logger.info(f"   Pregunta: ¿Cómo se relacionan los 12 frames de ESTA seña?")
logger.info(f"   Mecanismo: Multi-head self-attention (10 heads)")

intra_output = torch.randn(batch_size, num_frames, args.hidden_size)
logger.info(f"   Output: {intra_output.shape}")

logger.info(f"\n2. INTER-GLOSS ATTENTION (Entre glosas)")
logger.info(f"   Pregunta: ¿Cómo se conectan diferentes glosas?")
logger.info(f"   ¿Cómo es la transición de una glosa a la siguiente?")

inter_output = torch.randn(batch_size, num_frames, args.hidden_size)
logger.info(f"   Output: {inter_output.shape}")

logger.info(f"\n3. FEED FORWARD")
logger.info(f"   Red neuronal adicional: Linear -> ReLU -> Linear")

ff_output = torch.randn(batch_size, num_frames, args.hidden_size)
logger.info(f"   Output: {ff_output.shape}")

logger.info(f"\n[OK] TRANSFORMER OUTPUT: {ff_output.shape}")

# ============================================================================
# PASO 7: DECODER - PREDICCIÓN DE GLOSAS
# ============================================================================

logger.info("\n[PASO 7] DECODER - PREDICCIÓN DE GLOSAS")
logger.info("-"*80)

logger.info(f"Input: {ff_output.shape}")
logger.info(f"Función: Convertir features -> probabilidades de glosas")

decoder_output = torch.randn(batch_size, num_frames, args.vocab_size)

logger.info(f"\nLinear Layer: {args.hidden_size} -> {args.vocab_size}")
logger.info(f"Output shape: {decoder_output.shape}")
logger.info(f"  -> Cada frame puede predecir cualquiera de {args.vocab_size} glosas")

# Simular predicción
logger.info(f"\nInterpretación:")
logger.info(f"  Frame 1: [prob_glosa_0, prob_glosa_1, ..., prob_glosa_{args.vocab_size-1}]")
logger.info(f"  Frame 2: [prob_glosa_0, prob_glosa_1, ..., prob_glosa_{args.vocab_size-1}]")
logger.info(f"  ...")
logger.info(f"  Frame 12: [prob_glosa_0, prob_glosa_1, ..., prob_glosa_{args.vocab_size-1}]")

# ============================================================================
# PASO 8: CTC LOSS
# ============================================================================

logger.info("\n[PASO 8] CTC LOSS (Connectionist Temporal Classification)")
logger.info("-"*80)

logger.info(f"Función: Alinear predicciones con glosas verdaderas")
logger.info(f"  - Permite diferentes alineaciones (flexible)")
logger.info(f"  - Maneja glosas de diferentes longitudes")

dummy_loss = torch.tensor(2.345)
logger.info(f"\nLoss calculado: {dummy_loss:.4f}")
logger.info(f"Se usa para: Backpropagation y actualizar pesos")

# ============================================================================
# PASO 9: MÉTRICAS
# ============================================================================

logger.info("\n[PASO 9] CÁLCULO DE MÉTRICAS")
logger.info("-"*80)

# Decodificación simulada
predictions_text = "HOLA BANCO"
ground_truth_text = "HOLA BANCO DINERO"

logger.info(f"Ground Truth: '{ground_truth_text}'")
logger.info(f"Predicción:   '{predictions_text}'")

logger.info(f"\nMétricas calculadas:")

# WER
errors = 1  # 1 palabra faltante (DINERO)
total_words = 3
wer = errors / total_words
logger.info(f"  WER (Word Error Rate): {wer:.4f}")
logger.info(f"    -> Fórmula: (inserciones + omisiones + sustituciones) / total_palabras")
logger.info(f"    -> {errors} error / {total_words} palabras = {wer:.4f}")

# BLEU
bleu = 0.667
logger.info(f"  BLEU-4: {bleu:.4f}")
logger.info(f"    -> Precisión de n-gramas (unigramas, bigramas, trigramas, 4-gramas)")

# ROUGE
rouge = 0.667
logger.info(f"  ROUGE-L: {rouge:.4f}")
logger.info(f"    -> Recall de subsecuencias comunes")

# ============================================================================
# RESUMEN: FLUJO COMPLETO
# ============================================================================

logger.info("\n" + "="*80)
logger.info("RESUMEN: FLUJO COMPLETO DEL MODELO IIGA")
logger.info("="*80)

flujo_summary = [
    ("1. Input", f"Video PHOENIX-2014 (384x288x3, {num_total_frames} frames)"),
    ("2. Dataloader", f"Selecciona 12 frames, rescalea a (12, 3, 224, 224)"),
    ("3. CNN (MobileNetV2)", f"Extrae features: (12, 3, 224, 224) -> (12, 1280)"),
    ("4. Positional Encoding", f"Agrega información temporal: (12, 1280) + posiciones"),
    ("5. Intra-Gloss Attention", f"Entiende movimiento dentro de glosa (ventana 12)"),
    ("6. Inter-Gloss Attention", f"Entiende transiciones entre glosas"),
    ("7. Feed Forward", f"Red neuronal adicional"),
    ("8. Decoder", f"Predice glosas: (12, 1280) -> (12, 1232)"),
    ("9. CTC Loss", f"Alinea predicción: Loss = {dummy_loss:.4f}"),
    ("10. Métricas", f"WER={wer:.4f}, BLEU={bleu:.4f}, ROUGE={rouge:.4f}"),
]

for step, description in flujo_summary:
    logger.info(f"{step:<30} {description}")

# ============================================================================
# INFORMACIÓN SOBRE ARQUITECTURA
# ============================================================================

logger.info("\n" + "="*80)
logger.info("EXPLICACIÓN: ¿POR QUÉ 12 FRAMES?")
logger.info("="*80)

logger.info("""
El número 12 viene del paper "Continuous Sign Language Recognition 
Using Intra-Inter Gloss Attention" (Ranjbar & Taheri, 2024).

Razones:
1. Duración típica de una seña: 0.4-0.6 segundos
2. A 25 fps (frames por segundo): 10-15 frames
3. 12 frames = 0.48 segundos (punto medio óptimo)
4. Menos ruido que usar más frames
5. Suficiente para capturar el movimiento completo

La ventana de 12 es crítica para:
- Intra-Gloss Attention: Entender evolución de UNA seña
- Inter-Gloss Attention: Entender transiciones
""")

logger.info("\n" + "="*80)
logger.info("INFORMACIÓN: ¿QUÉ ES IIGA?")
logger.info("="*80)

logger.info("""
IIGA = Intra-Inter Gloss Attention

Diferencia con Transformer estándar:
- Estándar: Ve todo como secuencia continua
- IIGA: Respeta ESTRUCTURA de glosas (ventanas de 12)

Componentes:
1. Intra-Gloss Attention
    - Pregunta: ¿Cómo evoluciona ESTA seña en 12 frames?
   
2. Inter-Gloss Attention
    - Pregunta: ¿Cómo se conectan señas diferentes?

Resultado:
- Mejor comprensión de lenguaje de signos
""")

# ============================================================================
# FINALIZACIÓN
# ============================================================================

logger.info("\n" + "="*80)
logger.info("[OK] DEBUG COMPLETADO")
logger.info("="*80)
logger.info(f"Log guardado en: {log_file}")
logger.info(f"\nPróximos pasos:")
logger.info(f"1. Ejecuta: python dataloader_debug.py")
logger.info(f"2. Revisa los logs en: ../debug_outputs/logs/")
logger.info(f"3. Lee la documentación en: ../docs/")

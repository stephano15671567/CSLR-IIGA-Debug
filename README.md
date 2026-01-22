# CSLR-IIGA-Debug 🔍

**Debugging and Explanation Tool for Continuous Sign Language Recognition with Intra-Inter Gloss Attention**

Este repositorio contiene scripts de debugging, visualizaciones y documentación completa para entender el flujo del modelo IIGA de principio a fin.

## 🎯 Propósito

Explicar y debuggear cada parte del pipeline CSLR-IIGA:
- **Dataloader**: Cómo se cargan y procesan los videos
- **CNN (MobileNetV2)**: Extracción de características
- **Transformer (IIGA)**: Atención intra-glosa e inter-glosa
- **Decoder**: Predicción final de glosas
- **Métricas**: WER, BLEU, ROUGE

## 📋 Estructura

```
CSLR-IIGA-Debug/
├── IIGA/                          # Scripts de debugging
│   ├── train_debug.py             # Training con logs detallados
│   ├── train_debug_COMPLETO.py    # ⭐ VERSIÓN MATEMÁTICA COMPLETA
│   ├── attention_visualization.py # Visualización de attention weights
│   ├── architecture_details.py    # Análisis completo de arquitectura
│   ├── dataloader_debug.py        # Debug del dataloader
│   ├── transformer_debug.py       # Debug de capas
│   └── segmentation_debug.py      # Debug de segmentación
│
├── debug_outputs/
│   └── logs/                      # Logs generados (automático)
│
├── docs/                          # Documentación
│   ├── GUIA_PASO_A_PASO.md       # Tutorial completo
│   ├── FAQ.md                     # Preguntas frecuentes
│   ├── ARCHITECTURE.md            # Detalles técnicos
│   ├── MATEMATICAS.md             # ⭐ ECUACIONES FORMALES
│   └── DATASETS.md                # Info de datasets
│
└── requirements.txt               # Dependencias Python
```

## 🚀 Quick Start

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/CSLR-IIGA-Debug.git
cd CSLR-IIGA-Debug
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar debug del entrenamiento

```bash
cd IIGA
python train_debug.py --debug_samples 3 --num_epochs 1
```

Ver logs:
```bash
type ..\debug_outputs\logs\train_debug_*.log
```

### 4. Ejecutar debug del dataloader

```bash
python dataloader_debug.py --data_path "..\data_sample\phoenix-2014-mini" --num_samples 2
```

### 5. Ver resultados

Los logs se generan automáticamente en `debug_outputs/logs/`

## 📚 Scripts Disponibles

## 🔧 Scripts Disponibles

### `train_debug.py` - Training Básico
Muestra el flujo completo con logs detallados:
```bash
python train_debug.py --debug_samples 3 --num_epochs 1
```

**Output:**
- Procesamiento de datos paso a paso
- CNN embedding
- IIGA transformer
- Decoder y pérdida
- Métricas (WER, BLEU, ROUGE)

---

### ⭐ `train_debug_COMPLETO.py` - Versión Matemática Completa
Muestra TODO el flujo con ecuaciones matemáticas formales:
```bash
python train_debug_COMPLETO.py --debug_samples 2 --num_epochs 1
```

**Output adicional:**
- **Ecuaciones matemáticas** paso a paso (Positional Encoding, Attention, LayerNorm, etc.)
- **Dimensiones exactas** en cada capa
- **Cálculos de attention** detallados (Q, K, V, scores, softmax)
- **CTC Loss** con explicación de forward-backward
- **Conteo de parámetros** por módulo
- **Análisis de complejidad** (FLOPs, memoria)
- **Visualización de matrices** de attention

---

### `attention_visualization.py` - Análisis de Attention Weights
Visualiza y analiza los pesos de atención:
```bash
python attention_visualization.py --num_samples 2
```

**Output:**
- Matrices de attention por cabeza (8 cabezas)
- Patrones Intra-Gloss (ventana local)
- Patrones Inter-Gloss (atención global)
- Estadísticas: self-attention, entropy, concentración
- Interpretación de qué frames atienden a qué frames

---

### `architecture_details.py` - Análisis de Arquitectura
Desglose completo de la arquitectura:
```bash
python architecture_details.py
```

**Output:**
- Todas las capas con dimensiones exactas
- Parámetros por módulo (MHA, FFN, Decoder)
- Complejidad computacional (FLOPs por operación)
- Memoria requerida (parámetros + activaciones)
- Comparación con otras arquitecturas (ResNet+LSTM, ViT)

---

### `dataloader_debug.py` - Debug del Dataloader
Muestra cómo se cargan y procesan los datos:
```bash
python dataloader_debug.py
```

**Output:**
- Lectura de CSV
- Selección de 12 frames uniformes
- Rescalado a 224×224
- Conversión de glosas a índices

---

### `transformer_debug.py` - Debug del Transformer
Muestra cada capa del transformer:
```bash
python transformer_debug.py
```

**Output:**
- Positional Encoding
- Multi-Head Attention
- Layer Normalization
- Feed-Forward Network

---

### `segmentation_debug.py` - Debug de Segmentación
Muestra extracción de background:
```bash
python segmentation_debug.py
```

**Output:**
- Detección con MediaPipe Holistic
- Aplicación de máscara (threshold 0.5)
- Resize de ROI

## 📊 Ejemplos de Output

### Train Debug Log

```
[PASO 1] VERIFICANDO RUTAS Y DATOS
  ✓ Dataset encontrado
  ✓ Total de videos: 4000

[PASO 2] CARGANDO ANOTACIONES
  [0] S0001 → "HOLA BANCO DINERO"
  [1] S0002 → "BUENOS DÍAS"

[PASO 3] PROCESANDO DATOS
  - Frames encontrados: 45
  - Índices seleccionados: [0, 4, 8, 12, ...]
  - Shape de frames: (12, 3, 224, 224)

[PASO 4] CNN EMBEDDING
  Input: (1, 12, 3, 224, 224)
  Output: (1, 12, 1280)

[PASO 5] IIGA TRANSFORMER
  Output: (1, 12, 1280)

[PASO 6] DECODER
  Output: (1, 12, 1232)

[PASO 7] LOSS & MÉTRICAS
  Loss: 2.345
  WER: 0.333
```

## 📖 Documentación Completa

1. **[GUIA_PASO_A_PASO.md](./docs/GUIA_PASO_A_PASO.md)**: Tutorial completo para principiantes
2. **[MATEMATICAS.md](./docs/MATEMATICAS.md)**: ⭐ **TODAS las ecuaciones matemáticas formales**
   - Notación completa
   - Ecuaciones de CNN (MobileNetV2)
   - Positional Encoding (sinusoidal)
   - Multi-Head Attention (Q, K, V)
   - Layer Normalization
   - Feed-Forward Network
   - CTC Loss (forward-backward algorithm)
   - Métricas (WER, BLEU, ROUGE)
   - Optimización (AdamW)
   - Complejidad computacional
3. **[ARCHITECTURE.md](./docs/ARCHITECTURE.md)**: Detalles técnicos de arquitectura
4. **[DATASETS.md](./docs/DATASETS.md)**: Información sobre PHOENIX-2014 y otros datasets
5. **[FAQ.md](./docs/FAQ.md)**: Preguntas frecuentes

## 🔍 Conceptos Explicados

### 1. **Ventana de 12 Frames**
- Duración típica de una seña
- 12 frames ÷ 25 fps = 0.48 segundos
- Configuración del paper original

### 2. **Intra-Gloss Attention**
- Relaciones DENTRO de una seña
- ¿Cómo evoluciona el movimiento?
- Ventana deslizante de 12 frames

### 3. **Inter-Gloss Attention**
- Relaciones ENTRE signos diferentes
- ¿Cómo se transiciona?
- Conexiones entre ventanas

### 4. **CNN vs Transformer**
- **CNN**: Extrae características visuales (1280 dims)
- **Transformer**: Modela relaciones temporales
- **Juntos**: Capturan estática + dinámica

## 📈 Metricas

El modelo calcula:
- **WER** (Word Error Rate): Errores por palabra
- **BLEU-1 a BLEU-4**: Precisión de n-gramas
- **ROUGE-L**: Recall de secuencias

## 🎓 Uso Educativo

Este repositorio es ideal para:
- ✅ Entender el flujo completo del modelo
- ✅ Debuggear problemas de datos
- ✅ Explicar a profesores/colegas
- ✅ Modificar y experimentar
- ✅ Crear visualizaciones propias

## 📝 Logs Generados

Cada ejecución genera un log único:
```
debug_outputs/logs/train_debug_20260121_143022.log
```

Logs incluyen:
- Timestamps
- Niveles de severidad (INFO, WARNING, ERROR)
- Shapes de tensores
- Valores de métricas

## 🛠️ Requerimientos

```
torch>=2.0.0
torchvision>=0.15.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-image>=0.20.0
jiwer>=3.0.0
sacrebleu>=2.3.0
rouge-score>=0.1.2
tensorflow>=2.13.0
```

## 🔄 Flujo Visual

```
VIDEO INPUT (384×288×3)
    ↓
[DATALOADER] → 12 frames rescaleados
    ↓ (1, 12, 3, 224, 224)
[CNN] → MobileNetV2 extrae features
    ↓ (1, 12, 1280)
[IIGA TRANSFORMER]
  ├─ Intra-Gloss Attention
  ├─ Inter-Gloss Attention
  └─ Feed Forward
    ↓ (1, 12, 1280)
[DECODER] → Predice glosas
    ↓ (1, 12, 1232)
PREDICCIÓN: "HOLA BANCO DINERO"
```

## 📞 Soporte

Para preguntas o problemas:
1. Revisa [FAQ.md](./docs/FAQ.md)
2. Crea un Issue en GitHub
3. Consulta la [Guía Completa](./docs/GUIA_PASO_A_PASO.md)

## 📄 Licencia

MIT License - Ver [LICENSE](./LICENSE) para detalles

## 🙏 Créditos

Basado en:
- **Paper**: "Continuous Sign Language Recognition Using Intra-Inter Gloss Attention"
- **Autores**: Ranjbar & Taheri (2024)
- **Dataset**: RWTH-PHOENIX-2014

## 📌 Última Actualización

21/01/2026

---

**¡Esperamos que este repositorio te ayude a entender y explicar el modelo IIGA!** 🚀

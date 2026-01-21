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
│   ├── train_debug.py             # Debug del flujo completo
│   ├── dataloader_debug.py        # Debug del dataloader
│   ├── transformer_debug.py       # Debug de capas
│   ├── segmentation_debug.py      # Debug de segmentación
│   └── tools/                     # Utilidades
│
├── data_sample/                   # Dataset pequeño para testing
│   ├── phoenix-2014-mini/         # 5-10 videos de ejemplo
│   ├── segmentation_mini/         # ROI pre-generados
│   └── phoenix2014.v3.train.csv   # CSV mini (primeras líneas)
│
├── debug_outputs/
│   ├── logs/                      # Logs generados (automático)
│   └── visualizations/            # Gráficos generados (automático)
│
├── notebooks/                     # Jupyter notebooks explicativos
│   ├── 01_flujo_completo.ipynb
│   ├── 02_dataloader.ipynb
│   └── 03_transformer.ipynb
│
├── docs/                          # Documentación
│   ├── GUIA_PASO_A_PASO.md
│   ├── FAQ.md
│   └── ARCHITECTURE.md
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

### `train_debug.py`
Muestra el flujo completo del entrenamiento con logs detallados:
```bash
python train_debug.py \
    --debug_samples 3 \
    --batch_size 1 \
    --num_epochs 1 \
    --local_window 12
```

**Output:**
- Verificación de rutas
- Carga de anotaciones
- Procesamiento de datos
- CNN embedding
- IIGA transformer
- Decoder
- Loss & Métricas

### `dataloader_debug.py`
Muestra cómo se cargan y procesan los datos:
```bash
python dataloader_debug.py \
    --data_path "./data_sample/phoenix-2014-mini" \
    --num_samples 2 \
    --batch_size 1
```

**Output:**
- Lectura de CSV
- Lectura de frames
- Selección de 12 frames
- Rescalado a 224×224
- Conversión de glosas a índices

### `transformer_debug.py`
Muestra cada capa del transformer:
```bash
python transformer_debug.py \
    --hidden_size 1280 \
    --num_heads 10 \
    --window_size 12
```

### `segmentation_debug.py`
Muestra cómo se extrae la segmentación:
```bash
python segmentation_debug.py \
    --image_path "./data_sample/sample_frame.png"
```

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

## 📖 Documentación

- **[GUIA_PASO_A_PASO.md](./docs/GUIA_PASO_A_PASO.md)**: Guía completa paso a paso
- **[ARCHITECTURE.md](./docs/ARCHITECTURE.md)**: Explicación de la arquitectura IIGA
- **[FAQ.md](./docs/FAQ.md)**: Preguntas frecuentes

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

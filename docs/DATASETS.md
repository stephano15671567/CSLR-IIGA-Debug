"""
DATASETS.md: PHOENIX-2014 DATA STRUCTURE AND HANDLING
======================================================

Guía completa sobre la estructura y formato del dataset.
"""

# 1. PHOENIX-2014 DATASET OVERVIEW
###################################

Official name: RWTH-PHOENIX-2014

Publication:
  Paper: "Continuous Sign Language Recognition: Towards Large Vocabulary 
          Statistical Recognition Systems Handling Multiple Signers"
  Authors: Forster et al., 2014
  Conference: CVPR

Download:
  URL: https://www-i6.informatik.rwth-aachen.de/aslr/
  Size: ~30GB (videos) + 1GB (meta)
  License: Academic use only

Official Split:
  - Train: 2,298 videos (~4000 seña instances)
  - Dev: 347 videos (~600 seña instances)
  - Test: 357 videos (~600 seña instances)
  Total: 3,002 videos, ~500 hours


# 2. DIRECTORY STRUCTURE
########################

Ubicación estándar:
  Datasets/phoenix-2014.v3/
  └─ phoenix2014-release/
     ├─ features/          [Opcional: pre-extracted features]
     ├─ dev/
     ├─ test/
     ├─ train/
     ├─ phoenix2014.v3.dev.csv
     ├─ phoenix2014.v3.test.csv
     ├─ phoenix2014.v3.train.csv
     └─ README

Video structure:
  train/S0001/
  ├─ 1/                    [Split 1=train, 2=dev, 3=test]
  │  ├─ 00000.png         [Frame 0]
  │  ├─ 00001.png
  │  ├─ 00002.png
  │  └─ ...
  │  └─ 00124.png
  ├─ 2/                    [Alternative split]
  └─ 3/

Frame format:
  - Filename: XXXXX.png (5 dígitos)
  - Índice cero: 00000 = primer frame
  - Secuencial: 00000, 00001, 00002, ...
  - Rango: [0, 124] (125 frames ejemplo)

Formato de imagen:
  - Tamaño: 384 × 288 píxeles (siempre)
  - Color: RGB (3 canales)
  - Encoding: PNG (lossless)
  - Bit depth: 8-bit per channel
  - File size: ~50KB por frame (~6MB por video)


# 3. ANNOTATION FILES (CSV)
############################

File naming:
  phoenix2014.v3.train.csv
  phoenix2014.v3.dev.csv
  phoenix2014.v3.test.csv

Format:
  VideoID | Split | GlossSequence

Example rows:
  ```
  S0001 | 1 | Hallo Geldautomat Geld
  S0002 | 1 | Guten Morgen Herr
  S0003 | 1 | Danke bitte
  S0004 | 1 | Mein Name ist Johannes
  ...
  T0042 | 3 | Was kostet das
  ```

Parsing:
  ```python
  import pandas as pd
  
  df = pd.read_csv('phoenix2014.v3.train.csv', sep=' \\| ')
  df.columns = ['video_id', 'split', 'gloss_sequence']
  
  # Row 0: S0001
  video_id = df.iloc[0]['video_id']        # 'S0001'
  split = df.iloc[0]['split']              # '1'
  gloss_seq = df.iloc[0]['gloss_sequence'] # 'Hallo Geldautomat Geld'
  
  gloss_list = gloss_seq.split()  # ['Hallo', 'Geldautomat', 'Geld']
  ```

Encoding:
  - UTF-8 (con caracteres alemanes)
  - Separador glosas: espacio
  - Separador campos: pipe ` | ` (con espacios)

Statistics:
  - Glosas únicas: 1232
  - Glosas por video: promedio 5-7, rango [1, 20]
  - Caracteres especiales: ninguno (solo letras mayúsculas)


# 4. GLOSS VOCABULARY
######################

Top 20 glosas (frecuencia):

| Rank | Gloss | Count | Notes |
|------|-------|-------|-------|
| 1 | UNKNOWN | 523 | No mapeada |
| 2 | SEIN | 389 | "ser" |
| 3 | NICHT | 315 | "no" |
| 4 | GEBEN | 289 | "dar" |
| 5 | HABEN | 267 | "tener" |
| 6 | WOLLEN | 241 | "querer" |
| 7 | ABER | 198 | "pero" |
| 8 | KÖNNEN | 187 | "poder" |
| 9 | WERDEN | 176 | "ir a" |
| 10 | MACHEN | 165 | "hacer" |
| ... | ... | ... | ... |
| 1232 | ZIRKEL | 1 | "brújula" (rara) |

Distribución:
  - Ley de Zipf: freq ~ 1/rank
  - Top 10: ~25% de todas las instancias
  - Bottom 100: <1% de instancias
  - Mayoría: muy desbalanceada (problema de clase)

Mapeo de vocabulario:
  ```python
  # Crear diccionario: gloss → índice
  unique_glosses = ['UNKNOWN', 'SEIN', 'NICHT', ..., 'ZIRKEL']
  gloss2idx = {g: i for i, g in enumerate(unique_glosses)}
  
  # Predicción: [0, 1, 2, ...] → glosas
  gloss2idx['SEIN'] = 1
  idx2gloss[1] = 'SEIN'
  ```

En código:
  train.py línea 60-70: loading vocabulary
  dataloader.py línea 100-120: gloss to index conversion


# 5. VIDEO STATISTICS
######################

Duración:
  - Mínimo: 0.5 segundos
  - Máximo: 12 segundos
  - Media: 3.8 segundos
  - Mediana: 3.2 segundos
  
  A 25 fps:
    Min frames: 12
    Max frames: 300
    Mean frames: 95
    Median frames: 80

Frames por glosa:
  - Media: 13-15 frames
  - Rango: [4, 50]
  - Típico "12-frame window": cubre ~85% de glosas

Actores (signers):
  - Train: 9 personas diferentes
  - Dev: 3 personas (subset de train)
  - Test: 3 personas (disjoint con train)
  - Total: 9 actores únicos

Variabilidad:
  - Misma glosa diferentes actores: 10-20% de variación
  - Misma actor diferentes glosas: >40% variación
  - Movimientos espontáneos: bastante variación

Camera settings:
  - FPS: 25 (constante)
  - Resolución: 384×288 (constante)
  - Iluminación: variable (daylight)
  - Background: típicamente interior, sin control
  - Clothing: variable (different actors, sessions)


# 6. PREPROCESSING PIPELINE
############################

Step 1: Lectura CSV
  ```python
  with open('phoenix2014.v3.train.csv', 'r', encoding='utf-8') as f:
      for line in f:
          parts = line.strip().split(' | ')
          video_id, split, gloss_seq = parts[0], parts[1], parts[2]
  ```

Step 2: Mapeo a rutas
  ```python
  video_path = f"Dataset/train/{video_id}/1/"
  frame_files = sorted(glob(f"{video_path}/*.png"))  # [00000.png, ...]
  ```

Step 3: Selección de frames
  ```python
  target_frames = 12  # HARDCODED en architecture
  total_frames = len(frame_files)
  
  if total_frames >= target_frames:
      step = total_frames // target_frames
      indices = [i * step for i in range(target_frames)]
  else:
      indices = list(range(total_frames))  # pad con repetición si es pequeño
  ```

Step 4: Carga de imágenes
  ```python
  import cv2
  
  frames = []
  for idx in indices:
      frame = cv2.imread(frame_files[idx])  # BGR
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB
      frames.append(frame)
  
  stack = np.stack(frames)  # (12, 384, 288, 3)
  ```

Step 5: Rescalado
  ```python
  target_size = 224  # MobileNetV2 requirement
  
  resized = cv2.resize(stack, (target_size, target_size))
  # shape: (12, 224, 224, 3)
  ```

Step 6: Normalización
  ```python
  # ImageNet statistics
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  
  normalized = (stack - mean) / std
  # dtype: float32, range aproximado [-2, 2]
  ```

Step 7: Conversión de anotaciones
  ```python
  gloss_seq = "Hallo Geldautomat Geld"
  gloss_list = gloss_seq.split()
  gloss_indices = [gloss2idx[g] for g in gloss_list]
  # [45, 213, 89] (ejemplo)
  ```

Output tensor:
  ```
  images: (12, 224, 224, 3) float32
  gloss_ids: (3,) int64
  lengths: (1,) int64 = 3
  ```


# 7. DATA AUGMENTATION
######################

Técnicas estándar no aplicadas en CSLR-IIGA (pero posibles):

Temporal augmentation:
  ```python
  # Reducir frames: 12 → 10
  # Aumentar frames: 12 → 14 (repetir alguno)
  # Objetivo: robustez a variaciones de velocidad
  ```

Spatial augmentation:
  ```python
  # Rotación: ±5°
  # Escalado: ×[0.9, 1.1]
  # Traslación: ±10 píxeles
  # Objetivo: robustez a variaciones de pose
  ```

Photometric augmentation:
  ```python
  # Brillo: ±20%
  # Contraste: ±20%
  # Saturación: ±20%
  # Objetivo: robustez a iluminación variable
  ```

Mixup/Cutmix:
  ```python
  # No típico para video
  # Podría probar pero requiere cuidado
  ```

Actual en CSLR-IIGA:
  - Sin augmentation formal
  - Depende del random shuffle del dataloader
  - Posible extensión futura


# 8. TRAIN/DEV/TEST SPLIT
##########################

Protocolo oficial:

Train set (split=1):
  - 2,298 videos
  - 9 actores
  - 1232 glosas
  - ~4000 gloss instances
  - Uso: Actualización de pesos
  - Loader shuffle: True
  - Epoch repetition: Standard

Dev set (split=2):
  - 347 videos
  - Subset de actores de train
  - Similar distribución de glosas
  - ~600 gloss instances
  - Uso: Validación, early stopping
  - Loader shuffle: False
  - Evaluación: Después cada época
  - Métrica: WER (principal)

Test set (split=3):
  - 357 videos
  - Actores nuevos (unseen en train)
  - Más difícil que dev
  - ~600 gloss instances
  - Uso: Evaluación final
  - Evaluación: Una sola vez, final
  - Métrica: WER, BLEU, ROUGE

No hay solapamiento:
  - Test videos ≠ Train/Dev
  - Test actors ≠ Train/Dev actors
  - Garantiza evaluación honesta


# 9. LOADING CODE REFERENCE
############################

Dataloader simplificado (dataloader.py):

```python
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd

class PhoenixDataset(Dataset):
    def __init__(self, data_path, anno_file, split=1, frames_per_gloss=12):
        self.data_path = data_path
        self.frames_per_gloss = frames_per_gloss
        
        # Leer anotaciones
        df = pd.read_csv(anno_file, sep=' \\| ')
        df.columns = ['video_id', 'split', 'gloss_seq']
        self.samples = df[df['split'] == split].reset_index()
        
        # Cargar vocabulario
        self.gloss2idx = self._load_vocab()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        video_id = sample['video_id']
        gloss_seq = sample['gloss_seq']
        
        # Cargar frames
        video_dir = f"{self.data_path}/train/{video_id}/1/"
        frame_files = sorted(glob(f"{video_dir}/*.png"))
        
        # Seleccionar 12 frames uniformes
        indices = self._select_frames(len(frame_files))
        frames = [cv2.imread(frame_files[i]) for i in indices]
        frames = np.stack(frames)  # (12, 384, 288, 3)
        
        # Rescalar
        frames = cv2.resize(frames, (224, 224))  # (12, 224, 224, 3)
        
        # Convertir glosas
        gloss_list = gloss_seq.split()
        gloss_ids = torch.tensor([self.gloss2idx[g] for g in gloss_list],
                                 dtype=torch.long)
        
        return {
            'frames': torch.tensor(frames, dtype=torch.float32),
            'gloss_ids': gloss_ids,
            'length': torch.tensor(len(gloss_list), dtype=torch.long)
        }
    
    def _select_frames(self, total):
        target = self.frames_per_gloss
        if total >= target:
            step = total // target
            return [i * step for i in range(target)]
        else:
            return list(range(total))
    
    def _load_vocab(self):
        # Tipicamente hardcoded o desde archivo
        gloss_list = ['UNKNOWN', 'SEIN', 'NICHT', ..., 'ZIRKEL']
        return {g: i for i, g in enumerate(gloss_list)}

# Uso:
train_dataset = PhoenixDataset(
    data_path='Datasets/phoenix-2014.v3/phoenix2014-release',
    anno_file='phoenix2014.v3.train.csv',
    split=1
)

train_loader = DataLoader(
    train_dataset,
    batch_size=25,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate  # Para batch heterogéneo
)
```


# 10. KNOWN ISSUES AND SOLUTIONS
#################################

Problema 1: Desbalanceo de clases
─────────────────────────────────
Issue: Algunas glosas >100 ejemplos, otras <5
Effect: Modelo sesgado, WER bajo en comunes, alto en raras
Solution: Class weighting en CTC Loss (implementado en train.py)

Problema 2: Actores unseen en test
──────────────────────────────────
Issue: Test tiene actores no vistos en train
Effect: WER test > WER dev (~3-5% diferencia)
Solution: Data augmentation, more diverse training data

Problema 3: Variabilidad de duración
────────────────────────────────────
Issue: Glosa misma puede durar 4 frames o 50 frames
Effect: 12-frame window es promedio, algunos no caben bien
Solution: CTC Loss maneja alineación automáticamente

Problema 4: Iluminación variable
────────────────────────────────
Issue: Algunos videos lit bien, otros oscuros
Effect: CNN features degradados en videos oscuros
Solution: Normalización ImageNet helps, augmentation potential

Problema 5: File I/O bottleneck
──────────────────────────────
Issue: Leer PNG desde disco ~50ms/frame, ×12 frames = 600ms/video
Effect: Data loading es cuello de botella
Solution: Pre-cache en RAM (si hay memoria), SSD rápido, dataloader workers


# 11. ALTERNATIVE DATASETS
############################

Otros datasets SLR comúnmente usados:

CSL-Daily:
  - Lenguaje: Sign Language chino
  - Size: ~200 horas
  - Glosas: ~500 (menos que PHOENIX)
  - Papers: CSL-DAILY paper (2022)

WLASL:
  - Lenguaje: American Sign Language (ASL)
  - Size: 21,083 videos
  - Glosas: 2000
  - Plus: Muy reciente, community
  - Minus: Baja calidad, variable

DGS Corpus:
  - Lenguaje: German Sign Language (como PHOENIX)
  - Size: Large (>700 horas)
  - Glosas: 2600+
  - Status: Restricted access

BSL Corpus:
  - Lenguaje: British Sign Language
  - Size: 300+ horas
  - Glosas: 1000+
  - Access: Academic license

Transfer learning (PHOENIX → CSL-Daily):
  ```python
  # Cargar BEST.pt (PHOENIX)
  model.load_state_dict(torch.load('BEST.pt'))
  
  # Adaptar vocabulary (1232 → 500)
  # Reentrenar decoder layer
  model.decoder = nn.Linear(1280, 500)  # Nuevo
  
  # Fine-tune en CSL-Daily
  # Congelar CNN, entrenar Transformer+Decoder
  ```
"""

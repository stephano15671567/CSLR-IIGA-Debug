# FUNDAMENTOS MATEMÁTICOS DEL MODELO IIGA

Este documento contiene todas las ecuaciones matemáticas formales utilizadas en el modelo IIGA.

---

## 1. NOTACIÓN

| Símbolo | Significado |
|---------|-------------|
| B | Batch size |
| T | Número de frames (12) |
| H, W | Alto y ancho de imagen (224×224) |
| C | Canales de imagen (3: RGB) |
| D | Dimensión del embedding (1280) |
| V | Tamaño del vocabulario (1232 glosas) |
| N | Número de cabezas de atención (8) |
| d_k | Dimensión por cabeza (D/N = 160) |
| L | Número de capas Transformer (2) |
| ⊙ | Producto elemento a elemento (Hadamard) |
| ⊕ | Concatenación |

---

## 2. PIPELINE COMPLETO

### 2.1 Input
Video representado como secuencia de frames:

```
X_input ∈ ℝ^(B × T × H × W × C)
```

Ejemplo: (2, 12, 224, 224, 3)

---

## 3. CNN FEATURE EXTRACTION (MobileNetV2)

### 3.1 Reshape para procesamiento independiente
```
X_reshaped = reshape(X_input, (B·T, H, W, C))
```

### 3.2 Convolución Inicial
```
X_0 = Conv2D(X_reshaped; kernel=3×3, stride=2, channels=32)
X_0 = BatchNorm(X_0)
X_0 = ReLU6(X_0)
```

Donde ReLU6(x) = min(max(0, x), 6)

### 3.3 Inverted Residual Block

Para cada bloque i:

**Expansion:**
```
X_expanded = Conv2D_1×1(X_i; channels=C_in · t)
X_expanded = BatchNorm(X_expanded)
X_expanded = ReLU6(X_expanded)
```

**Depthwise Convolution:**
```
X_depth = DepthwiseConv2D_3×3(X_expanded)
X_depth = BatchNorm(X_depth)
X_depth = ReLU6(X_depth)
```

**Projection:**
```
X_proj = Conv2D_1×1(X_depth; channels=C_out)
X_proj = BatchNorm(X_proj)
```

**Residual Connection (si stride=1 y C_in=C_out):**
```
X_i+1 = X_i + X_proj
```

### 3.4 Convolución Final
```
X_conv = Conv2D_1×1(X_17; channels=1280)
X_conv = BatchNorm(X_conv)
X_conv = ReLU6(X_conv)
```

### 3.5 Global Average Pooling
```
X_pooled[b, c] = (1/(H·W)) · Σ_{h=1}^H Σ_{w=1}^W X_conv[b, h, w, c]
```

Resultado: X_pooled ∈ ℝ^(B·T × 1280)

### 3.6 Reshape Final
```
X_cnn = reshape(X_pooled, (B, T, D))
```

Resultado: **X_cnn ∈ ℝ^(B × T × D)**

---

## 4. POSITIONAL ENCODING

### 4.1 Ecuación Sinusoidal

Para posición `pos` y dimensión `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/D))
PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
```

Donde:
- pos ∈ [0, 1, 2, ..., T-1] (posición temporal del frame)
- i ∈ [0, 1, 2, ..., D/2-1] (índice de dimensión)

### 4.2 Matriz de Positional Encoding

```
PE ∈ ℝ^(T × D)

PE = [
  [sin(0/10000^0), cos(0/10000^0), sin(0/10000^(2/D)), cos(0/10000^(2/D)), ...],
  [sin(1/10000^0), cos(1/10000^0), sin(1/10000^(2/D)), cos(1/10000^(2/D)), ...],
  ...
  [sin(T-1/10000^0), cos(T-1/10000^0), ..., ...]
]
```

### 4.3 Aplicación

```
X_pe = X_cnn + PE
```

Broadcasting: PE se repite para cada sample del batch.

Resultado: **X_pe ∈ ℝ^(B × T × D)**

---

## 5. MULTI-HEAD ATTENTION

### 5.1 Proyecciones Lineales

```
Q = X_pe @ W_Q + b_Q    ∈ ℝ^(B × T × D)
K = X_pe @ W_K + b_K    ∈ ℝ^(B × T × D)
V = X_pe @ W_V + b_V    ∈ ℝ^(B × T × D)
```

Donde W_Q, W_K, W_V ∈ ℝ^(D × D) y b_Q, b_K, b_V ∈ ℝ^D

### 5.2 División en Múltiples Cabezas

```
Q = reshape(Q, (B, T, N, d_k)).transpose(1, 2)  →  (B, N, T, d_k)
K = reshape(K, (B, T, N, d_k)).transpose(1, 2)  →  (B, N, T, d_k)
V = reshape(V, (B, T, N, d_k)).transpose(1, 2)  →  (B, N, T, d_k)
```

Donde d_k = D / N

### 5.3 Attention Scores

```
Scores = (Q @ K^T) / sqrt(d_k)    ∈ ℝ^(B × N × T × T)
```

Elemento (i, j) de la matriz de scores para una cabeza:
```
Scores[i, j] = (1/sqrt(d_k)) · Σ_{k=1}^{d_k} Q[i, k] · K[j, k]
```

### 5.4 Softmax

Para cada fila i:
```
AttentionWeights[i, j] = exp(Scores[i, j]) / Σ_{j'=1}^T exp(Scores[i, j'])
```

Propiedades:
- Σ_{j=1}^T AttentionWeights[i, j] = 1
- AttentionWeights[i, j] ∈ [0, 1]

### 5.5 Aplicar Attention a Valores

```
Output_heads = AttentionWeights @ V    ∈ ℝ^(B × N × T × d_k)
```

Elemento i del output:
```
Output_heads[i] = Σ_{j=1}^T AttentionWeights[i, j] · V[j]
```

### 5.6 Concatenar Cabezas

```
Output_concat = reshape(Output_heads.transpose(1, 2), (B, T, D))
```

### 5.7 Proyección Final

```
Output_mha = Output_concat @ W_O + b_O    ∈ ℝ^(B × T × D)
```

Donde W_O ∈ ℝ^(D × D) y b_O ∈ ℝ^D

**Ecuación completa de Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_N) @ W_O

donde head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)

y Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
```

---

## 6. LAYER NORMALIZATION

### 6.1 Ecuación General

```
LayerNorm(x) = γ ⊙ ((x - μ) / sqrt(σ² + ε)) + β
```

Donde:
- μ: media de x en la última dimensión
- σ²: varianza de x en la última dimensión
- γ, β: parámetros aprendibles (scale y shift)
- ε: constante pequeña para estabilidad numérica (típicamente 1e-5)

### 6.2 Cálculo de Media y Varianza

Para cada sample b y posición t:

```
μ[b, t] = (1/D) · Σ_{i=1}^D x[b, t, i]

σ²[b, t] = (1/D) · Σ_{i=1}^D (x[b, t, i] - μ[b, t])²
```

### 6.3 Normalización

```
x_norm[b, t, i] = (x[b, t, i] - μ[b, t]) / sqrt(σ²[b, t] + ε)
```

### 6.4 Transformación Afín

```
output[b, t, i] = γ[i] · x_norm[b, t, i] + β[i]
```

Donde γ, β ∈ ℝ^D son parámetros aprendibles.

---

## 7. RESIDUAL CONNECTION

### 7.1 Ecuación

```
X_residual = X_input + Sublayer(X_input)
```

Ejemplo con Multi-Head Attention:
```
X_1 = X_pe + LayerNorm(MultiHeadAttention(X_pe))
```

**Post-Norm vs Pre-Norm:**

Post-Norm (usado en IIGA):
```
X_out = X_in + LayerNorm(Sublayer(X_in))
```

Pre-Norm (alternativa):
```
X_out = X_in + Sublayer(LayerNorm(X_in))
```

---

## 8. FEED-FORWARD NETWORK

### 8.1 Ecuación Completa

```
FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2
```

Donde:
- W_1 ∈ ℝ^(D × D_ff), b_1 ∈ ℝ^(D_ff)
- W_2 ∈ ℝ^(D_ff × D), b_2 ∈ ℝ^D
- D_ff = 4·D (expansión típica)

### 8.2 ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x) = {
  x,  si x > 0
  0,  si x ≤ 0
}
```

### 8.3 Paso a Paso

**Expansión:**
```
X_hidden = ReLU(X @ W_1 + b_1)    ∈ ℝ^(B × T × D_ff)
```

**Compresión:**
```
X_ffn = X_hidden @ W_2 + b_2    ∈ ℝ^(B × T × D)
```

---

## 9. TRANSFORMER LAYER COMPLETA

### 9.1 Ecuación General

Una capa Transformer combina Multi-Head Attention + FFN:

```
# Sub-layer 1: Multi-Head Attention
X_attn = MultiHeadAttention(X_in, X_in, X_in)
X_1 = LayerNorm(X_in + X_attn)

# Sub-layer 2: Feed-Forward Network
X_ffn = FFN(X_1)
X_out = LayerNorm(X_1 + X_ffn)
```

### 9.2 IIGA: Intra-Gloss vs Inter-Gloss

**Intra-Gloss Attention** (ventana local):
```
Mask_intra[i, j] = {
  1,  si |i - j| ≤ w  (w = window size)
  0,  en otro caso
}

Scores_intra = Scores ⊙ Mask_intra - ∞·(1 - Mask_intra)
AttentionWeights_intra = softmax(Scores_intra)
```

**Inter-Gloss Attention** (global):
```
AttentionWeights_inter = softmax(Scores)  (sin máscara)
```

### 9.3 Stack de Capas IIGA

```
X_0 = X_pe

# Capa 1
X_1_intra = IntraGlossAttention(X_0)
X_1 = LayerNorm(X_0 + X_1_intra)
X_1_ffn = FFN(X_1)
X_1_out = LayerNorm(X_1 + X_1_ffn)

X_2_inter = InterGlossAttention(X_1_out)
X_2 = LayerNorm(X_1_out + X_2_inter)
X_2_ffn = FFN(X_2)
X_2_out = LayerNorm(X_2 + X_2_ffn)

# Capa 2 (igual que Capa 1)
X_3_intra = IntraGlossAttention(X_2_out)
...
X_transformer = X_4_out
```

---

## 10. DECODER

### 10.1 Proyección Lineal

```
Logits = X_transformer @ W_decoder + b_decoder    ∈ ℝ^(B × T × V)
```

Donde:
- W_decoder ∈ ℝ^(D × V)
- b_decoder ∈ ℝ^V
- V = 1232 (tamaño del vocabulario)

### 10.2 Softmax (para interpretación)

```
P(glosa_v | frame_t) = exp(Logits[t, v]) / Σ_{v'=1}^V exp(Logits[t, v'])
```

Propiedades:
- Σ_{v=1}^V P(glosa_v | frame_t) = 1
- P(glosa_v | frame_t) ∈ [0, 1]

---

## 11. CTC LOSS (Connectionist Temporal Classification)

### 11.1 Definición del Problema

- Input: Logits ∈ ℝ^(T × B × V)
- Target: Secuencia de glosas y = [y_1, y_2, ..., y_U] donde U ≤ T
- Objetivo: Alinear secuencia de frames (longitud T) con glosas (longitud U)

### 11.2 Vocabulario Extendido

```
Alphabet' = Alphabet ∪ {ε}
```

Donde ε es el token "blank" (silencio/transición).

### 11.3 Alineaciones Válidas

Una alineación π = [π_1, π_2, ..., π_T] es válida si:
1. Removing blanks: π con ε removidos da y
2. Colapsing repeats: "HHεOOLA" → "HOLA"

Ejemplo:
- Target: y = ["HOLA", "MUNDO"]
- Alineaciones válidas:
  - π_1 = [H, O, L, A, ε, M, U, N, D, O, ε, ε]
  - π_2 = [ε, H, ε, O, L, A, M, ε, U, N, D, O]
  - ...

### 11.4 Probabilidad de Alineación

```
P(π | X) = Π_{t=1}^T P(π_t | X)[t]
```

Donde P(π_t | X)[t] es la probabilidad del símbolo π_t en el frame t (del softmax de logits).

### 11.5 Probabilidad Total

```
P(y | X) = Σ_{π ∈ Align(y)} P(π | X)
```

Donde Align(y) es el conjunto de todas las alineaciones válidas para y.

### 11.6 CTC Loss

```
Loss_CTC = -log P(y | X) = -log(Σ_{π ∈ Align(y)} P(π | X))
```

### 11.7 Algoritmo Forward-Backward

**Variables Forward (α):**
```
α[t, s] = probabilidad de generar y[1:s] en frames [1:t]

α[1, 0] = P(ε | X)[1]
α[1, 1] = P(y_1 | X)[1]

α[t, s] = (α[t-1, s] + α[t-1, s-1]) · P(y'_s | X)[t]
```

Donde y' es la secuencia extendida con blanks: y' = [ε, y_1, ε, y_2, ε, ..., ε, y_U, ε]

**Variables Backward (β):**
```
β[t, s] = probabilidad de generar y[s+1:U] desde frames [t+1:T]

β[T, |y'|] = 1
β[T, |y'|-1] = 1

β[t, s] = (β[t+1, s] + β[t+1, s+1]) · P(y'_s | X)[t+1]
```

**Probabilidad Total:**
```
P(y | X) = Σ_s α[T, s] = Σ_s β[1, s]
```

### 11.8 Gradiente

```
∂Loss_CTC / ∂Logits[t, v] = P(v | X)[t] - (1/P(y|X)) · Σ_{s: y'_s=v} α[t, s] · β[t, s]
```

---

## 12. MÉTRICAS

### 12.1 Word Error Rate (WER)

```
WER = (S + D + I) / N
```

Donde:
- S: Número de sustituciones
- D: Número de deleciones
- I: Número de inserciones
- N: Número total de palabras en la referencia

### 12.2 Distancia de Levenshtein

Matriz de programación dinámica:

```
DP[0, 0] = 0
DP[i, 0] = i    (i deleciones)
DP[0, j] = j    (j inserciones)

DP[i, j] = {
  DP[i-1, j-1],                           si ref[i] == hyp[j]
  1 + min(DP[i-1, j],                     (deleción)
          DP[i, j-1],                     (inserción)
          DP[i-1, j-1]),                  (sustitución)
                                          en otro caso
}
```

Distancia = DP[N, M] donde N, M son longitudes de ref y hyp.

### 12.3 BLEU Score

```
BLEU = BP · exp(Σ_{n=1}^4 w_n · log(p_n))
```

Donde:
- p_n: precisión de n-gramas
- w_n: peso (típicamente 1/4)
- BP: brevity penalty = min(1, exp(1 - r/c))
  - r: longitud de referencia
  - c: longitud de candidato

**Precisión de n-gramas:**
```
p_n = (número de n-gramas coincidentes) / (total de n-gramas en candidato)
```

### 12.4 ROUGE Score

**ROUGE-L (Longest Common Subsequence):**
```
R_lcs = LCS(ref, hyp) / len(ref)         (recall)
P_lcs = LCS(ref, hyp) / len(hyp)         (precision)
F_lcs = (2 · R_lcs · P_lcs) / (R_lcs + P_lcs)
```

Donde LCS es la longitud de la subsecuencia común más larga.

---

## 13. OPTIMIZACIÓN

### 13.1 AdamW

Actualización de parámetros:

```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t           (primer momento)
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t²          (segundo momento)

m̂_t = m_t / (1 - β_1^t)                         (bias correction)
v̂_t = v_t / (1 - β_2^t)

θ_t = θ_{t-1} - η · (m̂_t / (sqrt(v̂_t) + ε) + λ · θ_{t-1})
```

Donde:
- g_t: gradiente en paso t
- β_1 = 0.9, β_2 = 0.999 (típicamente)
- η: learning rate
- λ: weight decay (0.0001 típicamente)
- ε = 1e-8

### 13.2 Learning Rate Schedule

**Warmup + Cosine Decay:**
```
lr(t) = {
  lr_max · (t / T_warmup),                                    si t < T_warmup
  lr_min + 0.5 · (lr_max - lr_min) · (1 + cos(π·(t-T_warmup)/(T_max-T_warmup))),   si t ≥ T_warmup
}
```

Donde:
- T_warmup: steps de calentamiento
- T_max: steps totales
- lr_max: learning rate máximo
- lr_min: learning rate mínimo

---

## 14. COMPLEJIDAD COMPUTACIONAL

### 14.1 Multi-Head Attention

**Tiempo:**
```
O(T² · D + T · D²)
```

Desglose:
- Q·K^T: O(T² · d_k · N) = O(T² · D)
- softmax: O(T²)
- Attn·V: O(T² · d_k · N) = O(T² · D)
- Proyecciones Q, K, V, O: O(T · D²)

**Espacio:**
```
O(T² · N + T · D)
```

### 14.2 Feed-Forward Network

**Tiempo:**
```
O(T · D · D_ff) = O(T · D²)    (con D_ff = 4D)
```

**Espacio:**
```
O(T · D_ff) = O(T · D)
```

### 14.3 Modelo Completo (por batch)

**Tiempo total:**
```
O(B · T · H · W · C) +           (CNN)
O(L · (T² · D + T · D²))        (Transformer)
```

Con valores típicos (B=2, T=12, H=W=224, D=1280, L=4):
- CNN: ~3.6 GFLOPS
- Transformer: ~0.8 GFLOPS
- Total: ~4.4 GFLOPS

---

## 15. REFERENCIAS MATEMÁTICAS

1. **Attention Mechanism:**
   - Vaswani et al. "Attention is All You Need" (2017)
   - Ecuación fundamental: Attention(Q,K,V) = softmax(QK^T/√d_k)V

2. **CTC Loss:**
   - Graves et al. "Connectionist Temporal Classification" (2006)
   - Forward-backward algorithm

3. **Layer Normalization:**
   - Ba et al. "Layer Normalization" (2016)

4. **AdamW:**
   - Loshchilov & Hutter "Decoupled Weight Decay Regularization" (2019)

5. **MobileNetV2:**
   - Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)

---

## 16. NOTAS ADICIONALES

### 16.1 Estabilidad Numérica

**Softmax con log-sum-exp trick:**
```
max_val = max(x)
softmax(x) = exp(x - max_val) / Σ exp(x - max_val)
```

Evita overflow/underflow.

**LayerNorm con epsilon:**
```
σ = sqrt((1/D)·Σ(x - μ)² + ε)
```

ε = 1e-5 previene división por cero.

### 16.2 Inicialización de Parámetros

**Xavier/Glorot para Linear layers:**
```
W ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
```

**Bias:**
```
b = 0
```

### 16.3 Dropout (si se usa)

```
Dropout(x, p) = {
  0,        con probabilidad p
  x/(1-p),  con probabilidad 1-p
}
```

Durante inferencia: Dropout(x, p) = x

---

**Este documento contiene todas las ecuaciones matemáticas necesarias para entender y reimplementar el modelo IIGA.**

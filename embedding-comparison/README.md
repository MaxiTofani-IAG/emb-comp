# Sistema de Comparación de Embeddings

Compara diferentes modelos de embeddings y genera reportes automáticos.

## 🚀 Inicio Rápido

### 1. Instalar dependencias
```bash
uv add openai sentence-transformers scikit-learn
```

### 2. Ejecutar comparación
```bash
uv run python emb_test.py
```

¡Listo! Los reportes se generarán en `embedding_reports/`.

## 📁 Archivos Principales

- `emb_test.py` - Script principal
- `data/work_orders.json` - Datos de ejemplo
- `data/test_queries.json` - Consultas de prueba
- `data/config.json` - Configuración

## 💾 ¿Dónde se Guardan los Embeddings?

**Los embeddings se generan en memoria y NO se guardan permanentemente.**

### Estructura en Memoria

Los embeddings se almacenan en esta estructura de datos:

```python
# Estructura principal
embeddings = {
    'minilm': np.array([[0.1, 0.2, ..., 0.384]]),      # Shape: (N_textos, 384)
    'mpnet': np.array([[0.3, 0.4, ..., 0.768]]),       # Shape: (N_textos, 768) 
    'bge': np.array([[0.5, 0.6, ..., 0.384]]),         # Shape: (N_textos, 384)
    'minilm_time': 0.123,                               # Tiempo de generación
    'mpnet_time': 0.456,                                # Tiempo de generación
    'bge_time': 0.789                                   # Tiempo de generación
}

# Luego se separan en:
query_embeddings = {
    'minilm': np.array([[0.1, 0.2, ..., 0.384]]),      # Solo la query
    'mpnet': np.array([[0.3, 0.4, ..., 0.768]]),       # Solo la query
    'bge': np.array([[0.5, 0.6, ..., 0.384]])          # Solo la query
}

doc_embeddings = {
    'minilm': np.array([[0.1, 0.2, ..., 0.384],        # Solo los documentos
                       [0.2, 0.3, ..., 0.384],
                       ...]),
    'mpnet': np.array([[0.3, 0.4, ..., 0.768],         # Solo los documentos
                       [0.4, 0.5, ..., 0.768],
                       ...]),
    'bge': np.array([[0.5, 0.6, ..., 0.384],           # Solo los documentos
                     [0.6, 0.7, ..., 0.384],
                     ...])
}
```

### Flujo de Datos

1. **Entrada**: `[query] + documents` → Lista de textos
2. **Generación**: `get_embeddings()` → Diccionario con arrays numpy
3. **Separación**: Query vs documentos → Dos diccionarios separados
4. **Cálculo**: Similitud coseno → Scores de similitud
5. **Liberación**: Memoria se libera automáticamente

### Tamaños de Embeddings

- **MiniLM-L6-v2**: 384 dimensiones por texto
- **MPNet-base-v2**: 768 dimensiones por texto  
- **BGE-small-en-v1.5**: 384 dimensiones por texto
- **Azure OpenAI Ada**: 1536 dimensiones por texto

**¿Por qué no se guardan?**
- Los embeddings son muy grandes (384-1536 dimensiones por texto)
- Se pueden regenerar fácilmente desde el texto original
- Los reportes contienen toda la información necesaria (scores, rankings, etc.)

**Si necesitas guardar embeddings:**
```python
# En emb_test.py, modifica get_embeddings() para guardar:
import pickle
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(results, f)
```

## ⚙️ Configuración Básica

### Editar datos de ejemplo
**`data/work_orders.json`**:
```json
[
  "REPLACED BRAKE PADS ON MAIN LANDING GEAR",
  "FUEL PUMP PRESSURE LOW - INSPECTION REQUIRED"
]
```

**`data/test_queries.json`**:
```json
[
  "brake system maintenance",
  "fuel pump pressure"
]
```

### Configurar normalización de texto
**`data/config.json`**:
```json
{
  "text_processing": {
    "normalize_to_lowercase": true
  }
}
```

## 📊 Reportes Generados

Cada ejecución crea una carpeta con:
- `embedding_comparison_TIMESTAMP.json` - Datos completos
- `similarity_scores_TIMESTAMP.csv` - Scores de similitud
- `rankings_TIMESTAMP.csv` - Rankings ordenados
- `summary_TIMESTAMP.txt` - Resumen ejecutivo

## 🔧 Uso Avanzado

### Probar normalización de texto
```bash
uv run python test_normalization.py
```

### Usar configuración personalizada
```bash
uv run python emb_test.py --config data/config_no_normalize.json
```

## 🐛 Problemas Comunes

**Error: "No module named 'openai'"**
```bash
uv run python emb_test.py
```

**Error: "Archivo no encontrado"**
Verifica que los archivos en `data/` existen y son JSON válidos.

---

**¡Simple y directo! 🚀**

# Sistema de Comparación de Embeddings

Sistema completo para comparar diferentes modelos de embeddings y generar reportes detallados.

## 🚀 Características

- **Múltiples Modelos**: Compara MiniLM-L6-v2, MPNet-base-v2 y Azure OpenAI Ada-002
- **Normalización de Texto**: Opción para convertir texto a minúsculas antes de generar embeddings
- **Reportes Automáticos**: Genera reportes en JSON, CSV y texto
- **Organización por Sesiones**: Cada ejecución se guarda en una carpeta con fecha y hora
- **Gestión de Sesiones**: Script para navegar y comparar sesiones anteriores
- **Datos Externos**: Carga work orders y queries desde archivos JSON
- **Configuración Flexible**: Ajusta thresholds y parámetros via archivos de configuración
- **Análisis Detallado**: Rankings, similitudes y métricas de performance

## 📁 Estructura del Proyecto

```
embedding-comparison/
├── emb_test.py              # Script principal
├── data_loader.py           # Cargador de datos externos
├── session_manager.py       # Gestor de sesiones
├── example_usage.py         # Ejemplos de uso
├── data/                    # Directorio de datos
│   ├── work_orders.json     # Work orders de ejemplo
│   ├── test_queries.json    # Queries de prueba
│   └── config.json          # Configuración del sistema
├── embedding_reports/       # Reportes generados
│   ├── sessions_index.txt   # Índice de todas las sesiones
│   └── embedding_report_27-7-2025-21:08/  # Sesión específica
│       ├── embedding_comparison_20250727_210800.json
│       ├── similarity_scores_20250727_210800.csv
│       ├── rankings_20250727_210800.csv
│       └── summary_20250727_210800.txt
└── README.md               # Este archivo
```

## 🛠️ Instalación

1. **Instalar dependencias**:
   ```bash
   uv add openai sentence-transformers scikit-learn
   ```

2. **Ejecutar con entorno virtual**:
   ```bash
   uv run python emb_test.py
   ```

## 📊 Uso Básico

### 1. Usar Datos por Defecto
```bash
uv run python emb_test.py
```

### 2. Personalizar Datos
Edita los archivos en `data/`:

**`data/work_orders.json`**:
```json
[
  "REPLACED BRAKE PADS ON MAIN LANDING GEAR",
  "FUEL PUMP PRESSURE LOW - INSPECTION REQUIRED",
  "AVIONICS DISPLAY SHOWING ERROR CODES"
]
```

**`data/test_queries.json`**:
```json
[
  "brake system maintenance",
  "fuel pump pressure",
  "avionics display error"
]
```

### 3. Modificar Configuración
**`data/config.json`**:
```json
{
  "confidence_thresholds": {
    "minilm": {"high": 0.70, "medium": 0.50},
    "mpnet": {"high": 0.75, "medium": 0.55}
  },
  "output_settings": {
    "default_output_dir": "my_reports"
  },
  "text_processing": {
    "normalize_to_lowercase": true,
    "description": "Convierte todo el texto a minúsculas antes de generar embeddings"
  }
}
```

### 4. Configuración de Normalización de Texto

El sistema incluye una opción para normalizar el texto a minúsculas antes de generar embeddings, lo que puede mejorar la consistencia de los resultados.

**Configuración con normalización** (`data/config.json`):
```json
{
  "text_processing": {
    "normalize_to_lowercase": true,
    "description": "Convierte todo el texto a minúsculas antes de generar embeddings"
  }
}
```

**Configuración sin normalización** (`data/config_no_normalize.json`):
```json
{
  "text_processing": {
    "normalize_to_lowercase": false,
    "description": "Mantiene la capitalización original del texto"
  }
}
```

**Probar diferencias de normalización**:
```bash
uv run python test_normalization.py
```

## 🔧 Ejemplos Avanzados

### Ejecutar Ejemplos de Uso
```bash
uv run python example_usage.py
```

### Usar Datos Personalizados
```python
from data_loader import DataLoader
from emb_test import EmbeddingComparator

# Cargar datos personalizados
data_loader = DataLoader()
work_orders = data_loader.load_work_orders("my_work_orders.json")
queries = data_loader.load_test_queries("my_queries.json")
config = data_loader.load_config("my_config.json")

# Ejecutar comparación
comparator = EmbeddingComparator(output_dir="my_reports", config=config)
for query in queries:
    comparator.compare_models(query, work_orders)
comparator.generate_reports()
```

### Procesamiento por Lotés
```python
# Procesar múltiples datasets
datasets = {
    "aviation": {"work_orders": [...], "queries": [...]},
    "automotive": {"work_orders": [...], "queries": [...]}
}

for name, data in datasets.items():
    comparator = EmbeddingComparator(output_dir=f"{name}_reports")
    for query in data["queries"]:
        comparator.compare_models(query, data["work_orders"])
    comparator.generate_reports()
```

## 📈 Reportes Generados

### 1. **JSON Completo** (`embedding_comparison_TIMESTAMP.json`)
- Todos los datos estructurados
- Ideal para análisis programático

### 2. **CSV Detallado** (`similarity_scores_TIMESTAMP.csv`)
- Cada fila = combinación query-documento-modelo
- Incluye scores, confianza y timestamps

### 3. **CSV de Rankings** (`rankings_TIMESTAMP.csv`)
- Rankings ordenados por similitud
- Fácil importación en Excel/Sheets

### 4. **Resumen Ejecutivo** (`summary_TIMESTAMP.txt`)
- Estadísticas agregadas por modelo
- Métricas de performance
- Recomendaciones

## ⚙️ Configuración

### Modelos Disponibles
- **MiniLM-L6-v2**: 384 dimensiones, rápido
- **MPNet-base-v2**: 768 dimensiones, balanceado
- **Azure OpenAI Ada**: 1536 dimensiones, alta calidad

### Thresholds de Confianza
```json
{
  "minilm": {"high": 0.60, "medium": 0.40},
  "mpnet": {"high": 0.65, "medium": 0.45},
  "azure_ada": {"high": 0.70, "medium": 0.50}
}
```

### Configuración de Salida
```json
{
  "output_settings": {
    "default_output_dir": "embedding_reports",
    "include_timestamp": true,
    "generate_all_reports": true
  }
}
```

## 🔍 Interpretación de Resultados

### Niveles de Confianza
- **🟢 Strong Match** (> 0.70): Excelente coincidencia
- **🟡 Good Match** (0.50-0.70): Buena coincidencia
- **🔴 Weak Match** (< 0.50): Coincidencia débil

### Comparación con pgvector
Los scores de similitud son equivalentes a:
```sql
SELECT *, 1 - (embeddings <=> %s::vector) AS similarity
FROM work_orders ORDER BY embeddings <=> %s::vector;
```

## 🚀 Casos de Uso

### 1. **Análisis de Work Orders**
- Comparar diferentes modelos para búsqueda semántica
- Evaluar calidad de matches en mantenimiento aeronáutico

### 2. **Optimización de Modelos**
- Probar diferentes thresholds de confianza
- Comparar performance vs calidad

### 3. **Migración de Sistemas**
- Validar equivalencia con pgvector
- Calibrar nuevos modelos

## 📝 Personalización

### Agregar Nuevos Modelos
1. Edita `data/config.json`
2. Agrega configuración del modelo
3. Actualiza `emb_test.py` si es necesario

### Nuevos Tipos de Datos
1. Crea archivos JSON en `data/`
2. Usa `DataLoader` para cargar
3. Ejecuta comparaciones

### Reportes Personalizados
1. Extiende `EmbeddingComparator`
2. Agrega métodos de reporte
3. Integra con `generate_reports()`

## 🐛 Solución de Problemas

### Error: "No module named 'openai'"
```bash
uv run python emb_test.py  # Usar entorno virtual
```

### Error: "Directorio de datos no encontrado"
```bash
python data_loader.py  # Crear datos de ejemplo
```

### Error: "Archivo de work orders no encontrado"
Verifica que `data/work_orders.json` existe y es válido.

## 📞 Soporte

Para problemas o preguntas:
1. Revisa los ejemplos en `example_usage.py`
2. Verifica la configuración en `data/config.json`
3. Consulta los reportes generados para debugging

---

**¡Disfruta comparando embeddings! 🚀**

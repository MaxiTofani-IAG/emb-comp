#!/usr/bin/env python3
import numpy as np
from sentence_transformers import SentenceTransformer, util
import time
import openai
from typing import List, Dict, Tuple
import os
import json
import csv
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import DataLoader

# Configurar Azure OpenAI (opcional)
# Descomenta y configura estas variables:

AZURE_OPENAI_API_KEY="5Gfll2wWy8CFoq05W89TwlKqf9Ok9hOT1HN9WhWlvbc4XRaxdVhmJQQJ99BAACYeBjFXJ3w3AAABACOG8EA4"
AZURE_OPENAI_API_VERSION="2024-08-01-preview"
AZURE_OPENAI_API_ENDPOINT="https://mro-azure-dev.openai.azure.com/"	

os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_API_ENDPOINT
os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
#os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "text-embedding-ada-002"  # Tu deployment name


class EmbeddingComparator:
    def __init__(self, output_dir: str = "reports", config: Dict = None):
        print("Cargando modelos...")
        
        # Cargar configuración
        if config is None:
            data_loader = DataLoader()
            config = data_loader.load_config()
        
        self.config = config
        self.model_configs = config.get('models', {})
        self.thresholds = config.get('confidence_thresholds', {})
        
        # Inicializar modelos
        self.model_small = SentenceTransformer(self.model_configs.get('minilm', {}).get('name', 'all-MiniLM-L6-v2'))
        self.model_large = SentenceTransformer(self.model_configs.get('mpnet', {}).get('name', 'all-mpnet-base-v2'))
        
        # Crear directorio base de reportes si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear subcarpeta con fecha y hora legible
        now = datetime.now()
        self.timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.readable_timestamp = now.strftime("%d-%m-%Y-%H-%M")
        
        # Crear subcarpeta específica para esta ejecución
        self.session_dir = os.path.join(output_dir, f"embedding_report_{self.readable_timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Datos para reportes
        self.all_results = []
        
        print("Modelos cargados!")
        print(f"📁 Reportes se guardarán en: {self.session_dir}")
    
    def get_embeddings(self, texts: List[str]) -> Dict[str, List[np.ndarray]]:
        """Genera embeddings con todos los modelos"""
        results = {}
        
        # Verificar si debemos normalizar a minúsculas
        normalize_lowercase = self.config.get('text_processing', {}).get('normalize_to_lowercase', False)
        
        if normalize_lowercase:
            print("🔄 Normalizando textos a minúsculas...")
            processed_texts = [text.lower() for text in texts]
        else:
            processed_texts = texts
        
        # Sentence Transformers
        print("Generando embeddings con MiniLM-L6-v2...")
        start_time = time.time()
        results['minilm'] = self.model_small.encode(processed_texts, normalize_embeddings=True)
        results['minilm_time'] = time.time() - start_time
        
        print("Generando embeddings con mpnet-base-v2...")
        start_time = time.time()
        results['mpnet'] = self.model_large.encode(processed_texts, normalize_embeddings=True)
        results['mpnet_time'] = time.time() - start_time
        
        # Azure OpenAI Ada (opcional - descomenta si tienes configuración)
        """
        if all(key in os.environ for key in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]):
            print("Generando embeddings con Azure OpenAI Ada-002...")
            start_time = time.time()
            
            # Configurar cliente Azure OpenAI
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
            )
            
            azure_embeddings = []
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "text-embedding-ada-002")
            
            for text in processed_texts:
                response = client.embeddings.create(
                    input=text,
                    model=deployment_name
                )
                azure_embeddings.append(np.array(response.data[0].embedding))
            
            results['azure_ada'] = np.array(azure_embeddings)
            results['azure_ada_time'] = time.time() - start_time
        """
        
        return results
    
    def calculate_similarities(self, query_emb: Dict, doc_embs: Dict) -> Dict[str, List[float]]:
        """Calcula similarities usando la misma métrica que pgvector: 1 - cosine_distance"""
        similarities = {}
        
        for model_name in ['minilm', 'mpnet', 'azure_ada']:  # Agregado azure_ada
            if model_name in query_emb and model_name in doc_embs:
                # Calcular similitud coseno directamente (ya es similarity, no distance)
                cosine_similarities = cosine_similarity(
                    query_emb[model_name], doc_embs[model_name]
                )
                # Convertir a lista y extraer la primera fila (query vs todos los docs)
                similarities[model_name] = cosine_similarities[0].tolist()
        
        return similarities
    
    def compare_models(self, query: str, documents: List[str]) -> None:
        """Compara modelos con query específica"""
        print(f"\n{'='*80}")
        print(f"QUERY: '{query}'")
        print(f"{'='*80}")
        
        # Generar embeddings
        all_texts = [query] + documents
        embeddings = self.get_embeddings(all_texts)
        
        # Separar query de documentos
        query_embeddings = {}
        doc_embeddings = {}
        
        for model_name in ['minilm', 'mpnet', 'azure_ada']:  # Agregado azure_ada
            if model_name in embeddings:
                query_embeddings[model_name] = embeddings[model_name][0:1]
                doc_embeddings[model_name] = embeddings[model_name][1:]
        
        # Calcular similarities
        similarities = self.calculate_similarities(query_embeddings, doc_embeddings)
        
        # Guardar resultados para reportes
        self.save_query_results(query, documents, similarities, embeddings)
        
        # Mostrar resultados
        self.print_results(query, documents, similarities, embeddings)
        
        # También mostrar el cálculo exacto como pgvector
        self.print_pgvector_equivalent(query, documents, similarities)
    
    def save_query_results(self, query: str, documents: List[str], similarities: Dict, embeddings: Dict):
        """Guarda resultados de una query para reportes"""
        query_result = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'documents': documents,
            'similarities': similarities,
            'text_processing': {
                'normalize_to_lowercase': self.config.get('text_processing', {}).get('normalize_to_lowercase', False),
                'description': 'Normalización de texto aplicada antes de generar embeddings'
            },
            'performance': {
                'minilm_time': embeddings.get('minilm_time', 0),
                'mpnet_time': embeddings.get('mpnet_time', 0),
                'azure_ada_time': embeddings.get('azure_ada_time', 0)
            },
            'model_info': {
                'minilm': {'dimensions': 384, 'model': 'all-MiniLM-L6-v2'},
                'mpnet': {'dimensions': 768, 'model': 'all-mpnet-base-v2'},
                'azure_ada': {'dimensions': 1536, 'model': 'text-embedding-ada-002'}
            }
        }
        
        self.all_results.append(query_result)
    
    def generate_reports(self):
        """Genera reportes en diferentes formatos"""
        print(f"\n{'='*60}")
        print("GENERANDO REPORTES...")
        print(f"{'='*60}")
        
        # Reporte JSON completo
        json_file = os.path.join(self.session_dir, f"embedding_comparison_{self.timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Reporte JSON completo: {json_file}")
        
        # Reporte CSV detallado
        csv_file = os.path.join(self.session_dir, f"similarity_scores_{self.timestamp}.csv")
        self.generate_csv_report(csv_file)
        print(f"✅ Reporte CSV detallado: {csv_file}")
        
        # Reporte de resumen
        summary_file = os.path.join(self.session_dir, f"summary_{self.timestamp}.txt")
        self.generate_summary_report(summary_file)
        print(f"✅ Reporte de resumen: {summary_file}")
        
        # Reporte de ranking
        ranking_file = os.path.join(self.session_dir, f"rankings_{self.timestamp}.csv")
        self.generate_ranking_report(ranking_file)
        print(f"✅ Reporte de rankings: {ranking_file}")
        
        # Crear archivo de índice de sesiones
        self.create_session_index()
        
        print(f"\n📁 Todos los reportes guardados en: {self.session_dir}")
        print(f"📅 Sesión: {self.readable_timestamp}")
    
    def create_session_index(self):
        """Crea un archivo de índice con todas las sesiones disponibles"""
        base_dir = os.path.dirname(self.session_dir)
        index_file = os.path.join(base_dir, "sessions_index.txt")
        
        # Obtener todas las carpetas de sesiones
        sessions = []
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and item.startswith("embedding_report_"):
                    # Extraer fecha y hora de la carpeta
                    session_name = item.replace("embedding_report_", "")
                    sessions.append((session_name, item_path))
        
        # Ordenar por fecha (más reciente primero)
        sessions.sort(reverse=True)
        
        # Crear archivo de índice
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("ÍNDICE DE SESIONES DE EMBEDDING COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de sesiones: {len(sessions)}\n\n")
            
            for i, (session_name, session_path) in enumerate(sessions, 1):
                f.write(f"{i}. {session_name}\n")
                f.write(f"   📁 Ruta: {session_path}\n")
                
                # Verificar archivos disponibles
                if os.path.exists(session_path):
                    files = [f for f in os.listdir(session_path) if f.endswith(('.json', '.csv', '.txt'))]
                    f.write(f"   📄 Archivos: {', '.join(files)}\n")
                
                f.write("\n")
        
        print(f"✅ Índice de sesiones actualizado: {index_file}")
    
    def generate_csv_report(self, filename: str):
        """Genera reporte CSV con todos los scores de similitud"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['query', 'document_id', 'document_text', 'model', 'similarity', 'confidence_level', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.all_results:
                query = result['query']
                documents = result['documents']
                similarities = result['similarities']
                
                for doc_id, doc_text in enumerate(documents):
                    for model_name, sims in similarities.items():
                        similarity = sims[doc_id]
                        confidence = self.get_confidence_level(similarity, model_name)
                        
                        writer.writerow({
                            'query': query,
                            'document_id': doc_id + 1,
                            'document_text': doc_text[:200] + '...' if len(doc_text) > 200 else doc_text,
                            'model': model_name.upper(),
                            'similarity': round(similarity, 4),
                            'confidence_level': confidence,
                            'timestamp': result['timestamp']
                        })
    
    def generate_ranking_report(self, filename: str):
        """Genera reporte CSV con rankings por modelo"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['query', 'model', 'rank', 'document_id', 'similarity', 'confidence_level']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.all_results:
                query = result['query']
                documents = result['documents']
                similarities = result['similarities']
                
                for model_name, sims in similarities.items():
                    # Ordenar por similitud descendente
                    ranked_docs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
                    
                    for rank, (doc_idx, similarity) in enumerate(ranked_docs, 1):
                        confidence = self.get_confidence_level(similarity, model_name)
                        
                        writer.writerow({
                            'query': query,
                            'model': model_name.upper(),
                            'rank': rank,
                            'document_id': doc_idx + 1,
                            'similarity': round(similarity, 4),
                            'confidence_level': confidence
                        })
    
    def generate_summary_report(self, filename: str):
        """Genera reporte de texto con resumen ejecutivo"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE COMPARACIÓN DE EMBEDDINGS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de queries analizadas: {len(self.all_results)}\n\n")
            
            # Estadísticas por modelo
            model_stats = {'minilm': [], 'mpnet': [], 'azure_ada': []}
            
            for result in self.all_results:
                for model_name, sims in result['similarities'].items():
                    model_stats[model_name].extend(sims)
            
            f.write("ESTADÍSTICAS POR MODELO:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, scores in model_stats.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    min_score = min(scores)
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"  - Promedio: {avg_score:.4f}\n")
                    f.write(f"  - Máximo: {max_score:.4f}\n")
                    f.write(f"  - Mínimo: {min_score:.4f}\n")
                    f.write(f"  - Total de scores: {len(scores)}\n")
            
            # Performance
            f.write(f"\nPERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            total_minilm_time = sum(r['performance']['minilm_time'] for r in self.all_results)
            total_mpnet_time = sum(r['performance']['mpnet_time'] for r in self.all_results)
            f.write(f"Tiempo total MiniLM: {total_minilm_time:.3f}s\n")
            f.write(f"Tiempo total MPNet: {total_mpnet_time:.3f}s\n")
            
            # Recomendaciones
            f.write(f"\nRECOMENDACIONES:\n")
            f.write("-" * 20 + "\n")
            f.write("""
1. CALIDAD:
   - mpnet-base-v2: Generalmente mejor precisión
   - MiniLM-L6-v2: Buena para casos simples
   - OpenAI Ada: Mejor calidad pero costoso

2. PERFORMANCE:
   - MiniLM-L6-v2: Más rápido, menos memoria
   - mpnet-base-v2: Más lento, más memoria
   - OpenAI Ada: Latencia de API + costos

3. STORAGE:
   - MiniLM-L6-v2: 384 dims = menos espacio
   - mpnet-base-v2: 768 dims = doble espacio
   - OpenAI Ada: 1536 dims = 4x espacio

RECOMENDACIÓN:
- Para POC: Prueba mpnet-base-v2 si performance no es crítica
- Para producción: Evalúa trade-off calidad vs velocidad
- OpenAI Ada: Solo si calidad es crítica y budget lo permite
            """)
    
    def print_pgvector_equivalent(self, query: str, documents: List[str], similarities: Dict):
        """Muestra equivalencia exacta con pgvector query"""
        print(f"\n{'*'*60}")
        print("EQUIVALENCIA CON TU QUERY PGVECTOR:")
        print(f"{'*'*60}")
        print(f"SQL: SELECT *, 1 - (embeddings <=> %s::vector) AS similarity")
        print(f"     FROM work_orders ORDER BY embeddings <=> %s::vector;")
        print(f"\nQuery vector: '{query}'")
        print(f"{'-'*60}")
        
        for model_name, sims in similarities.items():
            print(f"\n{model_name.upper()} - Ordenado como pgvector (DESC):")
            # Ordenar igual que pgvector: ORDER BY similarity DESC
            ranked_docs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            for rank, (doc_idx, similarity) in enumerate(ranked_docs, 1):
                print(f"  TOP {rank}: similarity = {similarity:.4f} | Doc {doc_idx+1}")
                print(f"         Text: '{documents[doc_idx][:80]}{'...' if len(documents[doc_idx]) > 80 else ''}'")
        
        print(f"\n📊 INTERPRETACIÓN PARA TU CÓDIGO:")
        print(f"- Similarity > 0.70: 🟢 Excelente match (equivale a tu mejor caso)")
        print(f"- Similarity 0.50-0.70: 🟡 Buen match (equivale a tus casos típicos)")  
        print(f"- Similarity < 0.50: 🔴 Match débil (equivale a casos pobres)")
    
    
    def print_results(self, query: str, documents: List[str], similarities: Dict, embeddings: Dict):
        """Imprime resultados comparativos"""
        
        # Información de procesamiento de texto
        normalize_lowercase = self.config.get('text_processing', {}).get('normalize_to_lowercase', False)
        if normalize_lowercase:
            print(f"\n📝 PROCESAMIENTO DE TEXTO:")
            print(f"✅ Normalización a minúsculas: ACTIVADA")
        else:
            print(f"\n📝 PROCESAMIENTO DE TEXTO:")
            print(f"❌ Normalización a minúsculas: DESACTIVADA")
        
        # Información de modelos
        print(f"\nINFORMACIÓN DE MODELOS:")
        print(f"- MiniLM-L6-v2: 384 dimensiones, tiempo: {embeddings.get('minilm_time', 0):.3f}s")
        print(f"- mpnet-base-v2: 768 dimensiones, tiempo: {embeddings.get('mpnet_time', 0):.3f}s")
        if 'azure_ada_time' in embeddings:
            print(f"- Azure OpenAI Ada-002: 1536 dimensiones, tiempo: {embeddings.get('azure_ada_time', 0):.3f}s")
        
        # Resultados por documento
        for i, doc in enumerate(documents):
            print(f"\n{'-'*60}")
            print(f"DOCUMENTO {i+1}:")
            print(f"'{doc[:100]}{'...' if len(doc) > 100 else ''}'")
            print(f"{'-'*60}")
            
            for model_name, sims in similarities.items():
                similarity = sims[i]
                confidence_level = self.get_confidence_level(similarity, model_name)
                print(f"{model_name.upper():12}: {similarity:.4f} ({confidence_level})")
        
        # Ranking comparison
        print(f"\n{'='*60}")
        print("RANKING COMPARISON:")
        print(f"{'='*60}")
        
        for model_name, sims in similarities.items():
            ranked_docs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            print(f"\n{model_name.upper()} ranking:")
            for rank, (doc_idx, similarity) in enumerate(ranked_docs, 1):
                confidence = self.get_confidence_level(similarity, model_name)
                print(f"  {rank}. Doc {doc_idx+1}: {similarity:.4f} ({confidence})")
    
    def get_confidence_level(self, similarity: float, model_name: str) -> str:
        """Determina nivel de confianza basado en similarity"""
        # Usar thresholds de la configuración
        thresh = self.thresholds.get(model_name, {'high': 0.60, 'medium': 0.40})
        
        if similarity >= thresh['high']:
            return "🟢 Strong Match"
        elif similarity >= thresh['medium']:
            return "🟡 Good Match" 
        else:
            return "🔴 Weak Match"

def main():
    # Cargar datos desde archivos externos
    data_loader = DataLoader()
    
    try:
        # Validar archivos de datos
        validation_results = data_loader.validate_data_files()
        if not all(validation_results.values()):
            print("❌ Algunos archivos de datos tienen errores. Creando datos de ejemplo...")
            from data_loader import create_sample_data
            create_sample_data()
            data_loader = DataLoader()  # Recargar después de crear datos
        
        # Cargar work orders y queries
        work_orders = data_loader.load_work_orders()
        test_queries = data_loader.load_test_queries()
        config = data_loader.load_config()
        
        print(f"\n📊 Datos cargados:")
        print(f"   - Work orders: {len(work_orders)}")
        print(f"   - Queries de prueba: {len(test_queries)}")
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        print("Usando datos de ejemplo por defecto...")
        
        # Datos de fallback
        work_orders = [
            "ON DESCENT ( EI-724) AUTO FLT FCU 1 FAULT. QRH RESET CARRIED OUT-IN FLIGHT. RESET SUCCESSFUL.",
            "ENG ENTRY #1 NOSE WHEEL WORN TO LIMITS",
            "POTENTIOMETER REPLACED AND SPEAKER TESTED SERVICEABLE. AIPC 23-51-08 REV 5/25 REFERS.",
            "CONFIRMED LOUDSPEAKER FUNCTION U/S. C/F FOR SPARES IAW MEL 23-51-04A"
        ]
        
        test_queries = [
            "nose wheel replacement",
            "speaker potentiometer fault",
            "FCU reset procedure",
            "loudspeaker malfunction"
        ]
        
        config = {}
    
    # Crear comparador con configuración
    output_dir = config.get('output_settings', {}).get('default_output_dir', 'embedding_reports')
    comparator = EmbeddingComparator(output_dir=output_dir, config=config)
    
    # Ejecutar comparaciones
    for query in test_queries:
        comparator.compare_models(query, work_orders)
    
    # Generar reportes
    comparator.generate_reports()
        
    print(f"\n{'='*80}")
    print("RESUMEN Y RECOMENDACIONES:")
    print(f"{'='*80}")
    print("""
    FACTORES A CONSIDERAR:
    
    1. CALIDAD:
       - mpnet-base-v2: Generalmente mejor precisión
       - MiniLM-L6-v2: Buena para casos simples
       - OpenAI Ada: Mejor calidad pero costoso
    
    2. PERFORMANCE:
       - MiniLM-L6-v2: Más rápido, menos memoria
       - mpnet-base-v2: Más lento, más memoria
       - OpenAI Ada: Latencia de API + costos
    
    3. STORAGE:
       - MiniLM-L6-v2: 384 dims = menos espacio
       - mpnet-base-v2: 768 dims = doble espacio
       - OpenAI Ada: 1536 dims = 4x espacio
    
    RECOMENDACIÓN:
    - Para POC: Prueba mpnet-base-v2 si performance no es crítica
    - Para producción: Evalúa trade-off calidad vs velocidad
    - OpenAI Ada: Solo si calidad es crítica y budget lo permite
    """)

if __name__ == "__main__":
    main()
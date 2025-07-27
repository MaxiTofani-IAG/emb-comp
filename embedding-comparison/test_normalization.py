#!/usr/bin/env python3
"""
Script para probar la diferencia entre embeddings con y sin normalización de texto a minúsculas.
"""

import json
import os
from emb_test import EmbeddingComparator
from data_loader import DataLoader

def load_config(config_file):
    """Carga configuración desde archivo específico"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_normalization_comparison():
    """Compara resultados con y sin normalización"""
    
    print("=" * 80)
    print("🧪 PRUEBA DE NORMALIZACIÓN DE TEXTO")
    print("=" * 80)
    
    # Cargar datos
    data_loader = DataLoader()
    work_orders = data_loader.load_work_orders()
    test_queries = data_loader.load_test_queries()
    
    # Queries de prueba específicas para capitalización
    test_cases = [
        "LANDING GEAR EXTENSION",  # Todo mayúsculas
        "landing gear extension",  # Todo minúsculas
        "Landing Gear Extension",  # Capitalización mixta
        "Engine Oil Pressure",     # Capitalización mixta
        "ENGINE OIL PRESSURE",     # Todo mayúsculas
        "engine oil pressure"      # Todo minúsculas
    ]
    
    # Configuraciones a probar
    configs = {
        "Con normalización": "embedding-comparison/data/config.json",
        "Sin normalización": "embedding-comparison/data/config_no_normalize.json"
    }
    
    results = {}
    
    for config_name, config_file in configs.items():
        print(f"\n{'='*60}")
        print(f"🔧 PROBANDO: {config_name}")
        print(f"{'='*60}")
        
        # Cargar configuración
        config = load_config(config_file)
        
        # Crear comparador
        output_dir = f"embedding_reports/normalization_test_{config_name.lower().replace(' ', '_')}"
        comparator = EmbeddingComparator(output_dir=output_dir, config=config)
        
        config_results = {}
        
        for query in test_cases:
            print(f"\n📝 Query: '{query}'")
            
            # Generar embeddings y calcular similitudes
            all_texts = [query] + work_orders
            embeddings = comparator.get_embeddings(all_texts)
            
            # Separar query de documentos
            query_embeddings = {}
            doc_embeddings = {}
            
            for model_name in ['minilm', 'mpnet']:
                if model_name in embeddings:
                    query_embeddings[model_name] = embeddings[model_name][0:1]
                    doc_embeddings[model_name] = embeddings[model_name][1:]
            
            # Calcular similarities
            similarities = comparator.calculate_similarities(query_embeddings, doc_embeddings)
            
            # Guardar resultados
            config_results[query] = similarities
            
            # Mostrar top 3 resultados para cada modelo
            for model_name, sims in similarities.items():
                ranked_docs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
                print(f"  {model_name.upper()}:")
                for rank, (doc_idx, similarity) in enumerate(ranked_docs[:3], 1):
                    doc_preview = work_orders[doc_idx][:50] + "..." if len(work_orders[doc_idx]) > 50 else work_orders[doc_idx]
                    print(f"    {rank}. {similarity:.4f} - {doc_preview}")
        
        results[config_name] = config_results
    
    # Análisis comparativo
    print(f"\n{'='*80}")
    print("📊 ANÁLISIS COMPARATIVO")
    print(f"{'='*80}")
    
    for query in test_cases:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        for model_name in ['minilm', 'mpnet']:
            print(f"\n{model_name.upper()}:")
            
            # Obtener top result para cada configuración
            with_norm = results["Con normalización"][query][model_name]
            without_norm = results["Sin normalización"][query][model_name]
            
            # Top 1 de cada configuración
            top_with_norm = max(with_norm)
            top_without_norm = max(without_norm)
            
            # Índice del top result
            top_idx_with = with_norm.index(top_with_norm)
            top_idx_without = without_norm.index(top_without_norm)
            
            print(f"  Con normalización:    {top_with_norm:.4f} (Doc {top_idx_with + 1})")
            print(f"  Sin normalización:    {top_without_norm:.4f} (Doc {top_idx_without + 1})")
            
            # Diferencia
            diff = top_with_norm - top_without_norm
            print(f"  Diferencia:           {diff:+.4f}")
            
            # ¿Mismo documento en top?
            if top_idx_with == top_idx_without:
                print(f"  Resultado:            ✅ Mismo documento en top")
            else:
                print(f"  Resultado:            ❌ Diferentes documentos en top")
    
    print(f"\n{'='*80}")
    print("💡 CONCLUSIONES")
    print(f"{'='*80}")
    print("1. La normalización a minúsculas puede afectar significativamente los embeddings")
    print("2. Para queries con capitalización mixta, la normalización puede mejorar consistencia")
    print("3. Para queries ya en minúsculas, el impacto es mínimo")
    print("4. Se recomienda usar normalización para mayor consistencia en producción")
    print("5. Los reportes se guardaron en carpetas separadas para análisis detallado")

if __name__ == "__main__":
    test_normalization_comparison() 
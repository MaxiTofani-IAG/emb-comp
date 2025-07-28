#!/usr/bin/env python3
"""
Módulo para cargar datos de entrada desde archivos externos
"""
import json
import os
from typing import List, Dict, Any
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        """
        Inicializa el cargador de datos
        
        Args:
            data_dir: Directorio donde están los archivos de datos
        """
        self.data_dir = Path(data_dir)
        
        # Verificar que el directorio existe
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directorio de datos no encontrado: {data_dir}")
    
    def load_work_orders(self, filename: str = "work_orders.json", normalize_to_lowercase: bool = False) -> List[str]:
        """
        Carga los work orders desde un archivo JSON
        
        Args:
            filename: Nombre del archivo JSON con los work orders
            normalize_to_lowercase: Si True, convierte todos los work orders a minúsculas
            
        Returns:
            Lista de strings con los work orders
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo de work orders no encontrado: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                work_orders = json.load(f)
            
            if not isinstance(work_orders, list):
                raise ValueError("El archivo debe contener una lista de work orders")
            
            # Validar que todos los elementos son strings
            for i, wo in enumerate(work_orders):
                if not isinstance(wo, str):
                    raise ValueError(f"Work order {i} no es un string: {type(wo)}")
            
            # Aplicar normalización si se solicita
            if normalize_to_lowercase:
                work_orders = [wo.lower() for wo in work_orders]
                print(f"✅ Cargados {len(work_orders)} work orders desde {filename} (normalizados a minúsculas)")
            else:
                print(f"✅ Cargados {len(work_orders)} work orders desde {filename}")
            
            return work_orders
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON en {filename}: {e}")
    
    def load_test_queries(self, filename: str = "test_queries.json") -> List[str]:
        """
        Carga las queries de prueba desde un archivo JSON
        
        Args:
            filename: Nombre del archivo JSON con las queries
            
        Returns:
            Lista de strings con las queries de prueba
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo de queries no encontrado: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            
            if not isinstance(queries, list):
                raise ValueError("El archivo debe contener una lista de queries")
            
            # Validar que todos los elementos son strings
            for i, query in enumerate(queries):
                if not isinstance(query, str):
                    raise ValueError(f"Query {i} no es un string: {type(query)}")
            
            print(f"✅ Cargadas {len(queries)} queries de prueba desde {filename}")
            return queries
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON en {filename}: {e}")
    
    def load_config(self, filename: str = "config.json") -> Dict[str, Any]:
        """
        Carga la configuración desde un archivo JSON
        
        Args:
            filename: Nombre del archivo JSON con la configuración
            
        Returns:
            Diccionario con la configuración
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if not isinstance(config, dict):
                raise ValueError("El archivo debe contener un diccionario de configuración")
            
            print(f"✅ Configuración cargada desde {filename}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON en {filename}: {e}")
    
    def save_work_orders(self, work_orders: List[str], filename: str = "work_orders.json") -> None:
        """
        Guarda work orders en un archivo JSON
        
        Args:
            work_orders: Lista de work orders a guardar
            filename: Nombre del archivo donde guardar
        """
        file_path = self.data_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(work_orders, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Work orders guardados en {filename}")
            
        except Exception as e:
            raise ValueError(f"Error al guardar work orders en {filename}: {e}")
    
    def save_test_queries(self, queries: List[str], filename: str = "test_queries.json") -> None:
        """
        Guarda queries de prueba en un archivo JSON
        
        Args:
            queries: Lista de queries a guardar
            filename: Nombre del archivo donde guardar
        """
        file_path = self.data_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Queries guardadas en {filename}")
            
        except Exception as e:
            raise ValueError(f"Error al guardar queries en {filename}: {e}")
    
    def list_available_files(self) -> List[str]:
        """
        Lista todos los archivos JSON disponibles en el directorio de datos
        
        Returns:
            Lista de nombres de archivos JSON
        """
        json_files = list(self.data_dir.glob("*.json"))
        return [f.name for f in json_files]
    
    def validate_data_files(self) -> Dict[str, bool]:
        """
        Valida que todos los archivos de datos necesarios existan y sean válidos
        
        Returns:
            Diccionario con el estado de validación de cada archivo
        """
        validation_results = {}
        
        # Verificar work orders
        try:
            self.load_work_orders()
            validation_results["work_orders.json"] = True
        except Exception as e:
            print(f"❌ Error en work_orders.json: {e}")
            validation_results["work_orders.json"] = False
        
        # Verificar test queries
        try:
            self.load_test_queries()
            validation_results["test_queries.json"] = True
        except Exception as e:
            print(f"❌ Error en test_queries.json: {e}")
            validation_results["test_queries.json"] = False
        
        # Verificar config
        try:
            self.load_config()
            validation_results["config.json"] = True
        except Exception as e:
            print(f"❌ Error en config.json: {e}")
            validation_results["config.json"] = False
        
        return validation_results


def create_sample_data():
    """
    Función de utilidad para crear datos de ejemplo si no existen
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Work orders de ejemplo
    sample_work_orders = [
        "ON DESCENT ( EI-724) AUTO FLT FCU 1 FAULT. QRH RESET CARRIED OUT-IN FLIGHT. RESET SUCCESSFUL.",
        "ENG ENTRY #1 NOSE WHEEL WORN TO LIMITS",
        "POTENTIOMETER REPLACED AND SPEAKER TESTED SERVICEABLE. AIPC 23-51-08 REV 5/25 REFERS.",
        "CONFIRMED LOUDSPEAKER FUNCTION U/S. C/F FOR SPARES IAW MEL 23-51-04A"
    ]
    
    # Queries de ejemplo
    sample_queries = [
        "nose wheel replacement",
        "speaker potentiometer fault",
        "FCU reset procedure",
        "loudspeaker malfunction"
    ]
    
    # Configuración de ejemplo
    sample_config = {
        "models": {
            "minilm": {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "description": "Modelo rápido y eficiente para casos simples"
            },
            "mpnet": {
                "name": "all-mpnet-base-v2",
                "dimensions": 768,
                "description": "Modelo balanceado entre velocidad y precisión"
            }
        },
        "confidence_thresholds": {
            "minilm": {"high": 0.60, "medium": 0.40},
            "mpnet": {"high": 0.65, "medium": 0.45}
        },
        "output_settings": {
            "default_output_dir": "embedding_reports",
            "include_timestamp": True,
            "generate_all_reports": True
        }
    }
    
    # Crear archivos si no existen
    loader = DataLoader()
    
    if not (data_dir / "work_orders.json").exists():
        loader.save_work_orders(sample_work_orders)
    
    if not (data_dir / "test_queries.json").exists():
        loader.save_test_queries(sample_queries)
    
    if not (data_dir / "config.json").exists():
        with open(data_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print("✅ Datos de ejemplo creados en el directorio 'data'")


if __name__ == "__main__":
    # Ejecutar como script independiente para crear datos de ejemplo
    create_sample_data() 
from pathlib import Path
import joblib
import json
import time


import os
from extractors.gliner_dataset_extractor import GlinerDatasetExtractor
from extractors.gliner_metric_extractor import GlinerMetricExtractor
from extractors.qwen_metric_extractor import QwenMetricExtractor
from extractors.llama_dataset_extractor import LlamaDatasetExtractor
from extractors.qwen_dataset_extractor import QwenDatasetExtractor
from extractors.gliner_extractor import GlinerExtractor
from extractors.llm_extractors import LlamaExtractor, QwenExtractor

TAXONOMY_CLASSIFIER_FILE = "taxonomy_classifier.joblib"



class ModelCardGenerator:
    def __init__(self):
        

        # Load the classifier (TF-IDF + SVC)
        pipeline_path = Path(os.getcwd()) / "utils" / TAXONOMY_CLASSIFIER_FILE
        self._classification_pipeline = joblib.load(pipeline_path)

        # Load llama, qwen and gliner
        self._gliner = GlinerExtractor()
        self._qwen = QwenExtractor()
        self._llama = LlamaExtractor()
        self._gliner_dataset_extractor = GlinerDatasetExtractor()
        self._gliner_metric_extractor = GlinerMetricExtractor()
        self._qwen_metric_extractor = QwenMetricExtractor()
        self._llama_dataset_extractor = LlamaDatasetExtractor()
        self._qwen_dataset_extractor = QwenDatasetExtractor()

        print("Generator ready!")
        
        
    def _extract_classification(self, text):
        predicted_class = self._classification_pipeline.predict([text]).tolist()
        return predicted_class[0]

    def evaluate_pipeline(self, pipeline_mode, paper):
        init_time = time.time()
        abstract = paper.get("abstract")
        full_text = paper.get("full_text")
        sections = paper.get("sections")
        

        
        if pipeline_mode == "efficient": 
            model_name_prediction = self._gliner.extract(abstract, ["model"])
            datasets_prediction = self._gliner_dataset_extractor.extract(paper.get("full_text"))
            metrics_prediction = self._gliner_metric_extractor.extract(paper.get("full_text"))
        else:
            model_name_prediction = self._llama.extract(paper.get("full_text"), question = "What is the name of the model presented in this paper?")
            
            metrics_text = f"{paper.get('abstract', '')}\n\n{paper.get('sections', '')}"
            tablas_html = []
            json_crudo_tablas = paper.get("tables", {})

            for page_data in json_crudo_tablas.get("results", []):
                for table_dict in page_data.get("tables", []):
                    if "html" in table_dict:
                        tablas_html.append(table_dict["html"])

            
            datasets_prediction = self._qwen_dataset_extractor.extract("", tablas_html)
            
            metrics_prediction = self._qwen_metric_extractor.extract("", tablas_html)
            

        
        tasks_prediction = self._llama.extract(abstract, question =  "What are the tasks addressed in this paper?")
        category_prediction = self._extract_classification(abstract)
        implementation_text =  f"{paper.get('abstract', '')}\n\n{paper.get('sections', '')}"
        implementation_prediction = self._llama.extract(paper.get("full_text"), question = "Return the URL link for the paper implementation (like GitHub). Return an empty list if not found. DO NOT  invent links")
        

        
        end_time = time.time()
        total_time = end_time - init_time
        result = {
            "time": total_time,
            "model_name_prediction": model_name_prediction,
            "tasks_prediction": tasks_prediction,
            "category_prediction": category_prediction,
            "implementation_prediction": implementation_prediction,
            "datasets_prediction": datasets_prediction,
            "metrics_prediction": metrics_prediction,
            "title": paper.get("title"),
            "local_xml_path": paper.get("local_xml_path"),
            "pwc_abstract": paper.get("pwc_abstract"),
            "abstract": paper.get("abstract"),
            "full_text": paper.get("full_text"),
            "sections": paper.get("sections"),
        }
        return result, total_time








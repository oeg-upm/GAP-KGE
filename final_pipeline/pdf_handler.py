import json
from pathlib import Path
import requests
from final_pipeline.model_card_generation_pipeline import ModelCardGenerator
from utils.XMLParser import XMLParser

from parsers.scipdf_parser import SciPdfParser
from parsers.lightocr_parser import LightOcrParser


class PDFHandler:
    def __init__(self):
        self._pdf_dir = Path("data") / "raw" / "pdfs"
        self._xml_dir = Path("data") / "interim" / "scipdf_xml"
        self._json_dir = Path("data") / "interim" / "lightocr_json"

        self._scipdf_parser = SciPdfParser()
        self._lightocr_parser = LightOcrParser()

        self._mcg = ModelCardGenerator()
        


    def _extract_id_from_url(self, url):
        """Extrae un ID único para nombrar los archivos (ej. de arXiv)."""
        return url.split("/")[-1].replace(".pdf", "")
    
    def _download_pdf(self, pdf_url, save_path):
        
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                pdf_file.write(chunk)

    def _process_with_scipdf(self, pdf_path, xml_save_path):
        self._scipdf_parser.process(pdf_path, xml_save_path)

    def _process_with_lightocr(self, pdf_path, json_save_path):
        self._lightocr_parser.process(pdf_path, json_save_path)

    def _extract_values(self, xml_path, json_path):
        self._xml_parser = XMLParser(xml_path)
        title = self._xml_parser.get_title()
        arxiv_id = self._xml_parser.get_arxiv_id()
        full_text = self._xml_parser.get_full_text()
        abstract = self._xml_parser.get_abstract()
        authors = self._xml_parser.get_authors()
        section_dict = self._xml_parser.get_sections(target_sections=['Experiments','Evaluation','Results'])
        sections = "\n\n".join(section_dict.values())

        with open(json_path, 'r', encoding='utf-8') as f:
            tables = json.load(f)
        

        
        extracted_data = {
            "title": title,
            "arxiv_id": arxiv_id,
            "authors": authors,
            "abstract": abstract,
            "full_text": full_text,
            "sections": sections,
            "tables": tables,
        }
        return extracted_data
    def _process_with_lightocr_mock(self, pdf_path, json_save_path):

        """
        Método de prueba para simular la extracción de tablas sin ejecutar el modelo real.
        Crea un archivo JSON de ejemplo con datos ficticios.
        """
        mock_data = {
            "documents": [
                {
                    "tables": [
                        {
                            "caption": "Tabla de ejemplo",
                            "data": [
                                ["FB15-k", "WN18", "WN18RR"],
                                ["Dato 1", "Dato 2", "Dato 3"],
                                ["Dato A", "Dato B", "Dato C"]
                            ]
                        }
                    ]
                }
            ]
        }

        # Nos aseguramos de que la carpeta de destino existe
        json_save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, indent=2, ensure_ascii=False)

    def handle_pdf(self, pdf_url):

        paper_id = self._extract_id_from_url(pdf_url)
        pdf_path = self._pdf_dir / f"{paper_id}.pdf"
        xml_path = self._xml_dir / f"{paper_id}.xml"
        json_path = self._json_dir / f"{paper_id}.json"


        print(f"Downloading PDF from {pdf_url} to {pdf_path}")
        self._download_pdf(pdf_url, pdf_path)
        self._process_with_scipdf(pdf_path, xml_path)
        self._process_with_lightocr_mock(pdf_path, json_path)


        extracted_data = self._extract_values(xml_path, json_path)
        modelcard = self._mcg.generate_modelcard(extracted_data)
        return modelcard
        
        

        
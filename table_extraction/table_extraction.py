import json
import deepdoctection as dd
import time
import io
import re
import pandas as pd
from pathlib import Path
import table_extraction_utils as teu

#Global variables
METRICS_LIST = ["Accuracy", "MRR", "Hits@1", "Hits@3", "Hits@10", "F1-Score"]

DATASET_LIST = ["WN18","Dataset-B1","FB15k", "Dataset_A2", "WD"]

path_pdf = r"pdfs_prueba/HyperKG- Hyperbolic Knowledge Graph Embeddings for Knowledge Base Completion.pdf"



#Only looks for the predefined set of metrics specified in METRICS_LIST
def metric_extraction_from_table(html_str,lista_objetivo):
    metricas_encontradas = set()

    try:
        dfs = pd.read_html(io.StringIO(html_str), header=[0, 1, 2])
    except:
        return []

    if not dfs: return []
    df = dfs[0]

    for col_tuple in df.columns:
        texto_cabecera = " ".join([str(x) for x in col_tuple if "Unnamed" not in str(x) and str(x) != "nan"])

        metrica = teu.normalizar_texto(texto_cabecera)
        if metrica:
            metricas_encontradas.add(metrica)

    try:
        primera_columna = df.iloc[:, 0].astype(str).tolist()
        for celda in primera_columna:
            metrica = teu.normalizar_texto(celda)
            if metrica:
                metricas_encontradas.add(metrica)
    except:
        pass

    metricas_presentes = list(metricas_encontradas.intersection(set(lista_objetivo)))

    return metricas_presentes

#Looks for the datasets in DATASET_LIST in the input HTML
def search_datasets_in_tables_html(html_list):
    possible_datasets = DATASET_LIST.copy()
    found_datasets = []
    for html in html_list:
        for dataset_name in possible_datasets:
            clean_dataset_name = "".join(filter(str.isalnum, dataset_name))
            regex_pattern = r"[\W_]*".join(re.escape(c) for c in clean_dataset_name)
            final_pattern = r"\b" + regex_pattern

            if re.search(final_pattern, html, re.IGNORECASE) and dataset_name not in found_datasets:
                found_datasets.append(dataset_name)


    found_datasets = []
    search_datasets_in_tables_html()
    return found_datasets

#Extracts pairings model-metric-dataset from the table:
def extract_values_from_paper(tables_json_route):
    with open(tables_json_route, "r", encoding="utf-8") as j:
        tables_json = json.load(j)

    all_tables_data = []
    id = 0
    for page_data in tables_json['results']:
        for table in page_data['tables']:
            try:
                extracted_data = teu.extract_values_from_html_table(table['html'])
                id += 1
                table_object = {
                    "id": id,
                    "page": page_data['page'],
                    "num_values": len(extracted_data),
                    "data": extracted_data
                }
                all_tables_data.append(table_object)
            except Exception as e:
                print(f"   ⚠️ SALTANDO TABLA: Estructura errónea o compleja.")
                print(f"      └── Causa: {e}")
    values_output_json = {
        "file_name": tables_json["file_name"],
        "total_num_tables": all_tables_data.__len__(),
        "table_values": all_tables_data
    }

    with open("values.json", "w", encoding="utf-8") as f:
        json.dump(values_output_json, f, indent=4, ensure_ascii=False)
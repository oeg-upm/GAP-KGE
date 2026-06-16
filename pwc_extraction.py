import os
from contextlib import nullcontext

import requests
import time
from pathlib import Path
import pandas as pd
import arxiv
import json
import re
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def limpiar_url_arxiv(url):
    """
    Elimina la versión de ArXiv (v1, v2...) y normaliza la URL
    para asegurar coincidencias entre el CSV y el JSON.
    """
    if not isinstance(url, str):
        return ""
    # 1. Quitar espacios y convertir a minúsculas
    url = url.strip().lower()
    # 2. Eliminar la versión al final (ej: v1, v2, v33)
    url = re.sub(r'v\d+$', '', url)
    # 3. Eliminar barras finales si las hay
    return url.rstrip('/')

def normalize_string(s):
    """
    Limpia el texto convirtiéndolo a minúsculas y eliminando todo lo que
    no sea una letra o un número. Así aseguramos el match.
    """
    if not s:
        return ""
    return re.sub(r'\W+', '', str(s).lower())


def extraer_id_desde_url(url):
    """
    Extrae el ID de ArXiv de casi cualquier formato de URL de ArXiv.
    Soporta: /abs/, /pdf/, /ftp/, versiones (v1), y extensiones (.pdf).
    """
    if not isinstance(url, str): return None

    # Patrón para IDs modernos (YYMM.NNNNN)
    modern_pattern = r'(\d{4}\.\d{4,5})'
    # Patrón para IDs antiguos (cat-name/YYMMNNN)
    old_pattern = r'([a-z\-]+(?:\.[a-z]{2})?/\d{7})'

    for pattern in [modern_pattern, old_pattern]:
        match = re.search(pattern, url.lower())
        if match:
            return match.group(1)
    return None


def buscar_id_por_titulo(titulo):
    """
    Consulta la API de ArXiv para encontrar el ID dado un título.
    """
    if not titulo or len(titulo) < 10: return None

    try:
        search = arxiv.Search(query=f'ti:"{titulo}"', max_results=1)
        for result in search.results():
            # Retorna el ID limpio (ej: 2103.12345)
            return extraer_id_desde_url(result.entry_id)
    except:
        return None
    return None

def generate_pwc_dataset(output_file="pwc_pwa.json"):
    target_tasks = ['knowledge graph embedding', 'knowledge graph embeddings', 'link prediction', 'knowledge graph completion']
    print("Cargando el dataset pwc/papers-with-abstracts...")
    dataset = load_dataset("pwc-archive/papers-with-abstracts", split='train')

    target_tasks_lower = {task.strip().lower() for task in target_tasks}
    # Campos que queremos conservar en nuestro dataset
    fields_to_keep = [
        "paper_url", "arxiv_id", "title", "abstract", "short_abstract",
        "url_abs", "url_pdf", "proceeding", "authors", "tasks", "methods"
    ]
    filtered_data = []
    # Iteramos sobre el dataset
    for entry in dataset:
        paper_tasks = entry.get("tasks", [])
        if paper_tasks is None:
            paper_tasks = []
        if any(task.lower() in target_tasks_lower for task in paper_tasks):
            filtered_sample = {field: entry.get(field) for field in fields_to_keep}
            filtered_data.append(filtered_sample)

    print(f"Guardando {len(filtered_data)} muestras en {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print("¡Generada versión base del dataset!")

def add_repo_url(input_json_file):
    print(f"Cargando el archivo local: {input_json_file}...")
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            my_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {input_json_file}.")
        return

    print(f"Se han cargado {len(my_dataset)} papers locales.")

    print("Cargando enlaces entre papers y código desde Hugging Face...")
    links_ds = load_dataset("pwc-archive/links-between-paper-and-code", split='train')

    print("Construyendo el índice de repositorios...")
    repo_map = {}
    for item in links_ds:
        url_abs = item.get("paper_url_abs")
        repo_url = item.get("repo_url")
        mentioned= item.get("mentioned_in_paper")

        if mentioned and url_abs and repo_url:
            if url_abs not in repo_map:
                repo_map[url_abs] = []
            repo_map[url_abs].append(repo_url)

    print("Añadiendo el campo 'repo_url' a tus datos...")
    for paper in my_dataset:
        current_url_abs = paper.get("url_abs")
        paper["paper_repo"] = repo_map.get(current_url_abs, [])

    # 5. Guardar el resultado
    print(f"Guardando el dataset actualizado en {input_json_file}...")
    with open(input_json_file, 'w', encoding='utf-8') as f:
        json.dump(my_dataset, f, ensure_ascii=False, indent=4)

    print("URLs de repositorios añadidas con éxito")

def add_dataset_metrics_info(input_json_file):
    target_tasks = ['knowledge graph embedding', 'knowledge graph embeddings', 'link prediction', 'knowledge graph completion']
    print(f"📖 Cargando tu JSON...")
    with open(input_json_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    url_to_paper = {p.get('url_abs'): p for p in papers if p.get('url_abs')}
    urls_interes = set(url_to_paper.keys())


    tasks_to_find = {t.lower().strip() for t in target_tasks}
    tasks_found = set()

    print("🚀 Cargando evaluación en memoria (este proceso es largo)...")
    eval_ds = load_dataset("pwc-archive/evaluation-tables", split="train")

    print(f"🔍 Buscando exactamente {len(tasks_to_find)} tareas...")

    for row in tqdm(eval_ds):
        if not tasks_to_find:
            break

        t_name = (row.get('task') or "").lower()

        if t_name in target_tasks:  # Solo si está en nuestra lista
            tasks_to_find.discard(t_name)
            tasks_found.add(t_name)

            datasets_list = row.get('datasets') or []
            for ds_entry in datasets_list:
                ds_name = ds_entry.get('dataset')
                sota_data = ds_entry.get('sota')
                if not sota_data: continue

                rows = sota_data.get('rows') or []
                for r in rows:
                    p_url = r.get('paper_url')
                    if p_url in urls_interes:
                        paper_obj = url_to_paper[p_url]

                        paper_obj['Metrics']=set()
                        #if 'Datasets' not in paper_obj: paper_obj['Datasets'] = set()
                        #if 'Metrics' not in paper_obj: paper_obj['Metrics'] = set()

                        #paper_obj['Datasets'].add(ds_name)

                        metrics_dict = r.get('metrics') or {}
                        for metric_key, metric_value in metrics_dict.items():
                            # Comprobamos que el valor exista y no sea una cadena de texto "None" o vacía
                            if metric_value is not None:
                                val_str = str(metric_value).strip().lower()
                                if val_str and val_str != "none":
                                    # Si pasa el filtro, añadimos el nombre real de la métrica original
                                    paper_obj['Metrics'].add(metric_key)

                        model_name = r.get('model_name') or []
                        paper_obj['model_name']=model_name

    # 4. Consolidación y Guardado
    print(f"\n✅ Búsqueda terminada. Tareas localizadas: {len(tasks_found)}/{len(target_tasks)}")

    for p in papers:
        p['Datasets'] = list(p.get('Datasets', []))
        p['Metrics'] = list(p.get('Metrics', []))

    with open(input_json_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)

    if tasks_to_find:
        print(f"⚠️ No se encontraron estas tareas en el dataset SOTA: {tasks_to_find}")

def add_model_type(path_json, path_csv):
    df = pd.read_csv(path_csv)
    categoria_map = {}

    print("Normalizando IDs del CSV...")
    for _, row in df.iterrows():
        id_csv = extraer_id_desde_url(row['url'])
        if id_csv:
            categoria_map[id_csv] = row['category']

    # 2. Cargar JSON
    with open(path_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    encontrados = 0

    for paper in tqdm(papers):
        id_final = extraer_id_desde_url(paper.get('arxiv_id'))
        if not id_final:
            id_final = extraer_id_desde_url(paper.get('url_abs'))

        if id_final in categoria_map:
            paper['model_category'] = categoria_map[id_final]
            encontrados += 1
        else:
            paper['model_category'] = None

    # 3. Guardar
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Se han categorizado {encontrados} papers .")


def download_pdfs(path_json, folder_dest):
    output_dir = Path(folder_dest)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📖 Cargando dataset desde {path_json}...")
    with open(path_json, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    papers_con_pdf = [p for p in papers if p.get('url_pdf')]
    print(f"📥 Preparado para descargar {len(papers_con_pdf)} archivos PDF.")

    # Headers para evitar que el servidor nos bloquee por parecer un bot básico
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    descargados = 0
    errores = 0

    for paper in tqdm(papers_con_pdf, desc="Descargando PDFs"):
        url = paper['url_pdf']

        # Generar un nombre de archivo seguro
        # Prioridad: arxiv_id > title > parte de la URL
        file_id = paper.get('arxiv_id') or paper.get('title')[:50]
        # Limpiar caracteres no permitidos en nombres de archivos
        file_name = "".join([c for c in str(file_id) if c.isalnum() or c in (' ', '.', '-', '_')]).strip()
        file_name = file_name.replace(' ', '_') + ".pdf"

        file_path = output_dir / file_name

        # Si el archivo ya existe, saltamos (ideal para retomar descargas interrumpidas)
        if file_path.exists():
            continue

        try:
            # Realizar la petición
            response = requests.get(url, headers=headers, timeout=20)

            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                descargados += 1
                paper['local_pdf_path']=file_path
            else:
                errores += 1
                paper['local_pdf_path']=None

            time.sleep(1.0)

        except Exception as e:
            errores += 1

    print(f"✅ Descargados con éxito: {descargados}")
    print(f"❌ Fallidos: {errores}")



def make_xml_files(path_json_input, xml_dir, grobid_url="http://localhost:8070"):
    """
    Procesa los PDFs locales usando el backend de SciPDF (GROBID),
    guarda el XML resultante y actualiza el archivo JSON.
    """
    os.makedirs(xml_dir, exist_ok=True)

    print(f"⏰ ASEGÚRATE DE QUE LA IMAGEN DOCKER DE GROBID ESTÁ EJECUTANDO!")

    print(f"📖 Cargando dataset desde {path_json_input}...")
    with open(path_json_input, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # Verificar si el servidor GROBID está activo antes de lanzar el bucle
    try:
        response = requests.get(f"{grobid_url}/api/isalive")
        if response.status_code != 200:
            print("❌ El servidor GROBID no está respondiendo correctamente.")
            return
    except requests.exceptions.ConnectionError:
        print(f"❌ No se pudo conectar a GROBID en {grobid_url}. ¿Está corriendo el servicio/Docker?")
        return

    print("🚀 Iniciando el procesamiento de PDFs con SciPDF (puede tardar un rato)...")
    procesados = 0
    errores = 0

    for paper in tqdm(papers):
        pdf_path = paper.get('local_pdf_path')

        # Saltamos el registro si no tiene un PDF local asociado
        if not pdf_path or not os.path.exists(pdf_path):
            paper['local_xml_path'] = None
            continue

        # Generamos un nombre seguro para el XML usando el arxiv_id o el índice del bucle
        id_documento = paper.get('arxiv_id') or ''.join(e for e in paper.get('title', 'doc') if e.isalnum())[:30]
        xml_filename = f"{id_documento}.tei.xml"
        xml_output_path = os.path.join(xml_dir, xml_filename)

        # Si el archivo XML ya existe de una ejecución previa, saltamos el procesamiento para ahorrar tiempo
        if os.path.exists(xml_output_path):
            paper['local_xml_path'] = xml_output_path
            procesados += 1
            continue

        # 3. Enviar el PDF al endpoint de GROBID (Proceso nativo de SciPDF)
        endpoint = f"{grobid_url}/api/processFulltextDocument"

        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': f}
                # Configuramos las opciones habituales de SciPDF (extraer texto completo y figuras)
                data = {
                    'generateTeiIds': '1',
                    'consolidateHeader': '1',
                    'consolidateConclusion': '1'
                }

                res = requests.post(endpoint, files=files, data=data, timeout=120)

                if res.status_code == 200:
                    with open(xml_output_path, 'w', encoding='utf-8') as xml_file:
                        xml_file.write(res.text)

                    paper['local_xml_path'] = xml_output_path
                    procesados += 1
                else:
                    print(f"⚠️ Error al procesar {pdf_path}: Código de estado {res.status_code}")
                    paper['local_xml_path'] = None
                    errores += 1

        except Exception as e:
            print(f"⚠️ Error crítico procesando {pdf_path}: {str(e)}")
            paper['local_xml_path'] = None
            errores += 1

    # 4. Guardar el JSON enriquecido final
    with open(path_json_input, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)

    print(f"\n✨ ¡Pipeline Completado!")
    print(f"📊 PDFs parseados correctamente: {procesados}")
    print(f"⚠️ Fallas o errores: {errores}")


import json
import os


def filtrar_dataset_pdf(ruta_entrada, ruta_salida):
    """
    Elimina los elementos del dataset que no tengan un PDF válido.
    Guarda el resultado en un nuevo archivo JSON.
    """
    try:
        # 1. Leer el archivo original
        with open(ruta_entrada, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)

        if not isinstance(datos, list):
            print("Error: El JSON principal debe ser una lista de papers.")
            return

        # Guardamos la cantidad inicial para el reporte final
        total_original = len(datos)

        # 2. Filtrar los elementos
        datos_filtrados = []
        for paper in datos:
            # Comprobamos si el campo existe y tiene texto
            if "local_pdf_path" in paper and paper["local_pdf_path"] is not None and paper["local_pdf_path"] != "":
                ruta_pdf = paper["local_pdf_path"]

                # --- OPCIÓN A: Solo comprobar que el campo tenga valor en el JSON ---
                datos_filtrados.append(paper)

                # --- OPCIÓN B: Comprobar si el archivo REALMENTE existe en tu ordenador ---
                # (Si quieres usar esta opción, borra la línea de arriba "datos_filtrados.append(paper)"
                # y quita las almohadillas '#' de las siguientes dos líneas):
                # if os.path.exists(ruta_pdf):
                #     datos_filtrados.append(paper)

        # 3. Guardar el nuevo dataset limpio
        with open(ruta_salida, 'w', encoding='utf-8') as archivo_nuevo:
            # indent=4 hace que el JSON sea legible para humanos
            # ensure_ascii=False mantiene bien los acentos y caracteres especiales
            json.dump(datos_filtrados, archivo_nuevo, ensure_ascii=False, indent=4)

        # 4. Mostrar estadísticas del proceso
        eliminados = total_original - len(datos_filtrados)
        print("¡Filtrado completado con éxito!")
        print(f" - Total de papers iniciales: {total_original}")
        print(f" - Papers eliminados (sin PDF): {eliminados}")
        print(f" - Papers guardados en el nuevo archivo: {len(datos_filtrados)}")

    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo '{ruta_entrada}'.")
    except json.JSONDecodeError:
        print(f"Error: El archivo '{ruta_entrada}' no tiene un formato JSON válido.")

if __name__ == "__main__":
    print('Extrayendo datos de pwc-archive')
    generate_pwc_dataset('pwc_dataset.json')
    add_repo_url('pwc_dataset.json')
    add_dataset_metrics_info('pwc_dataset.json')
    download_pdfs('pwc_dataset.json','../data/pdf_files')
    make_xml_files('pwc_dataset.json','../data/xml_files')

    add_model_type('pwc_dataset.json','../data/index.csv')
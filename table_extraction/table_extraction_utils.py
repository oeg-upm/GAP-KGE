import json
import deepdoctection as dd
import time
import io
import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

config_overwrite = ["USE_OCR=False", "USE_PDF_MINER=True"]
analyzer = dd.get_dd_analyzer(config_overwrite=config_overwrite)

PATRONES = {
    "Accuracy": re.compile(r"\b(?:acc(?:uracy)?|c\.?a\.?)\b", re.IGNORECASE),
    "MRR": re.compile(r"\b(?:mrr|mean\s+reciprocal\s+rank)\b", re.IGNORECASE),
    "F1-Score": re.compile(r"\bf-?1(?:-?(?:score|measure))?\b", re.IGNORECASE),

}
REGEX_HITS = re.compile(r"\b(?:hits?|h)(?:\s*@\s*|\s+at\s+|\s*)(\d{1,3})\b", re.IGNORECASE)


def read_table_output_json(path_json):
    with open(path_json, "r", encoding="utf-8") as j:
        tables_json = json.load(j)

    tables_html = []

    for page_data in tables_json['results']:
        for table in page_data['tables']:
            tables_html.append(table['html'])

    return tables_html


def normalizar_texto(texto):
    """Convierte texto sucio ('H@ 10') en métrica canónica ('Hits@10')."""
    texto = str(texto).strip()

    match = REGEX_HITS.search(texto)
    if match:
        return f"Hits@{match.group(1)}"

    for nombre_canonico, patron in PATRONES.items():
        if patron.search(texto):
            return nombre_canonico

    return None

    # Pruebas\pdfs_prueba\


def extract_table_deepdoctection(path_pdf):
    start_time = time.time()

    df = analyzer.analyze(path=path_pdf)
    df.reset_state()

    results_data = []

    for dp in df:
        print(f"\n--- Processing page {dp.page_number} ---")

        if len(dp.tables) > 0:
            print(f"{len(dp.tables)} tables found")
        else:
            print("No tables found")

        table_content = []
        for table in dp.tables:
            table_content.append({
                "csv": table.csv,
                "html": table.html
            })

        if table_content:
            page_data = {
                "page": dp.page_number + 1,
                "tables": table_content
            }
            results_data.append(page_data)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal time: {total_time:.2f} seconds")

    final_output = {
        "file_name": path_pdf,
        "runtime_seconds": round(total_time, 2),
        "total_num_tables": sum(len(p["tables"]) for p in results_data),
        "results": results_data
    }

    with open("deepdoctection_output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("Data saved in 'deepdoctection_output.json'")

def html_to_matrix(html_str):
    soup = BeautifulSoup(html_str, 'html.parser')
    rows = soup.find_all('tr')

    grid = {}
    max_cols = 0
    max_rows = len(rows)

    # Processing
    for r, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        c_idx = 0

        for cell in cells:
            while (r, c_idx) in grid:
                c_idx += 1

            text = cell.get_text(strip=True)
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))

            for i in range(rowspan):
                for j in range(colspan):
                    real_row = r + i
                    real_column = c_idx + j

                    grid[(real_row, real_column)] = text
                    if real_column >= max_cols:
                        max_cols = real_column + 1

            c_idx += colspan

    # Turn dictionary into lists (matrix)
    matrix = []
    for r in range(max_rows):
        current_row = []
        for c in range(max_cols):
            current_row.append(grid.get((r, c), ""))
        matrix.append(current_row)
    return matrix


def is_value(text):
    text = text.strip()
    value_regex = r'^[\d.,]+[%*]?$|^[-–]$'
    return bool(re.match(value_regex, text))


def clean_and_convert_to_float(value_str):
    if not value_str: return None
    clean_str = str(value_str).strip().replace("*", "").replace(",", "")
    if clean_str in ["-", "–", "—", "nan", "N/A"]:
        return None
    try:
        return float(clean_str)
    except ValueError:
        return None


def split_header(matrix):
    if not matrix: return [], []
    stub_header = matrix[0][0]
    split_idx = 0

    for i, row in enumerate(matrix):
        cells_to_evaluate = row[1:]
        if not cells_to_evaluate: continue

        cells_total = len(cells_to_evaluate)
        num_count = sum(1 for c in cells_to_evaluate if is_value(c))
        different_stub = (row[0] != stub_header)

        if (num_count / cells_total) > 0.5 and different_stub:
            split_idx = i
            break

    headers = matrix[:split_idx]
    values = matrix[split_idx:]
    return headers, values


def extract_tuples(headers, values):
    context_by_column = []
    num_of_columns = len(headers[0])

    for col_idx in range(1, num_of_columns):
        linked_parts = []
        last_seen_value = ""

        for header_row in headers:
            current_value = header_row[col_idx]
            if current_value and current_value != last_seen_value:
                linked_parts.append(current_value)
                last_seen_value = current_value

        final_context = " | ".join(linked_parts)
        context_by_column.append(final_context)

    tuples = []

    for value_row in values:
        row_title = value_row[0]
        all_values = value_row[1:]
        for context, value in zip(context_by_column, all_values):
            if value in ["-", "–", ""]:
                continue
            tuple = (row_title, context, value)
            print(f"{tuple}")
            tuples.append(tuple)

    return tuples


def extract_values_from_html_table(html):
    matrix = html_to_matrix(html)
    if not matrix:
        raise ValueError("La matriz generada está vacía (HTML sin estructura válida).")
    headers, values = split_header(matrix)
    if not values:
        raise ValueError("No se pudieron separar datos numéricos de las cabeceras.")
    tuples = extract_tuples(headers, values)
    structured_data = []

    for tuple in tuples:
        structured_data.append({
            "row": tuple[0],
            "column": tuple[1],
            "value": clean_and_convert_to_float(tuple[2])
        })

    return structured_data
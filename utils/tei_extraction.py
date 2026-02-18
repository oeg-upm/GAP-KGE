from lxml import etree
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

import re

def extract_sections_fulltext(tei_xml_str):
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = etree.fromstring(tei_xml_str.encode())

    divs = root.xpath('//tei:text/tei:body/tei:div', namespaces=ns)
    sections = []

    for div in divs:
        head_el = div.find('tei:head', namespaces=ns)
        title = head_el.text.strip() if head_el is not None and head_el.text else None

        # All <p> elements (deep search in case nested <div>s)
        paragraphs = []
        for p in div.xpath('.//tei:p', namespaces=ns):
            # Option 1: plain text with references inlined
            para_text = ''.join(p.itertext()).strip()
            if para_text:
                paragraphs.append(para_text)

            # Option 2 (alternative): include inline XML tags (uncomment if needed)
            # para_text = etree.tostring(p, encoding=str, method='xml')
            # paragraphs.append(para_text)

        if title or paragraphs:
            sections.append({
                "title": title,
                "paragraphs": paragraphs
            })
    return sections

def extract_abstract(tei_xml_str):
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}  # TEI uses XML namespaces
    root = etree.fromstring(tei_xml_str.encode())

    # Corrected XPath to include <div> between <abstract> and <p>
    abstract_paragraphs = root.xpath('//tei:abstract/tei:div/tei:p', namespaces=ns)
    
    # Join all paragraph texts, stripping whitespace
    abstract_text = '\n'.join(p.text.strip() for p in abstract_paragraphs if p.text)

    return abstract_text

def tei_to_full_raw_text(tei_xml: str, remove_ref = None) -> str:
    
    # Extract *all* human‐readable text from a GROBID TEI string,
    # preserving document order (header, abstract, body, references…).

    # 1) parse as XML
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    # 2) remove the references section
    if remove_ref:
        # Remove <ref> elements

        for ref_section in soup.find_all("div", attrs={"type": "references"}):
            ref_section.decompose()

    # 2) pull every text node, joining with newlines
    raw = soup.get_text(separator="\n", strip=True)

    # 3) fix common hyphenation artifacts ("word-\nnext")
    raw = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw)

    # 4) collapse any 3+ blank lines down to two
    raw = re.sub(r"\n{3,}", "\n", raw)

    return raw

# def extract_flat_sections_with_subtext(tei_xml_str):
#     ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
#     root = etree.fromstring(tei_xml_str.encode())

#     divs = root.xpath('//tei:text/tei:body/tei:div', namespaces=ns)
#     section_map = {}
#     top_sections = []

#     for div in divs:
#         head = div.find('tei:head', namespaces=ns)
#         if head is None or not head.text:
#             continue

#         title = head.text.strip()
#         n_attr = head.get('n')
#         section_num = n_attr.strip() if n_attr else None

#         # Get all paragraph text in this <div>
#         paragraphs = [
#             ''.join(p.itertext()).strip()
#             for p in div.xpath('.//tei:p', namespaces=ns)
#             if p.text or len(p)
#         ]
#         text = '\n\n'.join(paragraphs)

#         if section_num and re.fullmatch(r'\d+', section_num):
#             # It's a top-level section
#             section_entry = {
#                 "title": title,
#                 "text": text
#             }
#             section_map[section_num] = section_entry
#             top_sections.append(section_entry)

#         elif section_num and re.fullmatch(r'\d+\.\d+', section_num):
#             # It's a subsection: merge into parent
#             parent_num = section_num.split('.')[0]
#             if parent_num in section_map:
#                 if text:
#                     section_map[parent_num]["text"] += '\n\n' + text
#             else:
#                 # Orphan subsection — ignore or treat as top-level
#                 pass

#         else:
#             # No numeric heading: optionally add as top-level
#             if text:
#                 top_sections.append({
#                     "title": title,
#                     "text": text
#                 })

#     return top_sections
def extract_flat_sections_with_subtext(tei_xml_str):
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = etree.fromstring(tei_xml_str.encode())

    divs = root.xpath('//tei:text/tei:body/tei:div', namespaces=ns)
    section_map = {}
    top_sections = []
    last_section_num = None  # Track most recent valid section

    for div in divs:
        head = div.find('tei:head', namespaces=ns)
        if head is None or not head.text:
            continue

        title = head.text.strip()
        n_attr = head.get('n')
        section_num = n_attr.strip() if n_attr else None
        # print(f"Processing section: {title} (n={section_num})")
        # print(f"Last section number: {last_section_num}")


        # Get all paragraph text in this <div>
        paragraphs = [
            ''.join(p.itertext()).strip()
            for p in div.xpath('.//tei:p', namespaces=ns)
            if p.text or len(p)
        ]
        text = '\n\n'.join(paragraphs)

        if section_num and re.fullmatch(r'\d+', section_num):
            # It's a top-level section
            section_entry = {
                "title": title,
                "text": text
            }
            section_map[section_num] = section_entry
            top_sections.append(section_entry)
            last_section_num = section_num

        elif section_num and re.fullmatch(r'\d+(?:\.\d+)+', section_num):
            # It's a subsection: merge into parent
            parent_num = section_num.split('.')[0]
            if parent_num in section_map:
                if text:
                    section_map[parent_num]["text"] += '\n\n' + text
                last_section_num = parent_num
            else:
                # Orphan subsection — ignore or treat as top-level
                # print(f"Orphan subsection {section_num} found with title '{title}'")
                # add as top-level section
                section_map[section_num] = {
                    "title": title,
                    "text": text
                }
                top_sections.append(section_map[section_num])
                last_section_num = section_num

        else:
            # No numeric heading: optionally add as top-level
            if text:
                if last_section_num:
                    # Add as a new top-level section if we have a valid last section
                    section_map[last_section_num] = {
                        "title": title,
                        "text": text
                    }
                else:
                    top_sections.append({
                        "title": title,
                        "text": text
                    })
    return top_sections

def rank_sections_by_semantic_similarity(section_titles, queries,model):
    # Encode query list and section titles
    query_embs = model.encode(queries, convert_to_tensor=True)
    section_embs = model.encode(section_titles, convert_to_tensor=True)

    # Compute cosine similarities (queries × section_titles)
    sim_matrix = util.cos_sim(query_embs, section_embs)

    # For each section title, get the max similarity across all queries
    max_scores = sim_matrix.max(dim=0).values
    ranked = sorted(zip(section_titles, max_scores.tolist()), key=lambda x: -x[1])
    return ranked

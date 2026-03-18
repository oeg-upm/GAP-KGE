import os
import requests
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import io


class GrobidService:
    def __init__(self, base_url: str = "http://localhost:8070"):
        self.base_url = base_url.rstrip('/')
        # Endpoints específicos de la API de GROBID
        self.header_url = f"{self.base_url}/api/processHeaderDocument"
        self.fulltext_url = f"{self.base_url}/api/processFulltextDocument"

    def _send_to_grobid(self, url: str, pdf_path: str, params: dict) -> str:
        """Método privado para manejar la lógica de envío común."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No se encontró el archivo: {pdf_path}")

        with open(pdf_path, 'rb') as f:
            files = {'input': f}
            try:
                response = requests.post(url, files=files, data=params, timeout=300)
                if response.status_code == 200:
                    return response.text
                else:
                    raise Exception(f"GROBID error ({response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                raise Exception(f"Fallo de conexión con GROBID: {e}")

    def process_header(self, pdf_path: str) -> str:
        """
        Llama a GROBID para procesar el header del PDF y devuelve el TEI XML.
        """
        params = {
            'generateIDs': '0',
            'consolidateHeader': '1',
            'consolidateCitations': '0',
            'includeRawCitations': '0',
            'includeRawAffiliations': '0',
            'teiCoordinates': '0'
        }
        return self._send_to_grobid(self.header_url, pdf_path, params)

    def process_full_text(self, pdf_path: str) -> str:
        """
        Llama a GROBID para procesar el texto completo del PDF y devuelve el TEI XML.
        """
        params = {
            'generateIDs': '0',
            'consolidateHeader': '1',
            'consolidateCitations': '0',
            'includeRawCitations': '0',
            'includeRawAffiliations': '0',
            'teiCoordinates': '0'
        }
        return self._send_to_grobid(self.fulltext_url, pdf_path, params)

    def extract_authors(self, tei_xml: str) -> list[str]:
        """
        Parsea el TEI XML y extrae una lista de nombres completos de autores.
        """
        soup = BeautifulSoup(tei_xml, "lxml-xml")
        names = []

        # GROBID estructura los autores dentro de <sourceDesc> o <analytic>
        for author in soup.find_all("author"):
            pers = author.find("persName")
            if not pers:
                continue

            # Extraer nombres de pila (forenames)
            forenames = [fn.get_text(strip=True) for fn in pers.find_all("forename")]
            # Extraer apellido (surname)
            surname = pers.find("surname")
            surname_text = surname.get_text(strip=True) if surname else ""

            full_name = " ".join(forenames + ([surname_text] if surname_text else [])).strip()

            if full_name and full_name not in names:
                names.append(full_name)

        return names

    def extract_authors_from_pdf(self, pdf_path: str) -> list[str]:
        """
        Método de conveniencia: procesa el PDF y extrae autores en un paso.
        """
        tei = self.process_header(pdf_path)
        return self.extract_authors(tei)

    def procesar_pdf_desde_url(self,url_pdf) -> str:
        """
        Descarga un PDF desde una URL y lo procesa directamente con GROBID.
        Retorna el TEI XML como string.
        """

        headers_descarga = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            # 2. Descargar el PDF
            print(f"📥 Descargando PDF desde: {url_pdf}...")
            response_download = requests.get(url_pdf, headers=headers_descarga, timeout=30)

            if response_download.status_code != 200:
                return f"❌ Error al descargar el PDF ({response_download.status_code})"

            # 3. Validar que es un PDF real (revisando los primeros bytes)
            pdf_content = response_download.content
            if not pdf_content.startswith(b'%PDF'):
                return "❌ Error: El contenido descargado no parece ser un PDF válido (posible bloqueo de bot o página HTML)."

            # 4. Enviar a GROBID directamente desde la memoria (usando io.BytesIO)
            print("📡 Enviando a GROBID...")
            files = {
                'input': ('documento.pdf', io.BytesIO(pdf_content), 'application/pdf')
            }
            # Parámetros para mejorar la extracción
            data = {
                'generateIDs': '1',
                'consolidateHeader': '1',
                'consolidateCitations': '1'
            }

            response_grobid = requests.post(self.fulltext_url, files=files, data=data, timeout=300)

            if response_grobid.status_code == 200:
                print("✅ Procesamiento completado con éxito.")
                return response_grobid.text
            else:
                return f"❌ Error en GROBID ({response_grobid.status_code}): {response_grobid.text}"

        except requests.exceptions.Timeout:
            return "❌ Error: Tiempo de espera agotado (timeout)."
        except requests.exceptions.ConnectionError:
            return "❌ Error: No se pudo conectar con el servidor (verifica el URL o el Docker de GROBID)."
        except Exception as e:
            return f"❌ Error inesperado: {str(e)}"







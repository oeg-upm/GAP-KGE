from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup

class GrobidService:
    def __init__(self, config_path: str = "./Grobid/config.json"):
        self.client = GrobidClient(config_path=config_path)

    def process_header(self, pdf_path: str) -> str:
        """
        Calls GROBID to process the PDF header and returns the TEI XML.
        """
        _, status, tei = self.client.process_pdf(
            service="processHeaderDocument",
            pdf_file=pdf_path,
            generateIDs=False,
            consolidate_header=True,
            consolidate_citations=False,
            include_raw_citations=False,
            include_raw_affiliations=False,
            segment_sentences=False,
            tei_coordinates=False
        )
        return tei

    def process_full_text(self, pdf_path: str) -> str:
        """
        Calls GROBID to process the PDF text and returns the TEI XML.
        """
        _, status, tei = self.client.process_pdf(
            service="processFulltextDocument",
            pdf_file=pdf_path,
            generateIDs=False,
            consolidate_header=True,
            consolidate_citations=False,
            include_raw_citations=False,
            include_raw_affiliations=False,
            segment_sentences=False,
            tei_coordinates=False
        )
        return tei

    def extract_authors(self, tei_xml: str) -> list[str]:
        """
        Parses TEI XML and extracts a list of author full names.
        """
        soup = BeautifulSoup(tei_xml, "lxml-xml")
        names = []
        for author in soup.find_all("author"):
            pers = author.find("persName")
            if not pers:
                continue
            parts = [fn.get_text(strip=True) for fn in pers.find_all("forename")]
            if pers.surname:
                parts.append(pers.surname.get_text(strip=True))
            if parts:
                names.append(" ".join(parts))
        return names

    def extract_authors_from_pdf(self, pdf_path: str) -> list[str]:
        """
        Convenience method: process PDF and extract authors in one step.
        """
        tei = self.process_header(pdf_path)
        return self.extract_authors(tei)

# Instrucciones y documentación respectiva al pipeline de P2MC

En este documento intentaré plasmar un "plano" del pipeline, que partes tiene y cómo funciona. Si algo no queda los suficientemente claro estaré encantado de responder a cualquier duda o aclaración a través de mi slack mateo.campaya@upm.es.

### Concepto

Este pipeline tiene como finalidad la extracción de datos de PDFs de papers cientificos del campo de los grafos de conocimiento y su estructuración en forma de JSON-LD. Lo que recibe el pipeline es un enlace a un PDF, por ejemplo http://arxiv.org/pdf/1805.12393v1.pdf

El output del pipeline es un archivo json de tipo JSON-LD con los datos extraidos.

### Flujo

1. Se instancia la clase PDFHandler y se llama a su método pdf_handler que recibe como parámetro la url del PDF. Este método devuelve la modelcard. La URL introducida provendría de la página de streamlit de la demo.

Ejemplo de uso:
from pdf_handler import PDFHandler
pdf_handler = PDFHandler()
modelcard = pdf_handler.handle_pdf("http://arxiv.org/pdf/1805.12393v1.pdf")

2. Cuando se llama al método handle_pdf se descarga el pdf desde ese enlace y se almacena en data/raw.

3. Se llama a la clase SciPdfParser (se ubica en parsers) que procesa el PDF (recibe la ubicación del PDF descargado en data/raw) y genera un XML que se guarda en data/interim.

4. Se llama a la clase LightOcrParser (también ubicada en parsers) que también procesa el PDF, al igual que en el paso anterior, pero resultando en un JSON que se guarda en data/interim.

5. Teniendo esos dos archivos se llama al método generate_modelcard() de ModelCardGenerator (ubicado en model_card_generation_pipeline.py). Ese método se encarga de llamar a una variada colección de extractores, algunos utilizando llama3, otros gliner, otros gemma y cada uno encargado de completar obtener un campo. La extracción se realiza sobre los archivos XML y JSON generados por SciPDF y Lightocr. 
6. Algunos campos son identificados en SEMOpenAlex y LinkedPapersWithCode. Para eso existe una clase llamada UriFetcher en Utils. Algunas ids que no logramos encontrar en esas páginas o que debemos crear nosotros son creadas en UriBuilder, también en Utils.

7. ModelCardGenerator le da forma a todos los campos extraídos y construye el JSON con ellos. Finalmente se devuelve ese JSON-LD terminado que es a lo que llamamos Modelcard.

Notas: El código ha crecido mucho en los últimos días y hay elementos que no he podido probar y seguro que hay bugs que aún no he podido resolver. Si hubiese algún problema no dudes en preguntarme o indicarme dónde está para poder resolverlo.

# GAP-KGE 🔵->🔴
This project contains both the experiments made regarding the efficiency and the aptness of different approaches for the extraction of each section of the final model card.
The notebooks employed for the different experiments (both using LLMs and non LLM models) are in the *experiment_notebooks* folder.

## Project context
The goal of the project is to automatically mine model cards (similar to those in HuggingFace) for works on the area of KGE, since it is a task that is not covered by HuggingFace.
In this context, since both PapersWithCode (🕊️) and HuggingFace rely on manual input to generate the cards, making them sometimes scarce, limited and not descriptive, our goal is to automatize this process, building an optimal pipeline capable of generating these cards automatically from the PDF itself.
Once we have this pipeline capable of generating the cards in unstructured format, our next step is to convert them into a KG, using the FAIR4ML representation.

The fields that we are extracting to generate the model cards are:
- Authors ✅
- Tasks addressed by the model✅
- Title of the paper ✅
- Reference to the implementation in the paper ✅
- Reference to the implementation in external libraries 👷
- Name of the proposed model ✅
- Type of model according to the Shen et al. (2022) taxonomy ✅
- Limits and biases 👷
- Datasets used for the evaluation ✅
- Metrics used for the evaluation✅
- Achieved results👷

Points marked with ✅ have been already covered in the pipeline, while those marked with 👷 are currently in progress.
## Table extraction
The work already done using DeepDocTextion is on the folder called **table_extraction**. Here, you'll find the notebook developed by Mateo for testing our approach, and I made a .py version of it (haven't had the chance to test it yet though). We hace a corpus of 109 articles on KGE which are on the *data/pdf_files* folder.
Inside the *table_extraction* folder, you'll find another folder with a couple of PDFs Mateo used for testing the approach by manually revising the output.

## Pre-requisites
Some of the studied methods that rely on non-LLM approaches are based on external software, that needs to be installed beforehand.
One of the softwares employed in this project is [Grobid](https://github.com/kermitt2/grobid), as well as it finetuned version for research papers, [SciPDF](github.com/titipata/scipdf_parser)

### Running Grobid
Grobid is deployed as a Docker (🐳) service:

<pre lang="markdown">
docker pull lfoppiano/grobid:0.8.0
docker run -p 8070:8070 lfoppiano/grobid:0.8.0
</pre>


## Running the experiments
As previously mentioned, the *experiment_notebooks* folder contains all notebooks developed throughout the experimentation process. All data for the experiments is in the *data* folder. If unavailable, you can run the *pwc_extraction.py* script to automatically create the dataset from the PWC dumps available.
One notebook is devised per considered field. Inside each notebook, all cells required to run the experiments on the different model and content combinations are provided.
**Make sure to have Ollama running with the corresponding models before launching the experiments!**

## Running the pipeline
The notebook named *model_card_pipeline.ipynb* can be used to run the full pipeline, both in efficient and in reliable mode.

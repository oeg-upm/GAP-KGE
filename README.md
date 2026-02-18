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
- Name of the proposed model 👷
- Type of model according to the Shen et al. (2022) taxonomy ✅
- Limits and biases
- Datasets used for the evaluation 👷
- Metrics used for the evaluation👷
- Achieved results👷

Points marked with ✅ have been already covered in the pipeline, while those marked with 👷 are currently in progress.
## Table extraction
Hi Erick! 👋. To make your life a bit easier, I'll keep it short and sweet for this part. The work already done using DeepDocTextion is on the folder called **table_extraction**. Here, you'll find the notebook developed by Mateo for testing our approach, and I made a .py version of it (haven't had the chance to test it yet though). We hace a corpus of 109 articles on KGE which are on the *data/pdf_files* folder.
Inside the *table_extraction* folder, you'll find another folder with a couple of PDFs Mateo used for testing the approach by manually revising the output.

## Pre-requisites
Some of the studied methods that rely on non-LLM approaches are based on external software, that needs to be installed beforehand.
One of the softwares employed in this project is [Grobid](https://github.com/kermitt2/grobid).

### Running Grobid
Grobid is deployed as a Docker (🐳) service:

<pre lang="markdown">
docker pull lfoppiano/grobid:0.8.0
docker run -p 8070:8070 lfoppiano/grobid:0.8.0
</pre>

### Running DataStet
Another employed software is DataStet, from SoftCite, which is used to detect dataset mention annotations.

The easiest way to deploy and run is to use the Docker image, although there are other ways also that you can check out on their github: [DataStet](https://github.com/kermitt2/datastet).

Run the docker container

<pre lang="markdown">
docker pull grobid/datastet:0.8.1
docker run --rm --gpus all -it --init --ulimit core=0 -p 8060:8060 grobid/datastet:0.8.1
</pre>

This let you access to their web services for dataset extraction. But to exploit the DataStet service more efficiently, a Python client is available in [softcite/software_mentions_client](https://github.com/softcite/software_mentions_client) that can use DataStet to produce dataset mention annotations.

Install the python client

<pre lang="markdown">
git clone https://github.com/softcite/software_mentions_client.git
cd software_mentions_client/
</pre>

It is advised to setup first a virtual environment 

<pre lang="markdown">
virtualenv --system-site-packages -p python3 env
source env/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
</pre>

Now with the docker container running and virtual environment activated. For processing a single file, the resulting json being written as file at the indicated output path:
<pre lang="markdown">
python3 -m software_mentions_client.client --file-in toto.pdf --file-out toto.json
</pre>

For processing a repository:
<pre lang="markdown">
python3 -m software_mentions_client.client   --repo-in pdf_dir  --datastet
</pre>

Anntations will be added along the PDF and XML files, with extension *.software.json.

### Running SciRex
Finally, SciRex was also used in the experimentation for the extraction of tasks within the papers.
To install and train the SciRex model, check the links for Check the links for [installation](https://github.com/allenai/SciREX?tab=readme-ov-file#installation) and [training](https://github.com/allenai/SciREX?tab=readme-ov-file#training-scirex-baseline-model).

In this case we created a virtual environment instead of conda.
First make sure the virtual environment is activated.
<pre lang="markdown">
source scirex_env/bin/activate
</pre>

Export required variables for AllenNLP configs.
<pre lang="markdown">
export BERT_BASE_FOLDER=/mnt/c/Users/Che/GAP-KGE/SciREX-master/models/scibert/scibert_scivocab_uncased
export BERT_VOCAB=$BERT_BASE_FOLDER/vocab.txt
export BERT_WEIGHTS=$BERT_BASE_FOLDER/weights.tar.gz
export TRAIN_PATH=/mnt/c/Users/Che/GAP-KGE/scirex-master/scirex_dataset/release_data/train.jsonl
export DEV_PATH=/mnt/c/Users/Che/GAP-KGE/scirex-master/scirex_dataset/release_data/dev.jsonl
export TEST_PATH=/mnt/c/Users/Che/GAP-KGE/scirex-master/scirex_dataset/release_data/test.jsonl
export IS_LOWERCASE=true
export CUDA_DEVICE=0
</pre>

Run and time Scirex
<pre lang="markdown">
time PYTHONPATH=. python scirex/predictors/predict_ner.py \
  outputs/pwc_outputs/experiment_scirex_full/main \ 
  scirex_format.jsonl \ # input path
  test_outputs/pdfs/ner_predictions.jsonl \ # output path
  0
</pre>

## Running the experiments
As previously mentioned, the *experiment_notebooks* folder contains all notebooks developed throughout the experimentation process.
This experimentation comprised evaluating both LLM-based and non-LLM approaches to assess which model was the best fit for each section of the final model card.

- To run the **non-LLM experiments**, execute the [run_non_llm.ipynb](/experiment_notebooks/run_non_llm.ipynb)
- To run the **LLM-based experiments**, execute the [run_llm.ipynb](/experiment_notebooks/run_llms.ipynb)
- Finally, to run the experiments for the optimal pipeline, execute the [run_best_configuration.ipynb](/experiment_notebooks/run_best_configuration.ipynb)

## Summary of work so far

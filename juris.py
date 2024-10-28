from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.tokenizers import Tokenizer
from functools import reduce
from gradio_pdf import PDF
from pathlib import Path
import gradio as gr
import PyPDF2
import nltk
import time

#"""
import chromadb
"""
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
#"""

ingestion_time = 0
results_count = 3
embeddings = None

# Setup ChromaDB
print('Setting up client...')
client = chromadb.Client()

# Create a new collection for storing embeddings
print('Creating collection...')
collection = client.create_collection('text_collection')

# Load model for processing
print('Loading modal...')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialise and load summparizer
print('Initialising summarizer resources...')
nltk.download('punkt_tab')
summarizer = LuhnSummarizer()

def ingestPdfReadings(collection: chromadb.Collection, model: SentenceTransformer):

    # Measure image ingestion time
    print('Starting data ingestion...')
    start_ingestion_time = time.time()

    readings = {
        'text': [],
        'metadatas': [],
        'idx': []
    }

    pathlist = Path('data').glob('**/*.pdf')
    for idx, path in enumerate(pathlist):
        path_str = str(path) # because path is object not string
        print(f'Ingesting #{idx}: {path_str}')

        # Check if exist in collection already
        #collection.get(where={'filename': '08-1521 McDonald v. Chicago.pdf'})

        # Extract text from pdf
        with open(path_str, 'rb') as f:
            full_text = ''
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                full_text += page.extract_text()

        # Summarise the full_text to focus on main points
        parser = PlaintextParser.from_string(full_text, Tokenizer('english'))
        summary = summarizer(parser.document, 10)
        summary = ''.join([str(sentence) for sentence in summary])

        readings['text'].append(summary)
        readings['idx'].append(str(idx))
        readings['metadatas'].append({
            'path': path_str,
            'filename': path.name
        })

    global embeddings
    embeddings = model.encode(readings['text'])

    collection.add(
        embeddings=embeddings,
        metadatas=readings['metadatas'],
        ids=readings['idx']
    )

    # Measure total ingestion time
    global ingestion_time
    end_ingestion_time = time.time()
    ingestion_time = end_ingestion_time - start_ingestion_time

    # Log the ingestion performance
    print(f"Data ingestion time: {ingestion_time:.4f} seconds")

def search(query):
    if not query.strip():
        return None, "Oops! You forgot to type something on the query input!", ""

    # Generate an embedding for the query text
    query_embedding = model.encode(query)

    # Perform a vector search in the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=results_count  # Retrieve top x similar entries
    )

    # Extract path from metadata 
    filepaths = [x['path'] for x in results['metadatas'][0]]

    # calculate accuracy
    accuracy = []
    for _, matched_result_index in enumerate(results['ids'][0]):
        matched_result_embedding = embeddings[int(matched_result_index)]
        accuracy.append(cosine_similarity([matched_result_embedding],[query_embedding])[0][0])

    def map_acc_to_str(accuracy_score):
        return f'{accuracy_score:.4f}'

    accuracy = '\n'.join(list(map(map_acc_to_str, accuracy)))

    return filepaths, filepaths[0], accuracy

def on_select_file(value, evt: gr.SelectData):
    return 'data/' + evt.value

def on_results_count_change(value):
    value = value.strip()

    if len(value) <= 0 or not value.isnumeric() or int(value) <= 0:
        """
        No change if
        - empty string
        - string is not numeric
        - number is not larger than 1
        """
        return

    # Save new config
    global results_count
    results_count = max(1, int(value))

# Ingest information
ingestPdfReadings(collection, model)

# Gradio Interface Layout
with gr.Blocks(title='Juris FastSearch') as gr_interface:
    gr.Markdown("# Juris FastSearch")
    with gr.Row():
        # Left Panel
        with gr.Column():
            gr.Markdown("### Search")

            # Input box for custom query
            custom_query = gr.Textbox(
                label="What are you looking for?",
                placeholder="Enter your custom query here",
                lines=10,
                submit_btn='Search'
            )

            gr.Markdown("### Results")
            
            files = gr.Files(file_count='multiple', interactive=False)

            with gr.Accordion(label="Performance", open=True):

                # Display the ingestion time of image embeddings
                gr.Markdown(f"Ingestion Time: *{ingestion_time:.4f} seconds*")

                # Output for accuracy score and query time
                accuracy_output = gr.Textbox(label="Relevance (Accuracy)")

            with gr.Accordion(label="Preferences", open=False):
                results_count_selector = gr.Textbox(label="Number of Results", value=results_count)

        # Right Panel
        with gr.Column():
            gr.Markdown("### Viewer")
            viewer = PDF(label="Document", interactive=False)

    custom_query.submit(fn=search, inputs=custom_query, outputs=[files, viewer, accuracy_output])

    files.select(fn=on_select_file, inputs=files, outputs=viewer)

    results_count_selector.change(fn=on_results_count_change, inputs=results_count_selector, outputs=None)


if __name__ == '__main__':
    # Launch the Gradio interface
    print('Ready! Launching interface ...')
    gr_interface.launch()

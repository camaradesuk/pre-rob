This application is designed to assess the risk of bias in text documents (typically scientific manuscripts) using a collection of deep learning models. Here’s an overview of how it works:

---

### 1. Model Definitions (model.py)

- **Deep Learning Architectures:**  
  The code defines several neural network models using PyTorch:
  - **ConvNet:** A convolutional neural network that applies multiple 2D convolutions (with different filter sizes) over text embeddings, followed by max pooling and a fully connected layer.
  - **AttnNet:** An RNN-based model (using either LSTM or GRU) with an attention mechanism that learns to weight different parts of the sequence. This allows the model to focus on important words.
  - **Hierarchical Attention Network (HAN):** A two-level attention network.  
    - The first level (WordAttn) computes attention scores over words in a sentence.  
    - The second level (SentAttn) computes attention over sentences to produce a document-level representation.
  - **DistilClsConv:** A model that combines DistilBERT (from the Transformers library) with convolutional layers. It processes the contextualized embeddings from DistilBERT using a similar convolutional approach as in the ConvNet.

- **Attention Mechanisms:**  
  Both the AttnNet and HAN models incorporate learnable attention weights that not only contribute to the prediction but can also be used to extract and highlight the most influential sentences or words in the document.

---

### 2. Model Loading and Prediction Functions (rob_fn.py)

- **Loading Models and Associated Metadata:**  
  The functions in this file are responsible for:
  - **Loading the Model Configurations and Checkpoints:**  
    Models are instantiated based on configuration files (JSON) and restored from saved PyTorch checkpoints.  
  - **Loading Vocabulary/Field Information:**  
    The TEXT field (using `dill`) provides a mapping between words and their numerical indices, which is crucial for converting raw text into tensors.

- **Prediction Functions:**  
  - **`pred`:**  
    Tokenizes a document (using spaCy), pads/truncates it to a maximum length, converts tokens to indices, and then passes the tensor to a model (e.g., ConvNet, AttnNet) to output a probability—typically indicating the likelihood that a certain risk-of-bias item is reported.
  - **`pred_bert`:**  
    Uses a DistilBERT-based model. It first splits the document into sentences, encodes them with SentenceTransformer to compute cosine similarities against a pre-defined “risk-of-bias sentence” (specific to a risk item), selects the most similar sentences, and then classifies the concatenated similar sentences.
  - **`extract_sent`:**  
    Uses the HAN model’s sentence-level attention scores to extract and rank sentences. The top-ranked sentences are returned as those most indicative of the risk-of-bias criteria.

---

### 3. Main Application Workflow (rob.py)

- **Input Handling:**  
  The main script uses command-line arguments to specify:
  - A CSV file (or a folder) that contains paths to text documents.
  - An output CSV file path where the prediction results will be saved.
  - Optionally, the number of sentences to extract that best illustrate the bias-relevant content.

- **PreRob Class:**  
  - **Text Collection:**  
    It gathers all text file paths by checking if the input is a directory, a single text file, or a CSV file with relative paths.
  - **Text Processing:**  
    Before analysis, the text is preprocessed using regular expressions to remove extraneous sections (like references, citations, and non-ASCII characters). This cleaning helps focus the models on the core content of the manuscript.
  - **Prediction Execution:**  
    For each document, the class:
    - Reads and cleans the text.
    - Runs multiple prediction functions (using different models) for various risk-of-bias items (for instance, randomization, blinding, sample size calculation, etc.).
    - Optionally extracts the most attention-worthy sentences for each risk item if the user requests sentence extraction.
  - **Output:**  
    The results—consisting of file paths, prediction scores for different risk items, and optionally extracted sentences—are compiled into a CSV file.

- **Integration:**  
  The script loads multiple models (with different architectures) from disk. It then processes every text document as described and writes the final predictions to an output CSV file.

---

### Summary

In essence, the rob app is a robust pipeline for risk-of-bias assessment that:
- **Combines Multiple Models:** Uses both classical CNN/RNN/HAN networks and a modern transformer-based approach.
- **Processes and Cleans Input Text:** Ensuring that only the relevant parts of the manuscript are fed into the models.
- **Extracts Interpretable Features:** The attention mechanisms help identify and extract the key sentences that influence the bias assessment.
- **Outputs Structured Results:** Predictions for various risk-of-bias items, which are then saved in a CSV format for further analysis or reporting.

This architecture reflects a well-structured, modular design where model training, inference, and text processing are separated—an approach that resonates well with principles in software engineering and Domain Driven Design.

Below is a detailed explanation of how to use the rob app and how to organize your inputs:

---

## 1. Input Organization

### A. Text Files

- **File Format:**  
  All documents should be plain text files with a **.txt** extension.  
- **Content Requirements:**  
  The app is designed to process academic or research manuscripts. For example, it removes sections before “Introduction” and trims references at the end. Make sure your texts are structured (e.g., include an “Introduction” section) so that the cleaning steps work as expected.
- **Location:**  
  Your text files can be stored anywhere on your file system, but their paths must be correctly referenced in your CSV (see next section).

### B. CSV Document

The app is triggered by providing a CSV file (via the `-p` argument) that lists the text files you want to process. The CSV must be organized as follows:

- **Required Column – `path`:**  
  - This column should list the full or relative path for each text file.  
  - Example entry: `/home/chris/documents/article1.txt` or `./data/article1.txt`

- **Optional Column – `id`:**  
  - If you want to track or label your documents, include an `id` column with a unique identifier for each text file.
  - If no `id` is provided, the script will generate an incremental id.

- **CSV Format:**  
  - The CSV should use commas as separators.
  - Example CSV content:
  
    ```csv
    id,path
    1,/path/to/document1.txt
    2,/path/to/document2.txt
    3,/path/to/document3.txt
    ```

*Note:* Although the `PreRob.get_txt_path()` function can handle folders or a single text file, the main execution block in `rob.py` explicitly checks for a CSV input. Therefore, you must supply a CSV file when running the app.

---

## 2. Running the App

The app is executed as a command-line script. It uses Python’s `argparse` to read input arguments. The key arguments are:

- **`-p` or `--csv`:**  
  The absolute or relative path to your CSV file that contains the text file paths.

- **`-o` or `--output`:**  
  The absolute or relative path where you want the output CSV file to be saved.

- **`-s` or `--sent` (Optional):**  
  An integer specifying how many sentences (per risk category) to extract from each text. If provided, the app will also return a set of the most “attention-worthy” sentences from each document.

### Example Command

From a terminal or command prompt, you would run the app like this:

```bash
python rob.py -p input_files.csv -o results/output_results.csv -s 3
```

- **Explanation:**  
  - `-p input_files.csv`: Uses the CSV file `input_files.csv` which contains the list of text file paths.
  - `-o results/output_results.csv`: Saves the final prediction results in the specified CSV file. The script will create the output directory if it does not exist.
  - `-s 3`: For each text file, extract the top 3 sentences per risk-of-bias item using the attention mechanism.

---

## 3. What Happens Under the Hood

1. **CSV Processing:**  
   - The script reads the provided CSV and extracts the list of text file paths (and optionally, their associated IDs).

2. **Text File Processing:**  
   - Each text file is opened and its content is cleaned using regular expressions.  
   - This includes stripping out introductory material (everything before “Introduction”), references, citations, non-ASCII characters, and extra white spaces.

3. **Model Predictions:**  
   - The app loads several pre-trained models (including CNN, RNN with attention, HAN, and a DistilBERT-based model).
   - For each document, different risk-of-bias metrics are predicted. These predictions result in a probability score (e.g., likelihood of reporting randomization, blinding, etc.).
   - If sentence extraction is requested (via the `-s` flag), the script uses attention scores from the HAN model to identify and extract key sentences from the text.

4. **Output Generation:**  
   - All predictions, along with the file path and document ID (if provided), are compiled into a structured CSV file.
   - The CSV includes columns such as `txt_path`, `random`, `blind`, `interest`, `welfare`, `exclusion`, and if applicable, a `sentences` field with extracted text segments.

---

## 4. Summary of Steps

1. **Prepare Your Data:**  
   - Organize your text files (.txt) and note their file paths.
   - Create a CSV file that has a column `path` (and optionally `id`) listing these paths.

2. **Run the Script:**  
   - Use the command line to run `rob.py` with the required arguments (`-p` for the CSV file and `-o` for the output CSV file).
   - Optionally, add `-s` with the number of sentences to extract.

3. **Review Output:**  
   - Once the script finishes, check the output CSV file for prediction scores and any extracted sentences.

Following these instructions should allow you to successfully use the app to assess risk-of-bias in your documents.
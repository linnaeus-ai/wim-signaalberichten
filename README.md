# WIM-Signaalberichten Text-to-Knowledge-Graph Pipeline

Transforms Dutch text into Schema.org JSON-LD knowledge graphs using LLMs.

## Prerequisites

- Python 3.12 or higher
- Go (for the Schema.org validator)
- Azure OpenAI API access

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd wim-signaalberichten
```

### 2. Create and Activate Virtual Environment

On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example environment file to the project root:
```bash
cp example.env .env
```

2. Edit `.env` and add your Azure OpenAI credentials:
   - At minimum, configure the GPT4O settings (default model)
   - Configure the EMBEDDINGS settings (required for schema matching)
   - Optionally configure other models based on your needs

**Note:** The `.env` file should be placed in the project root directory (same level as `README.md`). The `load_dotenv()` call in the pipeline will automatically find it.

### 5. Set PYTHONPATH

To ensure proper imports when running scripts, set the PYTHONPATH to include the `src` directory:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Or add it to your virtual environment activation script:
```bash
echo 'export PYTHONPATH="${PYTHONPATH}:'$(pwd)'/src"' >> venv/bin/activate
```

### 6. Verify Schema Validator

The project includes a Go-based Schema.org validator. Ensure it's executable:

```bash
chmod +x src/graph/validator/schema-validator
```

## Architecture

LangGraph state machine with five nodes:
1. **Entity Extraction**: Parses text for entities and relationships
2. **Schema Mapping**: Semantic search for best Schema.org types
3. **JSON-LD Generation**: Builds Schema.org knowledge graph representation
4. **Validation**: Go-based validator with 3-tier exit codes (0: success, 1: recoverable errors, 2: infrastructure failures)
5. **Topic Labeling**: Dutch taxonomy classification (Onderwerp/Beleving signals)

State flows through nodes carrying text, entities, schemas, and JSON-LD.
Recoverable validation errors trigger retries; infrastructure failures halt pipeline.

## Project Structure

```
src/
├── graph/              # Main pipeline implementation
│   ├── base/          # Abstract node and router classes
│   ├── nodes/         # Pipeline stages (n1-n5)
│   ├── prompts/       # LLM prompts for each node
│   ├── routers/       # Routing logic
│   ├── utils/         # Helper functions and models
│   └── validator/     # Schema.org validation tools
├── data/              # Data files and schema definitions
├── config/            # Configuration files (models.yaml)
└── scripts/           # Utility scripts
```

Each node supports multiple LLM configurations. Database logging tracks all LLM calls for analysis.

## Dependencies

- LangGraph for state management
- Schema.org validator binary (Go)
- Azure OpenAI APIs
- SQLite for LLM logging

## Troubleshooting

### Azure OpenAI Errors
- Verify your API keys and endpoints in `.env`
- Ensure your Azure deployment names match the configuration
- Check that API versions are compatible with your models

### Schema Validator Issues
If the validator fails to run:
1. Check file permissions: `ls -la src/graph/validator/schema-validator`
2. Verify Go compatibility if needed
3. The validator requires the `schemaorg-all-http.ttl` file in the same directory

## Scripts

### test_usage.py

Verify all is working well by running one example:

```bash
python src/scripts/test_usage.py
```


### run_metrics.py

**Purpose:** Evaluates the pipeline's performance on labeled test data by comparing generated labels against gold labels. Calculates precision, recall, and F1 scores for Dutch customer service signal classification.

**Usage:**
```bash
# Default: Process 10 rows from HuggingFace dataset
python src/scripts/run_metrics.py

# Process 100 rows from HuggingFace dataset
python src/scripts/run_metrics.py --limit 100

# Process Excel file with custom column names
python src/scripts/run_metrics.py --excel-file src/data/Sample_10_teksten.xlsx \
  --text-column "Toelichting_masked" \
  --labels-column "Categorieën samengevoegd"

# Specify custom output locations
python src/scripts/run_metrics.py --output-excel custom_metrics.xlsx --db-path custom.db
```

**Command-line options:**
- `--excel-file`: Path to Excel file (alternative to HuggingFace dataset)
- `--text-column`: Column name containing text to process (required with --excel-file)
- `--labels-column`: Column name containing gold labels (required with --excel-file)
- `--hf-dataset`: HuggingFace dataset name (default: UWV/wim_synthetic_data_for_testing)
- `--limit`: Number of rows to process from HF dataset (default: 10)
- `--output-excel`: Path for metrics Excel file (default: src/data/metrics.xlsx)
- `--db-path`: Path for SQLite database (default: src/data/metrics.db)

**What it does:**
1. Loads test data from either HuggingFace dataset or Excel file
2. Standardizes column names to 'text' and 'gold_labels' internally
3. Processes each text through the full pipeline (with label generation enabled)
4. Compares generated labels against gold labels
5. Calculates metrics (TP, FP, FN, TN) for two signal types:
   - **Onderwerp signals**: Topic-based categories (e.g., "Bouwen en verbouwen", "Burgerzaken")
   - **Beleving signals**: Experience-based categories (e.g., "Informatievoorziening", "Afhandeling")
6. Saves results to SQLite database and Excel report with precision, recall, and F1 scores

**Required files:**
- `src/data/Hoofdklantsignalen - Subklantsignalen.xlsx` (label taxonomy)
- Configured `.env` with Azure OpenAI credentials

**Note:** When using HuggingFace dataset, the script tracks progress and resumes from the last processed item if interrupted. Excel files are processed in full.

## Development

For development, you may want to install additional tools:

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Format code
black src/

# Lint code
flake8 src/
```

## Authors

Fauve Wevers, Kai Knauf, Yeb Havinga
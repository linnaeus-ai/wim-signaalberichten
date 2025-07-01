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
3. **KG Generation**: Builds JSON-LD representation
4. **Validation**: Go-based Schema.org validator with retry routing
5. **Topic Labeling**: Optional taxonomy classification

State flows through nodes carrying text, entities, schemas, and JSON-LD.
Validation failures trigger retries up to configured limit.

## Usage

```python
from src.graph.text_to_kg_pipeline import TextToKGPipeline

# Initialize pipeline
pipeline = TextToKGPipeline(db_path="llm_logs.db")

# Process text
result = await pipeline.run(
    text="Een evenement in Amsterdam op 15 januari 2025",
    add_labels=True,
    max_retries=3
)
```

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

### run_metrics.py

**Purpose:** Evaluates the pipeline's performance on labeled test data by comparing generated labels against golden labels. Calculates precision, recall, and F1 scores for Dutch customer service signal classification.

**Usage:**
```bash
# From project root with PYTHONPATH set
python src/scripts/run_metrics.py
```

**What it does:**
1. Loads synthetic test data from HuggingFace dataset `UWV/wim_synthetic_data_for_testing`
2. Processes each text through the full pipeline (with label generation enabled)
3. Compares generated labels against validated labels from the dataset
4. Calculates metrics (TP, FP, FN, TN) for two signal types:
   - **Onderwerp signals**: Topic-based categories (e.g., "Bouwen en verbouwen", "Burgerzaken")
   - **Beleving signals**: Experience-based categories (e.g., "Informatievoorziening", "Afhandeling")
5. Saves results to:
   - SQLite database: `src/data/metrics.db`
   - Excel report: `src/data/metrics.xlsx`

**Required files:**
- `src/data/Hoofdklantsignalen - Subklantsignalen.xlsx` (label taxonomy)
- Configured `.env` with Azure OpenAI credentials
- HuggingFace access to the test dataset

**Note:** The script tracks progress in the database and can resume from the last processed item if interrupted.

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
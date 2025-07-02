import argparse
import ast
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

from datasets import load_dataset, Dataset
from graph.utils import azure_llm
from graph import TextToKGPipeline, TextToKGState

DATA = pd.read_excel("src/data/Hoofdklantsignalen - Subklantsignalen.xlsx")
ONDERWERP_SIGNALS = [
    "Bouwen en verbouwen",
    "Burgerzaken",
    "Dagelijks leven en sociale gelegenheden",
    "Financiële ondersteuning",
    "Maatschappelijke ondersteuning",
    "No subtopic found",
    "Opruimen, afval en onderhoud",
    "Parkeren",
    "Veiligheid en omgeving",
    "Vervoer",
    "Werk",
    "Wonenen en ondernemen",
    "Zorg",
]
BELEVING_SIGNALS = [
    "Informatievoorziening",
    "Houding & Gedrag medewerker",
    "Fysieke dienstverlening",
    "Digitale mogelijkheden",
    "Contact leggen met medewerker",
    "Algemene ervaring",
    "Afhandeling",
    "Processen",
    "Prijs & Kwaliteit",
    "No subtopic found",
    "Kennis & Vaardigheden medewerker",
]


def get_labels_from_json_ld(state: TextToKGState) -> tuple[list, list]:
    """
    Extracts labels from the JSON-LD content in the state.
    """
    beleving, onderwerp = [], []
    json_ld = json.loads(state["json_ld_contents"][-1])

    if "about" in json_ld:
        for item in json_ld["about"]:
            if item["inDefinedTermSet"]["name"] in ONDERWERP_SIGNALS:
                onderwerp.append(item["name"])
            elif item["inDefinedTermSet"]["name"] in BELEVING_SIGNALS:
                beleving.append(item["name"])
            else:
                # We don't raise an error, because we want to simply exclude labels that are incorrect, not stop the whole row
                print(
                    f"ValueError: Label '{item["inDefinedTermSet"]["name"]}' not found in predefined signals. Continuing with the next label."
                )
    else:
        raise ValueError("No 'about' key found in JSON-LD content.")
    return beleving, onderwerp


def get_labels_from_validated_list(validated_labels: list) -> tuple[list, list]:
    """
    Extracts labels from the validated labels list.
    """
    onderwerp_signals = DATA[DATA["Hoofd_klantsignaal"].isin(ONDERWERP_SIGNALS)][
        "Sub_klantsignaal"
    ].tolist()
    beleving_signals = DATA[DATA["Hoofd_klantsignaal"].isin(BELEVING_SIGNALS)][
        "Sub_klantsignaal"
    ].tolist()

    beleving, onderwerp = [], []
    for label in validated_labels:
        if label in onderwerp_signals:
            onderwerp.append(label)
        elif label in beleving_signals:
            beleving.append(label)
        else:
            print(
                f"ValueError: Label '{label}' not found in predefined signals. Continuing with the next label."
            )
    return beleving, onderwerp


def calculate_metrics_signals(
    signal_type: str,
    generated: list,
    actual: list,
    cursor: sqlite3.Cursor,
) -> None:
    """
    Calculate true positives, false positives, false negatives, and true negatives
    for the onderwerp signals.
    """

    # Select the appropriate signal list based on signal_type
    if signal_type == "onderwerp":
        signals = DATA[DATA["Hoofd_klantsignaal"].isin(ONDERWERP_SIGNALS)][
            "Sub_klantsignaal"
        ].tolist()
    elif signal_type == "beleving":
        signals = DATA[DATA["Hoofd_klantsignaal"].isin(BELEVING_SIGNALS)][
            "Sub_klantsignaal"
        ].tolist()
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    # Calculate true positives, false positives, false negatives, and true negatives for each label
    for label in signals:
        tp, tn, fp, fn = 0, 0, 0, 0
        if label in generated and label in actual:
            tp += 1

        elif label in generated and label not in actual:
            fp += 1

        elif label not in generated and label in actual:
            fn += 1

        elif label not in generated and label not in actual:
            tn += 1

        # Add the scores to the database
        cursor.execute(
            """
            INSERT INTO scores (signal_type, label, tp, fp, fn, tn)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_type, label) DO UPDATE SET
                tp = tp + excluded.tp,
                fp = fp + excluded.fp,
                fn = fn + excluded.fn,
                tn = tn + excluded.tn
            """,
            (
                signal_type,
                label,
                tp,
                fp,
                fn,
                tn,
            ),
        )


def setup_local_db(local_db_path: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(local_db_path)
    cursor = conn.cursor()

    # Create text_and_labels table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS texts_and_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            gold_labels TEXT,
            generated_labels TEXT
        )
    """
    )

    # Create scores table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_type TEXT,
            label TEXT,
            tp INTEGER,
            fp INTEGER,
            fn INTEGER,
            tn INTEGER,
            UNIQUE(signal_type, label)
        )
    """
    )

    conn.commit()
    return conn, cursor


def load_data_source(args, cursor):
    """
    Load data from either Excel file or HuggingFace dataset based on arguments.
    
    Returns:
        Dataset: HuggingFace dataset with 'text' and 'gold_labels' columns
    """
    if args.excel_file:
        # Load from Excel file
        try:
            df = pd.read_excel(args.excel_file)
            
            # Validate required columns exist
            missing_columns = []
            if args.text_column not in df.columns:
                missing_columns.append(f"text column '{args.text_column}'")
            if args.labels_column not in df.columns:
                missing_columns.append(f"labels column '{args.labels_column}'")
                
            if missing_columns:
                print(f"Error: The following columns were not found: {', '.join(missing_columns)}")
                print(f"\nAvailable columns in '{args.excel_file}':")
                for col in df.columns:
                    print(f"  - {col}")
                print("\nPlease specify the correct column names using:")
                print("  --text-column <column_name>   for the text to process")
                print("  --labels-column <column_name> for the validated labels")
                sys.exit(1)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                args.text_column: 'text',
                args.labels_column: 'gold_labels'
            })
            
            # Convert labels to string format if needed (Excel might have lists as strings)
            if 'gold_labels' in df.columns:
                df['gold_labels'] = df['gold_labels'].apply(
                    lambda x: str(x.split('; ')) if isinstance(x, str) and '; ' in x else str([x]) if pd.notna(x) else '[]'
                )
            
            # Convert to HuggingFace Dataset
            dataset = Dataset.from_pandas(df[['text', 'gold_labels']])
            print(f"Loaded {len(dataset)} rows from Excel file: {args.excel_file}")
            
        except FileNotFoundError:
            print(f"Error: Excel file not found: {args.excel_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            sys.exit(1)
    else:
        # Load from HuggingFace dataset
        print(f"Loading HuggingFace dataset: {args.hf_dataset}")
        
        # Get last processed ID from database
        cursor.execute("SELECT MAX(id) FROM texts_and_labels")
        last_id = cursor.fetchone()[0]
        if last_id is None:
            last_id = 0
            
        # Load dataset and apply limit
        raw_dataset = load_dataset(args.hf_dataset, split="test")
        
        # Calculate the range to process
        start_idx = last_id
        end_idx = min(last_id + args.limit, len(raw_dataset))
        
        if start_idx >= len(raw_dataset):
            print(f"All rows already processed (last_id={last_id}, dataset size={len(raw_dataset)})")
            sys.exit(0)
            
        dataset = raw_dataset.select(range(start_idx, end_idx))
        print(f"Processing rows {start_idx} to {end_idx-1} from HuggingFace dataset (limit={args.limit})")
        
        # Rename columns to match expected format
        dataset = dataset.rename_columns({
            'Synthetic Text': 'text',
            'validated_labels': 'gold_labels'
        })
    
    return dataset


def write_to_excel(cursor: sqlite3.Cursor, excel_path: str) -> None:
    """
    Write the texts and labels and scores to an Excel file.
    """
    # Get the texts and labels from the database
    cursor.execute("SELECT * FROM texts_and_labels")
    texts_and_labels = cursor.fetchall()
    df_texts = pd.DataFrame(
        texts_and_labels,
        columns=["id", "text", "gold_labels", "generated_labels"],
    )
    df_texts["gold_labels"] = df_texts["gold_labels"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_texts["generated_labels"] = df_texts["generated_labels"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_texts["id"] = df_texts["id"].astype(int)
    df_texts = df_texts.sort_values(by="id").reset_index(drop=True)

    # Get the scores from the database
    cursor.execute("SELECT * FROM scores")
    scores = cursor.fetchall()
    existing_scores = {}
    for row in scores:
        signal_type = row[1]
        label = row[2]
        tp = row[3]
        fp = row[4]
        fn = row[5]
        tn = row[6]
        if signal_type not in existing_scores:
            existing_scores[signal_type] = {}
        existing_scores[signal_type][label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # Write the texts and labels and scores to an Excel file
    try:
        with pd.ExcelWriter(excel_path) as writer:
            df_texts.to_excel(writer, sheet_name="texts_and_labels", index=False)

            for signal_type, scores in existing_scores.items():
                df = pd.DataFrame(scores).T

                df["precision"] = np.where(
                    (df["tp"] + df["fp"]) > 0, df["tp"] / (df["tp"] + df["fp"]), 0
                )
                df["recall"] = np.where(
                    (df["tp"] + df["fn"]) > 0, df["tp"] / (df["tp"] + df["fn"]), 0
                )
                df["f1_score"] = np.where(
                    (df["precision"] + df["recall"]) > 0,
                    (2 * df["precision"] * df["recall"])
                    / (df["precision"] + df["recall"]),
                    0,
                )
                df.to_excel(writer, sheet_name=f"{signal_type}_scores")

        print(f"Metrics written to {excel_path}")
    except Exception as e:
        print(f"Error writing to Excel: {e}")


def main(args) -> None:
    """
    Main function to run the metrics calculation.
    """
    # Setup local db to save metrics
    conn, cursor = setup_local_db(args.db_path)
    
    # Load data from appropriate source
    dataset = load_data_source(args, cursor)

    # Initialize the pipeline
    pipeline = TextToKGPipeline(
        llm=azure_llm(model_prefix="GPT41", temperature=0.0), add_labels=True
    ).compile()

    for row in dataset:
        # Get the actual labels from the dataset
        gold_labels = ast.literal_eval(row["gold_labels"])
        if not gold_labels:
            gold_labels = ["No subtopic found"]

        # Initialize the pipeline with the state and run it
        state = TextToKGState(
            text=row["text"],
        )
        state = pipeline.invoke(state)

        # Get tp, tn, fp, fn for each signal type
        try:
            # Get the generated sub_signal labels from the state
            beleving_generated, onderwerp_generated = get_labels_from_json_ld(state)

            # Get the actual sub_signal labels from the dataset
            beleving_actual, onderwerp_actual = get_labels_from_validated_list(
                gold_labels
            )

            try:
                # Add the text and labels to the database
                cursor.execute(
                    "INSERT INTO texts_and_labels (text, gold_labels, generated_labels) VALUES (?, ?, ?)",
                    (
                        row["text"],
                        str(gold_labels),
                        str(beleving_generated + onderwerp_generated),
                    ),
                )

                calculate_metrics_signals(
                    "onderwerp", onderwerp_generated, onderwerp_actual, cursor
                )
                calculate_metrics_signals(
                    "beleving", beleving_generated, beleving_actual, cursor
                )

                # Commit the changes to the database
                conn.commit()
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                continue
        except ValueError as e:
            print(f"ValueError processing labels: {e}")
            continue
        except KeyError as e:
            print(f"KeyError processing labels: {e}")
            continue

    # Write to Excel file
    write_to_excel(cursor, args.output_excel)

    conn.close()
    print(f"\nMetrics saved to: {args.output_excel}")


def create_argument_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run metrics calculation on text-to-knowledge-graph pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use HuggingFace dataset (default, process 10 rows)
  python run_metrics.py

  # Use Excel file with specific columns
  python run_metrics.py --excel-file src/data/Sample_10_teksten.xlsx \\
    --text-column "Toelichting_masked" \\
    --labels-column "Categorieën samengevoegd"

  # Process more rows from HuggingFace dataset
  python run_metrics.py --limit 100
        """
    )
    
    # Data source options
    data_group = parser.add_argument_group('data source options')
    data_group.add_argument(
        '--excel-file',
        type=str,
        help='Path to Excel file to process (if not provided, uses HuggingFace dataset)'
    )
    data_group.add_argument(
        '--text-column',
        type=str,
        help='Name of the text column in Excel file (required with --excel-file)'
    )
    data_group.add_argument(
        '--labels-column',
        type=str,
        help='Name of the labels column in Excel file (required with --excel-file)'
    )
    data_group.add_argument(
        '--hf-dataset',
        type=str,
        default='UWV/wim_synthetic_data_for_testing',
        help='HuggingFace dataset name (default: %(default)s)'
    )
    data_group.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of rows to process from HuggingFace dataset (default: %(default)s)'
    )
    
    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--output-excel',
        type=str,
        default='src/data/metrics.xlsx',
        help='Path for output Excel file with metrics (default: %(default)s)'
    )
    output_group.add_argument(
        '--db-path',
        type=str,
        default='src/data/metrics.db',
        help='Path for SQLite database (default: %(default)s)'
    )
    
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.excel_file:
        if not args.text_column or not args.labels_column:
            parser.error("--text-column and --labels-column are required when using --excel-file")
    
    main(args)

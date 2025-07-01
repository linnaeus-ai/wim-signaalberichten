import ast
import json
import sqlite3

import numpy as np
import pandas as pd

from datasets import load_dataset
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


def setup_local_db(local_db_pah: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(local_db_path)
    cursor = conn.cursor()

    # Create text_and_labels table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS texts_and_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            golden_labels TEXT,
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


def write_to_excel(cursor: sqlite3.Cursor, excel_path: str) -> None:
    """
    Write the texts and labels and scores to an Excel file.
    """
    # Get the texts and labels from the database
    cursor.execute("SELECT * FROM texts_and_labels")
    texts_and_labels = cursor.fetchall()
    df_texts = pd.DataFrame(
        texts_and_labels,
        columns=["id", "text", "golden_labels", "generated_labels"],
    )
    df_texts["golden_labels"] = df_texts["golden_labels"].apply(
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


def main(excel_path: str, local_db_path: str, hf_ds_path: str) -> None:
    """
    Main function to run the metrics calculation.
    """
    # Setup local db to save metrics
    conn, cursor = setup_local_db(local_db_path)

    # Read from db, check last id
    cursor.execute("SELECT MAX(id) FROM texts_and_labels")
    last_id = cursor.fetchone()[0]
    if last_id is None:
        last_id = 0

    # Load dataset from HF
    raw_dataset = load_dataset(hf_ds_path, split="test")
    dataset = raw_dataset.select(range(last_id, len(raw_dataset)))

    # Initialize the pipeline
    pipeline = TextToKGPipeline(
        llm=azure_llm(model_prefix="GPT41", temperature=0.0), add_labels=True
    ).compile()

    for row in dataset:
        # Get the actual labels from the dataset
        validated_labels = ast.literal_eval(row["validated_labels"])
        if not validated_labels:
            validated_labels = ["No subtopic found"]

        # Initialize the pipeline with the state and run it
        state = TextToKGState(
            text=row["Synthetic Text"],
        )
        state = pipeline.invoke(state)

        # Get tp, tn, fp, fn for each signal type
        try:
            # Get the generated sub_signal labels from the state
            beleving_generated, onderwerp_generated = get_labels_from_json_ld(state)

            # Get the actual sub_signal labels from the dataset
            beleving_actual, onderwerp_actual = get_labels_from_validated_list(
                validated_labels
            )

            try:
                # Add the text and labels to the database
                cursor.execute(
                    "INSERT INTO texts_and_labels (text, golden_labels, generated_labels) VALUES (?, ?, ?)",
                    (
                        row["Synthetic Text"],
                        str(validated_labels),
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
    write_to_excel(cursor, excel_path)

    conn.close()


if __name__ == "__main__":
    excel_path = "src/data/metrics.xlsx"
    local_db_path = "src/data/metrics.db"
    hf_ds_path = "UWV/wim_synthetic_data_for_testing"

    main(excel_path, local_db_path, hf_ds_path)

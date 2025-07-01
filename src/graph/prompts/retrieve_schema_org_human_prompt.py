RETRIEVE_SCHEMA_ORG_HUMAN_PROMPT = """Je taak is om de best passende schema.org class te selecteren voor een entiteit die gevonden is in de volgende tekst:

**Originele tekst**:
{ORIGINAL_TEXT}

Een van de entiteiten in deze tekst is `{ENTITY_NAME}`.
Een passende class hiervoor zou zijn `{CLASS_NAME}` met beschrijving `{DESCRIPTION}`
Maar we mogen alleen schema.org classes gebruiken.
Jouw taak is om voor entiteit zoals gebruikt in de originele tekst, de best passende schema.org class te kiezen.

**Mogelijke Schema.org kandidaten**:
{SCHEMA_CANDIDATES}

Instructies:
1. Beredeneer je keuze in één zin in het formaat: "Omdat [korte reden] is in de gegeven tekst [geselecteerde class] de meest geschikte schema.org class voor class {CLASS_NAME}." 
2. Geef de naam van de geselecteerde schema.org class.
3. Selecteer het nummer (1-5) van de schema.org class.

Let op: Houd je verklaring beknopt en focus op de belangrijkste reden voor je keuze."""

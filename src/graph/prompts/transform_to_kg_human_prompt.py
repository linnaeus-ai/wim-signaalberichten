TRANSFORM_TO_KG_HUMAN_PROMPT = """
Zet de onderstaande **Input Tekst** en bijbehorende **Schema Definities** om in een valide en gestructureerde JSON-LD object volgens de schema.org standaarden.

**Input Text**:
\"\"\"{INPUT_TEXT}\"\"\"

**Schema Definities**:
{SCHEMA_DEFINITIONS}

**JSON-LD Structure Requirements:**
- **`@context`**: Gebruik `{{"@context": "https://schema.org"}}`.
- **`@type`**: Selecteer het meest geschikte schema.org type voor het hoofdobject gebaseerd op de inhoud.
- **`identifier`**: Voeg unieke identificatoren toe voor alle entiteiten waar van toepassing. Gebruik duidelijke en traceerbare identifiers.
- **`keywords`**: Extraheer relevante trefwoorden uit de tekstinhoud en modelleer deze als een array, alleen passend bij CreativeWork, Event, Organization, Place, Product

**Entity Modeling:**
- **Locaties**: Model geografische verwijzingen als `Place` objecten met passende eigenschappen:
   - Gebruik `containedInPlace` voor hiërarchische locatierelaties
   - Voeg `addressLocality`, `addressCountry` (ISO 3166-1 alpha-2) toe waar relevant
   - Gebruik `contentLocation` voor locaties gerelateerd aan de inhoud
- **Personen en Organisaties**: Model als `Person` of `Organization` met relevante eigenschappen zoals naam, functie, rol.

**Hierarchical Relationship Modeling:**
VERMIJD vlakke structuren. Creëer diepe, onderling verbonden grafen met deze patronen:

1. **Ruimtelijke Insluiting** (voor Places):
   - Gebruik `containsPlace`/`containedInPlace` voor geneste locaties
   - Bouw ketens: Land → Regio → Stad → Gebouw → Kamer
   - Voor entiteiten IN een plaats: gebruik `location` of `containedInPlace`

2. **Organisatorische Hiërarchie**:
   - Gebruik `parentOrganization`/`subOrganization` 
   - Gebruik `department` voor interne structuur
   - Gebruik `memberOf`/`member` voor lidmaatschapsrelaties
   - Gebruik `affiliation` voor losse verbanden

3. **Deel-Geheel Relaties**:
   - Gebruik `hasPart`/`isPartOf` voor componenten
   - Gebruik specifieke eigenschappen zoals `workExample`, `exampleOfWork`

4. **Agentschap & Creatie**:
   - Gebruik `creator`/`author` met omgekeerde `subjectOf`
   - Gebruik `publisher`, `producer`, `contributor` voor rollen
   - Gebruik `owns`/`ownedBy` voor eigendom

5. **Referentie & Identiteit**:
   - Gebruik `"@id"` om refereerbare nodes te maken (bijv. `"@id": "#buurtschap"`)
   - Gebruik `{{"@id": "#nodeId"}}` om naar nodes elders te verwijzen
   - Dit creëert een echte graaf, niet alleen een boom

**Graaf Constructie Regels:**
- GEBRUIK NOOIT `about` of `mentions` als een catch-all voor relaties
- VERKIES ALTIJD specifieke relatie-eigenschappen
- CREËER bidirectionele links waar logisch
- GEBRUIK node referenties (@id) om duplicatie te vermijden
- BOUW de diepst mogelijke structuur uit de beschikbare informatie

**Relaties uit Input Analyse:**
{EXTRACTED_RELATIONS}
Gebruik deze om je relatiemodellering te sturen.

**BELANGRIJKE INSTRUCTIE - Maximale Property Vulling:**
- Voor elke entiteit die je modelleert, vul ZOVEEL MOGELIJK properties uit het bijbehorende schema
- Zoek actief in de input tekst naar informatie die overeenkomt met de beschikbare properties
- Voorbeelden:
  - Voor een Person: zoek naar jobTitle, affiliation, nationality, birthDate, knows, memberOf, etc.
  - Voor een Organization: zoek naar address, telephone, email, foundingDate, numberOfEmployees, etc.
  - Voor een Place: zoek naar geo coordinates, containedInPlace, address details, etc.
- Als je informatie kunt afleiden of redelijkerwijs kunt concluderen uit de context, gebruik het
- Het is beter om meer properties te vullen dan te weinig - wees uitgebreid

**Events & Temporal Data:**
- Model significante gebeurtenissen als `Event` objecten met:
    - `startDate`, `endDate` (gebruik ISO 8601: "YYYY-MM-DD")
    - `location` voor waar het plaatsvond
    - `organizer` voor wie het organiseerde
    - `about` voor wat het betreft
- Voor historische perioden, gebruik `temporalCoverage` met bereiken (bijv. "1400/1499").
- Gebruik `dateCreated`, `datePublished`, `dateModified` waar van toepassing.

**Relationships:**
- Verbind entiteiten met passende schema.org eigenschappen:
    - `knows`, `colleague`, `spouse` voor persoonrelaties
    - `actor`, `performer`, `attendee` voor deelname aan events
    - `containedInPlace`, `location` voor plaatsrelaties
    - `subOrganizationOf`, `memberOf` voor organisatierelaties
- Voor creatieve werken:
    - Gebruik `creator`/`author` voor makers
    - Gebruik `about` voor onderwerpen binnen CreativeWork
    - Gebruik `subjectOf` om te verwijzen naar CreativeWork over een entiteit
- Voor verhalen/narratieven:
    - Gebruik `character` voor personages (niet `mentions`)
    - Voeg eigenschappen toe zoals `jobTitle`, `skills` waar relevant

**Algemene Vereisten:**
- **Taalvereiste**: Alle tekstinhoud MOET in het Nederlands zijn. Schema.org eigenschapsnamen blijven in het Engels.
- **Gebruik uitsluitend standaard schema.org eigenschappen en types uit de core vocabulary.**
- **Wees volledig**: Extraheer alle relevante entiteiten, gebeurtenissen en relaties uit de tekst.
- **Wees precies**: Gebruik de meest specifieke en passende schema.org types en eigenschappen.
- **Maximaliseer property gebruik**: Voor elke entiteit, gebruik ALLE relevante properties uit het schema waarvoor je informatie hebt of kunt afleiden
- **Final Output**: Antwoord mag ALLEEN valide JSON-LD bevatten zonder markdown formatting of uitleg. Geen codeblokken of commentaar na de JSON-LD.

Focus op semantische rijkdom, correcte relaties, en web-embedbare output. Zorg ervoor dat alle entiteiten, eigenschappen en relaties volledig en correct worden gemodelleerd volgens schema.org standaarden.

**Input Text**:
\"\"\"{INPUT_TEXT}\"\"\"

{RERUN}
"""
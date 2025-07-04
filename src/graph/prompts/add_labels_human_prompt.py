ADD_LABELS_HUMAN_PROMPT = """Analyseer de onderstaande tekst en bepaal welke labels van toepassing zijn. 
Houd rekening met de context en de specifieke inhoud van de tekst. 
Selecteer alleen de labels die relevant zijn uit beide categorieën.

**Instructies**:
1. Er zijn twee categorieën labels: Onderwerp (waar het over gaat) en Beleving (hoe de service ervaren wordt).
2. Kies uit elke categorie één of meerdere passende labels. Als er geen passend label is, kies dan "No subtopic found".
3. Selecteer alleen labels die nauw aansluiten bij de inhoud van de tekst.
4. BELANGRIJK: Gebruik ALLEEN labels uit de onderstaande lijsten. Maak GEEN nieuwe labels aan.
5. Vertaal labels NIET naar andere woorden (bijvoorbeeld: gebruik "No subtopic found", niet vertalen naar Nederlands).

**Onderwerp labels** (waar gaat het over):
{ONDERWERP_LIST}

**Beleving labels** (hoe wordt de service ervaren):
{BELEVING_LIST}

**Outputvoorbeeld**:
Onderwerp: [Parkeervergunning/abonnement, Parkeerkosten]
Beleving: [Communicatie, Duidelijkheid]

**Tekst om te analyseren**:
\"\"\"
{INPUT_TEXT}
\"\"\"
"""

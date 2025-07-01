ENTITY_EXTRACTION_HUMAN_PROMPT = """Analyseer de gegeven Nederlandse tekst en voer de volgende taken uit:

1. Geef eerst een beknopte samenvatting van waar de tekst over gaat (maximaal 2 zinnen).

2. Identificeer alle belangrijke entiteiten in de tekst. Voor elke entiteit:
   - Gebruik de exacte naam zoals vermeld in de tekst
   - Bepaal het meest specifieke type/class dat bij deze entiteit past
   - Geef een korte Engelse beschrijving van wat dit type/class vertegenwoordigt
   - Zorg dat alle expliciete en sterk impliciete entiteiten worden opgenomen, inclusief numerieke gegevens, tijdsaanduidingen en programma-/dienstnamen

3. Identificeer de relaties tussen de gevonden entiteiten. Voor elke relatie:
   - Gebruik werkwoorden of relationele termen die de verbinding duidelijk maken
   - Zorg dat de relaties logisch voortvloeien uit de tekst en expliciet of sterk impliciet aanwezig zijn
   - Geef prioriteit aan relaties die actionable zijn voor knowledge graph constructie
   - Beschrijf de context van de relatie indien relevant, zoals financiering, locatie, betrokkenheid, rolverdeling, tijdsduur, of andere attributen
   - Zorg dat relaties tussen abstracte concepten en entiteiten correct worden vastgelegd

Formattering:
- Begin met de samenvatting van de tekst in een aparte sectie met <summary></summary> tags:
  <summary>
  ...
  </summary>
- Lijst daarna alle entiteiten op de volgende manier met <entiteiten></entiteiten> tags:
  <entiteiten>
  EntiteitNaam | SpecifiekeClass | Engelse beschrijving van de class
  ... | ... | ...
  </entiteiten>
- Lijst vervolgens alle relaties op de volgende manier met <relaties></relaties> tags:
  <relaties>
  EntiteitA | relatieType | EntiteitB
  ... | ... | ...
  </relaties>

Richtlijnen:
- Wees zo specifiek mogelijk bij het bepalen van classes (bijvoorbeeld 'SocialHousingProgram' in plaats van alleen 'Program')
- Gebruik standaard Engelse termen voor de class-beschrijvingen
- Extraheer alle entiteiten en relaties die expliciet of sterk impliciet in de tekst aanwezig zijn, inclusief numerieke gegevens, tijdsaanduidingen en programma-/dienstnamen
- Behoud Nederlandse namen voor entiteiten, maar gebruik Engelse termen voor classes en beschrijvingen
- Zorg dat relaties betekenisvol en actionable zijn voor knowledge graph constructie
- Controleer zorgvuldig op ontbrekende entiteiten en relaties, inclusief samenwerkingen, financiële verbanden, tijdsgebonden acties en doelgroepen
- Vermijd generieke of incorrecte classificaties door zorgvuldig de rol, functie, of context van de entiteit te analyseren

Tekst om te analyseren:
{INPUT_TEXT}
"""

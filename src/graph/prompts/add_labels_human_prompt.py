ADD_LABELS_HUMAN_PROMPT = """Analyseer de onderstaande tekst en bepaal welke labels van toepassing zijn. 
Houd rekening met de context en de specifieke inhoud van de tekst. 
Selecteer alleen de labels die relevant zijn en vermeld zowel het label als de set waarin het voorkomt.

**Instructies**:
1. De labels zijn gestructureerd als volgt: {{set1: [label1, label2, ...], set2: [label10, label11, ...]}}.
2. Kies alleen labels die nauw aansluiten bij de inhoud van de tekst.
3. Noteer het resultaat in het formaat: "Set: [Label]".

**Beschikbare labels**:
{TOPICS_LIST}

**Tekst om te analyseren**:
{INPUT_TEXT}

**Outputvoorbeeld**:
Set1: [Label1]
Set2: [Label10]
"""

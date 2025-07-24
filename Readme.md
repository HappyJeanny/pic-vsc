# Infos zum Projekt

## Wie kann ich unter den ähnlichen Bildgruppen das "schönste" Bild automatisch auswählen?
- Schönheit ist subjektiv --> Sichtung
- Technischer Ansatz:
  - Bildschärfe
  - Helligkeit/Kontrast
  - Auflösung

  
Ästhetik-Modelle wie NIMA von Goolge könnte man in Step 2 implementieren. Modell versucht die ästhtische Qualität von Bildern zu bewerten. 
  

# Erster Start
Erstellen einer neuen virtuellen Umgebung
# Neue virtuelle Umgebung erstellen
python -m venv venv311
## Korrekte Aktivierung in PowerShell
### Aktivieren mit vollständigem Pfad
.\venv311\Scripts\Activate.ps1
### Aktivierung in der Windows-Eingabeaufforderung (cmd)
#### Wechseln Sie zur CMD und führen Sie aus:
venv311\Scripts\activate.bat



# Schnellstart VSC
- Neues Projekt - Folder in VSC öffnen
- venv311\Scripts\activate
- CMD Powershell oder Anaconda : 
cd c:\Python>cd pic-Spyder
streamlit run check-pic.py

- alt, ohne App python check-pic.py


## Virtuelle Umgebung einrichten
Im Terminal Python 3.11 verknüpfen
PS C:\Python\pic> C:\Users\IsabellMader\AppData\Local\Programs\Python\Python311\python.exe -m venv venv311  
>> 
PS C:\Python\pic> venv311\Scripts\activate
(venv311) PS C:\Python\pic> python --version
Python 3.11.0
(venv311) PS C:\Python\pic> 

## Tools installieren
- python -m pip install --upgrade pip 
- pip install tensorflow
- pip install numpy scikit-learn 
- pip install Pillow
- Installation Pyhton 3.11 mit .exe von https://www.python.org/downloads/release/python-3110/
- pip install transformers




| Parameter         | Typ             | Bedeutung                                                                                  |
|-------------------|-----------------|-------------------------------------------------------------------------------------------|
| BILDER_VERZEICHNIS| str             | Pfad zum Quellordner mit zu prüfenden Bildern                                             |
| ZIEL_ORDNER       | str             | Zielordner für Duplikate/ähnliche Bilder                                                  |
| IMG_SIZE          | tuple (int, int)| Zielgröße (Breite, Höhe) für die Bildskalierung vor MobileNet                             |
| SCHWELLWERT       | float           | Schwellwert für die Ähnlichkeit (euklidischer Abstand der Featurevektoren)                |
| MIN_ABSTAND       | float           | Toleranz für exakt identische Bilder (sehr kleiner Wert, z.B. 1e-6)                       |





| Parameter / Metrik | Bedeutung                                                         | Hinweis                         |
|--------------------|-------------------------------------------------------------------|---------------------------------|
| loss               | Verlustfunktion, z. B. 'sparse_categorical_crossentropy'          | Gibt den Optimierungsfokus vor  |
| accuracy           | Anteil korrekt klassifizierter Beispiele                          | Standard für Klassifikation     |
| recall             | Sensitivität / Anteil der richtig erkannten Positiven             | Keras: keras.metrics.Recall()   |
| precision          | Positiver Vorhersagewert                                          | Keras: keras.metrics.Precision()|
| F1-Score           | Harm. Mittel aus Precision & Recall, robust bei Klassenungleichgew.| Addon: tfa.metrics.F1Score()    |


## Vergleich YOLOv8 vs OWL-ViT

| Merkmal                      | **YOLOv8** (You Only Look Once)       | **OWL-ViT** (Open-World Language-based Vision Transformer)                |
| ---------------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| **Art der Objekterkennung**  | **Fixe Klassen** (z. B. Mensch, Hund) | **Textgesteuert (Zero-Shot)** – du gibst „person“, „car“… selbst an       |
| **Flexibilität**             | Begrenzte Klassenanzahl               | Sehr flexibel: Du kannst schreiben: „fully visible person“, „man in suit“ |
| **Training**                 | Klassisch trainiert auf COCO etc.     | Vortrainiert auf viele Beschreibungen + CLIP                              |
| **Prompt-Steuerung**         | ❌ Nicht möglich                       | ✅ Möglich, z. B.: „person standing“, „dog sitting“                        |
| **Geschwindigkeit**          | ⚡ Sehr schnell (auch mobil)           | 🐢 Langsamer, GPU empfohlen                                               |
| **Einfachheit der Nutzung**  | Sehr einfach mit `ultralytics`-Lib    | Etwas komplexer mit `transformers` und Bild/Text-Handling                 |
| **Besonderheit**             | ✅ Bounding Boxes sofort verfügbar     | ✅ Du gibst „was“ du erkennen willst – ohne Training                       |
| **Kombination mit Qualität** | Gut geeignet zur Vorauswahl           | Gut geeignet für **feinere Unterscheidung nach Sprache**                  |



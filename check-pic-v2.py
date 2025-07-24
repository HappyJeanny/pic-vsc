# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:11:48 2025

@author: IsabellMader

Zusatzfunktion:
Der User soll in der Web-Applikation auch ein anderes Bild (aus dem Unterordner "schlechtere Bilder") als "bestes" auswählen können. 
"""

import streamlit as st
import os
import numpy as np
from PIL import Image
import cv2
import shutil
import time 
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialisiere Session-State für persistente Daten
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'gruppen_info' not in st.session_state:
    st.session_state.gruppen_info = []
if 'gruppen_bewertungen' not in st.session_state:
    st.session_state.gruppen_bewertungen = []
if 'ziel_ordner' not in st.session_state:
    st.session_state.ziel_ordner = "similar"

st.set_page_config(page_title="Bildgruppen-Analyse", layout="wide")
st.title("🔎 Ähnliche Bilder finden & bestes Bild pro Gruppe auswählen")

# Funktion zum Aktualisieren des besten Bildes einer Gruppe
def change_best_image(gruppe_nr, altes_best, neues_best):
    """Ändert das beste Bild einer Gruppe und aktualisiert alle Ordner"""
    i = gruppe_nr
    gruppe_id = f"group_{i:03d}"
    gruppen_ordner = os.path.join(st.session_state.ziel_ordner, gruppe_id)
    schlechter_ordner = os.path.join(gruppen_ordner, "schlechtere_bilder")
    alle_besseren_ordner = os.path.join(st.session_state.ziel_ordner, "alle_besseren_bilder")
    
    # 1. Aktuelles bestes Bild in schlechtere Bilder verschieben
    altes_best_pfad = os.path.join(gruppen_ordner, altes_best)
    neues_pfad_fuer_altes = os.path.join(schlechter_ordner, altes_best)
    shutil.move(altes_best_pfad, neues_pfad_fuer_altes)
    
    # 2. Neues bestes Bild in Hauptordner verschieben
    neues_best_pfad = os.path.join(schlechter_ordner, neues_best)
    neues_ziel_pfad = os.path.join(gruppen_ordner, neues_best)
    shutil.move(neues_best_pfad, neues_ziel_pfad)
    
    # 3. In alle_besseren_ordner aktualisieren
    alter_best_pfad_global = os.path.join(alle_besseren_ordner, altes_best)
    if os.path.exists(alter_best_pfad_global):
        os.remove(alter_best_pfad_global)
    shutil.copy(neues_ziel_pfad, os.path.join(alle_besseren_ordner, neues_best))
    
    # 4. Bewertungsdatei aktualisieren
    with open(os.path.join(gruppen_ordner, "bewertung.txt"), "a", encoding="utf-8") as f:
        f.write(f"\nMANUELL GEÄNDERT: Neues bestes Bild ist: {neues_best}\n")
    
    # 5. Gruppe in gruppen_info aktualisieren
    for idx, (nr, bilder, _, score) in enumerate(st.session_state.gruppen_info):
        if nr == i:
            st.session_state.gruppen_info[idx] = (nr, bilder, neues_best, score)
            break
    
    # 6. Erfolgsbenachrichtigung
    st.toast(f"Bestes Bild für Gruppe {i} auf {neues_best} geändert!")

# --- 1. Ordnerauswahl über file_uploader (moderne Alternative zu folder-picker) ---
if not st.session_state.analysis_complete:
    st.subheader("1. Bild-Ordner auswählen")
    st.markdown(
        "Wähle mehrere Bilder aus einem Ordner aus (STRG+A oder STRG+Mausklick für Mehrfachauswahl). "
        "Die Verzeichnisstruktur bleibt erhalten."
    )
    uploaded_files = st.file_uploader(
        "Bilder auswählen",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="files"
    )

    if not uploaded_files or len(uploaded_files) < 2:
        st.warning("Bitte mind. 2 Bilder hochladen.")
        st.stop()

    # --- 2. Modellwahl ---
    st.subheader("2. Modell wählen")
    model_typ = st.selectbox(
        "Feature-Extraktion mit Modell",
        ["MobileNet", "ResNet50"],
        help="MobileNet ist schneller (Schwellwert ca. 20), ResNet50 ist genauer aber langsamer (Schwellwert 50)."
    )

    # --- 3. Schwellwert ---
    st.subheader("3. Ähnlichkeitsschwelle einstellen")
    schwellwert = st.slider(
        "Schwellwert für Ähnlichkeit (empfohlene Werte: 10-50)",
        min_value=1.0, max_value=100.0, value=20.0, step=1.0,
        help="Je kleiner der Wert, desto strenger wird gruppiert. Probiere 20, bei vielen Gruppen auch mal 10 oder 30."
    )

    # --- 4. Zielverzeichnis (im Arbeitsspeicher, temporär) ---
    ZIEL_ORDNER = "similar"
    st.session_state.ziel_ordner = ZIEL_ORDNER
    os.makedirs(ZIEL_ORDNER, exist_ok=True)

    IMG_SIZE = (224, 224)

    # --- 5. Analyse starten ---
    if st.button("🔍 Analyse starten"):
        start_time = time.time()
        st.info(f"Starte Analyse mit Modell: `{model_typ}` und Schwellwert: {schwellwert}")
        

        # 5.1 Speichere alle hochgeladenen Bilder temporär ab
        bild_pfade = []
        for file in uploaded_files:
            if file is not None:
                filename = file.name
                temp_path = os.path.join(ZIEL_ORDNER, filename)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                bild_pfade.append(temp_path)
        st.write(f"{len(bild_pfade)} Bilder temporär gespeichert.")

        # 5.2 Modell wählen
        if model_typ == "MobileNet":
            base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = mobilenet_preprocess
        else:
            base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = resnet_preprocess

        # 5.3 Features berechnen
        features = []
        valid_paths = []
        progress = st.progress(0, text="Bilder werden verarbeitet ...")
        for i, pfad in enumerate(bild_pfade):
            try:
                img = load_img(pfad, target_size=IMG_SIZE)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_func(x)
                feat = base_model.predict(x, verbose=0)
                features.append(feat.flatten())
                valid_paths.append(pfad)
            except Exception as e:
                st.error(f"Fehler bei {pfad}: {e}")
            if len(bild_pfade) > 0:
                progress.progress((i+1)/len(bild_pfade), text=f"{i+1}/{len(bild_pfade)} Bilder")
        features = np.array(features)
        st.success(f"Extrahierte Features für {len(features)} Bilder.")

        if len(features) < 2:
            st.error("Nicht genug Bilder für Vergleich.")
            st.stop()

        # 5.4 Gruppenbildung nach Ähnlichkeit
        knn = NearestNeighbors(radius=schwellwert, metric='euclidean')
        knn.fit(features)
        nachbarn = knn.radius_neighbors(features, return_distance=False)

        gruppen = []
        zugeordnet = set()
        for idx, nb_idxs in enumerate(nachbarn):
            if idx not in zugeordnet:
                gruppe = set(nb_idxs)
                gruppe.add(idx)
                zugeordnet |= gruppe
                gruppen.append(list(gruppe))
        gruppen = [g for g in gruppen if len(g) > 1]
        st.success(f"{len(gruppen)} ähnliche Bildgruppen gefunden.")

        # 5.5 Bewertungsfunktionen
        def berechne_schaerfe(pfad):
            try:
                img = cv2.imread(pfad, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return 0
                return cv2.Laplacian(img, cv2.CV_64F).var()
            except Exception:
                return 0
        def berechne_helligkeit(pfad):
            try:
                img = Image.open(pfad).convert('L')
                arr = np.array(img)
                return arr.mean()
            except Exception:
                return 0
        def berechne_aufloesung(pfad):
            try:
                img = Image.open(pfad)
                return img.size[0] * img.size[1]
            except Exception:
                return 0
        def finde_bestes_bild(bildpfade):
            bewertungen = []
            for pfad in bildpfade:
                sch = berechne_schaerfe(pfad)
                hel = berechne_helligkeit(pfad)
                auf = berechne_aufloesung(pfad)
                score = 2*sch + hel + 0.0001*auf
                bewertungen.append((score, pfad))
            best_score, best_pfad = max(bewertungen)
            return best_pfad, best_score

        # um alle besseren Bilder nicht in Unterordner zu halten
        alle_besseren_ordner = os.path.join(ZIEL_ORDNER, "alle_besseren_bilder")
        os.makedirs(alle_besseren_ordner, exist_ok=True)

        gruppen_info = []
        gruppen_bewertungen = []  # für extra Bewertungs-Textdateien

        for i, gruppe in enumerate(gruppen, 1):
            gruppen_ordner = os.path.join(ZIEL_ORDNER, f"group_{i:03d}")
            schlechter_ordner = os.path.join(gruppen_ordner, "schlechtere_bilder")
            os.makedirs(gruppen_ordner, exist_ok=True)
            os.makedirs(schlechter_ordner, exist_ok=True)
            gruppen_bilder = [valid_paths[idx] for idx in gruppe]
            
            # Bewertung aller Bilder inkl. Einzelwerte
            bewertungen = []
            for pfad in gruppen_bilder:
                sch = berechne_schaerfe(pfad)
                hel = berechne_helligkeit(pfad)
                auf = berechne_aufloesung(pfad)
                score = 2*sch + hel + 0.0001*auf
                bewertungen.append((pfad, sch, hel, auf, score))
            # Bestes Bild finden
            best_tuple = max(bewertungen, key=lambda x: x[4])
            best_pfad, best_sch, best_hel, best_auf, best_score = best_tuple
            best_bildname = os.path.basename(best_pfad)
            for pfad, sch, hel, auf, score in bewertungen:
                bildname = os.path.basename(pfad)
                if pfad == best_pfad:
                    ziel_pfad = os.path.join(gruppen_ordner, bildname)
                else:
                    ziel_pfad = os.path.join(schlechter_ordner, bildname)
                shutil.copy(pfad, ziel_pfad)
            # Bestes Bild zusätzlich in Sammelordner
            shutil.copy(best_pfad, os.path.join(alle_besseren_ordner, best_bildname))
            gruppen_info.append((i, gruppen_bilder, best_bildname, best_score))
            gruppen_bewertungen.append((i, bewertungen, best_bildname, best_score))

            # Pro Gruppe: Bewertungsdatei
            txt_bewertung = os.path.join(gruppen_ordner, "bewertung.txt")
            with open(txt_bewertung, "w", encoding="utf-8") as f:
                f.write("Dateiname,Schärfe,Helligkeit,Auflösung,Score\n")
                for pfad, sch, hel, auf, score in bewertungen:
                    f.write(f"{os.path.basename(pfad)},{sch:.2f},{hel:.2f},{auf},{score:.2f}\n")

        # 5.6 Übersicht als Textdatei (Download anbieten)
        txt_path = os.path.join(ZIEL_ORDNER, "gruppen_übersicht.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
          for i, bewertungen, best_bildname, best_score in gruppen_bewertungen:
            f.write(f"Gruppe {i:03d}:\n")
            f.write(f"{'Dateiname':20}{'Schärfe':>10}{'Helligkeit':>12}{'Auflösung':>12}{'Score':>10}\n")
            for pfad, sch, hel, auf, score in bewertungen:
                mark = " <-- BESTES BILD" if os.path.basename(pfad) == best_bildname else ""
                f.write(f"{os.path.basename(pfad):20}{sch:10.2f}{hel:12.2f}{auf:12}{score:10.2f}{mark}\n")
            f.write(f"Das beste Bild ist: {best_bildname} (Score: {best_score:.2f})\n\n")
        
        # Ergebnisse im Session-State speichern
        st.session_state.gruppen_info = gruppen_info
        st.session_state.gruppen_bewertungen = gruppen_bewertungen
        st.session_state.analysis_complete = True
        
        end_time = time.time()
        dauer = end_time - start_time
        mins = int(dauer // 60)
        secs = int(dauer % 60)
        st.info(f"Die Analyse hat {mins} Minuten {secs} Sekunden gedauert.")
        
        # Seite neu laden, um zur Ergebnisanzeige zu gelangen
        st.rerun()

# Wenn die Analyse abgeschlossen ist, zeige die Ergebnisse
if st.session_state.analysis_complete:
    # Übersicht als Download anbieten
    txt_path = os.path.join(st.session_state.ziel_ordner, "gruppen_übersicht.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "rb") as f:
            st.download_button(
                label="Gruppen-Übersicht als TXT herunterladen",
                data=f,
                file_name="gruppen_übersicht.txt",
                mime="text/plain"
            )
    
    # --- Ergebnisse als Tabelle, mit Bildvorschau ---
    st.subheader("Vorschau der Gruppen und besten Bilder")
    for i, gruppen_bilder, best_bildname, best_score in st.session_state.gruppen_info:
        cols = st.columns([2, 5])
        with cols[0]:
            pfad = [p for p in gruppen_bilder if os.path.basename(p) == best_bildname][0]
            try:
                st.image(pfad, caption=f"Gruppe {i:03d} - Bestes Bild", width=150)
            except Exception:
                st.write("(Bildvorschau nicht möglich)")
        with cols[1]:
            st.markdown(f"**Gruppe {i:03d}**  \nBestes Bild: `{best_bildname}`  \nScore: `{best_score:.2f}`  \nAnzahl ähnliche Bilder: {len(gruppen_bilder)}")
    
    # --- 6. Manuelles Auswählen des besten Bildes pro Gruppe ---
    st.subheader("Bestes Bild manuell auswählen")
    st.markdown(
        "Hier kannst du für jede Gruppe ein anderes Bild als 'bestes Bild' festlegen, falls du mit der automatischen Auswahl nicht zufrieden bist."
    )
    
    for i, gruppen_bilder, best_bildname, best_score in st.session_state.gruppen_info:
        gruppe_id = f"group_{i:03d}"
        gruppen_ordner = os.path.join(st.session_state.ziel_ordner, gruppe_id)
        schlechter_ordner = os.path.join(gruppen_ordner, "schlechtere_bilder")
        
        # Alle Bilder in der Gruppe (aktuell bestes + schlechtere)
        schlechtere_bilder = []
        if os.path.exists(schlechter_ordner):
            schlechtere_bilder = [os.path.basename(f) for f in os.listdir(schlechter_ordner) 
                                if os.path.isfile(os.path.join(schlechter_ordner, f))]
        alle_bilder = [best_bildname] + schlechtere_bilder
        
        # Expander für jede Gruppe zum Platzsparen
        with st.expander(f"Gruppe {i:03d} - Aktuell bestes Bild: {best_bildname}"):
            col1, col2, col3 = st.columns([2, 3, 2])
            
            # Aktuell bestes Bild anzeigen
            with col1:
                st.markdown("**Aktuell bestes Bild:**")
                best_pfad = os.path.join(gruppen_ordner, best_bildname)
                try:
                    st.image(best_pfad, caption=best_bildname, width=150)
                except Exception:
                    st.write("(Bildvorschau nicht möglich)")
            
            # Dropdown für Bildauswahl + Vorschau
            with col2:
                # Dropdown für alle Bilder in dieser Gruppe
                neues_bestes_bild = st.selectbox(
                    "Wähle ein anderes Bild als 'bestes Bild':",
                    options=alle_bilder,
                    index=0,  # Default ist das aktuelle beste Bild
                    key=f"select_best_{i}"
                )
                
                # Wenn ein anderes als das aktuelle beste Bild ausgewählt wurde
                if neues_bestes_bild != best_bildname:
                    # Pfad zum neu ausgewählten Bild
                    neues_bild_pfad = os.path.join(schlechter_ordner, neues_bestes_bild)
                    try:
                        st.image(neues_bild_pfad, caption=f"Neu ausgewähltes Bild: {neues_bestes_bild}", width=150)
                    except Exception:
                        st.write("(Bildvorschau nicht möglich)")
                    
                    # Button zum Anwenden der Änderung
                    if st.button("Als bestes Bild festlegen", key=f"set_best_{i}"):
                        change_best_image(i, best_bildname, neues_bestes_bild)
                        st.rerun()  # Seite neu laden, um Änderungen anzuzeigen
            
            # Alle schlechteren Bilder als Galerie anzeigen
            with col3:
                if schlechtere_bilder:
                    st.markdown("**Schlechtere Bilder:**")
                    for bild in schlechtere_bilder[:3]:  # Nur die ersten 3 zur Übersicht
                        try:
                            st.image(os.path.join(schlechter_ordner, bild), caption=bild, width=100)
                        except Exception:
                            st.write(f"({bild} - Vorschau nicht möglich)")
                    if len(schlechtere_bilder) > 3:
                        st.write(f"+ {len(schlechtere_bilder) - 3} weitere Bilder")

    st.success("Fertig! Du findest die sortierten Gruppen im temporären Ordner.")

    # Button zum Zurücksetzen der Analyse (für neue Analyse)
    if st.button("Neue Analyse starten"):
        st.session_state.analysis_complete = False
        st.session_state.gruppen_info = []
        st.session_state.gruppen_bewertungen = []
        st.rerun()
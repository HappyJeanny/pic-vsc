# -*- coding: utf-8 -*-
"""
Created on Tuesday Jul 22 17:27:30 2025

@author: IsabellMader
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

st.set_page_config(page_title="Bildgruppen-Analyse", layout="wide")
st.title("🔎 Ähnliche Bilder finden & bestes Bild pro Gruppe auswählen")

# --- 1. Ordnerauswahl über file_uploader (moderne Alternative zu folder-picker) ---
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
import tempfile

ZIEL_ORDNER = "similar"
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
      
    with open(txt_path, "rb") as f:
        st.download_button(
            label="Gruppen-Übersicht als TXT herunterladen",
            data=f,
            file_name="gruppen_übersicht.txt",
            mime="text/plain"
        )

    # --- Ergebnisse als Tabelle, mit Bildvorschau ---
    st.subheader("Vorschau der Gruppen und besten Bilder")
    for i, gruppen_bilder, best_bildname, best_score in gruppen_info:
        cols = st.columns([2, 5])
        with cols[0]:
            pfad = [p for p in gruppen_bilder if os.path.basename(p) == best_bildname][0]
            try:
                st.image(pfad, caption=f"Gruppe {i:03d} - Bestes Bild", width=150)
            except Exception:
                st.write("(Bildvorschau nicht möglich)")
        with cols[1]:
            st.markdown(f"**Gruppe {i:03d}**  \nBestes Bild: `{best_bildname}`  \nScore: `{best_score:.2f}`  \nAnzahl ähnliche Bilder: {len(gruppen_bilder)}")

    st.success("Fertig! Du findest die sortierten Gruppen im temporären Ordner.")
    end_time = time.time()
    dauer = end_time - start_time
    mins = int(dauer // 60)
    secs = int(dauer % 60)
    st.info(f"Die Analyse hat {mins} Minuten {secs} Sekunden gedauert.")

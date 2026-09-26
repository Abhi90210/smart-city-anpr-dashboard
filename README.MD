# 🚘 Smart City ANPR Traffic & Surveillance Engine

An interactive multi-camera Automatic Number Plate Recognition (ANPR) traffic monitoring and vehicle trajectory tracking dashboard built using Python and Streamlit. 

This application uses graph theory (`networkx`) to map city junctions and corridors, calculating spatio-temporal vehicle routes, speed estimates between monitoring checkpoints, and live corridor congestion indexes.

---

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit
* **Geospatial Mapping:** PyDeck
* **Graph & Spatial Network Analysis:** NetworkX
* **Data Processing & Analytics:** Pandas, NumPy

---

## 📁 Project Structure

```text
├── app.py              # Streamlit web interface and UI layout
├── engine.py           # Core ANPR traffic engine and data structures
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
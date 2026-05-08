from flask import Flask, render_template
from routes import register_routes
import os
from pathlib import Path

# Percorso assoluto alla cartella del progetto (fondamentale per Render/Gunicorn)
BASE_DIR = Path(__file__).resolve().parent

# Inizializzazione Flask con percorsi assoluti per static e templates
app = Flask(
    __name__,
    static_folder=BASE_DIR / 'static',
    static_url_path='/static',
    template_folder=BASE_DIR / 'templates'
)

register_routes(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sistemi_dinamici.html")
def sistemi_dinamici():
    return render_template("sistemi_dinamici.html")

@app.route("/stabilita.html")
def stabilita():
    return render_template("stabilita.html")

@app.route("/linearizzazione.html")
def linearizzazione():
    return render_template("linearizzazione.html")

@app.route("/equazioni_differenziali.html")
def equazioni_differenziali():
    return render_template("equazioni_differenziali.html")

@app.route("/equazioni_alle_differenze.html")
def equazioni_differenze():
    return render_template("equazioni_alle_differenze.html")

@app.route("/da_soluzione_a_sistema.html")
def da_soluzione_a_sistema():
    return render_template("da_soluzione_a_sistema.html")

@app.route("/fratti_semplici.html")
def fratti_semplici():
    return render_template("fratti_semplici.html")

@app.route("/sistemi_meccanici.html")
def sistemi_meccanici():
    return render_template("sistemi_meccanici.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # debug=False è obbligatorio quando si usa gunicorn in produzione
    app.run(host="0.0.0.0", port=port, debug=False)

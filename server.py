from flask import Flask, render_template
from routes import register_routes
import os
import socket

app = Flask(__name__)
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
from .power_at import power_at_bp
from .exp_at import exp_at_bp
from .sistemi_dinamici import sistemi_dinamici_bp
from .linearizzazione import linearizzazione_bp
from .equazioni_differenziali import equazioni_differenziali_bp
from .condizioni_differenziali import condizioni_differenziali_bp
from .equazioni_alle_differenze import equazioni_alle_differenze_bp
from .condizioni_alle_differenze import condizioni_alle_differenze_bp
from .decomposizione import compute as decomposizione_view

def register_routes(app):
    app.register_blueprint(power_at_bp)
    app.register_blueprint(exp_at_bp)
    app.register_blueprint(linearizzazione_bp)
    app.register_blueprint(sistemi_dinamici_bp)
    app.register_blueprint(equazioni_differenziali_bp)
    app.register_blueprint(condizioni_differenziali_bp)
    app.register_blueprint(equazioni_alle_differenze_bp)
    app.register_blueprint(condizioni_alle_differenze_bp)
    app.add_url_rule("/api/decomposizione", view_func=decomposizione_view, methods=["POST"]) 
   
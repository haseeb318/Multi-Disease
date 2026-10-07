import os
import pickle
import logging
from flask import Flask, render_template
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _load_model(path: str):
    """Safely load a pickle model. Returns None if file is missing."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Loaded model: %s", path)
        return model
    except FileNotFoundError:
        logger.error("Model file not found: %s", path)
        return None
    except Exception as e:
        logger.error("Failed to load model %s: %s", path, e)
        return None


def create_app(config_name: str = 'default') -> Flask:
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # ── Load ML models ────────────────────────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    app.diabetes_model      = _load_model(os.path.join(models_dir, 'diabetes_model.pkl'))
    app.heart_model         = _load_model(os.path.join(models_dir, 'heart_disease_model.pkl'))

    # ── Register blueprints ───────────────────────────────────────────────────
    from app.routes.main     import main_bp
    from app.routes.diabetes import diabetes_bp
    from app.routes.heart    import heart_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(diabetes_bp)
    app.register_blueprint(heart_bp)

    # ── Error handlers ────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error("Internal server error: %s", e)
        return render_template('errors/500.html'), 500

    return app

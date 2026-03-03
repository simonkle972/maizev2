from authlib.integrations.flask_client import OAuth
from config import Config

oauth = OAuth()

def init_oauth(app):
    """Initialize OAuth with the Flask app and register Auth0 clients."""
    oauth.init_app(app)
    base_kwargs = {
        'client_kwargs': {'scope': 'openid profile email'},
        'server_metadata_url': f'https://{Config.AUTH0_DOMAIN}/.well-known/openid-configuration',
    }
    oauth.register(
        'auth0_professor',
        client_id=Config.AUTH0_CLIENT_ID,
        client_secret=Config.AUTH0_CLIENT_SECRET,
        **base_kwargs,
    )
    oauth.register(
        'auth0_student',
        client_id=Config.AUTH0_STUDENT_CLIENT_ID,
        client_secret=Config.AUTH0_STUDENT_CLIENT_SECRET,
        **base_kwargs,
    )

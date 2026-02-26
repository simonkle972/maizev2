from authlib.integrations.flask_client import OAuth
from config import Config

oauth = OAuth()

def init_oauth(app):
    """Initialize OAuth with the Flask app and register Auth0."""
    oauth.init_app(app)
    oauth.register(
        'auth0',
        client_id=Config.AUTH0_CLIENT_ID,
        client_secret=Config.AUTH0_CLIENT_SECRET,
        client_kwargs={'scope': 'openid profile email'},
        server_metadata_url=f'https://{Config.AUTH0_DOMAIN}/.well-known/openid-configuration'
    )

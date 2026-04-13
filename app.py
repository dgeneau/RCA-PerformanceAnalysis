

from dash import Dash, Input, Output, html, dcc, no_update, State, page_container
import dash
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_auth_external import DashAuthExternal

from layout import Footer, Navbar, ProfileCard


from settings import *

from auth_setup import server, auth


here = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(here, "assets")
server.static_folder = assets_path
server.static_url_path = "/assets"

# Initialize the Dash app
app = Dash(
    __name__,
    server=server,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"]
)
nav_links = [

    {'label':"Analysis Home",'url':"/home"},
    {'label':"GPS Processing",'url':"/gps"}, 
    {'label':"Live GPS",'url':"/live-gps"}
]
navbar = Navbar(nav_links, id="navbar", title="RCA Performance Analysis", expand="lg")
navbar.register_callbacks(app)

footer = Footer()

app.layout = html.Div([

    dcc.Location(id="redirect-to", refresh=True),
    dcc.Interval(
        id="init-interval",
        interval=500,  # e.g., 1 second after page load
        n_intervals=0,
        max_intervals=1  # This ensures it fires only once
    ),

    # Store to hold current user's first name (shared across pages)
    dcc.Store(id="current-user-first-name", data=None),

    navbar.render(),

    dash.page_container,

    footer.render(),
])

@dash.callback(
    Output("redirect-to", "href"),
    Input("init-interval", "n_intervals")
)
def initial_view(n):
    """
    On timeout, load filters
    """
    try:
        token = auth.get_token()
    except Exception as e:
        return APP_URL

    return no_update


if __name__ == "__main__":
    app.run(debug=True, port=8050)

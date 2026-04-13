

import dash
from dash import html, dcc, callback, Input, Output, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from utils import fetch_profile, fetch_profiles
from auth_setup import auth
from layout import ProfileCard
from settings import SITE_URL
import requests


# Register this file as a page in the larger app
dash.register_page(__name__, path="/home", name="Home")



def layout():
    return html.Div(
        className="monitoring-page",
        children=[
            # one-shot interval to trigger profile load once on page load
            dcc.Interval(id="home-init-interval", interval=500, n_intervals=0, max_intervals=1),

            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H2("Home", className="mb-0"),
                                html.P("Welcome — choose a page to get started.", className="text-muted"),
                            ]
                        ),
                        md=8,
                    ),

                    # profile container (populated by callback)
                    dbc.Col(html.Div(id="profile-container", className="d-flex justify-content-end"), md=4),
                ],
                className="align-items-center mb-3",
            ),

            dbc.Row(
                className="g-3",
                children=[
                    dbc.Col(
                        md=6,
                        children=dbc.Card(
                            className="shadow-sm",
                            children=[
                                dbc.CardHeader("GPS Post Processing"),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "Analyze Sessions using 'Insiders' GPS units",
                                            className="text-muted",
                                        ),
                                        dbc.Button(
                                            "Go to GPS Processing",
                                            href="/gps",
                                            color="danger",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ),
                    dbc.Col(
                        md=6,
                        children=dbc.Card(
                            className="shadow-sm",
                            children=[
                                dbc.CardHeader("Live GPS"),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "Monitor GPS data in real-time from 'Insiders' GPS units",
                                            className="text-muted",
                                        ),
                                        dbc.Button(
                                            "Go to Live GPS",
                                            href="/live-gps",
                                            color="secondary",
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ),
                ],
            ),

            html.Hr(),
        ],
    )




@callback(Output("profile-container", "children"), 
          Input("home-init-interval", "n_intervals"))
def load_profile(n_intervals):
    if not n_intervals:
        raise PreventUpdate

    try:
        token = auth.get_token()
    except Exception as e:
        print(f'No token: {e}')
        return html.Div("Not signed in", className="text-muted")

    if not token:
        return html.Div("No token", className="text-muted")

    # Call the /api/csiauth/me/ endpoint to get the current logged-in user
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        me_resp = requests.get(
            f"{SITE_URL}/api/csiauth/me/",
            headers=headers,
            timeout=5
        )
        me_resp.raise_for_status()
        current_user = me_resp.json()
        print(f"Current user from /me: {current_user}")
        
        # Extract profile_id from current_profile, not from id
        profile_id = current_user.get('current_profile', {}).get('id')
        
        if not profile_id:
            return html.Div("Could not get profile_id from current user", className="text-muted text-small")
        
        # Fetch the full profile details
        profile = fetch_profile(token, profile_id)
        print(f"Full profile: {profile}")
        
    except Exception as e:
        print(f"Failed to fetch current user: {e}")
        return html.Div(f"Could not fetch profile: {str(e)}", className="text-muted text-small")

    # Build display values
    try:
        # Get name from the /me endpoint data (simpler path)
        first = current_user.get('first_name')
        last = current_user.get('last_name')
        print(f"Extracted first_name: {first}, last_name: {last}")
        name = " ".join([p for p in [first, last] if p]) or "-"
        
        # Get role from current_profile
        role = current_user.get('current_profile', {}).get('role', {}).get('verbose_name') or "-"
        
        # Get organization from the full profile
        org = None
        if profile.get('organization'):
            org = profile['organization'].get('name')
        elif profile.get('current_nomination'):
            org = profile['current_nomination'].get('organization', {}).get('name')
        
        org = org or "-"
        
        card = ProfileCard(name=name, role=role, organization=org)
        return card.render(id="home-profile")
        
    except Exception as e:
        print(f"Error building profile card: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(f"Error building profile: {str(e)}", className="text-muted text-small")
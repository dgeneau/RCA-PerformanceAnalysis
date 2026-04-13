from dash import html
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

class Navbar:
    def __init__(self, buttons=None, id="navbar", title="CSIP Apps", expand="lg"):
        self.buttons = buttons or []
        self.id = id
        self.title = title
        self.expand = expand
        # unique IDs for toggler/collapse
        self.toggler_id = f"{self.id}-toggler"
        self.collapse_id = f"{self.id}-collapse"

    def _nav_items(self):
        return [
            dbc.NavItem(
                dbc.NavLink(item["label"], href=item["url"],
                            active="exact",
                            className="px-3 py-2")
            )
            for item in self.buttons
        ]

    def render(self):
        return dbc.Navbar(
            dbc.Container(
                [
                    # Brand / logo
                    dbc.NavbarBrand(
                        [
                            html.Img(src="assets/img/csi-pacific-logo-reverse.png", height="40px"),
                            html.Span(self.title, className="ms-2 h5 mb-0"),
                        ],
                        href="/builder",
                        className="d-flex align-items-center text-white text-decoration-none",
                    ),

                    # Hamburger toggler (shown below `expand`)
                    dbc.NavbarToggler(id=self.toggler_id, n_clicks=0, className="ms-auto"),

                    # Collapsible nav area
                    dbc.Collapse(
                        dbc.Nav(
                            self._nav_items(),
                            navbar=True,
                            className="flex-column flex-lg-row ms-lg-auto align-items-start align-items-lg-center gap-2",
                        ),
                        id=self.collapse_id,
                        navbar=True,
                        is_open=False,
                    ),
                ]
            ),
            color="dark",
            dark=True,
            className="p-3",
            id=self.id,
            expand=self.expand,   # collapse below 'lg' (change to 'md' if you want earlier)
        )

    def register_callbacks(self, app: dash.Dash):
        @app.callback(
            Output(self.collapse_id, "is_open"),
            Input(self.toggler_id, "n_clicks"),
            State(self.collapse_id, "is_open"),
            prevent_initial_call=True,
        )
        def _toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

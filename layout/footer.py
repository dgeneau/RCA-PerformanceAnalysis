import dash_bootstrap_components as dbc
from dash import html


class Footer():

    def render(self, id="footer"):
        """
        Dash footer component styled with a black background, white text,
        and fixed to the bottom of the viewport.
        """
        return html.Footer(

            [
                # Copyright text

                dbc.Container([

                    html.P(
                        "© 2025 CSI Pacific",
                        className="col-md-4 mb-0"
                    ),

                    # Logo or brand link
                    # html.A(
                    #     html.Img(
                    #         src="/assets/img/csi-pacific-logo-reverse.png",
                    #         height="60px",
                    #     ),
                    #     href="/",
                    #     className=(
                    #         "col-md-4 d-flex align-items-center justify-content-center mb-3"
                    #         " mb-md-0 me-md-auto text-decoration-none"
                    #     ),
                    #     **{"aria-label": "Bootstrap"}
                    # ),

                    # Navigation links
                    html.Ul(
                        [
                            # html.Li(html.A("Home", href="#", className="nav-link text-white px-2"), className="nav-item"),
                            # html.Li(html.A("Features", href="#", className="nav-link text-white px-2"), className="nav-item"),
                            # html.Li(html.A("Pricing", href="#", className="nav-link text-white px-2"), className="nav-item"),
                            # html.Li(html.A("FAQs", href="#", className="nav-link text-white px-2"), className="nav-item"),
                            # html.Li(html.A("About", href="#", className="nav-link text-white px-2"), className="nav-item"),
                        ],
                        className="nav col-md-4 justify-content-end"
                    ),
                ],
                    className="d-flex flex-wrap justify-content-between align-items-center py-3  "
                )],
            className="mt-4 bg-dark text-white border-top border-light fixed-bottom",
            id=id
        )

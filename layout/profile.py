from dataclasses import dataclass
from dash import html
import dash_bootstrap_components as dbc


@dataclass
class ProfileCard:
    """A Bootstrap-styled profile card for Dash."""
    name: str = "-"
    role: str = "-"
    organization: str = "-"
    id_prefix: str = "profile"
    avatar_bg: str = "#0d6efd"  # Bootstrap primary

    def _initials(self) -> str:
        parts = [p for p in (self.name or "").split() if p]
        return "".join(p[0].upper() for p in parts[:2]) or "?"

    def render(self, id: str | None = None):
        cid = id or self.id_prefix

        avatar = html.Div(
            self._initials(),
            id=f"{cid}-avatar",
            className="d-flex align-items-center justify-content-center fw-semibold",
            style={
                "width": "64px",
                "height": "64px",
                "borderRadius": "50%",
                "background": self.avatar_bg,
                "color": "white",
                "fontSize": "1.1rem",
                "letterSpacing": "0.02em",
            },
        )

        title_row = html.Div(
            [
                html.H5(self.name or "-", id=f"{cid}-name", className="mb-1"),
                html.Span(self.role or "-", id=f"{cid}-role",
                          className="badge bg-secondary ms-0 ms-sm-2"),
            ],
            className="d-flex flex-column flex-sm-row align-items-start align-items-sm-center gap-2",
        )

        org_row = html.Div(
            [
                html.I(className="bi bi-building me-2"),  # if bootstrap-icons loaded
                html.Span(self.organization or "-", id=f"{cid}-organization",
                          className="text-muted"),
            ],
            className="text-muted mt-1 d-flex align-items-center",
        )

        body = dbc.CardBody(
            [
                title_row,
                org_row,
            ],
            className="py-3",
        )

        return dbc.Card(
            dbc.Row(
                [
                    dbc.Col(avatar, width="auto", className="p-3"),
                    dbc.Col(body, className="ps-0"),
                ],
                className="g-0 align-items-center",
            ),
            className="shadow-sm border-0",
            style={"maxWidth": "560px"},
        )

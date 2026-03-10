import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ── Load data ────────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)

dfs = []
for fname in ["uc_professors_2010.csv", "uc_professors_2011.csv"]:
    dfs.append(pd.read_csv(os.path.join(BASE, fname)))

for fname in sorted(os.listdir(os.path.join(BASE, "professors"))):
    if fname.endswith(".csv"):
        dfs.append(pd.read_csv(os.path.join(BASE, "professors", fname)))

df = pd.concat(dfs, ignore_index=True)
df["Position"] = df["Position"].str.upper().str.strip()

# ── Classify into 5 types (priority order matters) ───────────────────────────

def classify(pos):
    if "CLIN" in pos:
        return "Clinical Professor"
    if "ADJ" in pos:
        return "Adjunct Professor"
    if "ASSOC" in pos:
        return "Associate Professor"
    if "ASST" in pos or "ASSISTANT" in pos:
        return "Assistant Professor"
    if "PROF" in pos:
        return "Professor"
    return None

df["Type"] = df["Position"].apply(classify)
df = df[df["Type"].notna()].copy()

# Keep only rows with valid wages
df["TotalWages"] = pd.to_numeric(df["TotalWages"], errors="coerce")
df = df[df["TotalWages"] > 0]

TYPES = [
    "Professor",
    "Associate Professor",
    "Assistant Professor",
    "Adjunct Professor",
    "Clinical Professor",
]

COLORS = {
    "Professor":           "#1f77b4",
    "Associate Professor": "#ff7f0e",
    "Assistant Professor": "#2ca02c",
    "Adjunct Professor":   "#d62728",
    "Clinical Professor":  "#9467bd",
}

campuses = sorted(df["EmployerName"].unique())
years    = sorted(df["Year"].unique())

# ── App layout ───────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = "UC Professor Salaries"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px",
           "margin": "0 auto", "padding": "24px"},
    children=[
        html.H1("UC Professor Salaries", style={"marginBottom": "4px"}),
        html.P("University of California system · 2010–2024",
               style={"color": "#666", "marginTop": "0"}),

        # ── Controls ──────────────────────────────────────────────────────────
        html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "1fr 1fr 1fr",
                   "gap": "24px", "marginBottom": "24px",
                   "background": "#f8f8f8", "padding": "20px",
                   "borderRadius": "8px"},
            children=[

                # Year range
                html.Div([
                    html.Label("Year Range", style={"fontWeight": "bold"}),
                    dcc.RangeSlider(
                        id="year-slider",
                        min=min(years), max=max(years), step=1,
                        value=[min(years), max(years)],
                        marks={y: str(y) for y in years},
                        tooltip={"placement": "bottom"},
                    ),
                ]),

                # Campus
                html.Div([
                    html.Label("Campus", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="campus-dropdown",
                        options=[{"label": "All Campuses", "value": "ALL"}]
                                + [{"label": c, "value": c} for c in campuses],
                        value="ALL",
                        clearable=False,
                    ),
                ]),

                # Metric
                html.Div([
                    html.Label("Salary Metric", style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="metric-radio",
                        options=[
                            {"label": " Median", "value": "median"},
                            {"label": " Mean",   "value": "mean"},
                        ],
                        value="median",
                        labelStyle={"marginRight": "16px"},
                    ),
                ]),
            ],
        ),

        # Professor type checkboxes
        html.Div(
            style={"marginBottom": "24px"},
            children=[
                html.Label("Professor Type", style={"fontWeight": "bold"}),
                dcc.Checklist(
                    id="type-checklist",
                    options=[{"label": f"  {t}", "value": t} for t in TYPES],
                    value=TYPES,
                    inline=True,
                    labelStyle={"marginRight": "24px", "cursor": "pointer"},
                ),
            ],
        ),

        # ── Charts ────────────────────────────────────────────────────────────
        dcc.Graph(id="trend-chart", style={"height": "480px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                   "gap": "24px", "marginTop": "24px"},
            children=[
                dcc.Graph(id="campus-bar",  style={"height": "360px"}),
                dcc.Graph(id="box-chart",   style={"height": "360px"}),
            ],
        ),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

def filter_df(year_range, campus, types):
    mask = (
        df["Year"].between(year_range[0], year_range[1]) &
        df["Type"].isin(types)
    )
    if campus != "ALL":
        mask &= df["EmployerName"] == campus
    return df[mask]


@app.callback(
    Output("trend-chart", "figure"),
    Input("year-slider",    "value"),
    Input("campus-dropdown","value"),
    Input("type-checklist", "value"),
    Input("metric-radio",   "value"),
)
def update_trend(year_range, campus, types, metric):
    d = filter_df(year_range, campus, types)
    agg = (d.groupby(["Year", "Type"])["TotalWages"]
             .agg(metric)
             .reset_index())

    fig = go.Figure()
    for t in TYPES:
        if t not in types:
            continue
        sub = agg[agg["Type"] == t]
        fig.add_trace(go.Scatter(
            x=sub["Year"], y=sub["TotalWages"],
            mode="lines+markers", name=t,
            line=dict(color=COLORS[t], width=2.5),
            marker=dict(size=7),
            hovertemplate="%{x}: $%{y:,.0f}<extra>" + t + "</extra>",
        ))

    label = metric.capitalize()
    camp  = campus if campus != "ALL" else "All Campuses"
    fig.update_layout(
        title=f"{label} Total Wages Over Time · {camp}",
        xaxis_title="Year",
        yaxis_title=f"{label} Total Wages ($)",
        yaxis_tickformat="$,.0f",
        legend_title="Type",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    return fig


@app.callback(
    Output("campus-bar", "figure"),
    Input("year-slider",    "value"),
    Input("type-checklist", "value"),
    Input("metric-radio",   "value"),
)
def update_campus_bar(year_range, types, metric):
    d = filter_df(year_range, "ALL", types)
    agg = (d.groupby("EmployerName")["TotalWages"]
             .agg(metric)
             .sort_values(ascending=True)
             .reset_index())

    # Shorten campus labels
    agg["Label"] = agg["EmployerName"].str.replace(
        "University of California, ", "UC ", regex=False)

    fig = go.Figure(go.Bar(
        x=agg["TotalWages"], y=agg["Label"],
        orientation="h",
        marker_color="#1f77b4",
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{metric.capitalize()} Wages by Campus",
        xaxis_title=f"{metric.capitalize()} Total Wages ($)",
        xaxis_tickformat="$,.0f",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=False),
        margin=dict(l=80),
    )
    return fig


@app.callback(
    Output("box-chart", "figure"),
    Input("year-slider",    "value"),
    Input("campus-dropdown","value"),
    Input("type-checklist", "value"),
)
def update_box(year_range, campus, types):
    d = filter_df(year_range, campus, types)

    fig = go.Figure()
    for t in TYPES:
        if t not in types:
            continue
        sub = d[d["Type"] == t]["TotalWages"]
        fig.add_trace(go.Box(
            y=sub, name=t,
            marker_color=COLORS[t],
            boxmean=True,
            hovertemplate="$%{y:,.0f}<extra>" + t + "</extra>",
        ))

    fig.update_layout(
        title="Wage Distribution by Type",
        yaxis_title="Total Wages ($)",
        yaxis_tickformat="$,.0f",
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)

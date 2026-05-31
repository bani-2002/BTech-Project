import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ---------- Load Rain Status Model Results ----------
rain_results_df = pd.read_csv("rain_classification_comparison.csv")

# ---------- Load AQI Dataset and Train XGBoost ----------
df = pd.read_csv("aqi cat.csv")
df.dropna(inplace=True)

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

df['AQI_Category'] = df['aqi value'].apply(get_aqi_category)
label_encoder = LabelEncoder()
df['AQI_Label'] = label_encoder.fit_transform(df['AQI_Category'])

features = ['dhtTemp', 'dhthum', 'LDR value', 'pressure', 'altitude', 'rainValue']
X = df[features]
y = df['AQI_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)
y_pred = xgb_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

importance = xgb_model.get_booster().get_score(importance_type='gain')
feat_imp = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values(by='Importance', ascending=False)


# ---------- Start Dash App ----------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Weather Intelligence Dashboard"


app.layout = dbc.Container([
    html.H1("Weather Intelligence Dashboard", className="my-4 text-center"),

    dbc.Tabs([
        dbc.Tab(label="Rain Model Comparison", tab_id="tab-1"),
        dbc.Tab(label="AQI Category Prediction", tab_id="tab-2"),
        dbc.Tab(label="Rain Model Table View", tab_id="tab-3")
    ], id="tabs", active_tab="tab-1"),

    html.Div(id="tab-content", className="p-4")
], fluid=True)

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "tab-1":
        return html.Div([
            html.H4("Compare Rain Prediction Models"),
            html.Label("Select Evaluation Metric:"),
            dcc.Dropdown(
                id='rain-metric-dropdown',
                options=[{'label': m, 'value': m} for m in ['Accuracy', 'Precision', 'Recall', 'F1 Score']],
                value='Accuracy',
                style={'width': '300px'}
            ),
            dcc.Graph(id='rain-bar-chart')
        ])

    elif active_tab == "tab-2":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H4("Model Accuracy"),
                    html.P(f"{accuracy:.4f}")
                ], width=3),
                dbc.Col([
                    html.H4("Classification Summary"),
                    html.Pre('\n'.join([
                        f"{cls}: Precision {v['precision']:.2f}, Recall {v['recall']:.2f}, F1 {v['f1-score']:.2f}"
                        for cls, v in class_report.items() if cls in label_encoder.classes_
                    ]), style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=go.Figure(
                            data=[go.Heatmap(
                                z=conf_matrix,
                                x=label_encoder.classes_,
                                y=label_encoder.classes_,
                                colorscale='Viridis'
                            )],
                            layout=go.Layout(
                                title="Confusion Matrix",
                                xaxis_title="Predicted",
                                yaxis_title="Actual"
                            )
                        )
                    )
                ], width=6),
                dbc.Col([
                    dcc.Graph(
                        figure=go.Figure(
                            data=[go.Bar(
                                x=feat_imp['Feature'],
                                y=feat_imp['Importance'],
                                marker_color='indianred'
                            )],
                            layout=go.Layout(
                                title="Feature Importance (Gain)",
                                yaxis_title="Importance"
                            )
                        )
                    )
                ], width=6),
            ])
        ])

    elif active_tab == "tab-3":
        return html.Div([
            html.H3("Rain Model Performance Table", className="text-center"),
            dcc.Graph(
                figure=go.Figure(
                    data=[go.Table(
                        header=dict(values=list(rain_results_df.columns), fill_color='lightblue', align='center'),
                        cells=dict(values=[rain_results_df[col] for col in rain_results_df.columns], fill_color='white', align='center')
                    )]
                )
            )
        ])
    return "No tab selected"

@app.callback(
    Output('rain-bar-chart', 'figure'),
    Input('rain-metric-dropdown', 'value')
)
def update_rain_chart(metric):
    return {
        'data': [go.Bar(
            x=rain_results_df['Model'],
            y=rain_results_df[metric],
            marker_color='skyblue'
        )],
        'layout': go.Layout(
            title=f'{metric} Comparison (Rain Prediction)',
            xaxis_title='Model',
            yaxis_title=metric,
            plot_bgcolor='whitesmoke'
        )
    }

if __name__ == '__main__':
    app.run(debug=True)

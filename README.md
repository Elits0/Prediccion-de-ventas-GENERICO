# Prediccion-de-ventas-GENERICO
Este es una predicción como tal de ventas a futuro es genérico, pero quiero presentar todas mis datos y mi trabajo

import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# Análisis de Ventas Genérico con Predicciones\n\n"
                                     "Notebook de ejemplo para análisis de ventas y forecasting orientado a ciencia de datos. "
                                     "Incluye generación de datos sintéticos, EDA, consultas SQL con SQLite, mapa Folium y un modelo predictivo "
                                     "basado en RandomForest para pronosticar ventas mensuales.\n\n"
                                     "Puedes ejecutar las celdas en orden en Jupyter Notebook o JupyterLab."))

cells.append(nbf.v4.new_code_cell("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta

# Modelado
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Visualización interactiva
import plotly.express as px
import plotly.graph_objects as go

# Mapa
import folium
from folium.plugins import MarkerCluster

print('librerías cargadas')"""))

cells.append(nbf.v4.new_code_cell("""# Generar dataset sintético de ventas
np.random.seed(42)

start = '2018-01-01'
end = '2024-08-31'
dates = pd.date_range(start, end, freq='D')

stores = [
    {'store_id': 1, 'region': 'Norte', 'lat': -0.1807, 'lon': -78.4678},
    {'store_id': 2, 'region': 'Centro', 'lat': -0.2100, 'lon': -78.5000},
    {'store_id': 3, 'region': 'Sur', 'lat': -0.2500, 'lon': -78.3500}
]

rows = []
for d in dates:
    for s in stores:
        # Baseline seasonal pattern + weekday effect + random noise
        month = d.month
        seasonal = (1 + 0.3 * np.sin((month - 1) / 12 * 2 * np.pi))
        weekday = 1.0 + (0.1 if d.weekday() >=5 else 0.0)  # slightly higher on weekends
        trend = 1 + 0.02 * ((d.year - 2018) + (d.month/12))  # small upward trend
        base = 2000 if s['region']=='Norte' else (1500 if s['region']=='Centro' else 1000)
        sales = base * seasonal * weekday * trend + np.random.normal(0, 150)
        transactions = max(1, int(sales/50 + np.random.poisson(5)))
        rows.append({
            'date': d,
            'store_id': s['store_id'],
            'region': s['region'],
            'lat': s['lat'],
            'lon': s['lon'],
            'sales': max(0, round(sales,2)),
            'transactions': transactions
        })

sales_df = pd.DataFrame(rows)
# Crear columna mes-año para agregaciones mensuales
sales_df['month'] = sales_df['date'].dt.to_period('M').dt.to_timestamp()

print('dataset generado:', sales_df.shape)
sales_df.head()"""))

cells.append(nbf.v4.new_code_cell("""# Agregación mensual y visualización de la serie temporal
monthly = sales_df.groupby(['month']).sales.sum().reset_index()

fig = px.line(monthly, x='month', y='sales', title='Ventas Totales Mensuales (synthetic)')
fig.update_layout(xaxis_title='Mes', yaxis_title='Ventas')
fig"""))

cells.append(nbf.v4.new_code_cell("""# Ventas por región
region_month = sales_df.groupby(['month','region']).sales.sum().reset_index()
fig2 = px.area(region_month, x='month', y='sales', color='region', title='Ventas Mensuales por Región (stacked)')
fig2"""))

cells.append(nbf.v4.new_code_cell("""# Guardar en SQLite y ejecutar algunas consultas SQL
conn = sqlite3.connect('sales_example.db')
sales_df.to_sql('sales', conn, if_exists='replace', index=False)

q = '''
SELECT region, COUNT(DISTINCT store_id) as n_stores, SUM(sales) as total_sales
FROM sales
GROUP BY region
ORDER BY total_sales DESC
'''
print(pd.read_sql(q, conn))"""))

cells.append(nbf.v4.new_code_cell("""# Mapa interactivo con Folium: ventas acumuladas por tienda (último mes disponible)
last_month = sales_df['month'].max()
agg_last = sales_df[sales_df['month']==last_month].groupby('store_id').agg({'sales':'sum','lat':'first','lon':'first','region':'first'}).reset_index()

m = folium.Map(location=[-0.21, -78.45], zoom_start=10)
mc = MarkerCluster().add_to(m)
for _, r in agg_last.iterrows():
    folium.Marker([r['lat'], r['lon']], popup=f"Store {int(r['store_id'])} ({r['region']}) - Sales: ${r['sales']:.2f}").add_to(mc)

m.save('sales_map.html')
print('Mapa guardado en sales_map.html')
agg_last"""))

cells.append(nbf.v4.new_code_cell("""# Preparación de datos para forecasting mensual
# Usaremos la agregación mensual y crearemos lags
monthly = sales_df.groupby('month').sales.sum().reset_index()
monthly = monthly.sort_values('month').reset_index(drop=True)

# Crear lags (1,2,3,6,12 meses)
for l in [1,2,3,6,12]:
    monthly[f'lag_{l}'] = monthly['sales'].shift(l)
monthly['month_num'] = np.arange(len(monthly))
monthly = monthly.dropna().reset_index(drop=True)
monthly.head()"""))

cells.append(nbf.v4.new_code_cell("""# Entrenamiento de modelo RandomForest para predecir ventas mensuales
features = [c for c in monthly.columns if c.startswith('lag_') or c=='month_num']
X = monthly[features]
y = monthly['sales']

# Mantener el orden temporal: usar los últimos 6 meses como test
test_size = 6
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred, squared=False)
mae = mean_absolute_error(y_test, pred)
print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')


# Guardar resultados en dataframe
results = monthly.iloc[-test_size:].copy()
results['pred'] = pred
results"""))

cells.append(nbf.v4.new_code_cell("""# Pronosticar los próximos 6 meses (enfoque recursivo usando lags)
last_row = monthly.iloc[-1:].copy()
future = []
current = last_row.copy()
for i in range(6):
    # construir features a partir de la fila current
    feat = {}
    # lag_1 es la venta actual de current, lag_2 es el antiguo lag_1, etc.
    feat['lag_1'] = current['sales']
    feat['lag_2'] = current['lag_1']
    feat['lag_3'] = current['lag_2']
    feat['lag_6'] = current['lag_3'] if 'lag_3' in current.index else current.get('lag_6', current['lag_1'])
    feat['lag_12'] = current.get('lag_12', current['lag_1'])
    feat['month_num'] = current['month_num'] + 1

    Xf = pd.DataFrame([feat])
    yhat = model.predict(Xf)[0]

    # preparar nueva fila para future\n    new_month = pd.to_datetime(current['month']) + pd.offsets.MonthBegin(1)
    new_row = {
        'month': new_month,
        'sales': yhat,
        'lag_1': feat['lag_1'],
        'lag_2': feat['lag_2'],
        'lag_3': feat['lag_3'],
        'lag_6': feat['lag_6'],
        'lag_12': feat['lag_12'],
        'month_num': feat['month_num']
    }
    future.append(new_row)

    # actualizar current: convertir new_row a Series\n    current = pd.Series(new_row)

future_df = pd.DataFrame(future)
future_df['month'] = pd.to_datetime(future_df['month'])
future_df"""))

cells.append(nbf.v4.new_code_cell("""# Visualización: historial + predicciones
hist = monthly[['month','sales']].copy()
pred_df = results[['month','pred']].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist['month'], y=hist['sales'], name='Actual'))
fig.add_trace(go.Scatter(x=pred_df['month'], y=pred_df['pred'], name='Predicción (test)'))
fig.add_trace(go.Scatter(x=future_df['month'], y=future_df['sales'], name='Pronóstico next 6m', line=dict(dash='dash')))
fig.update_layout(title='Ventas: actual vs predicción y forecast', xaxis_title='Mes', yaxis_title='Ventas')
fig"""))

cells.append(nbf.v4.new_markdown_cell("## Conclusiones\n\n- Este notebook muestra un pipeline completo desde generación y limpieza de datos, EDA, consultas SQL, "
                                     "mapa interactivo y un modelo predictivo simple para ventas mensuales.\n\n"
                                     "### Siguientes pasos sugeridos:\n"
                                     "- Probar modelos de series temporales dedicados (ARIMA, SARIMA, Prophet, LSTM).\n- Incluir variables exógenas (promociones, campañas de marketing, precios).\n- Hacer backtesting robusto con validación temporal.\n- Incorporar incertidumbre en los pronósticos (intervalos de confianza)."))

nb['cells'] = cells

out_path = '/mnt/data/Sales_Analysis_Prediction.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

out_path


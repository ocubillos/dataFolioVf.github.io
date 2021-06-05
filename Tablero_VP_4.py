#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install jupyter-dash -q


# In[ ]:


#pip install dash-cytoscape -q


# In[1]:


from jupyter_dash import JupyterDash  # pip install dash
import dash_cytoscape as cyto  # pip install dash-cytoscape==0.2.0 or higher
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
import pandas as pd  # pip install pandas
import plotly.express as px
import math
import numpy as np
import base64
from dash import no_update
from pandas_profiling import ProfileReport
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from plotly.tools import mpl_to_plotly
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#pip install --upgrade plotly


# In[42]:


basedash=pd.read_csv('Base Tablero.csv')
EAyP=basedash[basedash['Region']=='East Asia & Pacific']
Malaysia=basedash[basedash['Country Name']=='Malaysia']
Malaysia.fillna(0,inplace=True)
df=pd.read_csv("data_KL_j.csv",encoding='latin-1',sep=",",error_bad_lines=False)
sim_log = np.log(df['SIMULATED_ENERGY_CONSUMPTION_kWh'])
df['log_SIMULATED_ENERGY']=sim_log
df['DATE']=pd.to_datetime(df['FECHA'])
#fig_sun = px.sunburst(df, path=['INTERVENTION_BOOLEAN','HOUSEHOLD_SIZE_INT', 'HOUSEHOLD_INCOME_INT' ])
fig_sun =px.line(df, x="FECHA", y='DAILY_ENERGY_CONSUMPTION_KWH', color='HOUSEHOLD_SIZE_INT')
fig_sun.update_layout(
    title="Consumo diario de energía",
    xaxis_title="Fecha",
    yaxis_title="Consumo diario Kwh por Hogar",
    legend_title="Número habitaciones-hogar")
df1 =  pd.read_csv('grafica_treemap.csv')
fig2 = px.treemap(df1,  path= ['Short Name', 'Indicator Name'], values='2013')
fig2.update_layout(
    title="Energía no renovable Asia", legend_title="Asia")
fig_area = px.area(df, x="DATE", y="MEAN_RELATIVE_HUMIDITY_PERC")
fig_area.update_layout(
    title="Promedio humedad relativa",
    xaxis_title="Fecha",
    yaxis_title="Humedad relativa" )
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
ds=pd.read_csv('ds.csv')
ds_differenced=pd.read_csv('ds_diff.csv')


# In[44]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
image_filename = 'grafica1.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
app.layout = html.Div([
     dcc.Tabs([
        dcc.Tab(label='Mundial', children=[
            html.H1(children='EVOLUCION DEL CONSUMO Y GENERACION DE ENERGIA ELECTRIA '),
    html.Div([dcc.Slider(
      id='year--slider',
      min=basedash['Año'].min(),
      max=2015,
      step=1,
      value=1990,
      marks={str(year): str(year) for year in np.arange(1960,2015,4)}
    ),]),
    html.Div([
              html.Div([
                        dcc.Graph(id='no-renovable',style={'width':'95%'}),
              ],style={'width': '48%', 'display': 'inline-block'}),
              html.Div([
                        dcc.Graph(id='renovable',style={'width':'95%'}),
              ],style={'width': '48%', 'display': 'inline-block'}),
    ]),
    html.Div([
              html.Div([
                        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),style={'width':'80%'}),
              ],style={'width': '32%', 'display': 'inline-block','vertical-align': 'top'}),
              html.Div([
                        html.Div(id = 'output_1'),
              ],style={'width': '32%', 'display': 'inline-block','vertical-align': 'middle'}),
              html.Div([ 
                        dcc.Graph(id = 'output_2'),
              ],style={'width': '32%', 'display': 'inline-block','vertical-align': 'middle'}),
    ]),
        ]),
        dcc.Tab(label='Asia-Kuala Lumpur', children=[
                html.Div([
                        html.Div([
                        html.Div('Malasia es un país asiático cuya energía depende en gran parte de recursos no renovables, lo cual incrementa los costos del servicio eléctrico; a esto se suma la fuerte sensación de humedad constante al año.'),
              ],style={'vertical-align': 'top'}),
                    
              html.Div([
                        dcc.Graph(id='sun',figure=fig_sun,style={'width': '32vw', 'height': '73vh'}),
              ],style={'vertical-align': 'bottom'}),
              ],style={'width': '32%', 'display': 'inline-block','vertical-align': 'top'}),
              # Aqui acaba la primera columna
              html.Div([
                        html.Div([dcc.Graph(id='tree',figure=fig2,style={'width': '32vw', 'height': '100vh'})])]
              ,style={'width': '32%', 'display': 'inline-block','vertical-align': 'top'}),
              #Aqui acaba la segunda columna
              html.Div([ 
                        html.Div([
                        dcc.Graph(id='area',figure=fig_area,style={'width': '32vw', 'height': '42vh'}),
              ],style={'height': '42%', 'vertical-align': 'top'}),
              html.Div([
                        dcc.Graph(id='barras',style={'width': '32vw', 'height': '42vh'}),
              ],style={'height': '42%', 'vertical-align': 'top'}),
              html.Div([dcc.Slider(
                        id='bins',
                        min=1,
                        max=25,
                        step=1,
                        value=10),])
              ],style={'width': '32%', 'display': 'inline-block','vertical-align': 'middle'}),
        ]),
         dcc.Tab(label='Modelo', children=[html.H1(children='Modelo predictivo por series de tiempo'),html.Div([dcc.Slider(
                        id='prediccion',
                        min=100,
                        max=1000,
                        step=10,
                        value=10),]),
                        html.Div([
                        dcc.Graph(id='serie'),
                        ]),
          ]),                                  
    ])
])   
 
@app.callback(Output('serie', 'figure'),
              Input('prediccion','value'))
def update_graph(prediccion):
    lag_order = loaded_model.k_ar
    predicted = loaded_model.forecast(ds_differenced.values[-lag_order:], prediccion)
    forecast = pd.DataFrame(predicted, index = ds.index[-prediccion:], columns = ds.columns)
    p2 = loaded_model.plot_forecast(1)
    plotly_fig = mpl_to_plotly(p2)
    return plotly_fig
    
@app.callback(Output('barras', 'figure'),
              Input('bins','value'))
def update_graph(bins):
    fig = px.histogram(df, x='MEAN_TEMPERATURE_C', nbins=bins)
    fig.update_layout(
    title="Temperatura promedio",
    xaxis_title="Temperatura °C",
    yaxis_title="Frecuencia")
    return fig
      
@app.callback(Output('no-renovable', 'figure'),
              Input('year--slider', 'value'))
def update_graph(year_value):
  fig = px.choropleth(basedash[basedash['Año']==year_value], locations="Country Code",color="Electricity production from oil, gas and coal sources (% of total)", # lifeExp is a column of gapminder
      hover_name="Country Name", # column to add to hover information
      #animation_frame="Año", # column on which to animate
      #scope='asia',
      color_continuous_scale=px.colors.sequential.amp)
  fig.update_layout(
  # add a title text for the plot
      title_text = 'Producccion de Electricidad por fuentes no renovables (% del total)',
     # set projection style for the plot
      geo = dict(projection={'type':'natural earth'}) # by default,projection type is set to 'equirectangular'
      )
  fig.update_coloraxes(colorbar_title_text='% Total')
  return fig

@app.callback(Output('renovable', 'figure'),
              Input('year--slider', 'value'))
def update_graph(year_value):
  fig = px.choropleth(basedash[basedash['Año']==year_value], locations="Country Code",color="Electricity production from renewable sources, excluding hydroelectric (% of total)", # lifeExp is a column of gapminder
      hover_name="Country Name", # column to add to hover information
      #animation_frame="Año", # column on which to animate
      #scope='asia',
      color_continuous_scale=px.colors.sequential.algae)
  fig.update_layout(
  # add a title text for the plot
      title_text = 'Producccion de Electricidad por fuentes renovables (% del total)',
     # set projection style for the plot
      geo = dict(projection={'type':'natural earth'}) # by default,projection type is set to 'equirectangular'
      )
  fig.update_coloraxes(colorbar_title_text='% Total')
  return fig

@app.callback(
    Output(component_id='output_1', component_property='children'),
    Input(component_id='year--slider', component_property='value')
)
def update_output_div(input_value):
  dr=EAyP[EAyP['Año']==input_value]['Electricity production from renewable sources, excluding hydroelectric (% of total)'].count()
  per=EAyP[EAyP['Año']==input_value]['Electricity production from renewable sources, excluding hydroelectric (% of total)'].mean()
  if dr==0:
    return html.Div([
                  html.Div('En el año {}, no hay datos reportados de la region del Este Asiatico y el Pacifico.'.format(input_value), style={'color': 'black', 'fontSize': 25}),
                  ], style={'marginBottom': 150 }) 
  else:
    return html.Div([
                  html.Div('En el año {}, hay datos reportados de {} paises de la region del Este Asiatico y el Pacifico.'.format(input_value, dr), style={'color': 'black', 'fontSize': 25}),
                  html.P('De estos, el porcentaje promedio de produccion de energia de fuentes renovables es {}%'. format(round(per, 2)), style={'color': 'green', 'fontSize': 25})
                  ], style={'marginBottom': 150})
@app.callback(
    Output(component_id='output_2', component_property='figure'),
    Input(component_id='year--slider', component_property='value')
)
def update_graph(input_value):
  a1=max(1960,input_value-4)
  le=np.arange(a1,input_value+1,1)
  M=Malaysia[Malaysia['Año'].isin(le)]
  fig=px.bar(M, x='Año',y='Electricity production from renewable sources, excluding hydroelectric (% of total)')
  fig.update_layout(
    # add a title text for the plot
        title_text = 'Electricidad verde Malasia',
        xaxis_title="Año",
        yaxis_title="% del Total",)
  return fig
app.run_server(mode='inline', port=8080)


# In[ ]:





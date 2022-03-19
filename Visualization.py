# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:14:55 2022

@author: ctj
"""
#%%


from dash.dash_table import DataTable, FormatTemplate, Format

import dash
from dash import html,dcc
import pandas as pd
from jupyter_dash import JupyterDash
import yfinance as yf
import plotly.express as px
import datetime
#from datetime import date
import numpy as np
from scipy.stats import norm

import plotly.graph_objects as go
from dash import Input, Output, State

from abc import ABC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import dash_bootstrap_components as dbc
from dash import dcc
from dashtable import get_dashtable
import MLstrategies as ml

#%% Monte Carlo Simulation for Stock Price
#adding an memoization 
m={}
def memoize(f):
    def wrapper(*args):
        global m
        print(args)
        print(m)
        if args not in m.keys():
            m[args]=f(*args)
            print("new****************************")
        
        
        return(m[args])
    return(wrapper)


@memoize
def MCsimulation(trials,days,ticker,date):
    
    df=yf.download(ticker,period='5y', progress=False)
    close=df['Close'].loc[:date]
    logr=np.log(1+close.pct_change())
    u=logr.mean()
    var=logr.var()
    drift=u-0.5*var
    vol=logr.std()
    Z=norm.ppf(np.random.rand(days, trials))
    #print(Z.shape)
    #use random number to simulate daility return
    r=drift+vol*Z
    exp_r=np.exp(r)
    price_paths=np.zeros_like(exp_r)
    price_paths[0]=close[-1]
    for i in range(1,days):
        price_paths[i]=price_paths[i-1]*exp_r[i]
        
    xaxis=list(range(days+1))
   
    fig_all=px.line(price_paths)
    #print(price_paths)
    fig_mean=px.line(np.mean(price_paths,axis=1))
    return fig_all,fig_mean,price_paths
#%% Dash Plot
def get_figure(fig_title, xtitle = None):
    fig = go.Figure()
    
    fig.update_layout(
        # this is a function taking multiple kwargs where complex args have to be passed as dictionaries
        title = {
            'text': fig_title,
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22}
        },
        paper_bgcolor = 'white',
        plot_bgcolor = 'white',
        autosize = False,
        height = 400,
        xaxis = {
            'title': xtitle,
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        },
        yaxis = {
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        }
    )
    
    return(fig)
#%% Plot Dash Table
def get_table(mc_data,date,trails,days):
    #mc__data--array
    trails_list=['trail_{}'.format(i) for i in range(trails)]
    date_list=[date+datetime.timedelta(i) for i in range(days)]
    
    
    date_list=pd.Series(date_list)
    
    date_list=date_list.apply(lambda x:datetime.datetime.strftime(x,"%Y-%m-%d"))
    
    mc_df=pd.DataFrame(mc_data,columns=trails_list).round(2)#
    
    #insert index as a columns
    mc_df.insert(0,"date",date_list)
    
    row_style = [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#eaf1f8'
            },
            
        ]
    mc_datatable=get_dashtable(mc_df,"table_id",selectable = None, sorting = 'native',row_style = row_style)
    return mc_datatable

DROPDOWN_OPTIONS={"trials":[{'label': str(i), 'value': str(i)} for i in range(10,1000,10)],"predict_days":[{'label': str(i), 'value': str(i)}  for i in range(10,100,20)]}
#%%Get Dash App
app=JupyterDash(__name__)
#add a button, click the button can downlowad data
#img_url="aapl_data_plot.jpg"


app.layout = html.Div([
        html.Header("This is a Dash Web App", style = {'font_size':'6em','font-weight':'bolder','color':'grey'}),
        dcc.Location(id='url', refresh=False),
        dbc.Row([
        dbc.Col(dcc.Link(children="Time Series",href="/Time_series",id="link1",style = {'font_size':'6em','font-weight':'bolder','text-align': 'center', 'color': "blue"})),
        #html.Br(),
        dbc.Col(dcc.Link(children="Monte Carlo Simulation",href="/MCsimulation",id="link2",style = {'font_size':'6em','font-weight':'bolder','text-align': 'center', 'color': "blue"})),
        #html.Br(),
        dbc.Col(dcc.Link(children="ML Strategies",href="/ML_strategies",id="link3",style = {'font_size':'6em','font-weight':'bolder','text-align': 'center', 'color': "blue"})),
        ]),
        html.Div([
            html.Div([
            html.H1("Data Visualization",style = {'font_size':'6em','font-weight':'bolder','text-align': 'center', 'color': "grey"}),
            html.Br(),
            dcc.Dropdown(
                    options =[{"label":"GOOG","value":"GOOG"},
                              {"label":"AAPL","value":"AAPL"},
                              {"label":"FB","value":"FB"}],
                    value="AAPL",
                    placeholder = 'Select Ticker',
                    searchable = True,
                    multi = False,
                    id = 'ticker',
                    #style = {'width': '10em'}
                ),
            
        
            dcc.RadioItems(
                    options=[
                       {'label': 'daily', 'value': 'daily'},
                       {'label': 'intraday', 'value': 'intraday'},
                    
                   ],
                    style={'margin-top':"auto",
                           'color': "grey"},
                    value='daily',
                    id='time_period',
                ),
            html.Br(),       
            dcc.DatePickerRange(
                    id='date_range',
                    min_date_allowed=datetime.date(2017, 1, 1),
                    max_date_allowed=datetime.date.today(),
                    initial_visible_month=datetime.date(2020,1,1),
                    start_date=datetime.date(2017,1,1),
                    end_date=datetime.date.today()
                ),
            html.Div(id='date_pick_notice',style={'color': "grey"}),
            html.Div(
            [
                html.Button("Download Time Series Data",id='btn_csv'),
                dcc.Download(id="download-dataframe-csv"),
            ],style={
                "width":'200px',
                 "margin-right": '0%',
                "margin-left": "auto",
                'display': 'flex'
            }),  
            html.Div([

            html.H2(
                #children='Time Series Data of {}'.format(ticker),
                id="title1",
                style={
                    'textAlign': 'center',
                    'color': "grey"
                }),   
        
            dcc.Graph(
                  id='figure_update',
                #figure=fig,
                  style={
            
                    "display": "block",
                    #"margin-left": '10%',
                    "margin-right": "auto",

                }),
            ],style ={
            'margin': '2em',
            'border-radius': '1em',
            'border-style': 'solid', 
            'padding': '2em',
            'background': "white","display":"block"},
            id="showpage1",
            
                ),
            html.Div([
                html.H2("Monte Carlo Simulation",style = {'font_size':'4em','font-weight':'bolder','text-align': 'center','color': "grey",}),
                #Plot Monte-Carlo Simulation
                dcc.DatePickerSingle(
                id="mc_begin_date",
                min_date_allowed=datetime.date(2017, 1, 1),
                max_date_allowed=datetime.date.today()-datetime.timedelta(days=30),
                initial_visible_month=datetime.date(2020,1,1),
                date=datetime.date(2020,1,3),
                style = {'width':'10em'}
            
                ),
                html.Br(),
                html.Br(),
                
                html.Div([
                        dcc.Dropdown(
                            options =opt,
                            placeholder = 'Select {}'.format(name),
                            disabled = False if name == 'trials' else True,
                            searchable = True,
                            multi = False,
                            id = '{}_dd'.format(name),
                            style = {'width': '10em'}
                        ) for name, opt in DROPDOWN_OPTIONS.items()
                    ],style = {'width':'350px','display': 'flex','justify-content': 'space-between'}
                    ),
                
                html.Br(),
                
                dcc.RadioItems(
                        options=[
                           {'label': 'all price paths', 'value': 'all price paths'},
                           {'label': 'average price', 'value': 'average price'},
                       ],
                        id='graph_type',
                        style={'color': "grey"}
                    ),
                html.Br(),
                dcc.Graph(
                          id='figure_MC',
                        #figure=fig,
                          style={
                    
                            "display": "block",
        
                        }),
                
                html.Div([
                html.H2("Monte Carlo Simulation data",style = {'font_size':'4em','font-weight':'bolder','text-align': 'center','color': "grey"}),
            ],style={'display': 'none'},id="showtable"),
            html.Div(id='table'),]
            ,style ={
            'margin': '2em',
            'border-radius': '1em',
            'border-style': 'solid', 
            'padding': '2em',
            'background': "white","display":"none"},
            id="showpage2"),
            
            html.Div([
                 html.H2("Result of Trading Strategies",style = {'font_size':'4em','font-weight':'bolder','text-align': 'center','color': "grey"}),
        html.Br(),
        html.Div([
                dcc.DatePickerSingle(
                    id="strategy_start",
                    style = {'width': '10em',"height":'3em','font_size':'2em'},
                    min_date_allowed=datetime.date(2017,1,1),
                    max_date_allowed=datetime.date(2018,1,1),
                    initial_visible_month=datetime.date(2018,1,1),
                    date=datetime.date(2017,1,1)),
                
                dcc.Dropdown(
                            options =[
                                {"label":"NaiveStrategy","value":"NaiveStrategy"},
                                {"label":"LogisticStrategy","value":"LogisticStrategy"},
                                {"label":"SVMStrategy_rbf","value":"SVMStrategy_rbf"},
                                {"label":"SVMStrategy_sigmoid","value":"SVMStrategy_sigmoid"},
                                {"label":"GradientBootStrategy","value":"GradientBootStrategy"},
                                {"label":"RandomForestStrategy","value":"RandomForestStrategy"},
                                 
                                ],
                            value="NaiveStrategy",
                            placeholder = 'Select a Strategy',
                            searchable =False,
                            multi = False,
                            id = 'strategy',
                            style = {'width': '10em','height':"3em","text-align": 'center','font_size':'2em'}
                        )],style = {'width':'350px','display': 'flex','justify-content': 'space-between'}
        
        ),
        
        
        dcc.Graph(
                  id='figure_strategy',
                #figure=fig,
                  style={
            
                    "display": "block",

                }),
        
        html.Div(id="alert"),
        
        html.Div([
        html.H2("Strategy Performance",style = {'font_size':'4em','font-weight':'bolder','text-align': 'center','color': "grey"}),
        ],style={'display': 'none'},id="showperformance"),
                
        html.Div(id='performance table'),
     
                
            ],style ={
            'margin': '2em',
            'border-radius': '1em',
            'border-style': 'solid', 
            'padding': '2em',
            'background': "white","display":"none"},
            id="showpage3"
            )
        ],style ={
            'margin': '2em',
            'border-radius': '1em',
            'border-style': 'solid',
            'padding': '2em',
            'background': "white"},
            id="webcontent")
            
    ])
    ])
        
#%% Show the Web App in 3 pages
links={"/Time_series":"1","/MCsimulation":"2","/ML_strategies":"3"}
@app.callback(
        [Output(("showpage{}").format(name),"style")for name in links.values()],
        [Input("url",'pathname')]
        )
def show_diff_pages(*args):
    ctx=dash.callback_context
    
    if not ctx.triggered:
        from dash.exceptions import PreventUpdate
        print("not trigger")
        raise PreventUpdate
    else:
        mhref=ctx.triggered[0]['value']
    
    display=[{"display":"block"}]+[{"display":"none"}]+[{"display":"none"}]
    if mhref=="/Time_series":
        return display
    if mhref=="/MCsimulation":
        display=[{"display":"none"}]+[{"display":"block"}]+[{"display":"none"}]
    if mhref=="/ML_strategies":
        display=[{"display":"none"}]+[{"display":"none"}]+[{"display":"block"}]
    return display
    
#%%
def update_strategy_figure(myfigure,ticker,strategy):
    myfigure.update_layout(
            title = {
        'text': "Profit and Loss of Trading {} with {}".format(ticker,strategy),
        'y': 0.95,
        'x': 0.5,
        'font': {'size': 22}
        },
            xaxis = {
            'title': 'number of trading days',
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        },
        yaxis = {
            'title':'P/l',
            'showline': True, 
            'linewidth': 1,
            'linecolor': 'black'
        })
    return myfigure

#%%
df_strategy=pd.DataFrame(columns=['Data','Start Day','Strategy','P&L','Sharpe Ratio'])

def get_strategy_performance(data,date,strategy,pnl,sharpe):
    global df_strategy
    df_strategy=df_strategy.append({"Data":data,"Start Day":date,"Strategy":strategy,"P&L":pnl,"Sharpe Ratio":sharpe},ignore_index=True)
    df_strategy=df_strategy.drop_duplicates()
    row_style = [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#eaf1f8'
            },
            
        ]
    
    performance_table=get_dashtable(df_strategy,"table_2",selectable=None,sorting='naive',row_style=row_style)
    return performance_table
#%% Plot Strategy        
#splot strategy
@app.callback(
            [Output("figure_strategy","figure")]+
            [Output("showperformance","style")]+
            [Output("performance table","children")],
            
            Input("strategy_start","date"),
            Input("ticker","value"),
            Input("strategy","value"),
            prevent_initial_call=False
        )
def strategy(mstart_date,mticker,mstrategy):
    strategy_dict={""}
    fig_strategy=get_figure("")
    show={'display':'block'}
    strategy_class=ml.find_strategy_class(mstrategy)
    pnl_figure,matrix,dcy,sp,p_l,pnl_list=ml.mystrategy(mstart_date,symbol=mticker,strategy=strategy_class)
    performance_table=get_strategy_performance(mticker,mstart_date,mstrategy,p_l,sp)
    new_figure=update_strategy_figure(pnl_figure,mticker,mstrategy)
    return [new_figure]+[show]+[performance_table]

#%% Download Data 
#download data
@app.callback(
    
        
        Output("download-dataframe-csv", "data"),
     
        Input("btn_csv", "n_clicks"),
        Input("time_period","value"),
        Input("date_range","start_date"),
        Input("date_range","end_date"),
        Input("ticker","value"),
        
        prevent_initial_call=True,
        )
def download_data(n_clicks,data_type,start_,end_,ticker):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if (changed_id=="btn_csv.n_clicks"):
      
        if data_type=='daily':
           
            data=yf.download(ticker,period='5y',progress=False)
            data=data.loc[start_:end_]
            
        else:
            data=yf.download(ticker,start=start_,interval="1m",progress=False)
        return dcc.send_data_frame(data.to_csv ,"{}_{}_data.csv".format(ticker,data_type))       
 
    

#%%   plot mc and get data table

@app.callback(
 [Output('{}_dd'.format(name),'disabled')for name in DROPDOWN_OPTIONS.keys()]+
    [Output('figure_MC','figure')]+[Output("showtable","style")]+
    [ Output("table",'children')]
    ,
     [Input('{}_dd'.format(name), 'value') for name in DROPDOWN_OPTIONS.keys()]+[
         Input('graph_type','value')]+[
         Input("mc_begin_date","date")]+[
         Input("ticker",'value')]
    
    
 )
def plot_MC(*args):
    date=args[3]
    ticker=args[4]
    #print(date)
    date=datetime.datetime.strptime(date[:10],"%Y-%m-%d")
    #print(date)
    end_date=date+datetime.timedelta(days=1)

    data=yf.download(ticker,
                    start=date,
                    end=end_date,
                    interval="1d",
                    progress=False,
        )
    print(end_date)
    #print("!!!",data)
    s0=float(data["Adj Close"].iloc[0])
    ctx = dash.callback_context
    #print("previous trigger",ctx.triggered)
    if not ctx.triggered:
        from dash.exceptions import PreventUpdate
        print('hey!')
        raise PreventUpdate
    else:
       # print("trigger",ctx.triggered)
        
        dropdown = ctx.triggered[0]['prop_id'].split('.')[0]
        print(dropdown)
      
#     if len(args)==2:
#         alert='The selection would result in a Monte Carlo Simulation plot of {} trials for {} days prediction'.format(args)
#     else:
#         alert='Please chose fuether options'
        
    disabled=[True]*(len(args)-3)
    print(disabled)
    fig_all=get_figure("")
    fig_mean=get_figure("")
    show={'display':'block'}
    mc_table=get_table([0],date,1,1)
    if dropdown=="trials_dd"or dropdown=="ticker"or dropdown=="mc_begin_date":
        disabled=[False]+[False]
    if dropdown=="predict_days_dd":
        disabled=[False]+[True]
    if dropdown=="graph_type":
        disabled=[False]+[True]
    if args[0]and args[1]and args[2]:
        trial=int(args[0])
        day=int(args[1])
        fig_all,fig_mean,price=MCsimulation(trial,day,ticker,date)
        
        
        #get_table
        mc_table=get_table(price,date,trial,day)
        
        
        if args[2]=='all price paths':
            fig_all.update_layout(
                title = {
            'text': 'all paths of predicted price',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22}
            },
                xaxis = {
                'title': 'number of predicted days',
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black'
            },
            yaxis = {
                'title':'predicted close price',
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black'
            })
            print(disabled)
            return disabled+[fig_all]+[mc_table]
        elif args[2]=='average price':
            fig_mean.update_layout(
                title = {
            'text': 'average of predicted price',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22}
            },
                xaxis = {
                'title': 'number of predicted days',
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black'
            },
            yaxis = {
                'title':'predicted close price',
                'showline': True, 
                'linewidth': 1,
                'linecolor': 'black'
            })
            return disabled+[fig_mean]+[show]+[mc_table]
    print(disabled)
    
    return disabled+[fig_all]+[show]+[mc_table]       

#update title
@app.callback(
    Output("title1","children"),
    Input("ticker",'value'),
    prevent_initial_call=False,
)
def update_title(ticker):
    return 'Time Series Data of {}'.format(ticker)


#%%plot time series data
@app.callback(
    Output("figure_update","figure"),
    Output("date_range","max_date_allowed"),
    Output("date_range","min_date_allowed"),
    Output("date_range","initial_visible_month"),
    Output("date_range","end_date"),
    Output("date_pick_notice","children"),
    
    Input("ticker",'value'),
    Input("time_period","value"),
    Input("date_range","start_date"),
    Input("date_range","end_date"),
    
    
    #prevent_initial_call=False,
)

def update_figure(ticker,time_interval,start_date,end_date):
    print(ticker)
    start_date=datetime.datetime.strptime(start_date[:10],"%Y-%m-%d")
    end_date=datetime.datetime.strptime(end_date[:10],"%Y-%m-%d")
    if time_interval=="intraday":
        notice="Please select a date within 30 days from now."

        if start_date.date()<datetime.date.today()-datetime.timedelta(days=30):
            start_date=datetime.date.today()
            
        min_date_allowed=datetime.date.today()-datetime.timedelta(days=30)
        max_date_allowed=datetime.date.today()-datetime.timedelta(days=1)
        initial_visible_month=datetime.date.today()
        end_date=start_date+datetime.timedelta(days=1)
        data=yf.download(ticker,start=start_date,
                    end=end_date,
                    period="1d",  
                    interval="1m",
                    progress=False,
        )
        fig = px.line(data[:-1], y="Close", x=data.index[:-1])
        fig.update_layout()
        #AAPL=AAPL_intraday
    elif time_interval=="daily":
        notice="Please select date range."
        min_date_allowed=datetime.date(2017, 1, 1),
        max_date_allowed=datetime.date.today(),
        initial_visible_month=start_date
        data=yf.download(ticker,start=start_date,
                    end=end_date,
                    interval="1d",
                    progress=False,
        )
        fig = px.line(data, y="Close", x=data.index)
        #AAPL=AAPL_daily
        fig.update_layout(
            xaxis=dict(
                rangeselector = dict(
                    buttons = list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"
                             ),
                        dict(count=6,
                             label="6m",
                             step="month",

                             ),
                        dict(count=1,
                             label="1y",
                             step="year",
                             ),
                        dict(count=3,
                             label="2y",
                             step="year",
                             stepmode="backward"
                             ),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date",
            )
            
        )
    return fig,str(max_date_allowed),str(min_date_allowed),str(initial_visible_month),str(end_date),notice
#s

app.run_server(port='8051')



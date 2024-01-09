# dash_app.py
import dash
from dash import html, dcc, Input, Output, State, callback_context, callback
import requests
import base64
import uuid
import os
from datetime import datetime
import logging

# Obté el registre unificat de l'aplicació
logger = logging.getLogger('dash')

# Inicia l'aplicació Dash
dash_app = dash.Dash(__name__, requests_pathname_prefix='/dash/')

# Defineix el disseny de l'aplicació Dash
dash_app.layout = html.Div([
    html.H1("Interfície Frontend per Interaccionar amb Models RL i LLM", style={'textAlign': 'center'}),
    html.Div([
        # Left Column
        html.Div([
            html.Div(id='loading-output'),
            dcc.Loading(id="loading-indicator", children=[html.Div(id="loading-content")], type="circle"),
            # QA with Facts Section
            html.Div([
                html.H3("Enviar una pregunta al CyberVigilant (LLM) amb dues pistes (facts)"),
                dcc.Input(id='input-question', type='text', maxLength=1024, placeholder="Introdueix la pregunta"),
                dcc.Input(id='input-hint1', type='text', maxLength=150, placeholder="Pista 1"),
                dcc.Input(id='input-hint2', type='text', maxLength=150, placeholder="Pista 2"),
                html.Button('Analitzar', id='button-qa-with-facts')
            ], className='section'),

            # Analyze Questions Section
            html.Div([
                html.H3("Enviar una pregunta al CyberVigilant (LLM)"),
                dcc.Input(id='input-analyze-question', type='text', maxLength=1024, placeholder="Introdueix la pregunta"),
                html.Button('Analitzar', id='button-analyze-question')
            ], className='section'),
            
            # RL Model Training Section
            html.Div([
                html.H3("Entrenar el model d'aprenentatge reforçat RL - Utilitzat per el CyberVigilant per fer l'avaluació de les respostes"),
                dcc.Upload(
                    id='upload-train-rl', 
                    children=html.Button('Carregar fitxer JSONL'), 
                    accept='.jsonl'
                ),
                html.Button('Entrenar RL', id='button-train-rl')
            ], className='section'),

            # LLM Model Training Section
            html.Div([
                html.H3("Entrenar el model LLM - el CyberVigilant utilitza aquest model per generar respostes"),
                dcc.Upload(
                    id='upload-train-llm', 
                    children=html.Button('Carregar fitxer CSV'), 
                    accept='.csv'
                ),
                html.Button('Entrenar LLM', id='button-train-llm')
            ], className='section'),
        ], className='left-column'),
        
        # Right Column
        html.Div([
            html.H3("Responses", style={'textAlign': 'center'}),
            html.Div(id='graph-output'),
            html.Div(id='system-messages')
        ], className='right-column'),
    ], className='main-content'),
], className='body')

@dash_app.callback(
    Output('system-messages', 'children'),
    [Input('button-analyze-question', 'n_clicks'),
     Input('button-qa-with-facts', 'n_clicks'),
     Input('button-train-rl', 'n_clicks'),
     Input('button-train-llm', 'n_clicks')],
    [State('input-analyze-question', 'value'),
     State('input-question', 'value'),
     State('input-hint1', 'value'),
     State('input-hint2', 'value'),
     State('upload-train-rl', 'contents'),
     State('upload-train-llm', 'contents'),
     State('system-messages', 'children')],
    prevent_initial_call=True
)
def combined_callback(
    btn_analyze_n_clicks, btn_qa_n_clicks, btn_train_rl_n_clicks, btn_train_llm_n_clicks,
    analyze_question_text, qa_question, qa_hint1, qa_hint2,
    rl_file_contents, llm_file_contents, existing_messages
):
    logger.debug("Callback de l'aplicació Dash activat")
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_message = None
    server_busy = True

    if triggered_id == 'button-analyze-question' and analyze_question_text:
        # Handle analyze_question logic
        response = requests.post('http://127.0.0.1:8000/analyze_question/ep', json={"question": analyze_question_text})
        new_message = response.json()['resposta'] if response.status_code == 200 else "Error: no s'ha pogut processar la pregunta."

    elif triggered_id == 'button-qa-with-facts' and qa_question:
        # Handle qa_with_facts logic
        payload = {"question": qa_question, "hint1": qa_hint1, "hint2": qa_hint2}
        response = requests.post('http://127.0.0.1:8000/a_with_facts/ep_f', json=payload)
        new_message = response.json()['resposta'] if response.status_code == 200 else html.P("Error: no s'ha pogut processar la pregunta amb pistes.")

    elif triggered_id == 'button-train-rl' and rl_file_contents:
       # Handle RL model training logic and file upload
        content_type, content_string = rl_file_contents.split(',')
        decoded = base64.b64decode(content_string)
        filename = str(uuid.uuid4()) + '.jsonl'  # Nom de fitxer únic

        # Utilitza os.path.join per a construir la ruta de manera compatible amb diferents SO
        datasets_folder = os.path.join('datasets')  # Ruta relativa a la carpeta datasets
        os.makedirs(datasets_folder, exist_ok=True)  # Crea la carpeta si no existeix

        file_path = os.path.join(datasets_folder, filename)  # Camí complet del fitxer

        with open(file_path, 'wb') as f:
            f.write(decoded)

        # try:
            # Enviar sol·licitud al punt final de l'API
        payload = {'file_path': file_path}
        response = requests.post('http://127.0.0.1:8000/train_rl/ep_r', json=payload)
        data_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_message = response.json()['message'] if response.status_code == 200 else html.P(f"[{data_actual}] Error: ha fallat l'entrenament del model RL.")
        # finally:
        #    os.remove(file_path)  # Esborrem el fitxer després de l'ús

    elif llm_file_contents is not None:
        # Decodificar el contingut del fitxer CSV codificat en base64
        content_type, content_string = llm_file_contents.split(',')
        decoded = base64.b64decode(content_string)
        filename = str(uuid.uuid4()) + '.csv'  # Nom de fitxer únic

        # Crear la ruta al directori on es desaran els datasets
        datasets_folder = os.path.join('datasets')
        os.makedirs(datasets_folder, exist_ok=True)

        # Camí complet del fitxer on es desarà
        file_path = os.path.join(datasets_folder, filename)

        # Desar el contingut decodificat en un fitxer
        with open(file_path, 'wb') as f:
            f.write(decoded)

        try:
            # Enviar una sol·licitud al punt final de l'API per entrenar el model
            payload = {'file_path': file_path}
            response = requests.post('http://127.0.0.1:8000/train_llm/ep_l', json=payload)

            data_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_message = response.json()['message'] if response.status_code == 200 else html.P(f"[{data_actual}] Error: ha fallat l'entrenament del model LLM.")
        except Exception as e:
            # Manejar errors que poden sorgir durant la petició
            logger.error(f"Error en la petició a l'API: {e}")
            new_message = html.P(f"[{data_actual}] Error durant la petició a l'API.")
    else:
        data_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_message = html.P(f"[{data_actual}] No s'ha seleccionat cap arxiu.")

    logger.info(f"Resposta en l'app Dash desde {triggered_id}: {new_message}")
    server_busy = False
    # Afegim el nou missatge (resposta) als ja existints
    updated_messages = existing_messages + [html.P(new_message)] if existing_messages else [html.P(new_message)]
    return updated_messages

if __name__ == '__main__':
    dash_app.run_server(debug=True)
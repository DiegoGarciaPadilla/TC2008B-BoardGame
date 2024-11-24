from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import batch_run

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

import numpy as np
import pandas as pd

import time
import datetime

import seaborn as sns

import plotly.graph_objects as go

import networkx as nx

import json

# ----------------- Leer el archivo de configuración -----------------

def read_board_config():
    """
    Leer el archivo de configuración del tablero y devolver un diccionario con la configuración
    """

    # Abrir el archivo de configuración
    file = open("board-config.txt", "r")

    # Obtener la información del tablero
    config = file.readlines()

    # Cerrar el archivo
    file.close()

    # Crear un diccionario para almacenar la configuración del tablero
    board_config = {}

    # Configuración del tablero
    board_config['board'] = [x.replace("\n", "").split() for x in config[:6]]

    # Puntos de interés
    board_config['points_of_interest'] = [x.replace("\n", "").split() for x in config[6:9]]

    # Indicadores de fuego
    board_config['fire_indicators'] = [x.replace("\n", "").split() for x in config[9:19]]

    # Puertas
    board_config['doors'] = [x.replace("\n", "").split() for x in config[19:27]]

    # Puntos de entrada
    board_config['entry_points'] = [x.replace("\n", "").split() for x in config[27:31]]

    return board_config

board_config = read_board_config()

# ----------------- Función para crear el grafo -----------------

def read_board(board_config):
    """
    Inicializa el tablero como un grafo usando NetworkX.
    """

    # Crear un grafo vacío
    G = nx.Graph()

    # Expandir el tablero con un anillo exterior
    rows = len(board_config['board'])
    cols = len(board_config['board'][0])

    expanded_board = [['x' for _ in range(cols + 2)] for _ in range(rows + 2)]

    # Copiar la información del tablero original al centro del tablero expandido
    for i in range(rows):
        for j in range(cols):
            expanded_board[i + 1][j + 1] = board_config['board'][i][j]

    # Ajustar las coordenadas de las puertas
    doors = {}
    for door in board_config['doors']:
        doors[(int(door[0]), int(door[1]))] = (int(door[2]), int(door[3]))
        doors[(int(door[2]), int(door[3]))] = (int(door[0]), int(door[1]))  # Puerta bidireccional

    # Agregar nodos para cada celda del tablero expandido
    for i in range(rows + 2):
        for j in range(cols + 2):
            if i == 0 or j == 0 or i == rows + 1 or j == cols + 1:
                # Nodo en el anillo exterior
                G.add_node((i, j), fire=0, POI=None, is_entry_point=False, type='exterior')
            else:
                # Nodo dentro del tablero original
                G.add_node((i, j), fire=0, POI=None, is_entry_point=False, type='interior')

    # Crear las conexiones del tablero expandido
    for i in range(rows + 2):
        for j in range(cols + 2):
            current_cell = expanded_board[i][j] if 0 < i < rows + 1 and 0 < j < cols + 1 else None

            # Procesar las conexiones para cada dirección
            # Arriba
            if i > 0:
                neighbor_cell = expanded_board[i - 1][j] if i - 1 > 0 else None
                if (i, j) in doors and doors[(i, j)] == (i - 1, j):
                    add_door(G, i, j, i - 1, j)  # Puerta
                elif current_cell and neighbor_cell:
                    if current_cell[0] == '1' or neighbor_cell[2] == '1':
                        add_wall(G, i, j, i - 1, j)  # Muro
                    else:
                        add_path(G, i, j, i - 1, j)  # Camino
                elif not G.has_edge((i, j), (i - 1, j)):
                    add_wall(G, i, j, i - 1, j)  # Muro por defecto

            # Izquierda
            if j > 0:
                neighbor_cell = expanded_board[i][j - 1] if j - 1 > 0 else None
                if (i, j) in doors and doors[(i, j)] == (i, j - 1):
                    add_door(G, i, j, i, j - 1)  # Puerta
                elif current_cell and neighbor_cell:
                    if current_cell[1] == '1' or neighbor_cell[3] == '1':
                        add_wall(G, i, j, i, j - 1)  # Muro
                    else:
                        add_path(G, i, j, i, j - 1)  # Camino
                elif not G.has_edge((i, j), (i, j - 1)):
                    add_wall(G, i, j, i, j - 1)  # Muro por defecto

            # Abajo
            if i < rows + 1:
                neighbor_cell = expanded_board[i + 1][j] if i + 1 <= rows else None
                if (i, j) in doors and doors[(i, j)] == (i + 1, j):
                    add_door(G, i, j, i + 1, j)  # Puerta
                elif current_cell and neighbor_cell:
                    if current_cell[2] == '1' or neighbor_cell[0] == '1':
                        add_wall(G, i, j, i + 1, j)  # Muro
                    else:
                        add_path(G, i, j, i + 1, j)  # Camino
                elif not G.has_edge((i, j), (i + 1, j)):
                    add_wall(G, i, j, i + 1, j)  # Muro por defecto

            # Derecha
            if j < cols + 1:
                neighbor_cell = expanded_board[i][j + 1] if j + 1 <= cols else None
                if (i, j) in doors and doors[(i, j)] == (i, j + 1):
                    add_door(G, i, j, i, j + 1)  # Puerta
                elif current_cell and neighbor_cell:
                    if current_cell[3] == '1' or neighbor_cell[1] == '1':
                        add_wall(G, i, j, i, j + 1)  # Muro
                    else:
                        add_path(G, i, j, i, j + 1)  # Camino
                elif not G.has_edge((i, j), (i, j + 1)):
                    add_wall(G, i, j, i, j + 1)  # Muro por defecto
    
    # Conectar nodos exteriores entre sí
    for node in G.nodes:
        if G.nodes[node]['type'] == 'exterior':
            for neighbor in G.adj[node]:
                # Verificar si ambos nodos son exteriores
                if G.nodes[neighbor]['type'] == 'exterior':
                    # Asegurarse de que la conexión sea un camino
                    if not G.has_edge(node, neighbor):
                        G.add_edge(node, neighbor, weight=1, type='path')
                    elif G[node][neighbor].get('type') != 'path':
                        # Sobreescribir cualquier conexión incorrecta con un camino
                        G[node][neighbor]['type'] = 'path'
                        G[node][neighbor]['weight'] = 1


    # Configurar puntos de interés y fuego inicial
    for poi in board_config['points_of_interest']:
        add_POI(G, int(poi[0]), int(poi[1]), poi[2] == 'v')
    for fire in board_config['fire_indicators']:
        add_fire(G, int(fire[0]), int(fire[1]))

    # Conectar puntos de entrada a nodos externos
    for entry_point in board_config['entry_points']:
        entry_x, entry_y = int(entry_point[0]), int(entry_point[1])
        G.nodes[(entry_x, entry_y)]['isEntryPoint'] = True

        # Conectar al nodo externo más cercano
        neighbors = [
            (entry_x - 1, entry_y),  # Arriba
            (entry_x + 1, entry_y),  # Abajo
            (entry_x, entry_y - 1),  # Izquierda
            (entry_x, entry_y + 1)   # Derecha
        ]

        for neighbor in neighbors:
            if neighbor in G.nodes and G.nodes[neighbor]['type'] == 'exterior':
                # Asegurarse de que sea un camino, no un muro
                if G.has_edge((entry_x, entry_y), neighbor):
                    G[(entry_x, entry_y)][neighbor]['type'] = 'path'
                    G[(entry_x, entry_y)][neighbor]['weight'] = 1
                else:
                    # Crear una nueva conexión como un camino
                    G.add_edge((entry_x, entry_y), neighbor, type='path', weight=1)


    return G

def validate_graph(G):
    """
    Validar y actualizar los pesos de las aristas en el grafo en función del estado actual (fuego, humo, puertas).
    """

    for edge in G.edges(data=True):
        node1, node2, edge_data = edge

        # Determinar el estado de los nodos conectados
        fire_status1 = G.nodes[node1].get('fire', 0)  # Nodo 1: 0 (nada), 1 (humo), 2 (fuego)
        fire_status2 = G.nodes[node2].get('fire', 0)  # Nodo 2: 0 (nada), 1 (humo), 2 (fuego)
        edge_type = edge_data.get('type', 'path')  # Por defecto, consideramos 'path'

        # Determinar el peso en función del peor caso (fuego > humo > nada)
        fire_status = max(fire_status1, fire_status2)

        # Ajustar el peso según el estado del nodo y el tipo de conexión
        if fire_status == 2:  # Fuego
            if edge_type == 'path':
                G[node1][node2]['weight'] = 3
            elif edge_type == 'wall':
                G[node1][node2]['weight'] = 7
            elif edge_type == 'door':
                if edge_data.get('is_open', False):
                    G[node1][node2]['weight'] = 3
                else:
                    G[node1][node2]['weight'] = 4
        elif fire_status == 1:  # Humo
            if edge_type == 'path':
                G[node1][node2]['weight'] = 2
            elif edge_type == 'wall':
                G[node1][node2]['weight'] = 6
            elif edge_type == 'door':
                if edge_data.get('is_open', False):
                    G[node1][node2]['weight'] = 2
                else:
                    G[node1][node2]['weight'] = 3
        else:  # Sin fuego ni humo
            if edge_type == 'path':
                G[node1][node2]['weight'] = 1
            elif edge_type == 'wall':
                G[node1][node2]['weight'] = 5
            elif edge_type == 'door':
                if edge_data.get('is_open', False):
                    G[node1][node2]['weight'] = 1
                else:
                    G[node1][node2]['weight'] = 2

def initialize_board(board_config):
    """
    Inicializar el tablero como un grafo con la configuración del archivo de configuración.
    """

    G = read_board(board_config)

    # Validar y actualizar los pesos de las aristas
    validate_graph(G)

    return G

# ----------------- Funciones de visualización -----------------

def plot_graph(G, title='Flash Point: Fire Rescue'):
    """
    Graficar el grafo con Plotly.
    """

    # Definir las posiciones de los nodos como una cuadrícula
    pos = {node: (node[1], -node[0]) for node in G.nodes()}  # El eje y se invierte para que la visualización sea de arriba a abajo

    # Crear trazas para las aristas con colores según su tipo y mostrar peso y tipo
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_annotations = []

    # Definir los colores para cada tipo de arista
    edge_type_colors = {
        'wall': 'red',
        'path': 'green',
        'door': 'blue',
        'unknown': 'gray'  # Color por defecto para tipos no definidos
    }

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # Para separar las líneas de las aristas
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        # Asignar color según el tipo de la arista
        edge_type = edge[2].get('type', 'unknown')  # 'unknown' si no tiene tipo
        edge_colors.append(edge_type_colors.get(edge_type, 'gray'))  # Usar color definido o 'gray'

        # Guardar anotaciones para peso, tipo, y estado de apertura (si es una puerta)
        weight = edge[2].get('weight', '?')
        is_open = edge[2].get('is_open', None)  # Obtener el estado de apertura (None si no aplica)
        if edge_type == 'door':  # Solo agregar estado de apertura para puertas
            door_state = 'Abierta' if is_open else 'Cerrada'
            annotation_text = f'{weight}<br>{edge_type}<br>{door_state}'
        else:
            annotation_text = f'{weight}<br>{edge_type}'

        edge_annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=annotation_text,  # Mostrar peso, tipo y estado de apertura
                showarrow=False,
                font=dict(size=10, color='black')
            )
        )

    # Crear trazas para las aristas coloreadas
    edge_traces = []
    for idx in range(len(edge_colors)):
        edge_traces.append(
            go.Scatter(
                x=[edge_x[idx * 3], edge_x[idx * 3 + 1], None],
                y=[edge_y[idx * 3], edge_y[idx * 3 + 1], None],
                line=dict(width=2, color=edge_colors[idx]),
                hoverinfo='none',
                mode='lines'
            )
        )

    # Crear trazas para los nodos y asignar colores según las condiciones
    node_x = []
    node_y = []
    node_colors = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Determinar el color según las condiciones
        if G.nodes[node].get("fire") == 1:  # Humo
            node_colors.append('orange')
        elif G.nodes[node].get("fire") == 2:  # Fuego
            node_colors.append('red')
        elif G.nodes[node].get("POI") is not None:  # POI
            node_colors.append('blue')
        elif G.nodes[node].get("isEntryPoint"):  # Punto de entrada
            node_colors.append('green')
        else:
            node_colors.append('black')  # Color por defecto

        # Crear texto para hover
        adyacentes = []
        for neighbor in G.adj[node]:
            edge_data = G.get_edge_data(node, neighbor)
            edge_type = edge_data.get('type', 'unknown')
            adyacentes.append(f'{neighbor}: {edge_type}')

        # Obtener el tipo del nodo, si existe
        node_type = G.nodes[node].get("type", "undefined")

        # Agregar el número de conexiones (grados del nodo)
        num_connections = len(G.adj[node])

        node_text.append(
            f'Posición: {node}<br>'
            f'Tipo: {node_type}<br>'  # Mostrar el tipo del nodo
            f'Conexiones: {num_connections}<br>'
            f'Adyacentes: {", ".join(adyacentes)}<br>'
            f'Fuego: {G.nodes[node].get("fire")}<br>'
            f'POI: {G.nodes[node].get("POI")}<br>'
            f'Es punto de entrada: {G.nodes[node].get("isEntryPoint", False)}'
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_colors,  # Colores asignados según las condiciones
        ),
        text=node_text
    )

    # Crear la figura
    fig = go.Figure(
        data=edge_traces + [node_trace],  # Agregar trazas de aristas coloreadas y nodos
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=edge_annotations,  # Agregar anotaciones para pesos y tipos
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )

    fig.show()

def draw_board(graph):
    """
    Dibuja el tablero basado únicamente en la información contenida en el grafo.

    Args:
        graph (nx.Graph): Grafo que contiene información sobre paredes, fuego, humo, puertas, etc.
    """
    # Obtener dimensiones del tablero a partir de los nodos del grafo
    nodes = list(graph.nodes)
    rows = max(node[0] for node in nodes) + 1
    cols = max(node[1] for node in nodes) + 1

    # Crear la figura con un ajuste de márgenes
    fig, ax = plt.subplots(figsize=(cols + 2, rows + 2))  # Espacio extra en la figura

    # Dibujar las celdas
    for node in graph.nodes:
        y, x = node  # Coordenadas (fila, columna)
        coord_x = x - 1  # Ajuste para dibujar
        coord_y = rows - y  # Invertir el eje Y para visualización

        # Determinar color de fondo según el estado del nodo
        color = "white"  # Color por defecto

        if graph.nodes[node].get('fire') == 2:
            color = "red"  # Fuego
        elif graph.nodes[node].get('fire') == 1:
            color = "gray"  # Humo
        elif graph.nodes[node].get('POI') is not None:
            color = "blue"  # Punto de interés
        elif graph.nodes[node].get('isEntryPoint', False):
            color = "green"  # Punto de entrada

        # Dibujar el fondo de la celda
        ax.add_patch(patches.Rectangle((coord_x, coord_y), 1, 1, color=color))

        # Dibujar paredes y puertas
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)

            if edge_data.get('type') == 'wall':  # Dibujar pared
                if neighbor == (y - 1, x):  # Arriba
                    ax.plot([coord_x, coord_x + 1], [coord_y + 1, coord_y + 1], color="black", lw=2)
                elif neighbor == (y + 1, x):  # Abajo
                    ax.plot([coord_x, coord_x + 1], [coord_y, coord_y], color="black", lw=2)
                elif neighbor == (y, x - 1):  # Izquierda
                    ax.plot([coord_x, coord_x], [coord_y, coord_y + 1], color="black", lw=2)
                elif neighbor == (y, x + 1):  # Derecha
                    ax.plot([coord_x + 1, coord_x + 1], [coord_y, coord_y + 1], color="black", lw=2)
            elif edge_data.get('type') == 'door':  # Dibujar puerta
                if neighbor == (y - 1, x):  # Arriba
                    ax.plot([coord_x, coord_x + 1], [coord_y + 1, coord_y + 1], color="orange", lw=4)
                elif neighbor == (y + 1, x):  # Abajo
                    ax.plot([coord_x, coord_x + 1], [coord_y, coord_y], color="orange", lw=4)
                elif neighbor == (y, x - 1):  # Izquierda
                    ax.plot([coord_x, coord_x], [coord_y, coord_y + 1], color="orange", lw=4)
                elif neighbor == (y, x + 1):  # Derecha
                    ax.plot([coord_x + 1, coord_x + 1], [coord_y, coord_y + 1], color="orange", lw=4)

    # Ajustar límites para centrar el tablero
    margin_x = 2
    margin_y = 2
    ax.set_xlim(-margin_x, cols + margin_x - 1)
    ax.set_ylim(-margin_y, rows + margin_y - 1)

    # Mantener aspecto cuadrado y ocultar ejes
    ax.set_aspect('equal')
    ax.axis('off')

    # Mostrar el tablero
    plt.show()

# ----------------- Funciones de transformación -----------------

def graph_to_json(board_config, G):
    """
    Convertir el grafo a una matriz JSON para exportar a Unity.

    Args:
        G: Grafo de NetworkX.

    Returns:
        JSON: Matriz NxM con la información del grafo.
    """

    # Obtener la configuración del tablero
    board_config = board_config['board']

    # Obtener las dimensiones del tablero
    rows = len(board_config) + 2
    cols = len(board_config[0]) + 2

    # Crear una matriz 6x8 vacía
    JSON = {}

    for row in range(rows):

        # Agregar una fila vacía
        JSON[row] = {}

        for col in range(cols):

            # Obtener el nodo correspondiente
            node = (row, col)

            # Información sobre los muros
            wall_info = board_config[row - 1][col - 1] if 0 < row < rows - 1 and 0 < col < cols - 1 else ''

            # Información sobre las puertas
            door_to = []
            for neighbor in G.adj[node]:
                if G.get_edge_data(node, neighbor)['type'] == 'door':
                    door_to.append(neighbor)

            # Obtener el tipo del nodo
            node_type = G.nodes[node].get('type', 'undefined')

            # Obtener el estado del nodo
            fire_status = G.nodes[node].get('fire', 0)

            # Obtener el POI
            poi = G.nodes[node].get('POI', None)

            # Obtener las conexiones
            connections = node in G.adj and list(G.adj[node].keys()) or []

            # Crear un diccionario con la información del nodo
            node_info = {
                'wall_info': wall_info,
                'door_to': door_to,
                'type': node_type,
                'fire': fire_status,
                'poi': poi,
                'connections': connections
            }

            # Agregar el nodo
            JSON[row][col] = node_info

    return json.dumps(JSON)

# ----------------- Funciones de modelado -----------------

def add_path(G, x1, y1, x2, y2):
    """
    Agregar un camino al grafo
    """

    # Verificar si las celdas existen
    if (x1, y1) not in G.nodes or (x2, y2) not in G.nodes:
        return 

    # Agregar el camino al grafo
    G.add_edge((x1, y1), (x2, y2), weight=1, type='path')

def add_wall(G, x1, y1, x2, y2):
    """
    Agregar una pared al grafo
    """

    # Verificar si las celdas existen
    if (x1, y1) not in G.nodes or (x2, y2) not in G.nodes:
        return 

    # Agregar la pared al grafo
    G.add_edge((x1, y1), (x2, y2), type='wall', weight=5, life=2)

def add_door(G, x1, y1, x2, y2):
    """
    Agregar una puerta al grafo
    """

    # Verificar si las celdas existen
    if (x1, y1) not in G.nodes or (x2, y2) not in G.nodes:
        return 

    # Agregar la puerta al grafo
    G.add_edge((x1, y1), (x2, y2), type='door', weight=2, is_open=False)

def add_POI(G, x, y, is_victim):
    """
    Agregar un punto de interés al grafo en el nodo especificado
    """

    # Verificar si la celda existe
    if (x, y) not in G.nodes:
        return
    
    # Borrar fuego y humo de la celda
    G.nodes[(x, y)]['fire'] = 0

    # Cambiar peso de las aristas adyacentes a 1
    for neighbor in G.adj[(x, y)]:
        if G.get_edge_data((x, y), neighbor)['type'] == 'path':
            G[(x, y)][neighbor]['weight'] = 1

    # Agregar el punto de interés
    G.nodes[(x, y)]['POI'] = is_victim

def place_POI(G):
    """
    Coloca un punto de interés en un nodo aleatorio del grafo
    """

    # Obtener nodos interiores sin puntos de interés
    interior_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'exterior' and G.nodes[node].get('POI') is None]

    # Seleccionar un nodo aleatorio
    if interior_nodes:
        node = np.random.choice(interior_nodes)
        add_POI(G, node[0], node[1], np.random.choice([True, False]))

def add_fire(G, x, y):
    """
    Agregar fuego a una celda del grafo y cambiar el peso de las aristas adyacentes
    """

    # Verificar si la celda existe
    if (x, y) not in G.nodes:
        return 
    
    # Colocar fuego
    G.nodes[(x, y)]['fire'] = 2

    # Cambia el peso de las aristas adyacentes
    # Path: 2 puntos para apagar el fuego y 1 punto para moverse
    # Wall: 4 para tirar la pared, 2 para apagar el fuego y 1 para moverse
    # Door: Variable (3 o 4 puntos dependiendo de si está abierta o cerrada)
    for neighbor in G.adj[(x, y)]:
        if G.get_edge_data((x, y), neighbor)['type'] == 'path':
            G[(x, y)][neighbor]['weight'] = 3
        elif G.get_edge_data((x, y), neighbor)['type'] == 'wall':
            G[(x, y)][neighbor]['weight'] = 7
        elif G.get_edge_data((x, y), neighbor)['type'] == 'door':
            if is_door_open(G, x, y, neighbor[0], neighbor[1]):
                # Cambiar el peso de la arista a 3 (2 puntos para apagar el fuego y 1 punto para moverse)
                G[(x, y)][neighbor]['weight'] = 3 
            else:
                # Cambiar el peso de la arista a 4 (1 puntos para abrir la puerta, 2 puntos para apagar el fuego y 1 punto para moverse)
                G[(x, y)][neighbor]['weight'] = 4

def add_smoke(G, x, y):
    """
    Agregar humo a una celda del grafo
    """

    # Verificar si la celda existe
    if (x, y) not in G.nodes:
        return 
    
    # Colocar humo
    G.nodes[(x, y)]['fire'] = 1

    # Cambia el peso de las aristas adyacentes
    # Path: 1 puntos para apagar el humo y 1 punto para moverse
    # Wall: 4 para tirar la pared, 1 para apagar el humo y 1 para moverse
    # Door: Variable (2 o 3 puntos dependiendo de si está abierta o cerrada)
    for neighbor in G.adj[(x, y)]:
        if G.get_edge_data((x, y), neighbor)['type'] == 'path':
            G[(x, y)][neighbor]['weight'] = 2
        elif G.get_edge_data((x, y), neighbor)['type'] == 'wall':
            G[(x, y)][neighbor]['weight'] = 6
        elif G.get_edge_data((x, y), neighbor)['type'] == 'door':
            if is_door_open(G, x, y, neighbor[0], neighbor[1]):
            # Cambiar el peso de la arista a 3 (1 puntos para apagar el humo y 1 punto para moverse)
                G[(x, y)][neighbor]['weight'] = 2
            else:
            # Cambiar el peso de la arista a 4 (1 puntos para abrir la puerta, 1 puntos para apagar el humo y 1 punto para moverse)
                G[(x, y)][neighbor]['weight'] = 3

def extinguish(G, x, y):
    """
    Convertir el fuego en humo, y el humo en nada
    """

    # Verificar si la celda existe
    if (x, y) not in G.nodes:
        return 
    
    # Verificar si hay fuego
    if G.nodes[(x, y)]['fire'] == 2:
        G.nodes[(x, y)]['fire'] = 1  # Convertir fuego en humo

        # Cambia el peso de las aristas adyacentes
        # Path: 1 puntos para apagar el humo y 1 punto para moverse
        # Wall: 4 para tirar la pared, 1 para apagar el humo y 1 para moverse
        # Door: Variable (2 o 3 puntos dependiendo de si está abierta o cerrada)
        for neighbor in G.adj[(x, y)]:
            if G.get_edge_data((x, y), neighbor)['type'] == 'path':
                G[(x, y)][neighbor]['weight'] = 2
            elif G.get_edge_data((x, y), neighbor)['type'] == 'wall':
                G[(x, y)][neighbor]['weight'] = 6
            elif G.get_edge_data((x, y), neighbor)['type'] == 'door':
                if is_door_open(G, x, y, neighbor[0], neighbor[1]):
                    # Cambiar el peso de la arista a 2 (1 puntos para apagar el humo y 1 punto para moverse)
                    G[(x, y)][neighbor]['weight'] = 2
                else:
                    # Cambiar el peso de la arista a 3 (1 puntos para abrir la puerta, 1 puntos para apagar el humo y 1 punto para moverse)
                    G[(x, y)][neighbor]['weight'] = 3

    # Verificar si hay humo
    elif G.nodes[(x, y)]['fire'] == 1:
        G.nodes[(x, y)]['fire'] = 0  # Convertir humo en nada

        # Cambia el peso de las aristas adyacentes
        # Path: 1 punto para moverse
        # Wall: 4 para tirar la pared y 1 para moverse
        # Door: Variable (2 o 1 puntos dependiendo de si está abierta o cerrada)
        for neighbor in G.adj[(x, y)]:
            if G.get_edge_data((x, y), neighbor)['type'] == 'path':
                G[(x, y)][neighbor]['weight'] = 1
            elif G.get_edge_data((x, y), neighbor)['type'] == 'wall':
                G[(x, y)][neighbor]['weight'] = 5
            elif G.get_edge_data((x, y), neighbor)['type'] == 'door':
                if is_door_open(G, x, y, neighbor[0], neighbor[1]):
                    # Cambiar el peso de la arista a 1 (1 punto para moverse)
                    G[(x, y)][neighbor]['weight'] = 1
                else:
                    # Cambiar el peso de la arista a 4 (1 puntos para abrir la puerta y 1 punto para moverse)
                    G[(x, y)][neighbor]['weight'] = 2

def ignite_cell(G, x, y):
    """
    Colocar humo en una celda y resolver la propagación del fuego (igniciones y explosiones)
    """

    # Verificar si la celda existe
    if (x, y) not in G.nodes:
        return 
    
    print(f"Ignición en {(x, y)}")
    
    # Verificar si no hay fuego
    if G.nodes[(x, y)]['fire'] == 0:
        G.nodes[(x, y)]['fire'] = 1  # Colocar humo

    # Verificar si hay humo (Ignición)
    elif G.nodes[(x, y)]['fire'] == 1:
        # Convertir el humo en fuego
        add_fire(G, x, y)

        # Obtener los vecinos de la celda
        neighbors = G.adj[(x, y)]

        # Verificar si hay humo en los vecinos
        for neighbor in neighbors:
            # Verificar si hay humo y no hay una pared bloqueante 
            if G.nodes[neighbor]['fire'] == 1 and G.get_edge_data((x, y), neighbor)['type'] != 'wall':
                # Verificar si hay una puerta
                if G.get_edge_data((x, y), neighbor)['type'] == 'door':
                    # Convertir humo en fuego recursivamente
                    ignite_cell(G, neighbor[0], neighbor[1])

                    # Verificar si la puerta está abierta
                    if is_door_open(G, x, y, neighbor[0], neighbor[1]):
                        # Cambiar el peso de la arista a 3 (2 puntos para apagar el fuego y 1 punto para moverse)
                        G[(x, y)][neighbor]['weight'] = 3 
                    else:
                        # Cambiar el peso de la arista a 4 (1 puntos para abrir la puerta, 2 puntos para apagar el fuego y 1 punto para moverse)
                        G[(x, y)][neighbor]['weight'] = 4
                        
                # Si no hay puerta, verificar si hay un camino
                elif G.get_edge_data((x, y), neighbor)['type'] == 'path':
                    ignite_cell(G, neighbor[0], neighbor[1])  # Convertir humo en fuego recursivamente

    # Verificar si hay fuego (Explosión)
    elif G.nodes[(x, y)]['fire'] == 2:
        # Propagar explosión
        propagate_explosion(G, x, y)

def propagate_explosion(G, x, y):
    """
    Expande el fuego en forma de cruz desde una celda con fuego.
    Sigue expandiéndose hasta encontrar una celda vacía, una pared, una puerta cerrada o un borde del tablero.
    """
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Izquierda, Arriba, Derecha, Abajo

    for dx, dy in directions:
        current_x, current_y = x, y

        while True:
            # Calcular la siguiente celda en la dirección
            next_x, next_y = current_x + dx, current_y + dy

            # Verificar si la celda es válida
            if (next_x, next_y) not in G.nodes:
                break

            edge_data = G.get_edge_data((current_x, current_y), (next_x, next_y))
            if not edge_data:  # Sin conexión (fuera del tablero o bloqueado)
                break

            # Si hay una pared
            if edge_data['type'] == 'wall':
                if 'life' not in edge_data:
                    edge_data['life'] = 2  # Inicializar vida de la pared

                # Reducir la vida de la pared
                edge_data['life'] -= 1
                print(f"Pared entre {(current_x, current_y)} y {(next_x, next_y)} dañada (vida restante: {edge_data['life']})")

                # Si la pared se destruye, convertirla en camino y detener propagación
                if edge_data['life'] == 0:
                    edge_data['type'] = 'path'
                    edge_data['weight'] = 1
                    print(f"Pared entre {(current_x, current_y)} y {(next_x, next_y)} destruida")
                break  # Detener propagación en esta dirección

            # Si hay una puerta cerrada
            if edge_data['type'] == 'door' and not is_door_open(G, current_x, current_y, next_x, next_y):
                # La puerta es destruida y convertida en un camino
                edge_data['type'] = 'path'
                edge_data['weight'] = 1
                print(f"Puerta entre {(current_x, current_y)} y {(next_x, next_y)} destruida")
                break  # Detener propagación en esta dirección

            # Si la celda está vacía
            if G.nodes[(next_x, next_y)]['fire'] == 0:
                add_fire(G, next_x, next_y)
                print(f"Fuego propagado a {(next_x, next_y)}")
                break

            # Si ya hay fuego, continuar en la misma dirección
            elif G.nodes[(next_x, next_y)]['fire'] == 2:
                current_x, current_y = next_x, next_y
                continue

def solve_smoke(G):
    """
    Al final de cada turno, todo el humo en contacto con el fuego se convierte en fuego.
    """

    for node in G.nodes():
        if G.nodes[node]['fire'] == 1:
            for neighbor in G.adj[node]:
                # Verificar si hay fuego en los vecinos y no hay una pared bloqueante
                if G.nodes[neighbor]['fire'] == 2 and G.get_edge_data(node, neighbor)['type'] != 'wall':
                    # Verificar si hay una puerta y si está abierta
                    if G.get_edge_data(node, neighbor)['type'] == 'door' and is_door_open(G, node[0], node[1], neighbor[0], neighbor[1]):
                        add_fire(G, node[0], node[1])
                    # Si no hay puerta, verificar si hay un camino
                    elif G.get_edge_data(node, neighbor)['type'] == 'path':
                        add_fire(G, node[0], node[1])

def shortest_path(G, start, end):
    """
    Encontrar el camino más corto entre dos nodos con el algoritmo de Dijkstra
    """

    # Verificar si los nodos existen
    if start not in G.nodes or end not in G.nodes:
        return None

    # Calcular el camino más corto
    path = nx.shortest_path(G, source=start, target=end, weight='weight')

    # Calcula el peso total del camino
    total_weight = 0
    for i in range(len(path) - 1):
        total_weight += G.get_edge_data(path[i], path[i + 1])['weight']

    # Crear un diccionario con la información del camino más corto
    shortest = {}
    shortest['path'] = path
    shortest['total_weight'] = total_weight


    return shortest

def open_door(G, x1, y1, x2, y2):
    """
    Abrir una puerta
    """

    # Verificar si las celdas existen
    if (x1, y1) not in G.nodes or (x2, y2) not in G.nodes:
        return 

    # Verificar si hay una puerta cerrada
    if not is_door_open(G, x1, y1, x2, y2):
        # Abrir la puerta
        G[(x1, y1)][(x2, y2)]['is_open'] = True

        # Cambiar el peso de la arista a 1
        G[(x1, y1)][(x2, y2)]['weight'] = 1
    else:
        print("No hay una puerta cerrada en esa posición")

def close_door(G, x1, y1, x2, y2):
    """
    Cerrar una puerta
    """

    # Verificar si las celdas existen
    if (x1, y1) not in G.nodes or (x2, y2) not in G.nodes:
        return 

    # Verificar si hay una puerta abierta
    if is_door_open(G, x1, y1, x2, y2):
        # Cerrar la puerta
        G[(x1, y1)][(x2, y2)]['is_open'] = False

        # Cambiar el peso de la arista a 2
        G[(x1, y1)][(x2, y2)]['weight'] = 2
    else:
        print("No hay una puerta abierta en esa posición")

def is_door_open(G, x1, y1, x2, y2):
    """
    Verificar si una puerta está abierta
    """

    # Verificar si las celdas existen
    if (x1, y1) not in G.nodes or (x2, y2) not in G.nodes:
        return False

    # Verificar si hay una puerta abierta
    return G.get_edge_data((x1, y1), (x2, y2))['type'] == 'door' and G.get_edge_data((x1, y1), (x2, y2))['is_open']

# ----------------- Simulación -----------------

# Inicializar el tablero
G = initialize_board(board_config)

ignite_cell(G, 3, 5)

ignite_cell(G, 3, 5)

plot_graph(G)
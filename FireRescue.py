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

def initialize_board(board_config):
    """
    De acuerdo a la configuración del tablero, se inicializa el tablero como un grafo
    """

    # Crear un grafo vacío
    G = nx.Graph()

    # Información del tablero (transiciones y muros)
    # "TLDR" -> Top, Left, Down, Right
    board_info = board_config['board']

    # Puertas
    doors = {}
    for door in board_config['doors']:
        # Se resta 1 a las coordenadas para que coincidan con las coordenadas del tablero
        doors[(int(door[0]) - 1, int(door[1]) - 1)] = (int(door[2]) - 1, int(door[3]) - 1)

    # Agregar los nodos al grafo
    for i in range(len(board_info)):
        for j in range(len(board_info[i])):
            G.add_node((i, j), fire=0, POI=None, isEntryPoint=False)

    # Agregar las aristas al grafo
    for i in range(len(board_info)):
        for j in range(len(board_info[i])):

            # String con la información de la celda
            # "TLDR" -> Top, Left, Down, Right
            # Peso 1 -> Libre / Puerta abierta o destruida
            # Peso 2 -> Puerta cerrada
            # Peso 5 -> Pared
            
            # Verificar si hay una transición arriba
            if board_info[i][j][0] == '0':
                add_path(G, i, j, i - 1, j)
            else:
                # Verificar si hay una puerta
                if (i, j) in doors and doors[(i, j)] == (i - 1, j):
                    # Agregar puerta
                    add_door(G, i, j, i - 1, j)
                # Si no hay puerta, agregar pared
                elif i != 0 and not G.has_edge((i, j), (i - 1, j)):
                    add_wall(G, i, j, i - 1, j)


            # Verificar si hay una transición a la izquierda
            if board_info[i][j][1] == '0':
                add_path(G, i, j, i, j - 1)
            else:
                # Verificar si hay una puerta
                if (i, j) in doors and doors[(i, j)] == (i, j - 1):
                    # Agregar puerta
                    add_door(G, i, j, i, j - 1)
                # Si no hay puerta, agregar pared
                elif j != 0 and not G.has_edge((i, j), (i, j - 1)):
                    add_wall(G, i, j, i, j - 1)

            # Verificar si hay una transición abajo
            if board_info[i][j][2] == '0':
                add_path(G, i, j, i + 1, j)
            else:
                # Verificar si hay una puerta
                if (i, j) in doors and doors[(i, j)] == (i + 1, j):
                    # Agregar puerta
                    add_door(G, i, j, i + 1, j)
                # Si no hay puerta, agregar pared
                elif i != len(board_info) - 1 and not G.has_edge((i, j), (i + 1, j)):
                    add_wall(G, i, j, i + 1, j)

            # Verificar si hay una transición a la derecha
            if board_info[i][j][3] == '0':
                add_path(G, i, j, i, j + 1)
            else:
                # Verificar si hay una puerta
                if (i, j) in doors and doors[(i, j)] == (i, j + 1):
                    # Agregar puerta
                    add_door(G, i, j, i, j + 1)
                # Si no hay puerta, agregar pared
                elif j != len(board_info[i]) - 1 and not G.has_edge((i, j), (i, j + 1)):
                    add_wall(G, i, j, i, j + 1)

    # Agregar los puntos de interés
    # None -> No es un punto de interés
    # True -> Hay una víctima
    # False -> Es una falsa alarma
    for poi in board_config['points_of_interest']:
        add_POI(G, int(poi[0]) - 1, int(poi[1]) - 1, poi[2] == 'v')

    # Agregar los indicadores de fuego
    # 0 -> No hay fuego
    # 1 -> Hay humo
    # 2 -> Hay fuego
    for fire in board_config['fire_indicators']:
        add_fire(G, int(fire[0]) - 1, int(fire[1]) - 1)

    # Agregar los puntos de entrada
    for entry_point in board_config['entry_points']:
        G.nodes[(int(entry_point[0]) - 1, int(entry_point[1]) - 1)]['isEntryPoint'] = True


    # Retornar el grafo
    return G

# ----------------- Funciones de visualización -----------------

def plot_graph(G, title='Flash Point: Fire Rescue'):
    """
    Graficar el grafo con Plotly
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

        # Guardar anotaciones para peso y tipo
        weight = edge[2].get('weight', '?')
        edge_annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=f'{weight}<br>{edge_type}',  # Mostrar peso y tipo
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

        node_text.append(
            f'Posición: {node}<br>'
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
    G.add_edge((x1, y1), (x2, y2), type='door', weight=2)

def add_POI(G, x, y, is_victim):
    """
    Agregar un punto de interés al grafo
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

def add_fire(G, x, y):
    """
    Agregar fuego a una celda del grafo y cambiar el peso de las aristas adyacentes
    """

    # Verificar si la celda existe
    if (x, y) not in G.nodes:
        return 
    
    # Colocar fuego
    G.nodes[(x, y)]['fire'] = 2

    # Cambia el peso de las aristas adyacentes a la celda a 3
    # Significa que al agente le costará el doble de tiempo pasar por esa arista
    # Moverse en una celda con fuego cuesta 2 puntos de acción
    for neighbor in G.adj[(x, y)]:
        if G.get_edge_data((x, y), neighbor)['type'] == 'path':
            G[(x, y)][neighbor]['weight'] = 3

def add_smoke(G, x, y):
    """
    Agregar humo a una celda del grafo
    """

    # Verificar si la celda existe
    if (x, y) not in G.nodes:
        return 
    
    # Colocar humo
    G.nodes[(x, y)]['fire'] = 1

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

        # Cambiar el peso de las aristas adyacentes a 1
        for neighbor in G.adj[(x, y)]:
            # Verificar si hay un camino (no se cambia el peso de las paredes, puertas, ni fuego)
            if G.get_edge_data((x, y), neighbor)['type'] == 'path':
                G[(x, y)][neighbor]['weight'] = 1

    # Verificar si hay humo
    elif G.nodes[(x, y)]['fire'] == 1:
        G.nodes[(x, y)]['fire'] = 0  # Convertir humo en nada

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
            if edge_data['type'] == 'door':
                # La puerta es destruida y convertida en un camino
                edge_data['type'] = 'path'
                edge_data['weight'] = 1
                print(f"Puerta entre {(current_x, current_y)} y {(next_x, next_y)} destruida")
                break  # Detener propagación en esta dirección

            # Si la celda está vacía
            if G.nodes[(next_x, next_y)]['fire'] == 0:
                G.nodes[(next_x, next_y)]['fire'] = 2  # Colocar fuego
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

# ----------------- Simulación -----------------

# Inicializar el tablero
G = initialize_board(board_config)

print("-- Inicial")
plot_graph(G, title='Tablero inicial')

# Agregar humo 

add_smoke(G, 3, 2)
add_smoke(G, 4, 2)

print("-- Humo")
plot_graph(G, title='Tablero con humo')

# Extinción de fuego

extinguish(G, 3, 2)

print("-- Fuego extinguido en (3, 2)")
plot_graph(G, title='Tablero con fuego extinguido en (3, 2)')   

# Resolver humo

solve_smoke(G)

print("-- Humo resuelto")
plot_graph(G, title='Tablero con humo resuelto')
# TC2008B Modelación de Sistemas Multiagentes con gráficas computacionales
# Python server to interact with Unity via POST
# Sergio Ruiz-Loza, Ph.D. March 2021

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json
from FireRescue_Strategic import read_board_config, initialize_board, BoardModel

"""
Clase Server hereda de BaseHTTPRequestHandler, clase que maneja las 
solicitudes HTTP y genera respuestas HTTP.
"""
class Server(BaseHTTPRequestHandler):
    
    """
    Configura la respuesta HTTP con un código 200 y el tipo de contenido.
    """
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    
    """
    Maneja las solicitudes GET.
    """
    def do_GET(self):
        self._set_response()
        board_config = read_board_config()
        G = initialize_board(board_config)
        model = BoardModel(G, board_config)

        """
        # 1 PASO DE SIMULACION
        model.step()

        # Obtener el estado del tablero
        board_json = model.grid.build_json()

        self.wfile.write(json.dumps(board_json).encode('utf-8'))
        """

        # SIMULACION COMPLETA
        while model.running:
            model.step()

        # Obtener datos de la simulacion
        data = model.datacollector.get_model_vars_dataframe()['Board State'].to_list()

        response = {"boardStates" : data}

        # Convertir a JSON
        board_json = json.dumps(response)
        self.wfile.write(board_json.encode('utf-8'))

    """
    Maneja las solicitudes POST.
    """
    def do_POST(self):
        position = {
            "x" : 1,
            "y" : 2,
            "z" : 3
        }

        self._set_response()
        self.wfile.write(str(position).encode('utf-8'))

"""
Configura el servidor HTTP y lo ejecuta en el puerto 8585.
"""
def run(server_class=HTTPServer, handler_class=Server, port=8585):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info("Starting httpd...\n") # HTTPD is HTTP Daemon!
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:   # CTRL+C stops the server
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")

if __name__ == '__main__':
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()




# TC2008B Modelación de Sistemas Multiagentes con gráficas computacionales
# Python server to interact with Unity via POST
# Sergio Ruiz-Loza, Ph.D. March 2021

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json
from FireRescue_Strategic import read_board_config, initialize_board, BoardModel

class Server(BaseHTTPRequestHandler):
    
    model = None
    board_config = None

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
    
    def do_GET(self):
        self._set_response()
        
        if Server.model is None:
            # Inicializar el modelo y enviar el estado inicial
            Server.board_config = read_board_config()
            G = initialize_board(Server.board_config)
            Server.model = BoardModel(G, Server.board_config)
            data = Server.model.grid.build_json()
            response = {"boardState": data}
        else:
            # Avanzar un paso en la simulación y enviar el nuevo estado
            if Server.model.running:
                Server.model.step()
                data = Server.model.build_json()
                response = {"boardState": data}
            else:
                response = {"boardState": None}

        # Convertir a JSON
        board_json = json.dumps(response)
        self.wfile.write(board_json.encode('utf-8'))

    def do_POST(self):
        self._set_response()
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                     str(self.path), str(self.headers), post_data.decode('utf-8'))

        response = {"message": "POST request received"}
        response_json = json.dumps(response)
        self.wfile.write(response_json.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=Server, port=8585):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info("Starting httpd...\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")

if __name__ == '__main__':
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
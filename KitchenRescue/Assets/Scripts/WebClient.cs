// TC2008B Modelación de Sistemas Multiagentes con gráficas computacionales
// C# client to interact with Python server via GET
// Sergio Ruiz-Loza, Ph.D. March 2021

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

public class WebClient : MonoBehaviour
{
    public string url = "http://localhost:8585";
    public GameObject floorPrefab;
    public GameObject wallPrefab;
    public GameObject doorPrefab;
    public GameObject firePrefab;
    public GameObject smokePrefab;
    public GameObject poiPrefab;

    public GameObject blueAgentPrefab;
    public GameObject greenAgentPrefab;
    public GameObject redAgentPrefab;
    public GameObject orangeAgentPrefab;
    public GameObject yellowAgentPrefab;
    public GameObject lilaAgentPrefab;

    private Dictionary<int, GameObject> agentPrefabs;

    void Start()
    {   
        agentPrefabs = new Dictionary<int, GameObject>
        {
            { 0, blueAgentPrefab },
            { 1, greenAgentPrefab },
            { 2, redAgentPrefab },
            { 3, orangeAgentPrefab },
            { 4, yellowAgentPrefab },
            { 5, lilaAgentPrefab }
        };

        StartCoroutine(GetBoardData());
    }

    IEnumerator GetBoardData()
    {
        using (UnityWebRequest www = UnityWebRequest.Get(url))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.Log(www.error);
            }
            else
            {
                string jsonText = www.downloadHandler.text;
                Debug.Log("JSON recibido: " + jsonText); // Imprimir el JSON recibido
                BoardStateList boardStateList = JsonConvert.DeserializeObject<BoardStateList>(jsonText);
                if (boardStateList != null && boardStateList.boardStates != null && boardStateList.boardStates.Count > 0)
                {
                    BoardState boardState = boardStateList.boardStates[0];
                    if (boardState.board == null)
                    {
                        Debug.LogError("boardState.board es nulo después de la deserialización");
                    }
                    else
                    {
                        Debug.Log("boardState.board no es nulo");
                    }
                    VisualizeBoard(boardState);
                }
                else
                {
                    Debug.LogError("Error al deserializar el JSON o el JSON está vacío.");
                }
            }
        }
    }

    void VisualizeBoard(BoardState boardState)
    {   
        float cellSize = 4f;
        float wallThickness = 0.2f;

        // Crear el piso y las paredes para cada celda
        for (int row = 0; row < boardState.information.rows; row++)
        {
            for (int col = 0; col < boardState.information.cols; col++)
            {
                Vector3 position = new Vector3((col * cellSize), 0, (-row * cellSize));
                GameObject floor = Instantiate(floorPrefab, position, Quaternion.identity);
                floor.transform.localScale = new Vector3(cellSize, floor.transform.localScale.y, cellSize);
                
                Debug.Log($"Celda [{row}, {col}]: {boardState.board[row][col]} at {position}");
                string walls = boardState.board[row][col];
                if (walls[0] == '1') // Arriba
                {
                    Vector3 wallPosition = position + new Vector3(0, 2, cellSize / 2);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 0, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                }
                if (walls[1] == '1') // Izquierda
                {
                    Vector3 wallPosition = position + new Vector3(-cellSize / 2, 2, 0);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 90, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                }
                if (walls[2] == '1') // Abajo
                {
                    Vector3 wallPosition = position + new Vector3(0, 2, -cellSize / 2);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 0, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                }
                if (walls[3] == '1') // Derecha
                {
                    Vector3 wallPosition = position + new Vector3(cellSize / 2, 2, 0);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 90, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                }
            }
        }

        // Agregar puertas
        if (boardState.doors != null)
        {
            foreach (var door in boardState.doors)
            {
                Vector3 fromPosition = new Vector3(door.from[1] * cellSize, 0, -door.from[0] * cellSize);
                Vector3 toPosition = new Vector3(door.to[1] * cellSize, 0, -door.to[0] * cellSize);
                Vector3 doorPosition = (fromPosition + toPosition) / 2;
                
                // Determinar rotación de la puerta
                Quaternion rotation = Quaternion.identity;
                if (door.from[1] == door.to[1]) // Mismo número de columna
                {
                    rotation = Quaternion.Euler(0, 90, 0);
                }

                // Instanciar la puerta en la posición calculada con la rotación adecuada
                GameObject doorObject = Instantiate(doorPrefab, doorPosition, rotation);
                
                Debug.Log($"Puerta de [{door.from[0]}, {door.from[1]}] a [{door.to[0]}, {door.to[1]}], abierta: {door.is_open} en coordenadas {doorPosition}");
            }
        }

        // Agregar agentes
        if (boardState.agents != null)
        {
            foreach (var agent in boardState.agents)
            {
                Vector3 position = new Vector3(agent.position[1] * cellSize, 0f, -agent.position[0] * cellSize);
                
                // Seleccionar el prefab de acuerdo con en el ID del agente
                if (agentPrefabs.TryGetValue(agent.id, out GameObject agentPrefab))
                {
                    GameObject agentObject = Instantiate(agentPrefab, position, Quaternion.identity);
                    Debug.Log($"Agente {agent.id} en posición [{agent.position[0]}, {agent.position[1]}], con coordenadas {position}");
                }
                else
                {
                    Debug.LogError($"No se encontró un prefab para el agente {agent.id}.");
                }
            }
        }

        // Agregar fuego
        if (boardState.fire != null)
        {
            foreach (var fire in boardState.fire)
            {
                Vector3 position = new Vector3(fire[1] * cellSize, 0, -fire[0] * cellSize);
                Instantiate(firePrefab, position, Quaternion.identity);
                
                Debug.Log($"Fuego en posición [{fire[0]}, {fire[1]}], con coordenadas {position}");
            }
        }

        // Agregar humo
        if (boardState.smoke != null)
        {
            foreach (var smoke in boardState.smoke)
            {
                Vector3 position = new Vector3(smoke[1] * cellSize, 0, -smoke[0] * cellSize);
                Instantiate(smokePrefab, position, Quaternion.identity);
                
                Debug.Log($"Humo en posición [{smoke[0]}, {smoke[1]}], con coordenadas {position}");
            }
        }

        // Agregar puntos de interés
        if (boardState.points_of_interest != null)
        {
            foreach (var poi in boardState.points_of_interest)
            {
                Vector3 position = new Vector3(poi.position[1] * cellSize, 1.6f, -poi.position[0] * cellSize);

                Instantiate(poiPrefab, position, Quaternion.identity);
                
                Debug.Log($"Punto de interés en posición [{poi.position[0]}, {poi.position[1]}], tipo: {poi.type}, con coordenadas {position}");
            }
        }
    }
}

[System.Serializable]
public class BoardStateList
{
    public List<BoardState> boardStates;
}

[System.Serializable]
public class BoardState
{
    public Information information;
    public List<List<string>> board;
    public List<Door> doors;
    public List<DamagedWall> damaged_walls;
    public List<Agent> agents;
    public List<List<int>> fire;
    public List<List<int>> smoke;
    public List<PointsOfInterest> points_of_interest;
}

[System.Serializable]
public class Information
{
    public int rows;
    public int cols;
    public int step;
    public int actual_turn;
    public int total_damage;
    public int victims_rescued;
    public int victims_dead;
    public bool? win;
}

[System.Serializable]
public class Door
{
    public List<int> from;
    public List<int> to;
    public bool is_open;
}

[System.Serializable]
public class DamagedWall
{
    public List<int> from;
    public List<int> to;
}

[System.Serializable]
public class Agent
{
    public int id;
    public List<int> position;
}

[System.Serializable]
public class PointsOfInterest
{
    public List<int> position;
    public string type;
}
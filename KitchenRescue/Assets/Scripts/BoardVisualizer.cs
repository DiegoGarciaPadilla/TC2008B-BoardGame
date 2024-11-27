using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BoardVisualizer : MonoBehaviour
{
    public GameObject floorPrefab;
    public GameObject wallPrefab;
    public GameObject doorPrefab;
    public GameObject doorFramePrefab;
    public GameObject shelfPrefab;
    public GameObject firePrefab;
    public GameObject smokePrefab;
    public GameObject poiPrefab;
    public GameObject victimPrefab;
    public GameObject falseAlarmPrefab;

    public AgentVisualizer agentVisualizer;

    private List<GameObject> instantiatedObjects = new List<GameObject>();

    public void VisualizeBoard(BoardState boardState)
    {
        ClearBoard();

        float cellSize = 4f;
        float wallThickness = 0.2f;

        // Crear el piso y las paredes para cada celda
        for (int row = 0; row < boardState.information.rows; row++)
        {
            for (int col = 0; col < boardState.information.cols; col++)
            {
                Vector3 position = new Vector3(col * cellSize, 0, -row * cellSize);
                GameObject floor = Instantiate(floorPrefab, position, Quaternion.identity);
                floor.transform.localScale = new Vector3(cellSize, floor.transform.localScale.y, cellSize);
                instantiatedObjects.Add(floor);
                
                Debug.Log($"Celda [{row}, {col}]: {boardState.board[row][col]} at {position}");
                string walls = boardState.board[row][col];
                if (walls[0] == '1') // Arriba
                {
                    Vector3 wallPosition = position + new Vector3(0, 2, cellSize / 2);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 0, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                    instantiatedObjects.Add(wall);
                }
                if (walls[1] == '1') // Izquierda
                {
                    Vector3 wallPosition = position + new Vector3(-cellSize / 2, 2, 0);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 90, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                    instantiatedObjects.Add(wall);
                }
                if (walls[2] == '1') // Abajo
                {
                    Vector3 wallPosition = position + new Vector3(0, 2, -cellSize / 2);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 0, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                    instantiatedObjects.Add(wall);
                }
                if (walls[3] == '1') // Derecha
                {
                    Vector3 wallPosition = position + new Vector3(cellSize / 2, 2, 0);
                    GameObject wall = Instantiate(wallPrefab, wallPosition, Quaternion.Euler(0, 90, 0));
                    wall.transform.localScale = new Vector3(cellSize, wall.transform.localScale.y, wallThickness);
                    instantiatedObjects.Add(wall);
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
                GameObject doorObject = door.is_open ? Instantiate(doorFramePrefab, doorPosition, rotation) : Instantiate(doorPrefab, doorPosition, rotation);
                instantiatedObjects.Add(doorObject);
                
                Debug.Log($"Puerta de [{door.from[0]}, {door.from[1]}] a [{door.to[0]}, {door.to[1]}], abierta: {door.is_open} en coordenadas {doorPosition}");
            }
        }

        // Agregar paredes dañadas
        if (boardState.damaged_walls != null)
        {
            foreach (var damagedWall in boardState.damaged_walls)
            {
                Vector3 fromPosition = new Vector3(damagedWall.from[1] * cellSize, 0, -damagedWall.from[0] * cellSize);
                Vector3 toPosition = new Vector3(damagedWall.to[1] * cellSize, 0, -damagedWall.to[0] * cellSize);
                Vector3 shelfPosition = (fromPosition + toPosition) / 2;

                // Determinar la rotación de la estantería
                Quaternion rotation = Quaternion.identity;
                if (damagedWall.from[0] == damagedWall.to[0])
                {
                    rotation = Quaternion.Euler(0, 90, 0);
                }

                // Instanciar la estantería en la posición calculada con la rotación adecuada
                GameObject shelfObject = Instantiate(shelfPrefab, shelfPosition, rotation);
                instantiatedObjects.Add(shelfObject);

                Debug.Log($"Pared dañada de [{damagedWall.from[0]}, {damagedWall.from[1]}] a [{damagedWall.to[0]}, {damagedWall.to[1]}] en coordenadas {shelfPosition}");
            }
        }

        // Visualizar agentes
        if (boardState.agents != null)
        {
            agentVisualizer.VisualizeAgents(boardState.agents, cellSize);
        }

        // Agregar fuego
        if (boardState.fire != null)
        {
            foreach (var fire in boardState.fire)
            {
                Vector3 position = new Vector3(fire[1] * cellSize, 0.2f, -fire[0] * cellSize);
                GameObject fireObject = Instantiate(firePrefab, position, Quaternion.identity);
                instantiatedObjects.Add(fireObject);
                
                Debug.Log($"Fuego en posición [{fire[0]}, {fire[1]}], con coordenadas {position}");
            }
        }

        // Agregar humo
        if (boardState.smoke != null)
        {
            foreach (var smoke in boardState.smoke)
            {
                Vector3 position = new Vector3(smoke[1] * cellSize, 0 , -smoke[0] * cellSize);
                GameObject smokeObject = Instantiate(smokePrefab, position, Quaternion.identity);
                instantiatedObjects.Add(smokeObject);
                
                Debug.Log($"Humo en posición [{smoke[0]}, {smoke[1]}], con coordenadas {position}");
            }
        }

        // Agregar puntos de interés
        if (boardState.points_of_interest != null)
        {
            foreach (var poi in boardState.points_of_interest)
            {
                Vector3 position = new Vector3(poi.position[1] * cellSize, 1.6f, -poi.position[0] * cellSize);
                bool agentFound = false;

                // Verificar si hay un agente en la misma celda que el POI
                foreach (var agent in boardState.agents)
                {
                    if (agent.position[0] == poi.position[0] && agent.position[1] == poi.position[1])
                    {
                        GameObject poiPrefabToInstantiate = poi.type == "victim" ? victimPrefab : falseAlarmPrefab;
                        GameObject poiInstance = Instantiate(poiPrefabToInstantiate, position, Quaternion.identity);
                        instantiatedObjects.Add(poiInstance);
                        Debug.Log($"Instanciado {poi.type} en posición [{poi.position[0]}, {poi.position[1]}] con coordenadas {position}");
                        agentFound = true;
                        break;
                    }
                }

                // Si no hay un agente en la posición del POI, instanciar el prefab poiPrefab
                if (!agentFound)
                {
                    GameObject poiObject = Instantiate(poiPrefab, position, Quaternion.identity);
                    instantiatedObjects.Add(poiObject);
                    Debug.Log($"Punto de interés en posición [{poi.position[0]}, {poi.position[1]}], tipo: {poi.type}, con coordenadas {position}");
                }
            }
        }
    }

    private void ClearBoard()
    {
        foreach (GameObject obj in instantiatedObjects)
        {
            Destroy(obj);
        }
        instantiatedObjects.Clear();
        agentVisualizer.ClearAgents();
    }
}
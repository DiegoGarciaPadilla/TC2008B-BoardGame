using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentVisualizer : MonoBehaviour
{
    public GameObject blueAgentPrefab;
    public GameObject greenAgentPrefab;
    public GameObject redAgentPrefab;
    public GameObject orangeAgentPrefab;
    public GameObject yellowAgentPrefab;
    public GameObject lilaAgentPrefab;
    public GameObject victimPrefab;

    public AgentRotationHandler rotationHandler; // Referencia al script AgentRotationHandler

    private Dictionary<int, GameObject> agentPrefabs;
    private List<GameObject> instantiatedObjects = new List<GameObject>();

    void Start()
    {
        // Inicializar el diccionario de prefabs
        agentPrefabs = new Dictionary<int, GameObject>
        {
            { 0, blueAgentPrefab },
            { 1, greenAgentPrefab },
            { 2, redAgentPrefab },
            { 3, orangeAgentPrefab },
            { 4, yellowAgentPrefab },
            { 5, lilaAgentPrefab }
        };
    }

    public void VisualizeAgents(List<Agent> agents, float cellSize)
    {
        Dictionary<Vector2Int, List<Agent>> cellAgents = new Dictionary<Vector2Int, List<Agent>>();

        // Agrupar agentes por celda
        foreach (var agent in agents)
        {
            Vector2Int cell = new Vector2Int(agent.position[0], agent.position[1]);
            if (!cellAgents.ContainsKey(cell))
            {
                cellAgents[cell] = new List<Agent>();
            }
            cellAgents[cell].Add(agent);
        }

        // Visualizar agentes
        foreach (var cell in cellAgents)
        {
            Vector3 basePosition = new Vector3(cell.Key.y * cellSize, 0.2f, -cell.Key.x * cellSize);
            int agentCount = cell.Value.Count;
            float offset = 0.4f; // Desplazamiento para separar agentes

            for (int i = 0; i < agentCount; i++)
            {
                var agent = cell.Value[i];
                Vector3 position = basePosition + new Vector3((i % 2) * offset, 0, (i / 2) * offset);

                // Seleccionar el prefab de acuerdo con en el ID del agente
                if (agentPrefabs.TryGetValue(agent.id, out GameObject agentPrefab))
                {
                    // Obtener la rotación del agente
                    Quaternion rotation = rotationHandler.GetAgentRotation(agent.id, position);

                    GameObject agentObject = Instantiate(agentPrefab, position, rotation);
                    instantiatedObjects.Add(agentObject);
                    Debug.Log($"Agente {agent.id} en posición [{agent.position[0]}, {agent.position[1]}], con coordenadas {position}");

                    // Si el agente está cargando una víctima, instanciar también el victimPrefab
                    if (agent.carrying_victim)
                    {
                        GameObject victimObject = Instantiate(victimPrefab, position, Quaternion.identity);
                        instantiatedObjects.Add(victimObject);
                        Debug.Log($"Agente {agent.id} está cargando una víctima en posición [{agent.position[0]}, {agent.position[1]}] con coordenadas {position}");
                    }
                }
                else
                {
                    Debug.LogError($"No se encontró un prefab para el agente {agent.id}.");
                }
            }
        }
    }

    public void ClearAgents()
    {
        foreach (GameObject obj in instantiatedObjects)
        {
            Destroy(obj);
        }
        instantiatedObjects.Clear();
    }
}
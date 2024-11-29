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

    public AgentRotationHandler rotationHandler;

    private Dictionary<int, GameObject> agentPrefabs;
    private Dictionary<int, AgentInstance> agentsDictionary = new Dictionary<int, AgentInstance>();
    private float moveSpeed = 2.5f;

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
        // Agrupar agentes por posición
        var groupedAgents = new Dictionary<Vector2Int, List<Agent>>();
        foreach (var agent in agents)
        {
            Vector2Int pos = new Vector2Int(agent.position[0], agent.position[1]);
            if (!groupedAgents.ContainsKey(pos))
            {
                groupedAgents[pos] = new List<Agent>();
            }
            groupedAgents[pos].Add(agent);
        }

        foreach (var group in groupedAgents)
        {
            Vector3 basePosition = new Vector3(group.Key.y * cellSize, 0.2f, -group.Key.x * cellSize);
            int count = group.Value.Count;

            for (int i = 0; i < count; i++)
            {
                Agent agent = group.Value[i];
                Vector3 offset = CalculateOffset(i, count, cellSize);
                Vector3 finalPosition = basePosition + offset;

                if (agentsDictionary.TryGetValue(agent.id, out AgentInstance agentInstance))
                {
                    agentInstance.SetTargetPosition(finalPosition);
                    Quaternion rotation = rotationHandler.GetAgentRotation(agent.id, finalPosition);
                    agentInstance.SetRotation(rotation);

                    if (agent.carrying_victim && agentInstance.victimObject == null)
                    {
                        GameObject victimObject = Instantiate(victimPrefab, agentInstance.gameObject.transform);
                        agentInstance.victimObject = victimObject;
                    }
                    else if (!agent.carrying_victim && agentInstance.victimObject != null)
                    {
                        Destroy(agentInstance.victimObject);
                    }
                }
                else
                {
                    if (agentPrefabs.TryGetValue(agent.id, out GameObject agentPrefab))
                    {
                        Quaternion rotation = rotationHandler.GetAgentRotation(agent.id, finalPosition);
                        GameObject agentObject = Instantiate(agentPrefab, finalPosition, rotation);
                        AgentInstance newAgentInstance = new AgentInstance(agentObject);
                        agentsDictionary.Add(agent.id, newAgentInstance);
                        Debug.Log($"Instanciado nuevo agente {agent.id} en posición {group.Key}");
                    }
                    else
                    {
                        Debug.LogError($"No se encontró un prefab para el agente {agent.id}.");
                    }
                }
            }
        }

        // Eliminar agentes que ya no existen
        List<int> agentsToRemove = new List<int>();
        foreach (var existingAgentId in agentsDictionary.Keys)
        {
            if (!agents.Exists(a => a.id == existingAgentId))
            {
                Destroy(agentsDictionary[existingAgentId].gameObject);
                if (agentsDictionary[existingAgentId].victimObject != null)
                {
                    Destroy(agentsDictionary[existingAgentId].victimObject);
                }
                agentsToRemove.Add(existingAgentId);
            }
        }

        foreach (int id in agentsToRemove)
        {
            agentsDictionary.Remove(id);
        }
    }

    private Vector3 CalculateOffset(int index, int total, float cellSize)
    {
        float angle = index * Mathf.PI * 2 / total;
        float radius = cellSize * 0.1f;
        return new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle)) * radius;
    }

    public void ClearAgents()
    {
        foreach (var agentInstance in agentsDictionary.Values)
        {
            Destroy(agentInstance.gameObject);
            if (agentInstance.victimObject != null)
            {
                Destroy(agentInstance.victimObject);
            }
        }
        agentsDictionary.Clear();
    }

    private class AgentInstance
    {
        public GameObject gameObject;
        public GameObject victimObject;
        private Vector3 targetPosition;
        private Coroutine movementCoroutine;

        public AgentInstance(GameObject obj)
        {
            gameObject = obj;
            targetPosition = obj.transform.position;
        }

        public void SetTargetPosition(Vector3 position)
        {
            targetPosition = position;
            if (movementCoroutine != null)
            {
                AgentVisualizer.Instance.StopCoroutine(movementCoroutine);
            }
            movementCoroutine = AgentVisualizer.Instance.StartCoroutine(MoveToPosition(position));
        }

        public void SetRotation(Quaternion rotation)
        {
            gameObject.transform.rotation = rotation;
        }

        private IEnumerator MoveToPosition(Vector3 position)
        {
            while (Vector3.Distance(gameObject.transform.position, position) > 0.01f)
            {
                gameObject.transform.position = Vector3.MoveTowards(gameObject.transform.position, position, AgentVisualizer.Instance.moveSpeed * Time.deltaTime);
                yield return null;
            }
            gameObject.transform.position = position;
        }
    }

    private static AgentVisualizer _instance;
    public static AgentVisualizer Instance
    {
        get
        {
            if (_instance == null)
            {
                _instance = FindFirstObjectByType<AgentVisualizer>();
            }
            return _instance;
        }
    }
}
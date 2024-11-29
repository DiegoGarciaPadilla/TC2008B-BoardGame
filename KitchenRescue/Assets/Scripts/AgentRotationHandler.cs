using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentRotationHandler : MonoBehaviour
{
    private Dictionary<int, Vector3> previousAgentPositions = new Dictionary<int, Vector3>();

    public Quaternion GetAgentRotation(int agentId, Vector3 currentPosition)
    {
        Vector3 direction = Vector3.zero;
        if (previousAgentPositions.ContainsKey(agentId))
        {
            direction = currentPosition - previousAgentPositions[agentId];
        }
        previousAgentPositions[agentId] = currentPosition;

        if (direction != Vector3.zero)
        {
            return Quaternion.LookRotation(direction);
        }
        return Quaternion.identity;
    }
}
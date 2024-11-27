// WebClient.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

public class WebClient : MonoBehaviour
{
    public string url = "http://localhost:8585";
    public BoardVisualizer boardVisualizer;
    public InfoTextUpdater infoTextUpdater;

    void Start()
    {
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
                    BoardState boardState = boardStateList.boardStates[5];
                    if (boardState.board == null)
                    {
                        Debug.LogError("boardState.board es nulo después de la deserialización");
                    }
                    else
                    {
                        Debug.Log("boardState.board no es nulo");
                        boardVisualizer.VisualizeBoard(boardState);
                        infoTextUpdater.UpdateInfoText(boardState.information);
                    }

                    // Iniciar la animación de los estados del tablero
                    StartCoroutine(AnimateBoardStates(boardStateList.boardStates, 6));
                }
                else
                {
                    Debug.LogError("Error al deserializar el JSON o el JSON está vacío.");
                }
            }
        }
    }

    IEnumerator AnimateBoardStates(List<BoardState> boardStates, int startIndex)
    {
        for (int i = startIndex; i < boardStates.Count; i++)
        {
            BoardState boardState = boardStates[i];
            if (boardState.board == null)
            {
                Debug.LogError("boardState.board es nulo después de la deserialización");
            }
            else
            {
                Debug.Log("boardState.board no es nulo");
                boardVisualizer.VisualizeBoard(boardState);
                infoTextUpdater.UpdateInfoText(boardState.information);
            }

            // Esperar un segundo antes de pasar al siguiente estado
            yield return new WaitForSeconds(1f);
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
    public bool carrying_victim;
}

[System.Serializable]
public class PointsOfInterest
{
    public List<int> position;
    public string type;
}
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
    private bool gameEnded = false;

    void Start()
    {
        StartCoroutine(GetBoardData());
    }
    IEnumerator GetBoardData()
    {
        while (!gameEnded)
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
                    Debug.Log("JSON recibido: " + jsonText);
                    BoardStateResponse response = JsonConvert.DeserializeObject<BoardStateResponse>(jsonText);
                    if (response != null && response.boardState != null)
                    {
                        BoardState boardState = response.boardState;
                        if (boardState.board == null)
                        {
                            Debug.LogError("boardState.board es nulo después de la deserialización");
                        }
                        else
                        {
                            Debug.Log("boardState.board no es nulo");
                            boardVisualizer.VisualizeBoard(boardState);
                            infoTextUpdater.UpdateInfoText(boardState.information);

                            // Verificar si el juego ha terminado
                            if (boardState.information.win.HasValue)
                            {
                                gameEnded = true;
                                Debug.Log("El juego ha terminado.");
                            }
                        }
                    }
                    else
                    {
                        Debug.LogError("Error al deserializar el JSON o el JSON está vacío.");
                    }
                }
            }

            // Esperar antes de la siguiente petición
            yield return new WaitForSeconds(4f);
        }
    }
}

[System.Serializable]
public class BoardStateResponse
{
    public BoardState boardState;
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
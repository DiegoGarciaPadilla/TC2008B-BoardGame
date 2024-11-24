// TC2008B Modelación de Sistemas Multiagentes con gráficas computacionales
// C# client to interact with Python server via POST
// Sergio Ruiz-Loza, Ph.D. March 2021

using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

public class WebClient : MonoBehaviour
{
    /*
    // IEnumerator - yield return
    Envía datos a servidor Python mediante solicitud POST.
    Los datos se envían como JSON y se espera una respuesta JSON que se convierte en un Vector3.
    */
    IEnumerator SendData(string data)
    {
        WWWForm form = new WWWForm();
        form.AddField("bundle", "the data");
        string url = "http://localhost:8585";
        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(data);
            www.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
            //www.SetRequestHeader("Content-Type", "text/html");
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();          // Talk to Python
            if(www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.Log(www.error);
            }
            else
            {
                Debug.Log(www.downloadHandler.text);    // Answer from Python
                Vector3 tPos = JsonUtility.FromJson<Vector3>(www.downloadHandler.text.Replace('\'', '\"'));
                //Debug.Log("Form upload complete!");
                Debug.Log(tPos);
            }
        }

    }

    /*
    Crea un objeto Vector3 y lo convierte a JSON para enviarlo al servidor Python.
    // Start is called before the first frame update
    */
    void Start()
    {
        //string call = "What's up?";
        Vector3 fakePos = new Vector3(3.44f, 0, -15.707f);
        string json = EditorJsonUtility.ToJson(fakePos);
        //StartCoroutine(SendData(call));
        StartCoroutine(SendData(json));
        // transform.localPosition
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

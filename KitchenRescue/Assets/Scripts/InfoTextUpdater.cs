using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.Video;

public class InfoTextUpdater : MonoBehaviour
{
    public TextMeshProUGUI infoText;
    public VideoPlayer failVideo;
    public VideoPlayer winVideo;

    public void UpdateInfoText(Information info)
    {
        if (infoText != null)
        {
            infoText.text = $"Ronda actual: {info.step}\n" +
                            $"Reseñas negativas: {info.total_damage} \n" +
                            $"Platillos rescatados: {info.victims_rescued}\n" +
                            $"Platillos infectados: {info.victims_dead}";
            Debug.Log($"Turno actual: {info.step}, Reseñas negativas: {info.total_damage}, Platillos rescatados: {info.victims_rescued}, Platillos infectados: {info.victims_dead}");
        }

        failVideo.gameObject.SetActive(false);
        winVideo.gameObject.SetActive(false);

        if (info.win.HasValue)
        {
            if (info.win.Value == false)
            {
                Debug.Log("Perdiste");
                failVideo.gameObject.SetActive(true);
                failVideo.Play();
            }
            else if (info.win.Value == true)
            {
                Debug.Log("Ganaste");
                winVideo.gameObject.SetActive(true);
                winVideo.Play();
            }
        }
    }
}
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class InfoTextUpdater : MonoBehaviour
{
    public TextMeshProUGUI infoText;

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
    }
}
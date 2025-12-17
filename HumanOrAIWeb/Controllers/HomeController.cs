using Microsoft.AspNetCore.Mvc;
using HumanOrAIWeb.Services;
using System.Text.Json;
using System.Globalization;


namespace HumanOrAIWeb.Controllers
{
    public class HomeController : Controller
    {
        private readonly PredictionService _predict = new PredictionService();

        public IActionResult Index()
        {
            if (TempData["History"] != null)
                TempData.Keep("History");

            return View();
        }

        [HttpPost]
        public IActionResult Predict(string inputText)
        {
            if (string.IsNullOrWhiteSpace(inputText))
            {
                TempData["Error"] = "Lütfen bir metin giriniz.";
                return RedirectToAction("Index");
            }

            var result = _predict.PredictAsync(inputText).Result;

            float L = result["logistic"] / 10;
            float S = result["svm"] / 10;
            float R = result["rf"]/10 ;


            TempData["L"] = L.ToString("F1", CultureInfo.InvariantCulture);
            TempData["S"] = S.ToString("F1", CultureInfo.InvariantCulture);
            TempData["R"] = R.ToString("F1", CultureInfo.InvariantCulture);


            // Özet için
            string summary = inputText.Length > 80 ? inputText[..80] + "..." : inputText;

            float avg = (L + S + R) / 3;

            string historyItem =
                $"{DateTime.Now:HH:mm} | %{avg:F1} AI | {summary} | {L:F1} | {S:F1} | {R:F1}";

            List<string> history = TempData.ContainsKey("History")
                ? JsonSerializer.Deserialize<List<string>>(TempData["History"].ToString())
                : new List<string>();

            history.Add(historyItem);

            TempData["History"] = JsonSerializer.Serialize(history);
            TempData.Keep("History");

            return RedirectToAction("Index");
        }

        [HttpPost]
        public IActionResult DeleteHistory()
        {
            TempData.Remove("History");
            return RedirectToAction("Index");
        }
    }
}

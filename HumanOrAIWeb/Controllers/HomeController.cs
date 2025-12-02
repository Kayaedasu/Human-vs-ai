using Microsoft.AspNetCore.Mvc;
using HumanOrAIWeb.Services;
using System.Text.Json;

namespace HumanOrAIWeb.Controllers
{
    public class HomeController : Controller
    {
        private readonly LogisticService _logistic;
        private readonly SVMService _svm;
        private readonly RandomForestService _rf;

        public HomeController()
        {
            _logistic = new LogisticService();
            _svm = new SVMService();
            _rf = new RandomForestService();
        }

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

            // --- Tahminler ---
            float l = _logistic.Predict(inputText) * 100;
            float s = _svm.Predict(inputText) * 100;
            float r = _rf.Predict(inputText) * 100;

            TempData["L"] = l.ToString("F1");
            TempData["S"] = s.ToString("F1");
            TempData["R"] = r.ToString("F1");

            // --- Özet ---
            string summary = inputText.Length > 80
                ? inputText[..80] + "..."
                : inputText;

            float avg = (l + s + r) / 3;

            string historyItem =
                $"{DateTime.Now:HH:mm} | %{avg:F1} AI | {summary} | {l:F1} | {s:F1} | {r:F1}";

            // --- Geçmiþ listesi ---
            List<string> historyList = new();

            if (TempData.ContainsKey("History"))
                historyList = JsonSerializer.Deserialize<List<string>>(TempData["History"].ToString());

            historyList.Add(historyItem);

            TempData["History"] = JsonSerializer.Serialize(historyList);
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

using System.Text;
using System.Text.Json;

namespace HumanOrAIWeb.Services
{
    public class PredictionService
    {
        private readonly HttpClient _client = new HttpClient();

        public async Task<Dictionary<string, float>> PredictAsync(string text)
        {
            var url = "http://127.0.0.1:5000/predict";

            // JSON body oluştur
            var body = new { text = text };
            string json = JsonSerializer.Serialize(body);

            var content = new StringContent(json, Encoding.UTF8, "application/json");

            // POST isteği gönder
            var response = await _client.PostAsync(url, content);
            var jsonString = await response.Content.ReadAsStringAsync();

            var result = JsonSerializer.Deserialize<PredictionResponse>(jsonString);

            return new Dictionary<string, float>
            {
                { "logistic", result.logistic_regression.ai * 100 },
                { "svm", result.svm.ai * 100 },
                { "rf", result.random_forest.ai * 100 }
            };
        }
    }

    public class PredictionResponse
    {
        public Logistic logistic_regression { get; set; }
        public SVM svm { get; set; }
        public RandomForest random_forest { get; set; }
    }

    public class Logistic { public float ai { get; set; } public float human { get; set; } }
    public class SVM { public float ai { get; set; } public float human { get; set; } }
    public class RandomForest { public float ai { get; set; } public float human { get; set; } }
}

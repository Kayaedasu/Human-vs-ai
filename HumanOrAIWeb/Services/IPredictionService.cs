namespace HumanOrAIWeb.Services
{
    public interface IPredictionService
    {
        float[] Predict(string text);
    }
}

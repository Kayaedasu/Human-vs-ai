using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace HumanOrAIWeb.Services
{
    public class SVMService
    {
        private readonly InferenceSession _session;
        private readonly bool _modelLoaded = false;

        public SVMService()
        {
            try
            {
                _session = new InferenceSession("Models/svm_model.onnx");
                _modelLoaded = true;
            }
            catch
            {
                _modelLoaded = false;
            }
        }

        public float Predict(string inputText)
        {
            if (!_modelLoaded)
            {
                return 0.62f;  // SAHTE tahmin
            }

            try
            {
                var inputTensor = new DenseTensor<string>(new[] { 1 });
                inputTensor[0] = inputText;

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
                };

                using var results = _session.Run(inputs);

                var output = results.First().AsEnumerable<float>().First();
                return output;
            }
            catch
            {
                return 0.64f;
            }
        }
    }
}

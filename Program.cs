using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

namespace GPT2OnnxApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Path to the ONNX model
            var modelPath = "D:\\AI\\ML_Processing\\GPT2Export\\distilgpt2\\distilgpt2.onnx"; // Update if needed

            using var session = new InferenceSession(modelPath);

            var inputText = "Hello world";
            var tokenizer = new GPT2Tokenizer();
            var inputIds = tokenizer.Encode(inputText).ToArray();

            var inputIdsLong = inputIds.Select(i => (long)i).ToArray();

            var tensor = new DenseTensor<long>(inputIdsLong, new int[] { 1, inputIdsLong.Length });

            var inputs = new NamedOnnxValue[] {
            NamedOnnxValue.CreateFromTensor("input_ids", tensor)
        };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            var outputTensor = results.First().AsTensor<float>();

            Console.WriteLine("Tensor Shape: [" + string.Join(',', outputTensor.Dimensions.ToArray()) + "]");

            Console.WriteLine("First 10 output values:");
            int count = 0;
            foreach (var val in outputTensor)
            {
                Console.Write($"{val:F4} ");
                if (++count >= 10) break;
            }

            Console.WriteLine("\n\nDone.");
            Console.ReadLine();
        }
    }

    class GPT2Tokenizer
    {
        public int[] Encode(string text)
        {
            return text.Select(c => (int)c).ToArray();
        }
    }
}

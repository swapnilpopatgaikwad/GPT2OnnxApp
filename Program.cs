using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using System.Text.Json;

namespace GPT2OnnxApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // === Paths ===
            var modelPath = "D:\\AI\\ML_Processing\\GPT2Export\\distilgpt2\\distilgpt2.onnx";
            var vocabPath = "D:\\AI\\ML_Processing\\GPT2Export\\distilgpt2\\vocab.json";
            var mergesPath = "D:\\AI\\ML_Processing\\GPT2Export\\distilgpt2\\merges.txt";

            // === Load model and vocab ===
            using var session = new InferenceSession(modelPath);
            var vocab = new GPT2Vocab(vocabPath);

            // === Input text ===
                var inputText = "Hi, I have";
            var tokenizer = new GPT2Tokenizer();
            var inputIds = tokenizer.Encode(inputText);

            // === Create tensor (convert Int32 to Int64 as required) ===
            var inputIdsLong = inputIds.Select(id => (long)id).ToArray();
            var tensor = new DenseTensor<long>(inputIdsLong, new int[] { 1, inputIds.Length });

            // === Create ONNX input ===
            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", tensor)
        };

            // === Run inference ===
            var results = session.Run(inputs);
            foreach (var result in results)
            {
                Console.WriteLine($"Output Name: {result.Name}");
            }
            var logits = results.First(r => r.Name == "last_hidden_state").AsTensor<float>();

            // === Get logits for last token position ===
            int vocabSize = logits.Dimensions[2];
            int lastTokenIndex = inputIds.Length - 1;

            float[] lastTokenLogits = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++)
            {
                lastTokenLogits[i] = logits[0, lastTokenIndex, i];
            }

            // === Get top 5 token predictions ===
            var topIds = GetTopIndices(lastTokenLogits, 5);
            Console.WriteLine("Top predicted next tokens:");
            foreach (var id in topIds)
            {
                Console.WriteLine($"Token ID: {id}, Token: {vocab.Decode(id)}");
            }

            Console.WriteLine("\nDone.");
            Console.ReadLine();
        }

        // === Get top N indices from logits ===
        static int[] GetTopIndices(float[] array, int topN)
        {
            return array
                .Select((val, idx) => new { val, idx })
                .OrderByDescending(x => x.val)
                .Take(topN)
                .Select(x => x.idx)
                .ToArray();
        }
    }

    // === Simple byte-level GPT2 tokenizer ===
    // Replace this with HuggingFace tokenizer logic for production
    class GPT2Tokenizer
    {
        public int[] Encode(string text)
        {
            // Simple: just use ASCII bytes for demo (not real GPT2 encoding)
            return text.Select(c => (int)c).ToArray();
        }
    }

    // === GPT2 Vocabulary loader ===
    class GPT2Vocab
    {
        private readonly Dictionary<int, string> _vocab;

        public GPT2Vocab(string vocabPath)
        {
            string json = File.ReadAllText(vocabPath);
            var raw = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(json);
            _vocab = raw.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        }

        public string Decode(int id) => _vocab.ContainsKey(id) ? _vocab[id] : "[UNK]";
    }
}

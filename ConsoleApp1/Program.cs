using System;
using Microsoft.ML;
using Microsoft.ML.Data;

// Define your data structure
public class DishData
{
    [LoadColumn(0)] public string DishName;
    [LoadColumn(1)] public float PastThreeMonthsSales;
    [LoadColumn(2)] public float NextThreeMonthsSales;
}

public class DishPrediction
{
    [ColumnName("Score")]
    public float NextThreeMonthsSales;
}

enum Command
{
    Train = 1,
    Predict = 2
}

class Program
{
    private static MLContext context = new();
    private static string modelPath = "./model.zip";

    static void Main(string[] args)
    {
        args = new string[] { "2" };
        if (args.Length > 0 && Enum.TryParse(typeof(Command), args[0], out var command))
        {
            switch ((Command)command)
            {
                case Command.Train:
                    TrainModel();
                    Console.WriteLine("Model training completed.");
                    break;
                case Command.Predict:
                    var dishData = new DishData { DishName = "Tacos", PastThreeMonthsSales = float.Parse("75") };
                    var prediction = PredictSales(dishData);
                    Console.WriteLine($"Predicted sales for next three months: {prediction.NextThreeMonthsSales}");
                    Console.ReadLine();
                    break;
                default:
                    Console.WriteLine("Invalid command. Please use '1' to train the model or '2' to make a prediction.");
                    break;
            }
        }
        else
        {
            Console.WriteLine("Please provide a command. Use '1' to train the model or '2' to make a prediction.");
        }
    }

    static void TrainModel()
    {
        // Load your data
        IDataView data = context.Data.LoadFromTextFile<DishData>("./data.csv", separatorChar: ',', hasHeader: true, allowQuoting: true, allowSparse: false, trimWhitespace: true);

        // Print the loaded data
        var preview = data.Preview(maxRows: 10);
        Console.WriteLine("Loaded data preview:");
        Console.WriteLine(preview.ToString());

        // Split your data into a training and test set
        var tt = context.Data.TrainTestSplit(data);

        // Define the pipeline
        var pipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "NextThreeMonthsSales")
            .Append(context.Transforms.Categorical.OneHotEncoding(outputColumnName: "EncodedDishName", inputColumnName: "DishName"))
            .Append(context.Transforms.NormalizeMinMax(outputColumnName: "NormalizedPastThreeMonthsSales", inputColumnName: "PastThreeMonthsSales"))
            .Append(context.Transforms.Concatenate("Features", "NormalizedPastThreeMonthsSales", "EncodedDishName"))
            .Append(context.Regression.Trainers.Sdca());

        // Train the model
        var model = pipeline.Fit(tt.TrainSet);

        // Measure error on the test set
        var predictions = model.Transform(tt.TestSet);
        var metrics = context.Regression.Evaluate(predictions);
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

        // Save the model
        context.Model.Save(model, tt.TrainSet.Schema, modelPath);
        Console.WriteLine();
    }


    static DishPrediction PredictSales(DishData dishData)
    {
        // Load the model
        var model = context.Model.Load(modelPath, out var modelSchema);

        // Make a prediction
        var predictionEngine = context.Model.CreatePredictionEngine<DishData, DishPrediction>(model, modelSchema);
        var prediction = predictionEngine.Predict(dishData);

        return prediction;
    }
}

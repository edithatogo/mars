using System.Text.Json;
using System.Text.Json.Serialization;
using Xunit;

namespace MarsRuntime.Tests;

public sealed class RuntimeFixtureTests
{
    [Fact]
    public void MatchesCheckedInRuntimePortabilityFixtures()
    {
        var fixturesDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../../tests/fixtures"));
        var modelSpecs = Directory.GetFiles(fixturesDir, "model_spec_*.json").OrderBy(path => path).ToArray();

        Assert.NotEmpty(modelSpecs);

        foreach (var modelSpecPath in modelSpecs)
        {
            var suffix = Path.GetFileNameWithoutExtension(modelSpecPath)["model_spec_".Length..];
            var spec = Runtime.LoadModelSpec(File.ReadAllText(modelSpecPath));
            var fixturePath = Path.Combine(fixturesDir, $"runtime_portability_fixture_{suffix}.json");
            var fixture = JsonSerializer.Deserialize<RuntimeFixture>(File.ReadAllText(fixturePath), JsonOptions)
                ?? throw new InvalidOperationException($"Could not load fixture {fixturePath}");

            var probe = NormalizeMatrix(fixture.Probe);
            AssertMatrixClose(Runtime.DesignMatrix(spec, probe), NormalizeMatrix(fixture.DesignMatrix));
            AssertVectorClose(Runtime.Predict(spec, probe), NormalizeVector(fixture.Predict));
        }
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private sealed record RuntimeFixture(
        [property: JsonPropertyName("probe")] double?[][] Probe,
        [property: JsonPropertyName("design_matrix")] double?[][] DesignMatrix,
        [property: JsonPropertyName("predict")] double?[] Predict);

    private static double[][] NormalizeMatrix(double?[][] rows)
    {
        return rows.Select(NormalizeVector).ToArray();
    }

    private static double[] NormalizeVector(double?[] values)
    {
        return values.Select(value => value ?? double.NaN).ToArray();
    }

    private static void AssertMatrixClose(double[][] actual, double[][] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var row = 0; row < expected.Length; row++)
        {
            AssertVectorClose(actual[row], expected[row]);
        }
    }

    private static void AssertVectorClose(double[] actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            if (double.IsNaN(expected[index]))
            {
                Assert.True(double.IsNaN(actual[index]));
            }
            else
            {
                Assert.True(Math.Abs(actual[index] - expected[index]) <= 1e-12, $"{actual[index]} != {expected[index]} at {index}");
            }
        }
    }
}

public sealed class RuntimeTrainingTests
{
    [Fact]
    public void FitsModelThroughRustAndReplaysPredictions()
    {
        var request = new TrainingRequest(
            X: new[]
            {
                new[] { 0.0 },
                new[] { 1.0 },
                new[] { 2.0 },
            },
            Y: new[] { 1.0, 3.0, 5.0 },
            SampleWeight: null,
            Params: new TrainingParams(
                MaxTerms: 5,
                MaxDegree: 1,
                Penalty: 3.0,
                Minspan: 0.0,
                Endspan: 0.0,
                Threshold: 0.001,
                AllowLinear: true,
                AllowMissing: false));

        var spec = Runtime.FitModel(request);

        Assert.NotNull(spec);
        Assert.NotEmpty(spec.BasisTerms);
        Assert.NotEmpty(spec.Coefficients);

        var predictions = Runtime.Predict(spec, request.X);
        Assert.Equal(request.Y.Length, predictions.Length);
        AssertVectorClose(predictions, request.Y);
    }

    private static void AssertVectorClose(double[] actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
        {
            if (double.IsNaN(expected[index]))
            {
                Assert.True(double.IsNaN(actual[index]));
            }
            else
            {
                Assert.True(Math.Abs(actual[index] - expected[index]) <= 1e-12, $"{actual[index]} != {expected[index]} at {index}");
            }
        }
    }
}

using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MarsRuntime;

public sealed record FeatureSchema(
    [property: JsonPropertyName("n_features")] int? NFeatures,
    [property: JsonPropertyName("feature_names")] string[]? FeatureNames);

public sealed record BasisTerm(
    [property: JsonPropertyName("kind")] string Kind,
    [property: JsonPropertyName("variable_idx")] int? VariableIdx,
    [property: JsonPropertyName("knot_val")] double? KnotVal,
    [property: JsonPropertyName("is_right_hinge")] bool? IsRightHinge,
    [property: JsonPropertyName("category")] JsonElement? Category,
    [property: JsonPropertyName("parent1")] BasisTerm? Parent1,
    [property: JsonPropertyName("parent2")] BasisTerm? Parent2);

public sealed record ModelSpec(
    [property: JsonPropertyName("spec_version")] string SpecVersion,
    [property: JsonPropertyName("params")] JsonElement? Params,
    [property: JsonPropertyName("feature_schema")] FeatureSchema FeatureSchema,
    [property: JsonPropertyName("basis_terms")] BasisTerm[] BasisTerms,
    [property: JsonPropertyName("coefficients")] double[] Coefficients);

public sealed record TrainingParams(
    [property: JsonPropertyName("max_terms")] int MaxTerms,
    [property: JsonPropertyName("max_degree")] int MaxDegree,
    [property: JsonPropertyName("penalty")] double Penalty,
    [property: JsonPropertyName("minspan")] double Minspan,
    [property: JsonPropertyName("endspan")] double Endspan,
    [property: JsonPropertyName("threshold")] double Threshold,
    [property: JsonPropertyName("allow_linear")] bool AllowLinear,
    [property: JsonPropertyName("allow_missing")] bool AllowMissing,
    [property: JsonPropertyName("categorical_features")] int[]? CategoricalFeatures = null,
    [property: JsonPropertyName("feature_names")] string[]? FeatureNames = null);

public sealed record TrainingRequest(
    [property: JsonPropertyName("x")] double[][] X,
    [property: JsonPropertyName("y")] double[] Y,
    [property: JsonPropertyName("sample_weight")] double[]? SampleWeight,
    [property: JsonPropertyName("params")] TrainingParams Params);

public static class Runtime
{
    public static ModelSpec LoadModelSpec(string json)
    {
        var spec = JsonSerializer.Deserialize<ModelSpec>(json)
            ?? throw new InvalidOperationException("malformed artifact: empty model spec");
        Validate(spec);
        return spec;
    }

    public static void Validate(ModelSpec spec)
    {
        if (TryValidateWithRust(spec, out var runtimeError))
        {
            return;
        }

        if (runtimeError is not null)
        {
            throw runtimeError;
        }

        ValidatePure(spec);
    }

    public static double[][] DesignMatrix(ModelSpec spec, double[][] rows)
    {
        if (TryDesignMatrixWithRust(spec, rows, out var matrix, out var runtimeError))
        {
            return matrix;
        }

        if (runtimeError is not null)
        {
            throw runtimeError;
        }

        ValidatePure(spec);
        ValidateRowsPure(spec, rows);
        return rows.Select(row => spec.BasisTerms.Select(term => Evaluate(term, row)).ToArray()).ToArray();
    }

    public static double[] Predict(ModelSpec spec, double[][] rows)
    {
        if (TryPredictWithRust(spec, rows, out var predictions, out var runtimeError))
        {
            return predictions;
        }

        if (runtimeError is not null)
        {
            throw runtimeError;
        }

        return DesignMatrix(spec, rows)
            .Select(row => row.Select((value, idx) => value * spec.Coefficients[idx]).Sum())
            .ToArray();
    }

    public static ModelSpec FitModel(TrainingRequest request)
    {
        if (TryFitWithRust(request, out var spec, out var runtimeError))
        {
            return spec!;
        }

        if (runtimeError is not null)
        {
            throw runtimeError;
        }

        throw new InvalidOperationException("Rust training binary is not available");
    }

    private static bool TryValidateWithRust(ModelSpec spec, out InvalidOperationException? runtimeError)
    {
        runtimeError = null;
        return TryRunRustRuntime("validate", spec, null, out _, out runtimeError);
    }

    private static bool TryDesignMatrixWithRust(
        ModelSpec spec,
        double[][] rows,
        out double[][] matrix,
        out InvalidOperationException? runtimeError)
    {
        matrix = Array.Empty<double[]>();
        runtimeError = null;
        if (!TryRunRustRuntime("design-matrix", spec, rows, out var stdout, out runtimeError))
        {
            return false;
        }

        var payload = JsonSerializer.Deserialize<double?[][]>(stdout)
            ?? throw new InvalidOperationException("Rust runtime returned an empty design matrix");
        matrix = NormalizeMatrix(payload);
        return true;
    }

    private static bool TryPredictWithRust(
        ModelSpec spec,
        double[][] rows,
        out double[] predictions,
        out InvalidOperationException? runtimeError)
    {
        predictions = Array.Empty<double>();
        runtimeError = null;
        if (!TryRunRustRuntime("predict", spec, rows, out var stdout, out runtimeError))
        {
            return false;
        }

        var payload = JsonSerializer.Deserialize<double?[]>(stdout)
            ?? throw new InvalidOperationException("Rust runtime returned empty predictions");
        predictions = NormalizeVector(payload);
        return true;
    }

    private static bool TryFitWithRust(
        TrainingRequest request,
        out ModelSpec? spec,
        out InvalidOperationException? runtimeError)
    {
        spec = null;
        runtimeError = null;
        if (!TryRunRustTraining(request, out var stdout, out runtimeError))
        {
            return false;
        }

        spec = JsonSerializer.Deserialize<ModelSpec>(stdout)
            ?? throw new InvalidOperationException("Rust runtime returned an empty model spec");
        Validate(spec);
        return true;
    }

    private static bool TryRunRustRuntime(
        string command,
        ModelSpec spec,
        double[][]? rows,
        out string stdout,
        out InvalidOperationException? runtimeError)
    {
        stdout = string.Empty;
        runtimeError = null;

        var binary = FindRustRuntimeBinary();
        if (binary is null)
        {
            return false;
        }

        var specPath = WriteTempJson(JsonSerializer.Serialize(spec));
        string? rowsPath = null;
        try
        {
            if (rows is not null)
            {
                rowsPath = WriteTempJson(JsonSerializer.Serialize(ToNullableRows(rows)));
            }

            var psi = new ProcessStartInfo
            {
                FileName = binary,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add(command);
            psi.ArgumentList.Add("--spec-file");
            psi.ArgumentList.Add(specPath);
            if (rowsPath is not null)
            {
                psi.ArgumentList.Add("--rows-file");
                psi.ArgumentList.Add(rowsPath);
            }

            using var process = Process.Start(psi);
            if (process is null)
            {
                return false;
            }

            stdout = process.StandardOutput.ReadToEnd();
            var stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                runtimeError = new InvalidOperationException(string.IsNullOrWhiteSpace(stderr)
                    ? $"Rust runtime command failed: {command}"
                    : stderr.Trim());
                return false;
            }

            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.ComponentModel.Win32Exception)
        {
            return false;
        }
        finally
        {
            TryDelete(specPath);
            if (rowsPath is not null)
            {
                TryDelete(rowsPath);
            }
        }
    }

    private static bool TryRunRustTraining(
        TrainingRequest request,
        out string stdout,
        out InvalidOperationException? runtimeError)
    {
        stdout = string.Empty;
        runtimeError = null;

        var binary = FindRustRuntimeBinary();
        if (binary is null)
        {
            return false;
        }

        var requestPath = WriteTempJson(JsonSerializer.Serialize(request));
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = binary,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add("fit");
            psi.ArgumentList.Add("--request-file");
            psi.ArgumentList.Add(requestPath);

            using var process = Process.Start(psi);
            if (process is null)
            {
                return false;
            }

            stdout = process.StandardOutput.ReadToEnd();
            var stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                runtimeError = new InvalidOperationException(string.IsNullOrWhiteSpace(stderr)
                    ? "Rust runtime command failed: fit"
                    : stderr.Trim());
                return false;
            }

            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.ComponentModel.Win32Exception)
        {
            return false;
        }
        finally
        {
            TryDelete(requestPath);
        }
    }

    private static string? FindRustRuntimeBinary()
    {
        var envBinary = Environment.GetEnvironmentVariable("MARS_RUNTIME_BIN");
        if (IsExecutableFile(envBinary))
        {
            return envBinary;
        }

        foreach (var start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            var directory = new DirectoryInfo(Path.GetFullPath(start));
            while (directory is not null)
            {
                foreach (var path in CandidateBinaryPaths(directory.FullName))
                {
                    if (IsExecutableFile(path))
                    {
                        return path;
                    }
                }

                directory = directory.Parent;
            }
        }

        return null;
    }

    private static IEnumerable<string> CandidateBinaryPaths(string root)
    {
        var binaryName = OperatingSystem.IsWindows() ? "mars-runtime-cli.exe" : "mars-runtime-cli";
        yield return Path.Combine(root, "rust-runtime", "target", "debug", binaryName);
        yield return Path.Combine(root, "rust-runtime", "target", "release", binaryName);
    }

    private static bool IsExecutableFile(string? path)
    {
        return !string.IsNullOrWhiteSpace(path) && File.Exists(path);
    }

    private static string WriteTempJson(string content)
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.json");
        File.WriteAllText(path, content);
        return path;
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
        }
    }

    private static double?[][] ToNullableRows(double[][] rows)
    {
        return rows.Select(row => row.Select(value => double.IsNaN(value) ? (double?)null : value).ToArray()).ToArray();
    }

    private static double[][] NormalizeMatrix(double?[][] rows)
    {
        return rows.Select(row => NormalizeVector(row)).ToArray();
    }

    private static double[] NormalizeVector(double?[] values)
    {
        return values.Select(value => value ?? double.NaN).ToArray();
    }

    private static void ValidatePure(ModelSpec spec)
    {
        if (string.IsNullOrEmpty(spec.SpecVersion) || spec.SpecVersion.Length < 3 || spec.SpecVersion[1] != '.')
        {
            throw new InvalidOperationException("malformed artifact: spec_version must be '<major>.<minor>'");
        }

        if (!spec.SpecVersion.StartsWith("1.", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"unsupported artifact version: {spec.SpecVersion}");
        }

        if (spec.BasisTerms.Length != spec.Coefficients.Length)
        {
            throw new InvalidOperationException("malformed artifact: coefficients length must match basis_terms");
        }

        if (spec.FeatureSchema.NFeatures is not null && spec.FeatureSchema.FeatureNames is not null)
        {
            if (spec.FeatureSchema.FeatureNames.Length != spec.FeatureSchema.NFeatures.Value)
            {
                throw new InvalidOperationException("malformed artifact: feature_names length must match n_features");
            }
        }

        for (var index = 0; index < spec.BasisTerms.Length; index++)
        {
            ValidateBasis(spec, spec.BasisTerms[index], index);
        }
    }

    private static void ValidateRowsPure(ModelSpec spec, double[][] rows)
    {
        if (spec.FeatureSchema.NFeatures is null)
        {
            return;
        }

        for (var index = 0; index < rows.Length; index++)
        {
            if (rows[index].Length != spec.FeatureSchema.NFeatures.Value)
            {
                throw new InvalidOperationException(
                    $"feature-count mismatch: row {index} has {rows[index].Length} features, expected {spec.FeatureSchema.NFeatures.Value}");
            }
        }
    }

    private static void ValidateBasis(ModelSpec spec, BasisTerm basis, int index)
    {
        if (string.IsNullOrEmpty(basis.Kind))
        {
            throw new InvalidOperationException($"missing required field: basis term {index} has empty kind");
        }

        switch (basis.Kind)
        {
            case "constant":
                return;
            case "linear":
            case "missingness":
                ValidateVariableIdx(spec, basis.VariableIdx, index);
                return;
            case "hinge":
                ValidateVariableIdx(spec, basis.VariableIdx, index);
                if (basis.KnotVal is null)
                {
                    throw new InvalidOperationException("missing required field: hinge requires knot_val");
                }

                if (basis.IsRightHinge is null)
                {
                    throw new InvalidOperationException("missing required field: hinge requires is_right_hinge");
                }

                return;
            case "categorical":
                ValidateVariableIdx(spec, basis.VariableIdx, index);
                _ = CategoryValue(basis);
                return;
            case "interaction":
                if (basis.Parent1 is null || basis.Parent2 is null)
                {
                    throw new InvalidOperationException("missing required field: interaction requires parent1 and parent2");
                }

                return;
            default:
                throw new InvalidOperationException($"unsupported basis term: {basis.Kind}");
        }
    }

    private static void ValidateVariableIdx(ModelSpec spec, int? variableIdx, int basisIndex)
    {
        if (variableIdx is null)
        {
            throw new InvalidOperationException("missing required field: basis term requires variable_idx");
        }

        if (spec.FeatureSchema.NFeatures is not null && variableIdx.Value >= spec.FeatureSchema.NFeatures.Value)
        {
            throw new InvalidOperationException(
                $"malformed artifact: basis term {basisIndex} references variable outside feature count");
        }
    }

    private static double Evaluate(BasisTerm basis, double[] row)
    {
        return basis.Kind switch
        {
            "constant" => 1.0,
            "linear" => row[basis.VariableIdx!.Value],
            "hinge" => basis.IsRightHinge == true
                ? Math.Max(row[basis.VariableIdx!.Value] - basis.KnotVal!.Value, 0.0)
                : Math.Max(basis.KnotVal!.Value - row[basis.VariableIdx!.Value], 0.0),
            "categorical" => EvaluateCategorical(basis, row),
            "interaction" => EvaluateInteraction(basis, row),
            "missingness" => double.IsNaN(row[basis.VariableIdx!.Value]) ? 1.0 : 0.0,
            _ => throw new InvalidOperationException($"unsupported basis term: {basis.Kind}"),
        };
    }

    private static double EvaluateCategorical(BasisTerm basis, double[] row)
    {
        var category = CategoryValue(basis);
        var value = row[basis.VariableIdx!.Value];
        if (double.IsNaN(value))
        {
            return double.NaN;
        }

        return value == category ? 1.0 : 0.0;
    }

    private static double EvaluateInteraction(BasisTerm basis, double[] row)
    {
        var left = Evaluate(basis.Parent1!, row);
        var right = Evaluate(basis.Parent2!, row);
        return double.IsNaN(left) || double.IsNaN(right) ? double.NaN : left * right;
    }

    private static double CategoryValue(BasisTerm basis)
    {
        if (basis.Category is null || basis.Category.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            throw new InvalidOperationException("missing required field: categorical requires category");
        }

        if (!basis.Category.Value.TryGetDouble(out var value))
        {
            throw new InvalidOperationException("invalid categorical encoding: expected numeric category");
        }

        return value;
    }
}

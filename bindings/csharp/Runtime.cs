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
    [property: JsonPropertyName("feature_schema")] FeatureSchema FeatureSchema,
    [property: JsonPropertyName("basis_terms")] BasisTerm[] BasisTerms,
    [property: JsonPropertyName("coefficients")] double[] Coefficients);

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
        if (!spec.SpecVersion.StartsWith("1."))
        {
            throw new InvalidOperationException($"unsupported artifact version: {spec.SpecVersion}");
        }
        if (spec.BasisTerms.Length != spec.Coefficients.Length)
        {
            throw new InvalidOperationException("malformed artifact: coefficients length must match basis_terms");
        }
    }

    public static double[][] DesignMatrix(ModelSpec spec, double[][] rows)
    {
        Validate(spec);
        return rows.Select(row => spec.BasisTerms.Select(term => Evaluate(term, row)).ToArray()).ToArray();
    }

    public static double[] Predict(ModelSpec spec, double[][] rows)
    {
        return DesignMatrix(spec, rows)
            .Select(row => row.Select((value, idx) => value * spec.Coefficients[idx]).Sum())
            .ToArray();
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
            "categorical" => double.IsNaN(row[basis.VariableIdx!.Value])
                ? double.NaN
                : row[basis.VariableIdx!.Value] == basis.Category!.Value.GetDouble() ? 1.0 : 0.0,
            "interaction" => Interaction(basis, row),
            "missingness" => double.IsNaN(row[basis.VariableIdx!.Value]) ? 1.0 : 0.0,
            _ => throw new InvalidOperationException($"unsupported basis term: {basis.Kind}"),
        };
    }

    private static double Interaction(BasisTerm basis, double[] row)
    {
        var left = Evaluate(basis.Parent1!, row);
        var right = Evaluate(basis.Parent2!, row);
        return double.IsNaN(left) || double.IsNaN(right) ? double.NaN : left * right;
    }
}

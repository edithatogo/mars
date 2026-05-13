using Xunit;

namespace MarsRuntime.Tests;

public sealed class NativeAbiSmokeTests
{
    [Fact]
    public void LoadsTheRustAbiAndValidatesAModelSpec()
    {
        var error = new MarsForeignError();
        var status = NativeAbi.MarsRuntimeAbiVersion(out var version, ref error);

        Assert.Equal(MarsForeignStatus.Ok, status);
        Assert.Equal((uint)1, version.Major);
        Assert.Equal((uint)0, version.Minor);
        Assert.Equal((uint)0, version.Patch);
        Assert.Equal(IntPtr.Zero, error.Message);

        status = NativeAbi.MarsRuntimeAbiCheckCompatibility(1, 0, 0, ref error);
        Assert.Equal(MarsForeignStatus.Ok, status);
        Assert.Equal(IntPtr.Zero, error.Message);

        var specJson = """
            {
              "spec_version": "1.0",
              "params": {},
              "feature_schema": { "n_features": 1, "feature_names": ["x"] },
              "basis_terms": [{ "kind": "constant" }],
              "coefficients": [1.0]
            }
            """;

        var handleStatus = NativeAbi.MarsModelSpecFromJson(specJson, out var handle, ref error);
        Assert.Equal(MarsForeignStatus.Ok, handleStatus);
        Assert.NotEqual(IntPtr.Zero, handle);

        var validateStatus = NativeAbi.MarsModelSpecValidate(handle, ref error);
        Assert.Equal(MarsForeignStatus.Ok, validateStatus);
        Assert.Equal(IntPtr.Zero, error.Message);

        var batchJson = "[[1.0, null], [3.0, 4.0]]";
        var batchStatus = NativeAbi.MarsBatchMatrixFromJson(batchJson, out var matrix, ref error);
        Assert.Equal(MarsForeignStatus.Ok, batchStatus);
        Assert.Equal((nuint)2, matrix.Rows);
        Assert.Equal((nuint)2, matrix.Cols);
        Assert.Equal((nuint)4, matrix.Len);
        Assert.Equal(IntPtr.Zero, error.Message);

        NativeAbi.MarsModelSpecFree(handle);
        if (matrix.Data != IntPtr.Zero)
        {
            NativeAbi.MarsForeignMatrixFree(ref matrix);
        }
    }
}

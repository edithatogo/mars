using System.Runtime.InteropServices;
using System.Reflection;

namespace MarsRuntime;

internal static class NativeAbi
{
    private const string LibraryName = "pymars_runtime";
    private static readonly object LoadLock = new();
    private static IntPtr LoadedLibrary = IntPtr.Zero;

    static NativeAbi()
    {
        NativeLibrary.SetDllImportResolver(typeof(NativeAbi).Assembly, ResolveLibrary);
    }

    [DllImport(LibraryName, EntryPoint = "mars_runtime_abi_version")]
    internal static extern MarsForeignStatus MarsRuntimeAbiVersion(
        out MarsForeignAbiVersion version,
        ref MarsForeignError error);

    [DllImport(LibraryName, EntryPoint = "mars_runtime_abi_check_compatibility")]
    internal static extern MarsForeignStatus MarsRuntimeAbiCheckCompatibility(
        uint requestedMajor,
        uint requestedMinor,
        uint requestedPatch,
        ref MarsForeignError error);

    [DllImport(LibraryName, EntryPoint = "mars_model_spec_from_json")]
    internal static extern MarsForeignStatus MarsModelSpecFromJson(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string json,
        out IntPtr handle,
        ref MarsForeignError error);

    [DllImport(LibraryName, EntryPoint = "mars_model_spec_validate")]
    internal static extern MarsForeignStatus MarsModelSpecValidate(
        IntPtr handle,
        ref MarsForeignError error);

    [DllImport(LibraryName, EntryPoint = "mars_batch_matrix_from_json")]
    internal static extern MarsForeignStatus MarsBatchMatrixFromJson(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string json,
        out MarsForeignMatrix matrix,
        ref MarsForeignError error);

    [DllImport(LibraryName, EntryPoint = "mars_model_spec_free")]
    internal static extern void MarsModelSpecFree(IntPtr handle);

    [DllImport(LibraryName, EntryPoint = "mars_foreign_matrix_free")]
    internal static extern void MarsForeignMatrixFree(ref MarsForeignMatrix matrix);

    [DllImport(LibraryName, EntryPoint = "mars_foreign_error_free")]
    internal static extern void MarsForeignErrorFree(ref MarsForeignError error);

    private static IntPtr ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, LibraryName, StringComparison.Ordinal))
        {
            return IntPtr.Zero;
        }

        lock (LoadLock)
        {
            if (LoadedLibrary != IntPtr.Zero)
            {
                return LoadedLibrary;
            }

            var path = FindNativeLibrary();
            if (path is null)
            {
                return IntPtr.Zero;
            }

            LoadedLibrary = NativeLibrary.Load(path);
            return LoadedLibrary;
        }
    }

    private static string? FindNativeLibrary()
    {
        var envPath = Environment.GetEnvironmentVariable("MARS_RUNTIME_NATIVE_LIB");
        if (!string.IsNullOrWhiteSpace(envPath))
        {
            if (File.Exists(envPath))
            {
                return envPath;
            }

            if (Directory.Exists(envPath))
            {
                foreach (var candidate in CandidateNativeLibraryPaths(envPath))
                {
                    if (File.Exists(candidate))
                    {
                        return candidate;
                    }
                }
            }
        }

        foreach (var candidate in CandidateNativeLibraryPaths(string.Empty))
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private static IEnumerable<string> CandidateNativeLibraryPaths(string root)
    {
        var binaryName = OperatingSystem.IsWindows()
            ? "pymars_runtime.dll"
            : OperatingSystem.IsMacOS()
                ? "libpymars_runtime.dylib"
                : "libpymars_runtime.so";

        var relativeCandidates = new[]
        {
            Path.Combine("rust-runtime", "target", "debug", binaryName),
            Path.Combine("rust-runtime", "target", "release", binaryName),
            Path.Combine("rust-runtime", "target", "maturin", binaryName),
        };

        if (!string.IsNullOrWhiteSpace(root))
        {
            foreach (var relative in relativeCandidates)
            {
                yield return Path.Combine(root, Path.GetFileName(relative));
            }
        }

        foreach (var start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            var directory = new DirectoryInfo(Path.GetFullPath(start));
            while (directory is not null)
            {
                foreach (var relative in relativeCandidates)
                {
                    var candidate = Path.Combine(directory.FullName, relative);
                    yield return candidate;
                }
                directory = directory.Parent;
            }
        }
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct MarsForeignAbiVersion
{
    public uint Major;
    public uint Minor;
    public uint Patch;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MarsForeignError
{
    public MarsForeignStatus Status;
    public IntPtr Message;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MarsForeignMatrix
{
    public IntPtr Data;
    public nuint Len;
    public nuint Rows;
    public nuint Cols;
}

internal enum MarsForeignStatus
{
    Ok = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    MalformedArtifact = 3,
    UnsupportedArtifactVersion = 4,
    MissingRequiredField = 5,
    UnsupportedBasisTerm = 6,
    FeatureCountMismatch = 7,
    InvalidCategoricalEncoding = 8,
    NumericalEvaluationFailure = 9,
    NotYetImplemented = 10,
}

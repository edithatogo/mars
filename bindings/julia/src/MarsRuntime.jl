module MarsRuntime

using JSON

export load_model_spec, validate_model_spec, design_matrix, predict_model

function load_model_spec(path_or_json::AbstractString)
    raw = isfile(path_or_json) ? read(path_or_json, String) : path_or_json
    spec = JSON.parse(raw)
    validate_model_spec(spec)
    return spec
end

function validate_model_spec(spec)
    if try
        _validate_model_spec_rust(spec)
    catch
        false
    end
        return true
    end
    return _validate_model_spec_pure(spec)
end

function design_matrix(spec, rows)
    if _rust_runtime_available()
        try
            return _design_matrix_rust(spec, rows)
        catch
        end
    end
    return _design_matrix_pure(spec, rows)
end

function predict_model(spec, rows)
    if _rust_runtime_available()
        try
            return _predict_rust(spec, rows)
        catch
        end
    end
    return _predict_model_pure(spec, rows)
end

function _validate_model_spec_rust(spec)
    if !_rust_runtime_available()
        return false
    end
    _invoke_rust_runtime("validate", spec, nothing)
    return true
end

function _design_matrix_rust(spec, rows)
    return _invoke_rust_runtime("design-matrix", spec, rows)
end

function _predict_rust(spec, rows)
    return _invoke_rust_runtime("predict", spec, rows)
end

function _invoke_rust_runtime(command, spec, rows)
    binary = _rust_runtime_binary()
    binary == "" && error("Rust runtime binary is not available")

    spec_file = tempname() * ".json"
    open(spec_file, "w") do io
        JSON.print(io, spec)
    end

    args = String[command, "--spec-file", spec_file]
    if rows !== nothing
        rows_file = tempname() * ".json"
        open(rows_file, "w") do io
            JSON.print(io, _json_rows(rows))
        end
        push!(args, "--rows-file")
        push!(args, rows_file)
    end

    output = readchomp(Cmd([binary; args...]))
    if isempty(output)
        return true
    end
    parsed = JSON.parse(output)
    if command == "design-matrix"
        return _normalize_matrix(parsed)
    elseif command == "predict"
        return _normalize_vector(parsed)
    end
    return parsed
end

function _rust_runtime_available()
    return _rust_runtime_binary() != ""
end

function _rust_runtime_binary()
    env_binary = get(ENV, "MARS_RUNTIME_BIN", "")
    candidates = [
        env_binary,
        normpath(joinpath(@__DIR__, "..", "..", "..", "rust-runtime", "target", "debug", "mars-runtime-cli")),
        normpath(joinpath(@__DIR__, "..", "..", "..", "rust-runtime", "target", "release", "mars-runtime-cli")),
    ]
    for candidate in candidates
        if !isempty(candidate) && isfile(candidate)
            return candidate
        end
    end
    return ""
end

function _validate_model_spec_pure(spec)
    version = get(spec, "spec_version", "")
    occursin(r"^[0-9]+\.[0-9]+$", version) || error("malformed artifact: spec_version must be '<major>.<minor>'")
    startswith(version, "1.") || error("unsupported artifact version: $version")
    length(spec["basis_terms"]) == length(spec["coefficients"]) || error("malformed artifact: coefficients length must match basis_terms")
    return true
end

function _design_matrix_pure(spec, rows)
    _validate_model_spec_pure(spec)
    nfeatures = get(spec["feature_schema"], "n_features", nothing)
    if nfeatures !== nothing
        for (idx, row) in enumerate(rows)
            length(row) == nfeatures || error("feature-count mismatch: row $idx")
        end
    end
    return [[_evaluate_basis_pure(basis, row) for basis in spec["basis_terms"]] for row in rows]
end

function _predict_model_pure(spec, rows)
    matrix = _design_matrix_pure(spec, rows)
    coefficients = Float64.(spec["coefficients"])
    return [sum(row[i] * coefficients[i] for i in eachindex(coefficients)) for row in matrix]
end

function _evaluate_basis_pure(basis, row)
    kind = basis["kind"]
    if kind == "constant"
        return 1.0
    elseif kind == "linear"
        return row[basis["variable_idx"] + 1]
    elseif kind == "hinge"
        value = row[basis["variable_idx"] + 1]
        return basis["is_right_hinge"] ? max(value - basis["knot_val"], 0.0) : max(basis["knot_val"] - value, 0.0)
    elseif kind == "categorical"
        value = row[basis["variable_idx"] + 1]
        isnan(value) && return NaN
        return value == basis["category"] ? 1.0 : 0.0
    elseif kind == "interaction"
        left = _evaluate_basis_pure(basis["parent1"], row)
        right = _evaluate_basis_pure(basis["parent2"], row)
        return isnan(left) || isnan(right) ? NaN : left * right
    elseif kind == "missingness"
        return isnan(row[basis["variable_idx"] + 1]) ? 1.0 : 0.0
    end
    error("unsupported basis term: $kind")
end

function _json_rows(rows)
    return [[isnan(value) ? nothing : value for value in row] for row in rows]
end

function _normalize_matrix(payload)
    return [[value === nothing ? NaN : Float64(value) for value in row] for row in payload]
end

function _normalize_vector(payload)
    return [value === nothing ? NaN : Float64(value) for value in payload]
end

end

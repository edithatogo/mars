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
    version = get(spec, "spec_version", "")
    occursin(r"^[0-9]+\.[0-9]+$", version) || error("malformed artifact: spec_version must be '<major>.<minor>'")
    startswith(version, "1.") || error("unsupported artifact version: $version")
    length(spec["basis_terms"]) == length(spec["coefficients"]) || error("malformed artifact: coefficients length must match basis_terms")
    return true
end

function design_matrix(spec, rows)
    validate_model_spec(spec)
    nfeatures = get(spec["feature_schema"], "n_features", nothing)
    if nfeatures !== nothing
        for (idx, row) in enumerate(rows)
            length(row) == nfeatures || error("feature-count mismatch: row $idx")
        end
    end
    return [[evaluate_basis(basis, row) for basis in spec["basis_terms"]] for row in rows]
end

function predict_model(spec, rows)
    matrix = design_matrix(spec, rows)
    coefficients = Float64.(spec["coefficients"])
    return [sum(row[i] * coefficients[i] for i in eachindex(coefficients)) for row in matrix]
end

function evaluate_basis(basis, row)
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
        left = evaluate_basis(basis["parent1"], row)
        right = evaluate_basis(basis["parent2"], row)
        return isnan(left) || isnan(right) ? NaN : left * right
    elseif kind == "missingness"
        return isnan(row[basis["variable_idx"] + 1]) ? 1.0 : 0.0
    end
    error("unsupported basis term: $kind")
end

end

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::errors::{MarsError, MarsResult};
use crate::runtime::{design_matrix, load_model_spec_str, predict, validate_model_spec};
use serde::Deserialize;

pub const MARS_FOREIGN_ABI_VERSION_MAJOR: u32 = 1;
pub const MARS_FOREIGN_ABI_VERSION_MINOR: u32 = 0;
pub const MARS_FOREIGN_ABI_VERSION_PATCH: u32 = 0;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MarsForeignStatus {
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

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MarsForeignAbiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct MarsForeignError {
    pub status: MarsForeignStatus,
    pub message: *mut c_char,
}

impl Default for MarsForeignError {
    fn default() -> Self {
        Self {
            status: MarsForeignStatus::Ok,
            message: ptr::null_mut(),
        }
    }
}

impl Default for MarsForeignAbiVersion {
    fn default() -> Self {
        Self {
            major: MARS_FOREIGN_ABI_VERSION_MAJOR,
            minor: MARS_FOREIGN_ABI_VERSION_MINOR,
            patch: MARS_FOREIGN_ABI_VERSION_PATCH,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct MarsForeignMatrix {
    pub data: *mut f64,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Default for MarsForeignMatrix {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
            rows: 0,
            cols: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct MarsForeignVector {
    pub data: *mut f64,
    pub len: usize,
}

impl Default for MarsForeignVector {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ForeignBatchMatrix(Vec<Vec<Option<f64>>>);

#[repr(C)]
pub struct MarsModelSpecHandle {
    spec_json: CString,
}

impl MarsModelSpecHandle {
    fn new(spec_json: CString) -> Self {
        Self { spec_json }
    }

    fn spec(&self) -> MarsResult<crate::model_spec::ModelSpec> {
        load_model_spec_str(self.spec_json.as_c_str().to_str().map_err(|error| {
            MarsError::MalformedArtifact(format!("invalid UTF-8 in stored spec: {error}"))
        })?)
    }
}

/// # Safety
///
/// `out_version` and `out_error` must be valid pointers when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_runtime_abi_version(
    out_version: *mut MarsForeignAbiVersion,
    out_error: *mut MarsForeignError,
) -> MarsForeignStatus {
    if out_version.is_null() {
        return write_error(
            out_error,
            MarsForeignStatus::NullPointer,
            "out_version must not be null",
        );
    }

    *out_version = MarsForeignAbiVersion::default();
    clear_error(out_error);
    MarsForeignStatus::Ok
}

/// # Safety
///
/// `out_error` must be a valid pointer when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_runtime_abi_check_compatibility(
    requested_major: u32,
    requested_minor: u32,
    requested_patch: u32,
    out_error: *mut MarsForeignError,
) -> MarsForeignStatus {
    let current = MarsForeignAbiVersion::default();
    if requested_major != current.major {
        return write_error(
            out_error,
            MarsForeignStatus::UnsupportedArtifactVersion,
            format!(
                "requested ABI major {requested_major} is incompatible with current major {}",
                current.major
            ),
        );
    }
    if requested_minor > current.minor
        || (requested_minor == current.minor && requested_patch > current.patch)
    {
        return write_error(
            out_error,
            MarsForeignStatus::UnsupportedArtifactVersion,
            format!(
                "requested ABI version {requested_major}.{requested_minor}.{requested_patch} exceeds current {}.{}.{}",
                current.major, current.minor, current.patch
            ),
        );
    }

    clear_error(out_error);
    MarsForeignStatus::Ok
}

/// # Safety
///
/// `handle` must be a pointer previously returned by this module or null. It
/// must not be freed more than once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_model_spec_free(handle: *mut MarsModelSpecHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
///
/// `error` must point to a valid `MarsForeignError` allocated by the caller or
/// be null. Any owned message pointer inside it must come from this module.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_foreign_error_free(error: *mut MarsForeignError) {
    if error.is_null() {
        return;
    }
    if !(*error).message.is_null() {
        drop(CString::from_raw((*error).message));
        (*error).message = ptr::null_mut();
    }
}

/// # Safety
///
/// `matrix` must point to a valid `MarsForeignMatrix` allocated by this module
/// or be null. Any owned buffer inside it must come from this module.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_foreign_matrix_free(matrix: *mut MarsForeignMatrix) {
    if matrix.is_null() {
        return;
    }
    if !(*matrix).data.is_null() {
        let _ = Vec::from_raw_parts((*matrix).data, (*matrix).len, (*matrix).len);
        (*matrix).data = ptr::null_mut();
        (*matrix).len = 0;
        (*matrix).rows = 0;
        (*matrix).cols = 0;
    }
}

/// # Safety
///
/// `vector` must point to a valid `MarsForeignVector` allocated by this module
/// or be null. Any owned buffer inside it must come from this module.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_foreign_vector_free(vector: *mut MarsForeignVector) {
    if vector.is_null() {
        return;
    }
    if !(*vector).data.is_null() {
        let _ = Vec::from_raw_parts((*vector).data, (*vector).len, (*vector).len);
        (*vector).data = ptr::null_mut();
        (*vector).len = 0;
    }
}

/// # Safety
///
/// `json`, `out_handle`, and `out_error` must be valid pointers when non-null
/// and `out_handle` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_model_spec_from_json(
    json: *const c_char,
    out_handle: *mut *mut MarsModelSpecHandle,
    out_error: *mut MarsForeignError,
) -> MarsForeignStatus {
    if json.is_null() || out_handle.is_null() {
        return write_error(
            out_error,
            MarsForeignStatus::NullPointer,
            "json and out_handle must not be null",
        );
    }

    let cstr = match CStr::from_ptr(json).to_str() {
        Ok(value) => value,
        Err(error) => {
            return write_error(
                out_error,
                MarsForeignStatus::InvalidUtf8,
                format!("json must be UTF-8: {error}"),
            );
        }
    };

    match load_model_spec_str(cstr) {
        Ok(_) => {
            let stored = match CString::new(cstr) {
                Ok(value) => value,
                Err(error) => {
                    return write_error(
                        out_error,
                        MarsForeignStatus::MalformedArtifact,
                        format!("model spec JSON contains interior nul byte: {error}"),
                    );
                }
            };
            let handle = Box::new(MarsModelSpecHandle::new(stored));
            *out_handle = Box::into_raw(handle);
            clear_error(out_error);
            MarsForeignStatus::Ok
        }
        Err(error) => write_error_from_mars_error(out_error, error),
    }
}

/// # Safety
///
/// `handle` and `out_error` must be valid pointers when non-null. `handle`
/// must point to a model spec previously returned by this module.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_model_spec_validate(
    handle: *const MarsModelSpecHandle,
    out_error: *mut MarsForeignError,
) -> MarsForeignStatus {
    if handle.is_null() {
        return write_error(
            out_error,
            MarsForeignStatus::NullPointer,
            "handle must not be null",
        );
    }

    match (*handle).spec().and_then(|spec| validate_model_spec(&spec)) {
        Ok(()) => {
            clear_error(out_error);
            MarsForeignStatus::Ok
        }
        Err(error) => write_error_from_mars_error(out_error, error),
    }
}

/// # Safety
///
/// `handle`, `rows`, `out_matrix`, and `out_error` must be valid pointers when
/// non-null. `rows` must reference at least `n_rows * n_cols` contiguous `f64`
/// values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_model_spec_design_matrix(
    handle: *const MarsModelSpecHandle,
    rows: *const f64,
    n_rows: usize,
    n_cols: usize,
    out_matrix: *mut MarsForeignMatrix,
    out_error: *mut MarsForeignError,
) -> MarsForeignStatus {
    if handle.is_null() || rows.is_null() || out_matrix.is_null() {
        return write_error(
            out_error,
            MarsForeignStatus::NullPointer,
            "handle, rows, and out_matrix must not be null",
        );
    }

    let spec = match (*handle).spec() {
        Ok(spec) => spec,
        Err(error) => return write_error_from_mars_error(out_error, error),
    };
    let rows = slice_rows(rows, n_rows, n_cols);
    match design_matrix(&spec, &rows) {
        Ok(matrix) => write_matrix(out_matrix, out_error, matrix),
        Err(error) => write_error_from_mars_error(out_error, error),
    }
}

/// # Safety
///
/// `handle`, `rows`, `out_vector`, and `out_error` must be valid pointers when
/// non-null. `rows` must reference at least `n_rows * n_cols` contiguous `f64`
/// values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_model_spec_predict(
    handle: *const MarsModelSpecHandle,
    rows: *const f64,
    n_rows: usize,
    n_cols: usize,
    out_vector: *mut MarsForeignVector,
    out_error: *mut MarsForeignError,
) -> MarsForeignStatus {
    if handle.is_null() || rows.is_null() || out_vector.is_null() {
        return write_error(
            out_error,
            MarsForeignStatus::NullPointer,
            "handle, rows, and out_vector must not be null",
        );
    }

    let spec = match (*handle).spec() {
        Ok(spec) => spec,
        Err(error) => return write_error_from_mars_error(out_error, error),
    };
    let rows = slice_rows(rows, n_rows, n_cols);
    match predict(&spec, &rows) {
        Ok(values) => write_vector(out_vector, out_error, values),
        Err(error) => write_error_from_mars_error(out_error, error),
    }
}

/// # Safety
///
/// `json`, `out_matrix`, and `out_error` must be valid pointers when non-null
/// and `out_matrix` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mars_batch_matrix_from_json(
    json: *const c_char,
    out_matrix: *mut MarsForeignMatrix,
    out_error: *mut MarsForeignError,
) -> MarsForeignStatus {
    if json.is_null() || out_matrix.is_null() {
        return write_error(
            out_error,
            MarsForeignStatus::NullPointer,
            "json and out_matrix must not be null",
        );
    }

    let cstr = match CStr::from_ptr(json).to_str() {
        Ok(value) => value,
        Err(error) => {
            return write_error(
                out_error,
                MarsForeignStatus::InvalidUtf8,
                format!("json must be UTF-8: {error}"),
            );
        }
    };

    let ForeignBatchMatrix(rows) = match serde_json::from_str(cstr) {
        Ok(value) => value,
        Err(error) => {
            return write_error(
                out_error,
                MarsForeignStatus::MalformedArtifact,
                format!("failed to deserialize batch matrix JSON: {error}"),
            );
        }
    };

    if let Some(first_len) = rows.first().map(Vec::len) {
        if rows.iter().any(|row| row.len() != first_len) {
            return write_error(
                out_error,
                MarsForeignStatus::MalformedArtifact,
                "batch matrix rows must all have the same length",
            );
        }
    }

    let matrix = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    write_matrix(out_matrix, out_error, matrix)
}

fn slice_rows(rows: *const f64, n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    let flat = unsafe { std::slice::from_raw_parts(rows, n_rows.saturating_mul(n_cols)) };
    flat.chunks(n_cols).map(|chunk| chunk.to_vec()).collect()
}

fn write_matrix(
    out_matrix: *mut MarsForeignMatrix,
    out_error: *mut MarsForeignError,
    matrix: Vec<Vec<f64>>,
) -> MarsForeignStatus {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    let len = rows.saturating_mul(cols);
    let mut flat = matrix.into_iter().flatten().collect::<Vec<f64>>();
    let ptr = flat.as_mut_ptr();
    std::mem::forget(flat);
    unsafe {
        *out_matrix = MarsForeignMatrix {
            data: ptr,
            len,
            rows,
            cols,
        };
    }
    clear_error(out_error);
    MarsForeignStatus::Ok
}

fn write_vector(
    out_vector: *mut MarsForeignVector,
    out_error: *mut MarsForeignError,
    values: Vec<f64>,
) -> MarsForeignStatus {
    let len = values.len();
    let mut values = values;
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    unsafe {
        *out_vector = MarsForeignVector { data: ptr, len };
    }
    clear_error(out_error);
    MarsForeignStatus::Ok
}

fn write_error(
    out_error: *mut MarsForeignError,
    status: MarsForeignStatus,
    message: impl Into<String>,
) -> MarsForeignStatus {
    unsafe {
        if !out_error.is_null() {
            *out_error = MarsForeignError {
                status,
                message: CString::new(message.into())
                    .expect("error messages should not contain NUL")
                    .into_raw(),
            };
        }
    }
    status
}

fn clear_error(out_error: *mut MarsForeignError) {
    unsafe {
        if !out_error.is_null() {
            *out_error = MarsForeignError::default();
        }
    }
}

fn write_error_from_mars_error(
    out_error: *mut MarsForeignError,
    error: MarsError,
) -> MarsForeignStatus {
    write_error(
        out_error,
        foreign_status_from_error(&error),
        error.to_string(),
    )
}

fn foreign_status_from_error(error: &MarsError) -> MarsForeignStatus {
    match error {
        MarsError::MalformedArtifact(_) => MarsForeignStatus::MalformedArtifact,
        MarsError::UnsupportedArtifactVersion(_) => MarsForeignStatus::UnsupportedArtifactVersion,
        MarsError::MissingRequiredField(_) => MarsForeignStatus::MissingRequiredField,
        MarsError::UnsupportedBasisTerm(_) => MarsForeignStatus::UnsupportedBasisTerm,
        MarsError::FeatureCountMismatch { .. } => MarsForeignStatus::FeatureCountMismatch,
        MarsError::InvalidCategoricalEncoding(_) => MarsForeignStatus::InvalidCategoricalEncoding,
        MarsError::NumericalEvaluationFailure(_) => MarsForeignStatus::NumericalEvaluationFailure,
        MarsError::NotYetImplemented(_) => MarsForeignStatus::NotYetImplemented,
    }
}

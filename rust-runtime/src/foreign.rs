use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::errors::{MarsError, MarsResult};
use crate::runtime::{design_matrix, load_model_spec_str, predict, validate_model_spec};

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

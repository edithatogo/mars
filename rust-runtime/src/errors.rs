use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub type MarsResult<T> = Result<T, MarsError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarsError {
    MalformedArtifact(String),
    UnsupportedArtifactVersion(String),
    MissingRequiredField(String),
    UnsupportedBasisTerm(String),
    FeatureCountMismatch {
        row_index: usize,
        actual: usize,
        expected: usize,
    },
    InvalidCategoricalEncoding(String),
    NumericalEvaluationFailure(String),
}

impl MarsError {
    pub fn category(&self) -> &'static str {
        match self {
            Self::MalformedArtifact(_) => "malformed artifact",
            Self::UnsupportedArtifactVersion(_) => "unsupported artifact version",
            Self::MissingRequiredField(_) => "missing required field",
            Self::UnsupportedBasisTerm(_) => "unsupported basis term",
            Self::FeatureCountMismatch { .. } => "feature-count mismatch",
            Self::InvalidCategoricalEncoding(_) => "invalid categorical encoding",
            Self::NumericalEvaluationFailure(_) => "numerical evaluation failure",
        }
    }
}

impl Display for MarsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedArtifact(message)
            | Self::UnsupportedArtifactVersion(message)
            | Self::MissingRequiredField(message)
            | Self::UnsupportedBasisTerm(message)
            | Self::InvalidCategoricalEncoding(message)
            | Self::NumericalEvaluationFailure(message) => {
                write!(f, "{}: {}", self.category(), message)
            }
            Self::FeatureCountMismatch {
                row_index,
                actual,
                expected,
            } => write!(
                f,
                "{}: row {} has {} features, expected {}",
                self.category(),
                row_index,
                actual,
                expected
            ),
        }
    }
}

impl Error for MarsError {}

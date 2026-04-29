use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct FeatureSchema {
    pub n_features: Option<usize>,
    #[serde(default)]
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct BasisTermSpec {
    pub kind: String,
    pub variable_idx: Option<usize>,
    pub variable_name: Option<String>,
    pub knot_val: Option<f64>,
    pub is_right_hinge: Option<bool>,
    pub category: Option<Value>,
    pub gcv_score: Option<f64>,
    pub rss_score: Option<f64>,
    pub parent1: Option<Box<BasisTermSpec>>,
    pub parent2: Option<Box<BasisTermSpec>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ModelSpec {
    pub spec_version: String,
    pub params: Value,
    pub feature_schema: FeatureSchema,
    pub basis_terms: Vec<BasisTermSpec>,
    pub coefficients: Vec<f64>,
}

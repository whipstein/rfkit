use std::fmt;

/// Additional error types for matrix inversion
#[derive(Debug, PartialEq)]
pub enum InversionError {
    NotSquare(String),
    Singular(String),
    DimensionMismatch(String),
}

impl std::fmt::Display for InversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InversionError::NotSquare(msg) => write!(f, "Matrix is not square: {}", msg),
            InversionError::Singular(msg) => write!(f, "Matrix is singular: {}", msg),
            InversionError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
        }
    }
}

impl std::error::Error for InversionError {}

/// Error types for optimizers
#[derive(Debug)]
pub enum MinimizerError {
    ConstraintViolation,
    FileError(String),
    FunctionEvaluationError,
    GradientEvaluationError,
    HessianEvaluationError,
    InfeasibleStartingPoint,
    InvalidBracket,
    InvalidDimension,
    InvalidDirectionSet,
    InvalidInitialPoints,
    InvalidInitialSimplex,
    InvalidParameters(String),
    InvalidStepSize,
    InvalidTolerance,
    LinearAlgebraError(String),
    LinearSearchFailed,
    LinearSystemSingular,
    LineSearchFailed,
    MaxIterationsExceeded,
    NumericalError(String),
    NoMinimumFound,
    NumericalInstability,
    NumericalOverflow,
    SameSignError,
    SingularHessianApproximation,
    ZeroDerivative,
    ZeroGradient,
}

impl fmt::Display for MinimizerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MinimizerError::ConstraintViolation => write!(f, "Constraint violation detected"),
            MinimizerError::FileError(msg) => write!(f, "File error: {}", msg),
            MinimizerError::FunctionEvaluationError => {
                write!(f, "Function evaluation returned invalid value")
            }
            MinimizerError::GradientEvaluationError => write!(f, "Gradient evaluation error"),
            MinimizerError::HessianEvaluationError => write!(f, "Hessian evaluation error"),
            MinimizerError::InfeasibleStartingPoint => {
                write!(f, "Starting point violates constraints")
            }
            MinimizerError::InvalidBracket => {
                write!(f, "Invalid bracket: ensure a < b")
            }
            MinimizerError::InvalidDimension => write!(f, "Invalid dimension or empty vector"),
            MinimizerError::InvalidDirectionSet => {
                write!(f, "Invalid or linearly dependent direction set")
            }
            MinimizerError::InvalidInitialPoints => {
                write!(f, "Invalid initial points: ensure a < b")
            }
            MinimizerError::InvalidInitialSimplex => {
                write!(f, "Invalid initial simplex configuration")
            }
            MinimizerError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
            MinimizerError::InvalidStepSize => {
                write!(f, "Step size must be positive and finite")
            }
            MinimizerError::InvalidTolerance => write!(f, "Tolerance must be positive"),
            MinimizerError::LinearAlgebraError(msg) => write!(f, "Linear algebra error: {}", msg),
            MinimizerError::LinearSearchFailed => write!(f, "Line search failed to converge"),
            MinimizerError::LinearSystemSingular => write!(f, "Linear system is singular"),
            MinimizerError::LineSearchFailed => write!(f, "Line search failed to find valid step"),
            MinimizerError::MaxIterationsExceeded => write!(f, "Maximum iterations exceeded"),
            MinimizerError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            MinimizerError::NumericalInstability => write!(f, "Numerical instability detected"),
            MinimizerError::NoMinimumFound => {
                write!(f, "No minimum bracket found within search limits")
            }
            MinimizerError::NumericalOverflow => {
                write!(f, "Numerical overflow during bracket expansion")
            }
            MinimizerError::SameSignError => {
                write!(
                    f,
                    "Function values at bracket endpoints must have opposite signs"
                )
            }
            MinimizerError::SingularHessianApproximation => {
                write!(f, "Hessian approximation became singular")
            }
            MinimizerError::ZeroDerivative => {
                write!(f, "Encountered zero derivative, cannot continue")
            }
            MinimizerError::ZeroGradient => write!(f, "Zero gradient encountered"),
        }
    }
}

impl std::error::Error for MinimizerError {}

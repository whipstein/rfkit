use ndarray::{Dim, Dimension, IntoDimension, Ix, Ix1, IxDyn, IxDynImpl};

impl IntoDimension for index!(tuple_type [Ix] 3) {
    type Dim = Dim<[Ix; 3]>;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        Dim::new(index!(array_expr [self] 3))
    }
}

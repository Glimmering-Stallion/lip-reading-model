//! Module summary visitor for model architecture inspection.
//!
//! This module provides a `SummaryVisitor` that traverses any Burn `Module` and prints
//! a table of layer paths, tensor shapes, and parameter counts—similar to TensorFlow's
//! `model.summary()`.



// imports
use burn::{
    module::{
        Module,
        ModuleVisitor,
        Param,
    },
    tensor::{
        backend::Backend,
        Int,
        Tensor,
    },
};

/// Visitor that prints a summary table of module parameters (path, shape, count).
///
/// Burn's Module::visit calls `visit_float`/`visit_int` (not the `_with_path` variants),
/// so we override those and build path from `enter_module`/`exit_module`.
#[derive(Default)]
pub struct SummaryVisitor {
    pub total_params: usize,
    path: Vec<String>,
}



impl SummaryVisitor {
    /// creates new summary visitor
    pub fn new() -> Self { Self::default() }

    /// prints header row for summary table
    pub fn print_header() {
        println!("{:-<100}", "");
        println!("{:<50} | {:<25} | {:>19}", "Layer (Path)", "Shape", "Count");
        println!("{:-<100}", "");
    }

    /// prints footer with total parameter count
    pub fn print_footer(&self) {
        println!("{:-<100}", "");
        println!("Total Trainable Parameters: {}\n", self.total_params);
    }

    /// runs visitor on a module and prints full summary
    pub fn summarize<B: Backend, M: Module<B>>(module: &M) {
        let mut visitor = Self::new();
        Self::print_header();
        module.visit(&mut visitor);
        visitor.print_footer();
    }
}



impl<B: Backend> ModuleVisitor<B> for SummaryVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let shape = param.lazy_shape();
        let num_params = shape.num_elements();
        self.total_params += num_params;

        println!(
            "{:<50} | {:<25} | {:>12} params",
            self.path.join("."),
            format!("{:?}", shape.dims),
            num_params
        );
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        let shape = param.lazy_shape();
        let num_params = shape.num_elements();
        self.total_params += num_params;

        println!(
            "{:<50} | {:<25} | {:>12} params (Int)",
            self.path.join("."),
            format!("{:?}", shape.dims),
            num_params
        );
    }
}

// Utility Functions



// imports
use burn::tensor::{backend::Backend, Tensor};
use num_traits::Float;
use zip; // for extracting zip files



// #[inline]
// pub fn mean(data: &Vec<f32>) -> f32 {
//     let count = data.len() as f32;
//     let sum: f32 = data.iter().sum();
//     sum / count
// }

// #[inline]
// pub fn std_dev(data: &Vec<f32>) -> f32 {
//     let count = data.len();
//     let mean = mean(data);
//     let variance: f32 = data.iter().map(|x| x - mean).map(|x| x * x).sum::<f32>() / count as f32;
//     variance.sqrt()
// }

#[inline]
pub fn mean<T: Float>(data: &[T]) -> T {
    let count = T::from(data.len()).unwrap();
    let sum = data.iter().copied().fold(T::zero(), |acc, x| acc + x);
    sum / count
}

#[inline]
pub fn std_dev<T: Float>(data: &[T]) -> T {
    let count = T::from(data.len()).unwrap();
    let mean = mean(data);
    let variance = data.iter().copied().fold(T::zero(), |acc, x| acc + ((x - mean) * (x - mean))) / count;
    variance.sqrt()
}

#[inline]
pub fn log_sum_exp_2<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, D> {
    let max = a.clone().max_pair(b.clone()); // element-wise maxxing
    let sum = (a - max.clone()).exp().add((b - max.clone()).exp());
    let lse = max.clone().add(sum.log());

    // handle pairwise (-inf, -inf) cases (to avoid NaNs from -inf - -inf)
    let nan_mask = lse.clone().is_nan();
    lse.mask_fill(nan_mask, f32::NEG_INFINITY)
}

#[inline]
pub fn log_sum_exp_3<B: Backend, const D: usize>(
    a: Tensor<B, D>,
    b: Tensor<B, D>,
    c: Tensor<B, D>,
) -> Tensor<B, D> {
    log_sum_exp_2(log_sum_exp_2(a, b), c)
}

pub fn extract_zip(zip_path: &str, extract_to: &str) {
    let mut archive =
        zip::ZipArchive::new(std::fs::File::open(zip_path).expect("Failed to open zip file."))
            .expect("Failed to read zip file.");

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("Failed to read file from zip.");
        let out_path = std::path::Path::new(extract_to).join(file.sanitized_name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&out_path).expect("Failed to create directory.");
        } else {
            if let Some(p) = out_path.parent() {
                std::fs::create_dir_all(p).expect("Failed to create parent directory.");
            }
            let mut outfile = std::fs::File::create(&out_path).expect("Failed to create file.");
            std::io::copy(&mut file, &mut outfile).expect("Failed to write file.");
        }
    }
    println!("Extracted zip file to {}", extract_to);
}

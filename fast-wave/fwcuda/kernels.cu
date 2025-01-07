// Copyright (c) 2024, ETH =Zurich

#include "kernels.hpp"
#include "wrappers.hpp"

template <typename S>
__global__ void propagate_analytically_kernel(DevComplex<S> *d_u, SimParams params, double x_source,
                                              double z) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    // This part is really sensitive to float precision so we use
    // doubles here.
    const double x = static_cast<double>(i - params.N / 2) * params.dx - x_source;
    const double r = sqrt(x * x + z * z);
    const double phase = r * -2.0f * M_PI / params.wl;

    const double inv_sqrt_r = 1.0 / sqrt(r);
    d_u[i].x = static_cast<S>(cos(phase) * inv_sqrt_r);
    d_u[i].y = static_cast<S>(sin(phase) * inv_sqrt_r);
}

template <typename S>
void propagate_analytically(DevComplex<S> *d_u, const SimParams &params, double x_source,
                            double z) {
    propagate_analytically_kernel<S><<<(params.N + 63) / 64, 64>>>(d_u, params, x_source, z);
}

template <typename S> [[nodiscard]] __host__ __device__ S fftfreq(int i, int N, S dx) {
    const int shifted = i - N * (i >= N / 2);
    return static_cast<S>(shifted) / (static_cast<S>(N) * dx);
}

template <typename S>
[[nodiscard]] __device__ DevComplex<S> complex_mult(DevComplex<S> a, DevComplex<S> b) {
    return DevComplex<S>{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template <typename S>
__global__ void propagate_convolve_step_kernel(DevComplex<S> *d_U, SimParams params, double dz,
                                               double cutoff_freq) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    const double ff = fftfreq<double>(i, params.N, params.dx);

    // Only perform the calculation for the entries that don't get zeroed out
    // in the frequency cutoff.
    if (abs(ff) <= cutoff_freq) {
        const double angle = M_PI * dz * (-2 / params.wl + params.wl * ff * ff);
        d_U[i] = complex_mult<S>(
            d_U[i], DevComplex<S>{static_cast<S>(cos(angle)), static_cast<S>(sin(angle))});

    } else {
        d_U[i] = DevComplex<S>{0.f, 0.f};
    }
}

template <typename S>
void propagate_convolve_step(DevComplex<S> *d_U, const SimParams &params, double dz,
                             double cutoff_freq) {
    propagate_convolve_step_kernel<S><<<(params.N + 63) / 64, 64>>>(d_U, params, dz, cutoff_freq);
}

template <typename S>
__global__ void apply_grating_factors_kernel(DevComplex<S> *d_u, SimParams params,
                                             DevComplex<S> factor_a, DevComplex<S> factor_b,
                                             double pitch, double dc, double x_position) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    // todo: gratings seem to be one off from python version
    const double x = static_cast<double>(i - params.N / 2) * params.dx + x_position;

    const double t = x / pitch + dc * 0.5f;
    const double phase = t - floor(t);
    const DevComplex<S> factor = phase < dc ? factor_a : factor_b;
    d_u[i] = complex_mult<S>(d_u[i], factor);
}

template <typename S>
void apply_grating_factors(DevComplex<S> *d_u, const SimParams &params, Complex<S> factor_a,
                           Complex<S> factor_b, double pitch, double dc, double x_position) {
    apply_grating_factors_kernel<S>
        <<<params.N / 64, 64>>>(d_u, params, c2dc(factor_a), c2dc(factor_b), pitch, dc, x_position);
}

template <typename S>
__global__ void apply_env_grating_factors_kernel(DevComplex<S> *d_u, SimParams params,
                                             DevComplex<S> factor_a, DevComplex<S> factor_b,
                                             double pitch0, double pitch1, double dc0, 
                                             double dc1, double x_position) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    const double x = static_cast<double>(i - params.N / 2) * params.dx + x_position;

    const double t0 = x / pitch0 + dc0 * 0.5f;
    const double t1 = x / pitch1;
    const double phase0 = t0 - floor(t0);
    const double phase1 = t1 - floor(t1);
    const DevComplex<S> factor = (phase0 < dc0 && phase1 < dc1) ? factor_a : factor_b;
    d_u[i] = complex_mult<S>(d_u[i], factor);
}

template <typename S>
void apply_env_grating_factors(DevComplex<S> *d_u, const SimParams &params, Complex<S> factor_a,
                           Complex<S> factor_b, double pitch0, double pitch1, 
                           double dc0, double dc1, double x_position) {
    apply_env_grating_factors_kernel<S><<<params.N / 64, 64>>>(d_u, params, c2dc(factor_a),
                                                        c2dc(factor_b), pitch0, pitch1, dc0, dc1, x_position);
}

template <typename S>
__global__ void apply_sample_factors_kernel(DevComplex<S> *d_u, SimParams params, double dz,
                                            int8_t *d_sample, double pixel_size_x,
                                            std::size_t x_len, DevComplex<double> *d_deltabetas,
                                            int z_slice_index, double x_position) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= params.N)
        return;

    const double x =
        static_cast<double>(i - params.N / 2) * params.dx + x_position + pixel_size_x * x_len * 0.5;
    const double x_index = x / pixel_size_x;
    const std::size_t x_floor = static_cast<std::size_t>(x_index);
    const double x_frac = x_index - static_cast<double>(x_floor);

    if (x_index < 0 || x_floor + 1 >= x_len) {
        return;
    }

    const DevComplex<double> db_lower = d_deltabetas[d_sample[x_len * z_slice_index + x_floor]];
    const DevComplex<double> db_upper = d_deltabetas[d_sample[x_len * z_slice_index + x_floor + 1]];

    DevComplex<double> interpolated;
    interpolated.x = db_lower.x * (1 - x_frac) + db_upper.x * x_frac;
    interpolated.y = db_lower.y * (1 - x_frac) + db_upper.y * x_frac;

    const DevComplex<double> exponent =
        complex_mult<double>(interpolated, DevComplex<double>{0, 2.0 * M_PI * dz / params.wl});
    const double exp_r = exp(exponent.x);
    const double angle = exponent.y;
    d_u[i] = complex_mult<S>(d_u[i], DevComplex<S>{static_cast<float>(exp_r * cos(angle)),
                                                   static_cast<float>(exp_r * sin(angle))});
}

template <typename S>
void apply_sample_factors(DevComplex<S> *d_u, const SimParams &params, double dz, int8_t *d_sample,
                          double pixel_size_x, std::size_t x_len, DevComplex<double> *d_deltabetas,
                          int z_slice_index, double x_position) {
    apply_sample_factors_kernel<S><<<params.N / 64, 64>>>(
        d_u, params, dz, d_sample, pixel_size_x, x_len, d_deltabetas, z_slice_index, x_position);
}

template <typename S>
__global__ void scale_kernel(DevComplex<S> *d_u, DevComplex<S> scale, const int N) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;

    d_u[i] = complex_mult<S>(d_u[i], scale);
}

template <typename S> void scale(DevComplex<S> *d_u, Complex<S> scale, const int N) {
    scale_kernel<S><<<(N + 63) / 64, 64>>>(d_u, c2dc(scale), N);
    check_cuda_result("scale kernel", cudaPeekAtLastError());
}

template <typename S>
__global__ void square_and_downsample_kernel(DevComplex<S> *d_u, int N, S *d_out, int outsize,
                                             double detector_pixel_size_x,
                                             double detector_pixel_size_y, double current_z,
                                             double dx) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= outsize)
        return;

    const double min_x = static_cast<double>(i - outsize / 2) * detector_pixel_size_x;
    const double max_x = static_cast<double>(i + 1 - outsize / 2) * detector_pixel_size_x;
    const int min_idx = max(static_cast<int>(min_x / dx) + N / 2, 0);
    const int max_idx = min(static_cast<int>(max_x / dx) + N / 2, N);

    double sum = 0;
    for (int j = min_idx; j < max_idx; ++j) {
        // This is not optimal because global memory access won't be coalesced.
        // The advantage is that the code is much simpler: the kernel is pretty
        // much independent of block size and we don't need shared memory or
        // atomics.
        sum += d_u[j].x * d_u[j].x + d_u[j].y * d_u[j].y;
    }
    double r = sqrt(min_x * min_x + current_z * current_z);
    double angle = atan(min_x / current_z);
    d_out[i] = sum * dx * detector_pixel_size_y * cos(angle) / r;
}

template <typename S>
void square_and_downsample(DevComplex<S> *d_u, int N, S *d_out, int outsize,
                           double detector_pixel_size_x, double detector_pixel_size_y,
                           double current_z, double dx) {
    square_and_downsample_kernel<S><<<(N + 63) / 64, 64>>>(
        d_u, N, d_out, outsize, detector_pixel_size_x, detector_pixel_size_y, current_z, dx);
    check_cuda_result("square_and_downsample kernel", cudaPeekAtLastError());
}

template <typename S>
__global__ void analytical_history_row_kernel(S *row, int nr_detector_pixels,
                                              double detector_pixel_size_x,
                                              double detector_pixel_size_y, double x_source,
                                              double z, double cutoff_gratient) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nr_detector_pixels)
        return;
    const double x =
        static_cast<double>(i - nr_detector_pixels / 2) * detector_pixel_size_x - x_source;
    if (abs(x) > cutoff_gratient * z)
        return;

    const double angle = atan(x / z);
    const double r2 = x * x + z * z;
    const double intensity = detector_pixel_size_x * detector_pixel_size_y * cos(angle) / r2;
    row[i] = static_cast<S>(intensity);
}

template <typename S>
void analytical_history_row(S *row, int nr_detector_pixels, double detector_pixel_size_x,
                            double detector_pixel_size_y, double x_source, double z,
                            double cutoff_gratient) {
    analytical_history_row_kernel<S><<<(nr_detector_pixels + 63) / 64, 64>>>(
        row, nr_detector_pixels, detector_pixel_size_x, detector_pixel_size_y, x_source, z,
        cutoff_gratient);
    check_cuda_result("analytical_history_row kernel", cudaPeekAtLastError());
}

// This is a bit awkward.. usually the simpler solution would be to just
// move the function definitions to the header so that we don't have to
// write down the template instantiations, but we want to make the cuda
// kernel calls invisible to non-cuda-cpp files so we can't put the wrapper
// bodies in a header that is included in a cpp file.

// explicit instantiations for float
template void propagate_analytically<float>(DevComplex<float> *, const SimParams &, double, double);
template __host__ __device__ float fftfreq<float>(int, int, float);
template void propagate_convolve_step<float>(DevComplex<float> *, const SimParams &, double,
                                             double);
template void apply_grating_factors<float>(DevComplex<float> *, const SimParams &,
                                           Complex<float>, Complex<float>, double, double, double);
template void apply_env_grating_factors<float>(DevComplex<float> *, const SimParams &,
                                           Complex<float>, Complex<float>, double, double, double, double, double);
template void apply_sample_factors<float>(DevComplex<float> *, const SimParams &, double, int8_t *,
                                          double, std::size_t, DevComplex<double> *, int, double);
template void scale<float>(DevComplex<float> *, Complex<float>, const int);
template void square_and_downsample<float>(DevComplex<float> *, int, float *, int, double, double,
                                           double, double);
template void analytical_history_row<float>(float *, int, double, double, double, double, double);

// explicit instantiations for double
template void propagate_analytically<double>(DevComplex<double> *, const SimParams &, double,
                                             double);
template __host__ __device__ double fftfreq<double>(int, int, double);
template void propagate_convolve_step<double>(DevComplex<double> *, const SimParams &, double,
                                              double);
template void apply_grating_factors<double>(DevComplex<double> *, const SimParams &,
                                            Complex<double>, Complex<double>, double, double,
                                            double);
template void apply_env_grating_factors<double>(DevComplex<double> *, const SimParams &,
                                            Complex<double>, Complex<double>, double, double,
                                            double, double, double);
template void apply_sample_factors<double>(DevComplex<double> *u, const SimParams &, double,
                                           int8_t *, double, std::size_t, DevComplex<double> *, int,
                                           double);
template void scale<double>(DevComplex<double> *, Complex<double>, const int);
template void square_and_downsample<double>(DevComplex<double> *, int, double *, int, double,
                                            double, double, double);
template void analytical_history_row<double>(double *, int, double, double, double, double, double);

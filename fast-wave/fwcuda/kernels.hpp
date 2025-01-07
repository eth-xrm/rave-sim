// Copyright (c) 2024, ETH Zurich

#ifndef _FWCUDA_KERNELS_HPP
#define _FWCUDA_KERNELS_HPP

#include "types.hpp"

template <typename S>
__global__ void propagate_analytically_kernel(DevComplex<S> *d_u, SimParams params, double x_source,
                                              double z);

template <typename S> [[nodiscard]] __host__ __device__ S fftfreq(int i, int N, S dx);

template <typename S>
__global__ void propagate_convolve_step_kernel(DevComplex<S> *d_U, SimParams params, double dz,
                                               double cutoff_freq);

template <typename S>
__global__ void apply_grating_factors_kernel(DevComplex<S> *d_u, SimParams params,
                                             DevComplex<S> factor_a, DevComplex<S> factor_b,
                                             double pitch, double dc, double x_position);

template <typename S>
__global__ void apply_env_grating_factors_kernel(DevComplex<S> *d_u, SimParams params,
                                             DevComplex<S> factor_a, DevComplex<S> factor_b,
                                             double pitch0, double pitch1, double dc0, double dc1, double x_position);

template <typename S>
__global__ void apply_sample_factors_kernel(DevComplex<S> *d_u, SimParams params, double dz,
                                            int8_t *d_sample, double pixel_size_x,
                                            std::size_t x_len, DevComplex<double> *d_deltabetas,
                                            int z_slice_index, double x_position);

template <typename S>
__global__ void scale_kernel(DevComplex<S> *d_u, DevComplex<S> scale, const int N);

template <typename S>
__global__ void square_and_downsample_kernel(DevComplex<S> *d_u, int N, S *d_out, int outsize,
                                             double detector_pixel_size_x,
                                             double detector_pixel_size_y, double current_z,
                                             double dx);

template <typename S>
__global__ void analytical_history_row_kernel(S *row, int nr_detector_pixels,
                                              double detector_pixel_size_x,
                                              double detector_pixel_size_y, double x_source,
                                              double z, double cutoff_gratient);

#endif // _FWCUDA_KERNELS_HPP

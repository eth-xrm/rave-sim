// Copyright (c) 2024, ETH Zurich

#ifndef _FWCUDA_WARPPERS_HPP
#define _FWCUDA_WARPPERS_HPP

#include "types.hpp"

// This file should be safe to include from non-cuda files.

template <typename S>
void propagate_analytically(DevComplex<S> *d_u, const SimParams &params, double x_source, double z);

template <typename S>
void propagate_convolve_step(DevComplex<S> *d_U, const SimParams &params, double dz,
                             double cutoff_freq);

template <typename S>
void apply_grating_factors(DevComplex<S> *d_u, const SimParams &params, Complex<S> factor_a,
                           Complex<S> factor_b, double pitch, double dc, double x_position);

template <typename S>
void apply_env_grating_factors(DevComplex<S> *d_u, const SimParams &params, Complex<S> factor_a,
                           Complex<S> factor_b, double pitch0, double pitch1, double dc0, double dc1, double x_position);

template <typename S>
void apply_sample_factors(DevComplex<S> *d_u, const SimParams &params, double dz, int8_t *d_sample,
                          double pixel_size_x, std::size_t x_len, DevComplex<double> *d_deltabetas,
                          int z_slice_index, double x_position);

template <typename S> void scale(DevComplex<S> *d_u, Complex<S> scale, const int N);

template <typename S>
void square_and_downsample(DevComplex<S> *d_u, int N, S *d_out, int outsize,
                           double detector_pixel_size_x, double detector_pixel_size_y,
                           double current_z, double dx);

template <typename S>
void analytical_history_row(S *row, int nr_detector_pixels, double detector_pixel_size_x,
                            double detector_pixel_size_y, double x_source, double z,
                            double cutoff_gratient);

#endif // _FWCUDA_WARPPERS_HPP

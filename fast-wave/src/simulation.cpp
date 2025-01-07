// Copyright (c) 2024, ETH Zurich

#include <Npy++.h>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <spdlog/spdlog.h>

#include "fft.hpp"
#include <config_parsing.hpp>
#include <simulation.hpp>

class CudaTimerSection {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

  public:
    CudaTimerSection() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    void start() { cudaEventRecord(start_event); }

    float stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return milliseconds;
    }
};

template <typename S>
void propagate(SimParams const &params, FFT<S> const &fft, double dz, double cutoff_freq,
               DevComplex<S> *d_u, DevComplex<S> *d_U) {
    fft.forward(d_u, d_U);
    propagate_convolve_step<S>(d_U, params, dz, cutoff_freq);
    fft.inverse(d_U, d_u);
}

/// The history class is used to store a downsampled version of the wavefield at multiple
/// z-positions throughout the simulation. This can then be used to visualize the setup.
template <typename S> class History {
    S *d_history;
    std::size_t capacity;
    std::vector<double> z_positions;

  public:
    /// The z-step size in empty space. Inside or at the boundary of optical elements, different
    /// step sizes might apply.
    double dz;
    std::size_t nr_pixels;

    History(std::size_t nr_pixels, std::size_t capacity, double dz)
        : capacity(capacity), dz(dz), nr_pixels(nr_pixels) {
        check_cuda_result("malloc d_sample", cudaMalloc((void **)&(this->d_history),
                                                        capacity * nr_pixels * sizeof(S)));
    }

    /// Add an entry to the history. Returns a pointer to where the downsampled output should be
    /// written to. This expects that exactly `nr_pixels` entries are written, and that the entries
    /// are written before `save()` is called.
    S *new_row(double z_position) {
        const auto nr_rows = this->z_positions.size();
        if (nr_rows >= capacity) {
            throw std::runtime_error("History is full");
        }
        this->z_positions.push_back(z_position);

        return d_history + nr_rows * nr_pixels;
    }

    /// Store the history as a .npy file
    void save(const std::filesystem::path &out_dir) const {
        const auto nr_rows = this->z_positions.size();
        spdlog::info("Saving history with {} rows and {} pixels", nr_rows, nr_pixels);
        const std::size_t nr_scalars = nr_pixels * nr_rows;
        std::vector<S> history(nr_scalars);

        cudaMemcpyAsync(history.data(), this->d_history, nr_scalars * sizeof(S),
                        cudaMemcpyDeviceToHost);
        check_cuda_result("sync after copy history", cudaDeviceSynchronize());
        npypp::Save(out_dir / "history.npy", history, {nr_rows, nr_pixels}, "w");
        npypp::Save(out_dir / "history_z.npy", this->z_positions, {nr_rows}, "w");
    }
};

template <typename S>
void propagate_with_history(SimParams const &params, FFT<S> const &fft, double dz,
                            double cutoff_freq, DevComplex<S> *d_u, DevComplex<S> *d_U,
                            double start_z, std::optional<History<S>> &history) {
    if (history) {
        History<S> &hist = history.value();
        if (dz == 0) {
            return;
        }

        double current_z = start_z;
        while (current_z + hist.dz < start_z + dz) {

            double current_dz = hist.dz;
            // Ensure that we don't have a tiny distance left over for the last step, instead we'd
            // rather extend this last step a bit to the end.
            if (current_z + current_dz * 1.5 >= start_z + dz) {
                current_dz = start_z + dz - current_z;
            }

            propagate<S>(params, fft, current_dz, cutoff_freq, d_u, d_U);
            current_z += current_dz;
            S *d_hist_row = hist.new_row(current_z);
            square_and_downsample<S>(d_u, params.N, d_hist_row, hist.nr_pixels,
                                     params.detector_pixel_size_x, params.detector_pixel_size_y,
                                     current_z, params.dx);
        }

        const double remaining_z = start_z + dz - current_z;
        if (remaining_z > 0) {
            propagate<S>(params, fft, remaining_z, cutoff_freq, d_u, d_U);
            S *d_hist_row = hist.new_row(start_z + dz);
            square_and_downsample<S>(d_u, params.N, d_hist_row, hist.nr_pixels,
                                     params.detector_pixel_size_x, params.detector_pixel_size_y,
                                     start_z + dz, params.dx);
        }

    } else {
        propagate<S>(params, fft, dz, cutoff_freq, d_u, d_U);
    }
}

template <typename S>
void apply_grating(Grating g, DevComplex<S> *d_u, DevComplex<S> *d_U, SimParams const &params,
                   FFT<S> const &fft, double cutoff_freq, std::size_t phase_step) {
    const double dz = g.thickness / g.nr_steps;
    const auto fac_a = Complex<S>(material_factor(g.deltabeta_a, dz, params.wl));
    const auto fac_b = Complex<S>(material_factor(g.deltabeta_b, dz, params.wl));
    for (int i = 0; i < g.nr_steps; ++i) {
        const S dc = g.dc[0] + i * (g.dc[1] - g.dc[0]) / g.nr_steps;

        apply_grating_factors<S>(d_u, params, fac_a, fac_b, g.pitch, dc, g.x_positions[phase_step]);
        propagate<S>(params, fft, dz, cutoff_freq, d_u, d_U);
    }
    if (g.substrate_thickness > 0) {
        const auto fac_substrate =
            Complex<S>(material_factor(g.deltabeta_substrate, g.substrate_thickness, params.wl));

        scale<S>(d_u, fac_substrate, params.N);
        propagate<S>(params, fft, g.substrate_thickness, cutoff_freq, d_u, d_U);
    }
}

template <typename S>
void apply_envgrating(EnvGrating e, DevComplex<S> *d_u, DevComplex<S> *d_U, SimParams const &params,
                   FFT<S> const &fft, double cutoff_freq, std::size_t phase_step) {
    const double dz = e.thickness / e.nr_steps;
    const auto fac_a = Complex<S>(material_factor(e.deltabeta_a, dz, params.wl));
    const auto fac_b = Complex<S>(material_factor(e.deltabeta_b, dz, params.wl));
    for (int i = 0; i < e.nr_steps; ++i) {
        const S dc0 = e.dc0[0] + i * (e.dc0[1] - e.dc0[0]) / e.nr_steps;
        const S dc1 = e.dc1[0] + i * (e.dc1[1] - e.dc1[0]) / e.nr_steps;

        apply_env_grating_factors<S>(d_u, params, fac_a, fac_b, e.pitch0, e.pitch1, dc0, dc1,
                                 e.x_positions[phase_step]);
        propagate<S>(params, fft, dz, cutoff_freq, d_u, d_U);
    }
    if (e.substrate_thickness > 0) {
        const auto fac_substrate =
            Complex<S>(material_factor(e.deltabeta_substrate, e.substrate_thickness, params.wl));

        scale<S>(d_u, fac_substrate, params.N);
        propagate<S>(params, fft, e.substrate_thickness, cutoff_freq, d_u, d_U);
    }
}

template <typename S>
void apply_sample(Sample s, DevComplex<S> *d_u, DevComplex<S> *d_U, SimParams const &params,
                  FFT<S> const &fft, double cutoff_freq, std::size_t phase_step) {
    const double dz = s.pixel_size_z;

    int8_t *d_sample;
    DevComplex<double> *d_deltabetas;
    const std::size_t sample_size_bytes = s.grid.size() * sizeof(int8_t);
    const std::size_t deltabetas_size_bytes = s.deltabetas.size() * sizeof(DevComplex<double>);
    check_cuda_result("malloc d_sample", cudaMalloc((void **)&d_sample, sample_size_bytes));
    check_cuda_result("malloc d_deltabetas",
                      cudaMalloc((void **)&d_deltabetas, deltabetas_size_bytes));

    cudaMemcpyAsync(d_sample, s.grid.data(), sample_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_deltabetas, s.deltabetas.data(), deltabetas_size_bytes,
                    cudaMemcpyHostToDevice);

    for (std::size_t i = 0; i < s.z_len; ++i) {
        apply_sample_factors<S>(d_u, params, dz, d_sample, s.pixel_size_x, s.x_len, d_deltabetas, i,
                                s.x_positions[phase_step]);
        propagate<S>(params, fft, dz, cutoff_freq, d_u, d_U);
    }

    check_cuda_result("sample", cudaPeekAtLastError());
    check_cuda_result("sync", cudaDeviceSynchronize());

    cudaFree(d_sample);
    cudaFree(d_deltabetas);
}

template <typename S>
void apply_optical_element(OpticalElement *el, DevComplex<S> *d_u, DevComplex<S> *d_U,
                           SimParams const &params, FFT<S> const &fft, double cutoff_freq,
                           std::size_t phase_step, std::size_t element_index,
                           std::size_t nr_elements) {
    CudaTimerSection timer;
    check_cuda_result("sync", cudaDeviceSynchronize());
    spdlog::info("Simulating optical element {}/{}", element_index + 1, nr_elements);
    timer.start();
    switch (el->type) {
    case OpticalElementType::Grating:
        apply_grating<S>(*reinterpret_cast<Grating *>(el), d_u, d_U, params, fft, cutoff_freq,
                         phase_step);
        break;
    case OpticalElementType::EnvGrating:
        apply_envgrating<S>(*reinterpret_cast<EnvGrating *>(el), d_u, d_U, params, fft, cutoff_freq,
                         phase_step);
        break;
    case OpticalElementType::Sample:
        apply_sample<S>(*reinterpret_cast<Sample *>(el), d_u, d_U, params, fft, cutoff_freq,
                        phase_step);
        break;
    default:
        throw std::runtime_error("this optical element type is not handled yet");
    }

    const float ms = timer.stop();
    spdlog::info("Elapsed time for optical element: {} ms", ms);
}

/// Check if the configuration is valid and if yes return the total number of
/// phase steps.
std::size_t check_validity(const Config &config) {
    std::size_t nr_phase_steps = 1;

    // accept small z-overlaps to prevent problems in cases where the overlap is just due
    // to float-decimal conversions.
    const double z_tolerance = 1e-8;

    double last_z = 0;
    for (const auto &el : config.optical_elements) {
        if (el->z_start < last_z - z_tolerance) {
            throw std::runtime_error("Optical element starts at z " + std::to_string(el->z_start) +
                                     ", smaller than the previous z " + std::to_string(last_z));
        }

        last_z = el->z_start + el->total_thickness();

        const auto el_steps = el->x_positions.size();
        if (el_steps != nr_phase_steps && nr_phase_steps != 1) {
            throw std::runtime_error("Number of phase steps is not consistent "
                                     "across optical elements: " +
                                     std::to_string(el_steps) + " vs " +
                                     std::to_string(nr_phase_steps));
        }
        nr_phase_steps = std::max(el_steps, nr_phase_steps);
    }

    if (last_z > config.sim_params.z_detector + z_tolerance) {
        throw std::runtime_error("Last optical element ends at z " + std::to_string(last_z) +
                                 ", larger than detector z " +
                                 std::to_string(config.sim_params.z_detector));
    }

    if (config.sim_params.detector_pixel_size_x <= 0 || config.sim_params.detector_size <= 0) {
        throw std::runtime_error("Detector size and pixel size must be positive");
    }

    return nr_phase_steps;
}

template <typename S>
void save_vector(const std::filesystem::path &outpath, DevComplex<S> *d_u, int N) {
    std::vector<Complex<S>> u;
    u.reserve(N);

    cudaMemcpyAsync(u.data(), d_u, N * sizeof(Complex<S>), cudaMemcpyDeviceToHost);
    check_cuda_result("sync after copying u", cudaDeviceSynchronize());

    npypp::Save(outpath, u, {static_cast<std::size_t>(N)}, "w");
}

double get_next_z(const Config &config, std::size_t current_element) {
    if (current_element < config.optical_elements.size() - 1) {
        return config.optical_elements[current_element + 1]->z_start;
    } else {
        return config.sim_params.z_detector;
    }
}

struct Snapshot {
    double z;
    std::size_t element_idx;
};

/// A reasonably close upper bound for how many entries will be pushed to the history. We could make
/// this tight but it's probably not worth the effort.
std::size_t calculate_required_history_capacity(const Config &config, double history_dz) {
    std::size_t history_capacity = 0;

    double first_z = config.sim_params.z_detector;
    if (!config.optical_elements.empty()) {
        first_z = config.optical_elements[0]->z_start;
    }

    // analytical propagation
    history_capacity += first_z / history_dz;

    double current_z = first_z;
    for (std::size_t i = 0; i < config.optical_elements.size(); ++i) {
        const auto el = config.optical_elements[i].get();
        history_capacity += el->nr_history_entries();

        current_z = el->z_start + el->total_thickness();
        const double next_z = get_next_z(config, i);
        const double remaining_z = next_z - current_z;
        if (remaining_z > 0) {
            history_capacity += remaining_z / history_dz + 1;
        }
    }

    return history_capacity;
}

/// Propagate from the source to the first optical element.
template <typename S>
void propagate_source(DevComplex<S> *d_u, DevComplex<S> *d_U, const Source *source,
                      const SimParams &params, const FFT<S> &fft, double cutoff_freq,
                      double next_z) {
    switch (source->type) {
    case SourceType::Point: {
        const PointSource &point_source = *reinterpret_cast<const PointSource *>(source);
        propagate_analytically<S>(d_u, params, point_source.x, next_z);
        // propagate by zero distance just to apply the frequency cutoff
        propagate<S>(params, fft, 0., cutoff_freq, d_u, d_U);
        break;
    }
    case SourceType::Vector: {
        const VectorSource &vector_source = *reinterpret_cast<const VectorSource *>(source);
        const auto loaded = npypp::LoadFull<Complex<S>>(vector_source.input_path, true);
        assert(loaded.shape.size() == 1);
        assert(loaded.shape[0] == static_cast<std::size_t>(params.N));

        // copy the loaded data to the GPU
        cudaMemcpy(d_u, loaded.data.data(), params.N * sizeof(Complex<S>), cudaMemcpyHostToDevice);
        check_cuda_result("sync after copying u", cudaDeviceSynchronize());

        const double propagate_distance = next_z - vector_source.z;
        if (propagate_distance > 0) {
            propagate<S>(params, fft, propagate_distance, cutoff_freq, d_u, d_U);
        }
        break;
    }
    }
}

template <typename S>
void generate_analytical_history(History<S> &history, double dz, double first_z, double x_source,
                                 const SimParams &params, int nr_pixels, double cutoff_angle) {
    const std::size_t nr_steps = first_z / dz;
    const double cutoff_gradient = tan(cutoff_angle);

    for (std::size_t i = 0; i < nr_steps; ++i) {
        const double z = i * dz;
        const auto row = history.new_row(z);
        analytical_history_row<S>(row, nr_pixels, params.detector_pixel_size_x,
                                  params.detector_pixel_size_y, x_source, z, cutoff_gradient);
    }
}

template <typename S>
void run_simulation_inner(const Config &config, const std::filesystem::path &sub_dir,
                          std::optional<double> history_dz) {
    spdlog::info("Running simulation {}", sub_dir.string());

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    const double z_tolerance = 1e-8;

    const std::size_t nr_phase_steps = check_validity(config);
    const std::size_t nr_pixels =
        config.sim_params.detector_size / config.sim_params.detector_pixel_size_x;

    const std::size_t sim_size_bytes = sizeof(Complex<S>) * config.sim_params.N;
    const std::size_t detector_size_bytes = sizeof(S) * nr_pixels;

    DevComplex<S> *d_u;
    DevComplex<S> *d_U;
    DevComplex<S> *d_u_snapshot;
    S *d_detector_output;
    check_cuda_result("malloc d_u", cudaMalloc((void **)&d_u, sim_size_bytes));
    check_cuda_result("malloc d_U", cudaMalloc((void **)&d_U, sim_size_bytes));
    check_cuda_result("malloc detector output",
                      cudaMalloc((void **)&d_detector_output, detector_size_bytes));

    std::optional<History<S>> hist = std::nullopt;
    if (history_dz) {
        hist = History<S>(nr_pixels, calculate_required_history_capacity(config, *history_dz),
                          *history_dz);
    }

    if (nr_phase_steps > 1) {
        check_cuda_result("malloc snapshot", cudaMalloc((void **)&d_u_snapshot, sim_size_bytes));
    }
    std::vector<S> detector_output(nr_pixels * nr_phase_steps);

    FFT<S> fft(config.sim_params.N);

    double cutoff_freq =
        convert_cutoff_angle_to_frequency(config.cutoff_angles[0], config.sim_params.wl);

    double current_z = config.sim_params.z_detector;
    if (!config.optical_elements.empty()) {
        current_z = config.optical_elements[0]->z_start;
    }

    if (hist && config.source.get()->type == SourceType::Point) {
        const PointSource &point_source =
            *reinterpret_cast<const PointSource *>(config.source.get());
        generate_analytical_history<S>(*hist, *history_dz, current_z, point_source.x,
                                       config.sim_params, nr_pixels, config.cutoff_angles[0]);
    }

    propagate_source(d_u, d_U, config.source.get(), config.sim_params, fft, cutoff_freq, current_z);

    // save_vector<S>(sub_dir / ("keypoint_after_pa.npy"), d_u, config.sim_params.N);

    std::optional<Snapshot> opt_snapshot = std::nullopt;

    for (std::size_t i = 0; i < config.optical_elements.size(); ++i) {
        const auto el = config.optical_elements[i].get();

        if (el->x_positions.size() > 1 && !opt_snapshot) {
            opt_snapshot = Snapshot{el->z_start, i};
            cudaMemcpy(d_u_snapshot, d_u, sim_size_bytes, cudaMemcpyDeviceToDevice);
        }

        cutoff_freq =
            convert_cutoff_angle_to_frequency(config.cutoff_angles[i + 1], config.sim_params.wl);

        // uncomment those lines to save keypoints of the current u vector throughout the simulation
        // save_vector<S>(sub_dir / ("keypoint_cuda_" + std::to_string(i) + "_0.npy"), d_u,
        //                config.sim_params.N);
        apply_optical_element(el, d_u, d_U, config.sim_params, fft, cutoff_freq, 0, i,
                              config.optical_elements.size());

        current_z = el->z_start + el->total_thickness();
        // save_vector<S>(sub_dir / ("keypoint_cuda_" + std::to_string(i) + "_1.npy"), d_u,
        //                config.sim_params.N);

        const double next_z = get_next_z(config, i);
        const double remaining_z = next_z - current_z;
        if (remaining_z > z_tolerance) {
            propagate_with_history<S>(config.sim_params, fft, remaining_z, cutoff_freq, d_u, d_U,
                                      current_z, hist);
            current_z = next_z;
        }
    }

    std::vector<Complex<S>> u;
    if (config.save_final_u_vectors) {
        u.resize(config.sim_params.N);
        cudaMemcpyAsync(u.data(), d_u, sim_size_bytes, cudaMemcpyDeviceToHost);
    }
    // While the u vector is being copied over we can already compute the
    // downsampled vector.

    square_and_downsample<S>(d_u, config.sim_params.N, d_detector_output, nr_pixels,
                             config.sim_params.detector_pixel_size_x,
                             config.sim_params.detector_pixel_size_y, config.sim_params.z_detector,
                             config.sim_params.dx);

    cudaMemcpyAsync(&detector_output[0], d_detector_output, detector_size_bytes,
                    cudaMemcpyDeviceToHost);

    if (config.save_final_u_vectors) {
        check_cuda_result("sync before save u", cudaDeviceSynchronize());

        npypp::Save(sub_dir / "u_0000.npy", u, {static_cast<std::size_t>(config.sim_params.N)},
                    "w");
    }

    if (opt_snapshot) {
        for (std::size_t phase_step = 1; phase_step < nr_phase_steps; ++phase_step) {
            check_cuda_result("sync before running phase step", cudaDeviceSynchronize());
            spdlog::info("Running phase step {}/{}", phase_step + 1, nr_phase_steps);

            if (phase_step < nr_phase_steps - 1) {
                cudaMemcpy(d_u, d_u_snapshot, sim_size_bytes, cudaMemcpyDeviceToDevice);
            } else {
                std::swap(d_u, d_u_snapshot);
            }

            current_z = opt_snapshot->z;
            for (std::size_t i = opt_snapshot->element_idx; i < config.optical_elements.size();
                 ++i) {
                const auto el = config.optical_elements[i].get();

                cutoff_freq = convert_cutoff_angle_to_frequency(config.cutoff_angles[i + 1],
                                                                config.sim_params.wl);

                apply_optical_element(el, d_u, d_U, config.sim_params, fft, cutoff_freq, phase_step,
                                      i, config.optical_elements.size());

                current_z = el->z_start + el->total_thickness();

                const double next_z = get_next_z(config, i);
                const double remaining_z = next_z - current_z;
                if (remaining_z > z_tolerance) {
                    propagate<S>(config.sim_params, fft, remaining_z, cutoff_freq, d_u, d_U);
                    current_z = next_z;
                }
            }

            if (config.save_final_u_vectors) {
                cudaMemcpyAsync(u.data(), d_u, sim_size_bytes, cudaMemcpyDeviceToHost);
            }

            square_and_downsample<S>(d_u, config.sim_params.N, d_detector_output, nr_pixels,
                                     config.sim_params.detector_pixel_size_x,
                                     config.sim_params.detector_pixel_size_y,
                                     config.sim_params.z_detector, config.sim_params.dx);
            cudaMemcpyAsync(&detector_output[nr_pixels * phase_step], d_detector_output,
                            detector_size_bytes, cudaMemcpyDeviceToHost);

            if (config.save_final_u_vectors) {
                check_cuda_result("sync before save u", cudaDeviceSynchronize());

                npypp::Save(sub_dir / ("u_" + zeropad(phase_step, 4) + ".npy"), u,
                            {static_cast<std::size_t>(config.sim_params.N)}, "w");
            }
        }
    }

    if (hist) {
        // todo: this history is transposed compared to the big-wave history
        (*hist).save(sub_dir);
    }

    check_cuda_result("sync before saving detector output", cudaDeviceSynchronize());
    npypp::Save(sub_dir / "detected.npy", detector_output, {nr_phase_steps, nr_pixels}, "w");

    std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start_time;
    spdlog::info("Simulation finished in {} seconds", duration.count());
}

void run_simulation(const Config &config, const std::filesystem::path &sub_dir,
                    std::optional<double> history_dz) {
    if (config.dtype == DType::C8) {
        run_simulation_inner<float>(config, sub_dir, history_dz);
    } else if (config.dtype == DType::C16) {
        run_simulation_inner<double>(config, sub_dir, history_dz);
    }
}

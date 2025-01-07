// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_SIMULATION_HPP
#define _FAST_WAVE_SIMULATION_HPP

#include <filesystem>
#include <memory>
#include <optional>
#include <types.hpp>
#include <vector>

#include "optical_element.hpp"
#include <wrappers.hpp>

namespace fs = std::filesystem;

/// planck constant in mˆ2 kg / s
const double h = 6.62607004e-34;
/// Vacuum speed of light in m / s
const double c_0 = 299792458.0;
/// equivalent energy of one electron volt in jules
const double eV_to_joule = 1.602176634e-19;

enum class SourceType {
    Point,
    Vector,
};

struct Source {
    SourceType type;
};

struct PointSource: public Source {
    double x;
    double z;
};

struct VectorSource: public Source {
    fs::path input_path;
    double z;
};

enum class DType {
    C8,
    C16,
};

struct Config {
    SimParams sim_params;
    std::vector<std::unique_ptr<OpticalElement>> optical_elements;
    std::unique_ptr<Source> source;
    DType dtype;
    bool save_final_u_vectors;
    std::vector<double> cutoff_angles;
};

[[nodiscard]] inline Complex<double> material_factor(Complex<double> deltabeta, double thickness,
                                                     double wl) {
    const Complex<double> exponent = deltabeta * Complex<double>(0, 2.0 * M_PI * thickness / wl);

    return std::exp(exponent);
}

[[nodiscard]] inline double convert_cutoff_angle_to_frequency(double angle, double wl) {
    return std::sin(angle) / wl;
}

/// Given an energy in eV, calculate the wavelength in metres or vice versa
[[nodiscard]] inline double convert_energy_wavelength(double energy_or_wavelength) {
    return h * c_0 / (energy_or_wavelength * eV_to_joule);
}

void run_simulation(const Config &config, const std::filesystem::path &sub_dir,
                    std::optional<double> history_dz);

#endif // _FAST_WAVE_SIMULATION_HPP

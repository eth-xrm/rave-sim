// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_CONFIG_PARSING_HPP
#define _FAST_WAVE_CONFIG_PARSING_HPP

#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "simulation.hpp"

namespace fs = std::filesystem;

using DeltabetaTable = std::vector<std::pair<Material, Complex<double>>>;

[[nodiscard]] Complex<double> db_table_lookup(const Material &mat, const DeltabetaTable &db_table);

[[nodiscard]] Grating parse_grating(const YAML::Node &node, const DeltabetaTable &db_table);

[[nodiscard]] EnvGrating parse_envgrating(const YAML::Node &node, const DeltabetaTable &db_table);

[[nodiscard]] Sample parse_sample(const YAML::Node &node, const DeltabetaTable &db_table,
                                  const fs::path &config_path);

[[nodiscard]] std::unique_ptr<OpticalElement> parse_optical_element(const YAML::Node &node,
                                                                    const DeltabetaTable &db_table,
                                                                    const fs::path &sim_dir);

[[nodiscard]] std::vector<std::unique_ptr<OpticalElement>>
parse_optical_elements(const YAML::Node &node, const DeltabetaTable &db_table,
                       const fs::path &sim_dir);

[[nodiscard]] DeltabetaTable parse_deltabeta_table(const YAML::Node &node);

[[nodiscard]] SimParams parse_sim_params(const YAML::Node &node, double wl);

[[nodiscard]] PointSource parse_point_source(const YAML::Node &node);

[[nodiscard]] DType parse_dtype(const YAML::Node &node);

[[nodiscard]] std::vector<double> parse_cutoff_angles(const YAML::Node &node);

[[nodiscard]] fs::path get_subdir(const fs::path &sim_dir, int source_idx);

[[nodiscard]] Config parse_config(const fs::path &sim_dir, int source_idx);

std::string zeropad(int number, std::size_t length);

#endif // _FAST_WAVE_CONFIG_PARSING_HPP

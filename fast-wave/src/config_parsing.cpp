// Copyright (c) 2024, ETH Zurich

#include <Npy++.h>

#include <config_parsing.hpp>

std::string stringify_key(const char *key) { return std::string(key); }

std::string stringify_key(int key) { return std::to_string(key); }

template <typename Key> [[nodiscard]] double get_scalar(YAML::Node const &node, const Key &key) {
    if (!node[key].IsDefined()) {
        throw std::runtime_error("Value " + stringify_key(key) + " is not defined");
    }
    if (node[key].IsScalar()) {
        return node[key].template as<double>();
    } else {
        throw std::runtime_error("Value " + stringify_key(key) + " is not a scalar");
    }
}

template <typename Key>
[[nodiscard]] Material get_material(YAML::Node const &node, const Key &key) {
    if (!node[key].IsDefined()) {
        throw std::runtime_error("Value " + stringify_key(key) + " is not defined");
    }
    if (!node[key].IsSequence()) {
        throw std::runtime_error("Value " + stringify_key(key) + " is not a sequence");
    }
    if (!node[key][0].IsDefined()) {
        throw std::runtime_error("First component (material name) of material " +
                                 stringify_key(key) + " is missing");
    }
    if (!node[key][1].IsScalar()) {
        throw std::runtime_error("Second component (density) of material " + stringify_key(key) +
                                 " is missing");
    }

    return Material{node[key][0].template as<std::string>(), get_scalar(node[key], 1)};
}

[[nodiscard]] Complex<double> db_table_lookup(const Material &mat, const DeltabetaTable &db_table) {
    using SCPair = std::pair<double, Complex<double>>;

    std::vector<SCPair> candidates;
    candidates.reserve(db_table.size());

    for (const auto &pair : db_table) {
        if (pair.first.name == mat.name) {
            candidates.emplace_back(std::abs(pair.first.density - mat.density), pair.second);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const SCPair &a, const SCPair &b) { return a.first < b.first; });

    if (candidates.empty()) {
        throw std::runtime_error("Material " + mat.name + " not found in deltabeta_table");
    }
    if (candidates[0].first > 1e-5) {
        throw std::runtime_error("Material " + mat.name +
                                 " has no matching density in deltabeta_table");
    }
    return candidates[0].second;
}

template <typename Key>
[[nodiscard]] Complex<double> parse_optional_material(const YAML::Node &node, const Key &key,
                                                      const DeltabetaTable &db_table) {
    if (!node[key].IsDefined()) {
        throw std::runtime_error("Material " + stringify_key(key) + " is not defined");
    }
    if (node[key].size() == 0) {
        return Complex<double>{};
    }
    return db_table_lookup(get_material(node, key), db_table);
}

[[nodiscard]] Grating parse_grating(const YAML::Node &node, const DeltabetaTable &db_table) {
    const auto z_start = get_scalar(node, "z_start");
    const auto dc = node["dc"].as<std::array<double, 2>>();
    const auto thickness = get_scalar(node, "thickness");
    const auto pitch = get_scalar(node, "pitch");
    const auto nr_steps = node["nr_steps"].as<int>();
    auto x_positions = node["x_positions"].as<std::vector<double>>();
    const auto substrate_thickness = get_scalar(node, "substrate_thickness");

    const auto deltabeta_a = parse_optional_material(node, "mat_a", db_table);
    const auto deltabeta_b = parse_optional_material(node, "mat_b", db_table);
    const auto deltabeta_substrate = parse_optional_material(node, "mat_substrate", db_table);

    Grating g;
    g.z_start = z_start;
    g.x_positions = std::move(x_positions);
    g.type = OpticalElementType::Grating;
    g.thickness = thickness;
    g.pitch = pitch;
    g.dc = dc;
    g.nr_steps = nr_steps;
    g.substrate_thickness = substrate_thickness;
    g.deltabeta_a = deltabeta_a;
    g.deltabeta_b = deltabeta_b;
    g.deltabeta_substrate = deltabeta_substrate;

    return g;
}

[[nodiscard]] EnvGrating parse_envgrating(const YAML::Node &node, const DeltabetaTable &db_table) {
    const auto z_start = get_scalar(node, "z_start");
    const auto dc0 = node["dc0"].as<std::array<double, 2>>();
    const auto dc1 = node["dc1"].as<std::array<double, 2>>();
    const auto thickness = get_scalar(node, "thickness");
    const auto pitch0 = get_scalar(node, "pitch0");
    const auto pitch1 = get_scalar(node, "pitch1");
    const auto nr_steps = node["nr_steps"].as<int>();
    auto x_positions = node["x_positions"].as<std::vector<double>>();
    const auto substrate_thickness = get_scalar(node, "substrate_thickness");

    const auto deltabeta_a = parse_optional_material(node, "mat_a", db_table);
    const auto deltabeta_b = parse_optional_material(node, "mat_b", db_table);
    const auto deltabeta_substrate = parse_optional_material(node, "mat_substrate", db_table);

    EnvGrating e;
    e.z_start = z_start;
    e.x_positions = std::move(x_positions);
    e.type = OpticalElementType::EnvGrating;
    e.thickness = thickness;
    e.pitch0 = pitch0;
    e.pitch1 = pitch1;
    e.dc0 = dc0;
    e.dc1 = dc1;
    e.nr_steps = nr_steps;
    e.substrate_thickness = substrate_thickness;
    e.deltabeta_a = deltabeta_a;
    e.deltabeta_b = deltabeta_b;
    e.deltabeta_substrate = deltabeta_substrate;

    return e;
}

[[nodiscard]] Sample parse_sample(const YAML::Node &node, const DeltabetaTable &db_table,
                                  const fs::path &sim_dir) {
    const auto z_start = get_scalar(node, "z_start");
    const auto pixel_size_x = get_scalar(node, "pixel_size_x");
    const auto pixel_size_z = get_scalar(node, "pixel_size_z");
    auto x_positions = node["x_positions"].as<std::vector<double>>();

    std::vector<Complex<double>> deltabetas = {{0., 0.}};
    const auto materials_node = node["materials"];
    for (std::size_t i = 0; i < materials_node.size(); ++i) {
        const auto mat = get_material(materials_node, i);
        deltabetas.emplace_back(db_table_lookup(mat, db_table));
    }

    const std::string grid_path = sim_dir / node["grid_path"].as<std::string>();
    npypp::MultiDimensionalArray<uint8_t> arr = npypp::LoadFull<uint8_t>(grid_path, false);

    Sample s;
    s.z_start = z_start;
    s.x_positions = std::move(x_positions);
    s.type = OpticalElementType::Sample;
    s.pixel_size_x = pixel_size_x;
    s.pixel_size_z = pixel_size_z;
    s.deltabetas = std::move(deltabetas);
    s.grid = std::move(arr.data);
    s.z_len = arr.shape[0];
    s.x_len = arr.shape[1];
    return s;
}

[[nodiscard]] std::unique_ptr<OpticalElement> parse_optical_element(const YAML::Node &node,
                                                                    const DeltabetaTable &db_table,
                                                                    const fs::path &sim_dir) {
    const auto type = node["type"].as<std::string>();
    if (type == "grating") {
        return std::make_unique<Grating>(parse_grating(node, db_table));
    } else if (type == "env_grating") {
        return std::make_unique<EnvGrating>(parse_envgrating(node, db_table));
    } else if (type == "sample") {
        return std::make_unique<Sample>(parse_sample(node, db_table, sim_dir));
    } else {
        throw std::runtime_error("Unknown optical element type: " + type);
    }
}

[[nodiscard]] std::vector<std::unique_ptr<OpticalElement>>
parse_optical_elements(const YAML::Node &node, const DeltabetaTable &db_table,
                       const fs::path &sim_dir) {
    std::vector<std::unique_ptr<OpticalElement>> elements;
    for (std::size_t i = 0; i < node.size(); ++i) {
        elements.push_back(parse_optical_element(node[i], db_table, sim_dir));
    }
    return elements;
}

[[nodiscard]] DeltabetaTable parse_deltabeta_table(const YAML::Node &node) {
    DeltabetaTable db_table;
    for (const auto &entry : node) {
        const auto mat = get_material(entry, 0);
        const Complex<double> db(get_scalar(entry[1], 0), get_scalar(entry[1], 1));

        db_table.emplace_back(mat, db);
    }
    return db_table;
}

[[nodiscard]] SimParams parse_sim_params(const YAML::Node &node, double wl) {
    const auto N = node["N"].as<int>();
    const auto dx = get_scalar(node, "dx");
    const auto z_detector = get_scalar(node, "z_detector");
    const auto detector_size = get_scalar(node, "detector_size");
    const auto detector_pixel_size_x = get_scalar(node, "detector_pixel_size_x");
    const auto detector_pixel_size_y = get_scalar(node, "detector_pixel_size_y");
    return SimParams{N, dx, z_detector, detector_size, detector_pixel_size_x, detector_pixel_size_y, wl};
}

[[nodiscard]] PointSource parse_point_source(const YAML::Node &node) {
    const auto x = get_scalar(node, "x");
    const auto z = get_scalar(node, "z");
    PointSource s;
    s.type = SourceType::Point;
    s.x = x;
    s.z = z;
    return s;
}

[[nodiscard]] VectorSource parse_vector_source(const YAML::Node &node) {
    const auto input_path = node["input_path"].as<std::string>();
    const auto z = get_scalar(node, "z");
    VectorSource s;
    s.type = SourceType::Vector;
    s.input_path = std::move(input_path);
    s.z = z;
    return s;
}

[[nodiscard]] std::unique_ptr<Source> parse_source(const YAML::Node &node) {
    const auto type = node["type"].as<std::string>();

    if (type == "point") {
        return std::make_unique<PointSource>(parse_point_source(node));
    } else if (type == "vector") {
        return std::make_unique<VectorSource>(parse_vector_source(node));
    } else {
        throw std::runtime_error("unknown source type: " + type);
    }
}

[[nodiscard]] DType parse_dtype(const YAML::Node &node) {
    if (!node.IsDefined()) {
        throw std::runtime_error("dtype is not defined in config");
    }

    const auto dtype = node.as<std::string>();
    if (dtype == "c8") {
        return DType::C8;
    } else if (dtype == "c16") {
        return DType::C16;
    } else {
        throw std::runtime_error("Unknown dtype: " + dtype);
    }
}

[[nodiscard]] std::vector<double> parse_cutoff_angles(const YAML::Node &node) {
    std::vector<double> angles;
    for (std::size_t i = 0; i < node.size(); ++i) {
        angles.push_back(get_scalar(node, i));
    }
    return angles;
}

std::string zeropad(int number, std::size_t length) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(length) << number;
    return ss.str();
}

[[nodiscard]] fs::path get_subdir(const fs::path &sim_dir, int source_idx) {
    return sim_dir / zeropad(source_idx, 8);
}

[[nodiscard]] Config parse_config(const fs::path &sim_dir, int source_idx) {
    const auto config_path = sim_dir / "config.yaml";
    const auto subconfig_path = get_subdir(sim_dir, source_idx) / "subconfig.yaml";
    const auto computed_path = sim_dir / "computed.yaml";

    YAML::Node config_node = YAML::LoadFile(config_path.string());
    YAML::Node subconfig_node = YAML::LoadFile(subconfig_path.string());
    YAML::Node computed_node = YAML::LoadFile(computed_path.string());

    const DType dtype = parse_dtype(config_node["dtype"]);
    const double energy = get_scalar(subconfig_node, "energy");

    const SimParams sim_params =
        parse_sim_params(config_node["sim_params"], convert_energy_wavelength(energy));
    const auto db_table = parse_deltabeta_table(subconfig_node["deltabeta_table"]);
    auto optical_elements = parse_optical_elements(config_node["elements"], db_table, sim_dir);
    auto source = parse_source(subconfig_node["source"]);

    auto cutoff_angles = parse_cutoff_angles(computed_node["cutoff_angles"]);

    const bool save_final_u_vectors = config_node["save_final_u_vectors"].as<bool>();

    return Config{
        sim_params, std::move(optical_elements), std::move(source),
        dtype,      save_final_u_vectors,        std::move(cutoff_angles),
    };
}

# MINCO-Python GCOPTER 绑定 MVP 方案

## 用户需求整理

基于用户需求，我们需要：
1. **自定义 flatness 模型**：用户完全控制 flatness 实现，通过 YAML 文件配置
2. **自定义约束惩罚**：用户完全控制约束惩罚逻辑
3. **现代 C++17 接口**：清晰的输入输出，避免输出参数
4. **Python 友好绑定**：易于使用的 Python API

## 核心设计

### Flatness 接口设计

```cpp
namespace minco::flatness {

template <typename T>
concept ForwardQuery = requires(T value) {
    { value.velocity } -> std::convertible_to<Eigen::Vector3d>;
    { value.acceleration } -> std::convertible_to<Eigen::Vector3d>;
    { value.jerk } -> std::convertible_to<Eigen::Vector3d>;
};

template <typename T>
concept ForwardResult = requires(T value) {
    { value.thrust } -> std::convertible_to<double>;
    { value.quaternion } -> std::convertible_to<Eigen::Vector4d>;
    { value.angular_velocity } -> std::convertible_to<Eigen::Vector3d>;
};

template <typename T>
concept BackwardQuery = requires(T value) {
    { value.position_gradient } -> std::convertible_to<Eigen::Vector3d>;
    { value.velocity_gradient } -> std::convertible_to<Eigen::Vector3d>;
    { value.thrust_gradient } -> std::convertible_to<double>;
    { value.quaternion_gradient } -> std::convertible_to<Eigen::Vector4d>;
    { value.angular_velocity_gradient } -> std::convertible_to<Eigen::Vector3d>;
};

template <typename T>
concept BackwardResult = requires(T value) {
    { value.position_total_gradient } -> std::convertible_to<Eigen::Vector3d>;
    { value.velocity_total_gradient } -> std::convertible_to<Eigen::Vector3d>;
    { value.acceleration_total_gradient } -> std::convertible_to<Eigen::Vector3d>;
    { value.jerk_total_gradient } -> std::convertible_to<Eigen::Vector3d>;
};

template <typename Model>
concept FlatnessModel = requires(Model model,
                                 const typename Model::ConfigType &config,
                                 const typename Model::ForwardQuery &fwd_query,
                                 const typename Model::BackwardQuery &bwd_query) {
    ForwardQuery<typename Model::ForwardQuery>;
    ForwardResult<typename Model::ForwardResult>;
    BackwardQuery<typename Model::BackwardQuery>;
    BackwardResult<typename Model::BackwardResult>;

    { model.configure(config) } -> std::same_as<void>;
    { model.configure_from_file(std::declval<std::string>()) } -> std::same_as<void>;
    { model.forward(fwd_query) } -> std::same_as<typename Model::ForwardResult>;
    { model.backward(bwd_query) } -> std::same_as<typename Model::BackwardResult>;
};

} // namespace minco::flatness
```

```cpp
template <minco::flatness::FlatnessModel FlatnessModelT>
class GCOPTER_PolytopeSFC {
   public:
    using FlatnessModel      = FlatnessModelT;
    using ForwardQuery   = typename FlatnessModel::ForwardQuery;
    using ForwardResult  = typename FlatnessModel::ForwardResult;
    using BackwardQuery  = typename FlatnessModel::BackwardQuery;
    using BackwardResult = typename FlatnessModel::BackwardResult;

    FlatnessModel &flatness() { return flatness_model_; }

    void configure_from_file(const std::string &file_path)
    {
        const std::string path = file_path.empty() ? kDefaultGcopterConfigPath
                                                   : file_path;
        auto node = YAML::LoadFile(path);
        flatness_model_.configure_from_file(path);
        cost_config_.configure_from_node(node);
        const auto gcopter_node = node["gcopter"];
        if (gcopter_node && gcopter_node["yaw_smooth"])
        {
            yaw_smooth_ = gcopter_node["yaw_smooth"].as<double>();
        }
        else
        {
            yaw_smooth_ = 1.0e-6;
        }
    }

   private:
    static constexpr const char *kDefaultGcopterConfigPath =
        "config/default_gcopter.yaml";
    FlatnessModel flatness_model_{};
    double        yaw_smooth_{1.0e-6};
    CostConfig   &cost_config_{CostConfig::getInstance()};
};
```


### 约束惩罚接口设计

```cpp
template <minco::flatness::FlatnessModel FlatnessModel>
class ConstraintPenaltyFunctional {
   public:
    void compute(
        const Eigen::VectorXd &times,
        const Eigen::MatrixX3d &coefficients,
        double smooth_factor,
        int integral_resolution,
        const CostConfig &config,
        FlatnessModel &flatness,
        double &total_cost,
        Eigen::VectorXd &time_gradient,
        Eigen::MatrixX3d &coefficient_gradient);
};
```

## 样本文件结构

### 1. 默认 Flatness 实现 (包装现有 `flatness::FlatnessMap`)

```cpp
// flatness_default.hpp
struct DefaultFlatness {
    struct Parameters {
        std::array<double, 6> data{1.0, 9.81, 0.0, 0.0, 0.0, 1.0e-3};
    };

    struct ForwardQueryType {
        Eigen::Vector3d velocity;
        Eigen::Vector3d acceleration;
        Eigen::Vector3d jerk;
        double          yaw      = 0.0;
        double          yaw_rate = 0.0;
    };

    struct ForwardResultType {
        double          thrust           = 0.0;
        Eigen::Vector4d quaternion       = Eigen::Vector4d::Zero();
        Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    };

    struct BackwardQueryType {
        Eigen::Vector3d position_gradient      = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity_gradient      = Eigen::Vector3d::Zero();
        double          thrust_gradient        = 0.0;
        Eigen::Vector4d quaternion_gradient    = Eigen::Vector4d::Zero();
        Eigen::Vector3d angular_velocity_gradient = Eigen::Vector3d::Zero();
    };

    struct BackwardResultType {
        Eigen::Vector3d position_total_gradient     = Eigen::Vector3d::Zero();
        Eigen::Vector3d velocity_total_gradient     = Eigen::Vector3d::Zero();
        Eigen::Vector3d acceleration_total_gradient = Eigen::Vector3d::Zero();
        Eigen::Vector3d jerk_total_gradient         = Eigen::Vector3d::Zero();
        double          yaw_gradient                = 0.0;
        double          yaw_rate_gradient           = 0.0;
    };

    using ConfigType    = Parameters;
    using ForwardQuery  = ForwardQueryType;
    using ForwardResult = ForwardResultType;
    using BackwardQuery = BackwardQueryType;
    using BackwardResult = BackwardResultType;

    void configure(const minco::flatness::Config &config)
    {
        config_ = config;
        map_.reset(config_.mass, config_.gravity, config_.horizontal_drag,
                   config_.vertical_drag, config_.parasitic_drag,
                   config_.speed_smooth);
    }


    ForwardResult forward(const ForwardQuery &query)
    {
        ForwardResult result;
        map_.forward(query.velocity, query.acceleration, query.jerk,
                     query.yaw, query.yaw_rate, result.thrust,
                     result.quaternion, result.angular_velocity);
        return result;
    }

    BackwardResult backward(const BackwardQuery &query)
    {
        BackwardResult result;
        map_.backward(query.position_gradient, query.velocity_gradient,
                      query.thrust_gradient, query.quaternion_gradient,
                      query.angular_velocity_gradient,
                      result.position_total_gradient,
                      result.velocity_total_gradient,
                      result.acceleration_total_gradient,
                      result.jerk_total_gradient, result.yaw_gradient,
                      result.yaw_rate_gradient);
        return result;
    }

    ConfigType config_{};
    ::flatness::FlatnessMap map_;
};
```

### 2. 配置示例文件

```yaml
# config/default_gcopter.yaml
flatness:
  mass: 1.0                    # 质量 (kg)
  gravity: 9.81                # 重力加速度 (m/s²)
  horizontal_drag: 0.10        # 水平阻力系数
  vertical_drag: 0.10          # 垂直阻力系数
  parasitic_drag: 0.01         # 寄生阻力系数
  speed_smooth: 1.0e-3         # 速度平滑因子

cost:
  v_max: 5.0
  omg_x_max: 1.0
  omg_y_max: 2.0
  omg_z_max: 1.0
  acc_max: 50.0
  thrust_min: -20.0
  thrust_max: 20.0
  pos_weight: 1.0
  vel_weight: 0.0
  acc_weight: 0.0
  omg_x_weight: 0.0
  omg_y_weight: 0.0
  omg_z_weight: 0.0
  thrust_weight: 0.0
  time_weight: 0.0
  omg_consistent_weight: 0.0

gcopter:
  yaw_smooth: 1.0e-6
```


## 实现步骤

### 阶段 1：核心接口实现
1. 在 C++ 中定义现代接口
2. 提供基于现有代码的默认实现
3. 修改 GCOPTER_PolytopeSFC 使用接口

### 阶段 2：Python 绑定
1. 使用 pybind11 绑定核心类
2. 实现 Python 接口包装器
3. 支持 NumPy 数组转换

### 阶段 3：配置系统
1. 支持 YAML 配置文件
2. 提供配置验证工具
3. 添加配置示例

## 关键特性

1. **完全自定义**：用户完全控制 flatness 和约束惩罚实现
2. **配置驱动**：通过 YAML 文件管理配置
3. **现代接口**：清晰的输入输出，无输出参数
4. **Python 友好**：易于使用的 Python API
5. **高性能**：保持 C++ 后端性能

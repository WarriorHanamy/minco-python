# MINCO-Python GCOPTER 绑定 MVP 方案

## 用户需求整理

基于用户需求，我们需要：
1. **自定义 flatness 模型**：用户完全控制 flatness 实现，通过 YAML 文件配置
2. **自定义约束惩罚**：用户完全控制约束惩罚逻辑
3. **现代 C++17 接口**：清晰的输入输出，避免输出参数
4. **Python 友好绑定**：易于使用的 Python API

## 核心设计

### Flatness 接口设计

#### 1. Flatness 协议概念

```cpp
// 这不是具体的类型，而是概念定义
namespace flatness_interface {

// ForwardInput 概念：前向传播输入必须满足的接口
template<typename T>
concept ForwardInput = requires(T input) {
    { input.vel } -> std::convertible_to<Eigen::Vector3d>;
    { input.acc } -> std::convertible_to<Eigen::Vector3d>;
    { input.jerk } -> std::convertible_to<Eigen::Vector3d>;
};

// ForwardResult 概念：前向传播输出必须满足的接口
template<typename T>
concept ForwardResult = requires(T result) {
    { result.quat } -> std::convertible_to<Eigen::Vector4d>;
    { result.ang_vel } -> std::convertible_to<Eigen::Vector3d>;
};

// BackwardInput 概念：反向传播输入必须满足的接口
template<typename T>
concept BackwardInput = requires(T input) {
// NULL 
};

// BackwardResult 概念：反向传播输出必须满足的接口
template<typename T>
concept BackwardResult = requires(T result) {
    { result.pos_grad } -> std::convertible_to<Eigen::Vector3d>;
    { result.vel_grad } -> std::convertible_to<Eigen::Vector3d>;
    { result.acc_grad } -> std::convertible_to<Eigen::Vector3d>;
    { result.jerk_gradient } -> std::convertible_to<Eigen::Vector3d>;
};

} // namespace flatness_interface
```

#### 2. Flatness 接口（基于概念）

```cpp
class FlatnessInterface {
public:
    virtual ~FlatnessInterface() = default;

    // 协议方法：用户自定义类型必须满足概念要求
    template<flatness_interface::ForwardInput InputType,
             flatness_interface::ForwardResult ResultType>
    virtual ResultType forward(const InputType& input) = 0;

    template<flatness_interface::BackwardInput InputType,
             flatness_interface::BackwardResult ResultType>
    virtual ResultType backward(const InputType& input) = 0;

    virtual bool isConfigured() const = 0;
};
```


### 约束惩罚接口设计

```cpp
class ConstraintPenaltyInterface {
public:
    virtual ~ConstraintPenaltyInterface() = default;

    // 计算约束惩罚和梯度
    virtual void computePenalty(
        const Eigen::VectorXd& times,              // 时间分配
        const Eigen::MatrixX3d& coefficients,      // 轨迹系数
        const double smooth_factor,                // 平滑因子
        const int integral_resolution,             // 积分分辨率
        const CostConfig& config,                  // 成本配置
        FlatnessInterface& flatness,               // flatness 模型
        double& total_cost,                        // 总惩罚成本
        Eigen::VectorXd& time_gradient,            // 时间梯度
        Eigen::MatrixX3d& coefficient_gradient     // 系数梯度
    ) = 0;
};
```

## 样本文件结构

### 1. 默认 Flatness 实现 (基于现有代码)

```cpp
// default_flatness.hpp
class DefaultFlatness : public FlatnessInterface {
private:
    // 从 YAML 文件加载的配置
    double mass_ = 1.0;
    double gravity_ = 9.81;
    double horizontal_drag_ = 0.0;
    double vertical_drag_ = 0.0;
    double parasitic_drag_ = 0.0;
    double speed_smooth_ = 1e-3;

    bool configured_ = false;

public:
    // 从 YAML 文件加载配置
    bool loadFromYAML(const std::string& config_path);

    bool isConfigured() const override { return configured_; }

    ForwardResult forward(const InputState& input) override {
        // 基于现有 flatness.hpp 的实现
        // 但返回清晰的结构体

        // 现有计算逻辑...
        double thrust = ...;
        Eigen::Vector4d quat = ...;
        Eigen::Vector3d omg = ...;

        return ForwardResult(thrust, quat, omg);
    }

    BackwardResult backward(const BackwardInput& input) override {
        // 基于现有 backward 实现
        // 返回清晰的结构体

        Eigen::Vector3d pos_grad = ...;
        Eigen::Vector3d vel_grad = ...;
        Eigen::Vector3d acc_grad = ...;
        Eigen::Vector3d jerk_grad = ...;
        double yaw_grad = ...;
        double yaw_rate_grad = ...;

        return BackwardResult(pos_grad, vel_grad, acc_grad, jerk_grad, yaw_grad, yaw_rate_grad);
    }
};
```

### 2. 配置示例文件

```yaml
# config/flatness_config.yaml
flatness:
  mass: 1.2                    # 质量 (kg)
  gravity: 9.81                # 重力加速度 (m/s²)
  horizontal_drag: 0.1         # 水平阻力系数
  vertical_drag: 0.05          # 垂直阻力系数
  parasitic_drag: 0.01         # 寄生阻力系数
  speed_smooth: 1e-3           # 速度平滑因子

constraints:
  max_velocity: 5.0            # 最大速度 (m/s)
  max_acceleration: 10.0       # 最大加速度 (m/s²)
  max_thrust: 20.0             # 最大推力 (N)
  min_thrust: 2.0              # 最小推力 (N)
```

### 3. Python 绑定示例

```python
# minco_trajectory/gcopter.py
import numpy as np
from typing import Tuple, Optional

class FlatnessInterface:
    """Python Flatness 接口"""

    def forward(self, velocity: np.ndarray, acceleration: np.ndarray,
                jerk: np.ndarray, yaw_angle: float, yaw_rate: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        前向传播

        Args:
            velocity: 速度向量 (3,)
            acceleration: 加速度向量 (3,)
            jerk: 加加速度向量 (3,)
            yaw_angle: 偏航角 (rad)
            yaw_rate: 偏航角速度 (rad/s)

        Returns:
            tuple: (thrust, quaternion, angular_velocity)
        """
        pass

    def backward(self, position_gradient: np.ndarray, velocity_gradient: np.ndarray,
                 thrust_gradient: float, quaternion_gradient: np.ndarray,
                 angular_velocity_gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        反向传播

        Args:
            position_gradient: 位置梯度 (3,)
            velocity_gradient: 速度梯度 (3,)
            thrust_gradient: 推力梯度
            quaternion_gradient: 四元数梯度 (4,)
            angular_velocity_gradient: 角速度梯度 (3,)

        Returns:
            tuple: (position_total_grad, velocity_total_grad, acceleration_total_grad,
                   jerk_total_grad, yaw_grad, yaw_rate_grad)
        """
        pass

class GCOPTEROptimizer:
    """GCOPTER 优化器"""

    def __init__(self):
        self._optimizer = None  # C++ 后端
        self._flatness = None
        self._constraint_penalty = None

    def set_flatness(self, flatness: FlatnessInterface):
        """设置自定义 flatness 模型"""
        self._flatness = flatness

    def set_constraint_penalty(self, penalty):
        """设置自定义约束惩罚"""
        self._constraint_penalty = penalty

    def optimize(self, initial_pva: np.ndarray, terminal_pva: np.ndarray,
                 corridors: list, rel_cost_tol: float = 1e-6) -> dict:
        """执行轨迹优化"""
        pass
```

### 4. 使用示例

```python
# example_custom_flatness.py
import numpy as np
import yaml
from minco_trajectory import GCOPTEROptimizer, FlatnessInterface

class CustomFlatness(FlatnessInterface):
    def __init__(self, config_path: str):
        # 从 YAML 文件加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['flatness']

    def forward(self, velocity, acceleration, jerk, yaw_angle, yaw_rate):
        # 基于配置参数实现自定义 flatness 逻辑
        mass = self.config['mass']
        gravity = self.config['gravity']

        # 自定义计算逻辑
        thrust = mass * np.linalg.norm(acceleration) + mass * gravity
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 简单四元数
        angular_velocity = np.array([0.0, 0.0, yaw_rate])

        return thrust, quaternion, angular_velocity

    def backward(self, position_gradient, velocity_gradient, thrust_gradient,
                 quaternion_gradient, angular_velocity_gradient):
        # 自定义反向传播逻辑
        position_total_grad = position_gradient.copy()
        velocity_total_grad = velocity_gradient.copy()
        acceleration_total_grad = np.zeros(3)
        jerk_total_grad = np.zeros(3)
        yaw_grad = 0.0
        yaw_rate_grad = 0.0

        return position_total_grad, velocity_total_grad, acceleration_total_grad, jerk_total_grad, yaw_grad, yaw_rate_grad

# 使用示例
if __name__ == "__main__":
    # 创建优化器
    optimizer = GCOPTEROptimizer()

    # 设置自定义 flatness
    custom_flatness = CustomFlatness("config/flatness_config.yaml")
    optimizer.set_flatness(custom_flatness)

    # 设置问题
    initial_pva = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    terminal_pva = np.array([[10, 10, 5], [0, 0, 0], [0, 0, 0]])

    # 执行优化
    result = optimizer.optimize(initial_pva, terminal_pva, corridors=[])
    print(f"优化完成，最终成本: {result['cost']}")
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

这个设计完全满足用户需求，提供了灵活的自定义能力，同时保持了代码的清晰性和性能。
# Goal
这个项目来完成python-first的多旋翼/固定翼轨迹生成，基于[MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER.git)。

原始的MINCO包含：
1. 前端飞行走廊生成；
2. 基于多旋翼的forward-flatness 和 backward-flatness的model来进行平坦空间的轨迹生成；
3. 纯ROS，cpp的项目。

为了便捷动力学轨迹生成这一单一任务，为其他learning-based control服务，这个项目的改进要点为：
1. 去除前端飞行走廊部分
2. flatness前向方向部分基于casadi来进行代码生成，集成。
3. 暴露python接口，用户不应体验cpp的细节。

## WARNING
make sure you are familiar with ros2 rules.


## pybind11接口设计
1. setup_intial_trajectory
    Args:
        1. headPVAJ, tailPVAJ, initTimeAllooc, init
    Params:
        1. lbfgs setting



## Config设计
1. costfunc_config
2. lbgfs_config
3. sfc_config




## Installation
首先同步python依赖
```shell
uv sync
```

python工具链自动构建cpp-pybind11
```shell
uv pip install -e . --no-deps
```

生成pybind11静态类型提示
```shell
uv run pybind11-stubgen minco
```
然后将生成的stub/*.pyi 移动至与*.so同级别路径下。

（似乎最新配置的uv sync直接一步到位了？？？）
（TODO：应该移动到 site-packages下面）
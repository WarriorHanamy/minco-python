from __future__ import annotations

from typing import Tuple

import numpy as np

import minco
import matplotlib

matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


from matplotlib.patches import Patch


def _cube_polytope() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, -1.0, -1.0],
        ]
    )


def visualize_3d_planes(
    h_poly: np.ndarray,
    plot_size: Tuple[float, float] = (-1.5, 1.5),
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    Visualize a set of 3D planes defined by the inequality a*x + b*y + c*z + d <= 0

    Parameters:
    -----------
    h_poly : numpy.ndarray
        Array of shape (n, 4) where each row is [a, b, c, d] representing a plane
    plot_size : tuple
        Range for the x, y, and z axes
    figsize : tuple
        Size of the matplotlib figure
    """
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # 为绘图定义一个统一的坐标范围
    plot_range = np.linspace(plot_size[0], plot_size[1], 10)

    # 定义颜色和图例代理
    colors = ["gray", "gray", "gray", "gray", "gray", "gray"]
    legend_patches = []

    # 遍历并绘制每一个平面
    for i, plane in enumerate(h_poly):
        a, b, c, d = plane
        color = colors[i % len(colors)]
        label = f"Plane {i + 1}"

        # 为图例创建一个代理对象，因为plot_surface本身不支持label
        legend_patches.append(Patch(color=color, label=label, alpha=0.5))

        # --- 核心修正：在循环内部根据平面方向创建网格 ---
        if c != 0:
            # 标准平面, z = f(x, y)，在 x-y 平面创建网格
            xx, yy = np.meshgrid(plot_range, plot_range)
            zz = (-a * xx - b * yy - d) / c
            ax.plot_surface(xx, yy, zz, alpha=0.5, color=color)
        elif b != 0:
            # 垂直于 y 轴的平面, y = f(x, z)，在 x-z 平面创建网格
            xx, zz = np.meshgrid(plot_range, plot_range)
            yy = (-a * xx - c * zz - d) / b
            ax.plot_surface(xx, yy, zz, alpha=0.5, color=color)
        elif a != 0:
            # 垂直于 x 轴的平面, x = f(y, z)，在 y-z 平面创建网格
            yy, zz = np.meshgrid(plot_range, plot_range)
            xx = (-b * yy - c * zz - d) / a
            ax.plot_surface(xx, yy, zz, alpha=0.5, color=color)
        else:
            # 无效的平面方程 [0, 0, 0, d]
            print(f"Skipping invalid plane: {plane}")
            continue

    # 设置标签和标题
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Planes Visualization")

    # 设置统一的坐标轴范围，防止图像变形
    ax.set_xlim(plot_size)
    ax.set_ylim(plot_size)
    ax.set_zlim(plot_size)

    # 使用代理对象显示图例
    ax.legend(handles=legend_patches)
    ax.view_init(elev=20, azim=45)  # 调整视角

    plt.tight_layout()
    plt.show()


def test_cube_polytope_vertices_and_overlap() -> None:
    h_poly = _cube_polytope()

    ok, interior = minco.geo_utils.find_interior(h_poly)
    assert ok
    np.testing.assert_allclose(interior, np.array([0.0, 0.0, 0.0]), atol=1e-6)

    vertices = minco.geo_utils.enumerate_vertices(h_poly, interior)
    # print(f"vertices:\n{vertices.mT}")
    ok_auto, auto_vertices = minco.geo_utils.enumerate_vertices_auto(h_poly)
    assert ok_auto
    np.testing.assert_allclose(auto_vertices, vertices)
    # print(f"vertices:\n{vertices.mT}")

    decimal_precision = 8  # 1e-8
    obtained = {
        tuple(np.round(vertices[:, i], decimal_precision))
        for i in range(vertices.shape[1])
    }
    expected = {
        (1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
    }
    assert obtained == expected

    h_poly_up_1 = np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, -2.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )

    h_poly_z_up_almost2 = np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, -1.0 - 1.9],
            [0.0, 0.0, -1.0, -1.0 + 1.9],
        ]
    )
    h_poly_z_up_2 = np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, -1.0 - 2.0],
            [0.0, 0.0, -1.0, -1.0 + 2.0],
        ]
    )
    h_poly_z_up_3 = np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, -1.0 - 3.0],
            [0.0, 0.0, -1.0, -1.0 + 3.0],
        ]
    )

    assert minco.geo_utils.overlap(h_poly, h_poly_up_1)
    assert minco.geo_utils.overlap(h_poly, h_poly_z_up_almost2)
    assert not minco.geo_utils.overlap(h_poly, h_poly_z_up_2)
    assert not minco.geo_utils.overlap(h_poly, h_poly_z_up_3)


if __name__ == "__main__":
    visualize_3d_planes(_cube_polytope())

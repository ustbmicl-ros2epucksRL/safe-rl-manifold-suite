# 示例：如何实现圆形和楔形编队
# 基于 RMPflow 框架

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from rmp import RMPRoot, RMPNode
from rmp_leaf import FormationDecentralized, GoalAttractorUni, CollisionAvoidanceDecentralized


def create_circular_formation_distances(N, radius):
    """
    创建圆形编队的期望距离矩阵
    
    参数:
        N: 机器人数量
        radius: 圆形编队的半径
    
    返回:
        dists: NxN 矩阵，dists[i,j] 是机器人 i 和 j 之间的期望距离
    """
    # 计算每个机器人在圆上的角度
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / N)
    
    # 计算每个机器人在圆上的坐标
    coords = np.stack((np.cos(theta) * radius, np.sin(theta) * radius), axis=1)
    
    # 计算所有机器人对之间的距离
    dists = squareform(pdist(coords))
    
    return dists


def create_wedge_formation_distances(N, base_width, depth):
    """
    创建楔形（V形）编队的期望距离矩阵
    
    参数:
        N: 机器人数量（奇数）
        base_width: 底边宽度
        depth: 深度（从底边到顶点的距离）
    
    返回:
        dists: NxN 矩阵，dists[i,j] 是机器人 i 和 j 之间的期望距离
    """
    assert N % 2 == 1, "楔形编队需要奇数个机器人"
    
    coords = []
    # 顶点（leader）
    coords.append([0, depth])
    
    # 两侧的机器人
    for i in range(1, (N + 1) // 2):
        offset = base_width * i / ((N + 1) // 2)
        coords.append([-offset, 0])  # 左侧
        coords.append([offset, 0])    # 右侧
    
    coords = np.array(coords)
    
    # 计算所有机器人对之间的距离
    dists = squareform(pdist(coords))
    
    return dists


def create_formation_rmp_tree(N, formation_type='circular', **formation_params):
    """
    创建编队控制的 RMP 树
    
    参数:
        N: 机器人数量
        formation_type: 'circular' 或 'wedge'
        **formation_params: 编队参数
            - 圆形: radius
            - 楔形: base_width, depth
    """
    r = RMPRoot('root')
    robots = []
    
    # 创建机器人节点
    def create_mappings(i):
        phi = lambda y, i=i: np.array([[y[2 * i, 0]], [y[2 * i + 1, 0]]])
        J = lambda y, i=i: np.concatenate((
                np.zeros((2, 2 * i)),
                np.eye(2),
                np.zeros((2, 2 * (N - i - 1)))), axis=1)
        J_dot = lambda y, y_dot: np.zeros((2, 2 * N))
        return phi, J, J_dot
    
    for i in range(N):
        phi, J, J_dot = create_mappings(i)
        robot = RMPNode('robot_' + str(i), r, phi, J, J_dot)
        robots.append(robot)
    
    # 创建编队控制节点
    fcs = []
    
    if formation_type == 'circular':
        radius = formation_params.get('radius', 1.0)
        dists = create_circular_formation_distances(N, radius)
    elif formation_type == 'wedge':
        base_width = formation_params.get('base_width', 1.0)
        depth = formation_params.get('depth', 1.0)
        dists = create_wedge_formation_distances(N, base_width, depth)
    else:
        raise ValueError(f"Unknown formation type: {formation_type}")
    
    # 为每对机器人创建编队控制节点
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            fc = FormationDecentralized(
                'fc_robot_' + str(i) + '_robot_' + str(j),
                robots[i],
                robots[j],
                d=dists[i, j],  # 使用计算出的期望距离
                gain=1,
                eta=2,
                w=10)
            fcs.append(fc)
    
    # 添加碰撞避免（可选，但推荐）
    iacas = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            iaca = CollisionAvoidanceDecentralized(
                'ca_robot_' + str(i) + '_robot_' + str(j),
                robots[i],
                robots[j],
                R=0.3,  # 安全距离
                eta=1)
            iacas.append(iaca)
    
    return r, robots, fcs, iacas


# 使用示例
if __name__ == "__main__":
    N = 5
    
    # 创建圆形编队
    r_circle, robots_circle, fcs_circle, iacas_circle = create_formation_rmp_tree(
        N, 
        formation_type='circular',
        radius=2.0
    )
    
    # 创建楔形编队
    r_wedge, robots_wedge, fcs_wedge, iacas_wedge = create_formation_rmp_tree(
        N,
        formation_type='wedge',
        base_width=1.5,
        depth=2.0
    )
    
    print("圆形编队 RMP 树已创建")
    print("楔形编队 RMP 树已创建")


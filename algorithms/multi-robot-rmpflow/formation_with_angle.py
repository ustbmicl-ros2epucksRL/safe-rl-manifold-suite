# 带角度控制的编队节点
# 可以实现更精确的编队形状（如圆形、楔形等）

import numpy as np
from numpy.linalg import norm
from rmp_leaf import RMPLeaf


class FormationWithAngle(RMPLeaf):
    """
    带距离和角度控制的编队节点
    可以同时控制两个机器人之间的距离和相对角度
    """
    
    def __init__(self, name, parent, parent_param, 
                 d=1.0, theta_desired=0.0, 
                 gain_distance=1.0, gain_angle=1.0, 
                 eta=2.0, w=1.0):
        """
        参数:
            d: 期望距离
            theta_desired: 期望相对角度（弧度）
            gain_distance: 距离控制增益
            gain_angle: 角度控制增益
        """
        assert parent_param is not None
        self.d = d
        self.theta_desired = theta_desired
        self.gain_distance = gain_distance
        self.gain_angle = gain_angle
        
        psi = None
        J = None
        J_dot = None
        
        def RMP_func(x, x_dot):
            """
            x: [距离误差, 角度误差]
            x_dot: [距离误差变化率, 角度误差变化率]
            """
            # 分离距离和角度误差
            x_dist = x[0, 0]
            x_angle = x[1, 0]
            x_dot_dist = x_dot[0, 0]
            x_dot_angle = x_dot[1, 0]
            
            # 距离控制（与 FormationDecentralized 相同）
            G_dist = w
            grad_Phi_dist = gain_distance * x_dist * w
            Bx_dot_dist = eta * w * x_dot_dist
            f_dist = -grad_Phi_dist - Bx_dot_dist
            
            # 角度控制
            G_angle = w
            grad_Phi_angle = gain_angle * x_angle * w
            Bx_dot_angle = eta * w * x_dot_angle
            f_angle = -grad_Phi_angle - Bx_dot_angle
            
            # 组合
            M = np.eye(2) * w
            f = np.array([[f_dist], [f_angle]])
            
            return (f, M)
        
        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)
    
    def update_params(self):
        """更新映射函数"""
        z = self.parent_param.x  # 另一个机器人的位置
        z_dot = self.parent_param.x_dot
        
        c = z
        d = self.d
        theta_d = self.theta_desired
        
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        
        N = c.size
        
        def psi(y):
            """计算距离误差和角度误差"""
            # 相对位置向量
            rel_pos = y - c
            r = norm(rel_pos)
            
            # 距离误差
            dist_error = r - d
            
            # 角度误差（相对角度 - 期望角度）
            if r > 1e-6:
                # 计算当前相对角度
                theta_current = np.arctan2(rel_pos[1, 0], rel_pos[0, 0])
                angle_error = theta_current - theta_d
                # 归一化到 [-pi, pi]
                angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
            else:
                angle_error = 0.0
            
            return np.array([[dist_error], [angle_error]])
        
        def J(y):
            """雅可比矩阵：2xN"""
            rel_pos = y - c
            r = norm(rel_pos)
            
            if r < 1e-6:
                # 如果距离太近，使用单位矩阵
                J_dist = np.zeros((1, N))
                J_angle = np.zeros((1, N))
            else:
                # 距离的梯度
                J_dist = (1.0 / r) * rel_pos.T
                
                # 角度的梯度
                # theta = arctan2(y[1]-c[1], y[0]-c[0])
                # dtheta/dy = [-sin(theta)/r, cos(theta)/r]
                J_angle = (1.0 / (r ** 2)) * np.array([[-rel_pos[1, 0], rel_pos[0, 0]]])
            
            return np.concatenate([J_dist, J_angle], axis=0)
        
        def J_dot(y, y_dot):
            """雅可比的时间导数"""
            rel_pos = y - c
            rel_vel = y_dot - z_dot
            r = norm(rel_pos)
            
            if r < 1e-6:
                return np.zeros((2, N))
            
            # 距离的雅可比时间导数
            r_dot = np.dot(rel_pos.T, rel_vel) / r
            J_dist_dot = (1.0 / r) * rel_vel.T - (r_dot / (r ** 2)) * rel_pos.T
            
            # 角度的雅可比时间导数
            # 更复杂的计算，这里简化处理
            J_angle_dot = np.zeros((1, N))
            
            return np.concatenate([J_dist_dot, J_angle_dot], axis=0)
        
        self.psi = psi
        self.J = J
        self.J_dot = J_dot


class CircularFormationNode(RMPLeaf):
    """
    专门用于圆形编队的节点
    控制机器人相对于编队中心的位置
    """
    
    def __init__(self, name, parent, center_param, 
                 radius=1.0, angle_desired=0.0,
                 gain_radius=1.0, gain_angle=1.0,
                 eta=2.0, w=1.0):
        """
        参数:
            center_param: 编队中心的节点（可以是虚拟的或某个leader机器人）
            radius: 期望半径
            angle_desired: 期望角度（在圆上的位置）
        """
        assert center_param is not None
        self.radius = radius
        self.angle_desired = angle_desired
        self.gain_radius = gain_radius
        self.gain_angle = gain_angle
        
        psi = None
        J = None
        J_dot = None
        
        def RMP_func(x, x_dot):
            """x: [半径误差, 角度误差]"""
            x_radius = x[0, 0]
            x_angle = x[1, 0]
            x_dot_radius = x_dot[0, 0]
            x_dot_angle = x_dot[1, 0]
            
            # 半径控制
            f_radius = -gain_radius * x_radius * w - eta * w * x_dot_radius
            
            # 角度控制
            f_angle = -gain_angle * x_angle * w - eta * w * x_dot_angle
            
            M = np.eye(2) * w
            f = np.array([[f_radius], [f_angle]])
            
            return (f, M)
        
        RMPLeaf.__init__(self, name, parent, center_param, psi, J, J_dot, RMP_func)
    
    def update_params(self):
        """更新映射函数"""
        center = self.parent_param.x
        
        if center.ndim == 1:
            center = center.reshape(-1, 1)
        
        N = center.size
        r_desired = self.radius
        theta_d = self.angle_desired
        
        def psi(y):
            """计算相对于中心的半径和角度误差"""
            rel_pos = y - center
            r = norm(rel_pos)
            
            radius_error = r - r_desired
            
            if r > 1e-6:
                theta_current = np.arctan2(rel_pos[1, 0], rel_pos[0, 0])
                angle_error = theta_current - theta_d
                angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
            else:
                angle_error = 0.0
            
            return np.array([[radius_error], [angle_error]])
        
        def J(y):
            """雅可比矩阵"""
            rel_pos = y - center
            r = norm(rel_pos)
            
            if r < 1e-6:
                J_radius = np.zeros((1, N))
                J_angle = np.zeros((1, N))
            else:
                J_radius = (1.0 / r) * rel_pos.T
                J_angle = (1.0 / (r ** 2)) * np.array([[-rel_pos[1, 0], rel_pos[0, 0]]])
            
            return np.concatenate([J_radius, J_angle], axis=0)
        
        def J_dot(y, y_dot):
            """雅可比时间导数"""
            return np.zeros((2, N))
        
        self.psi = psi
        self.J = J
        self.J_dot = J_dot


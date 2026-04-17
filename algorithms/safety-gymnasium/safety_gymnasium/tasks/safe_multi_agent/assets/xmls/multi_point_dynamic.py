# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================
"""Generate ``multi_point`` MuJoCo XML for an arbitrary number of planar point agents."""

from __future__ import annotations

# Two-agent colors match legacy multi_point.xml; extra agents get distinct rgba.
_POINT_RGBA = [
    (0.7412, 0.0431, 0.1843, 1.0),
    (0.0039, 0.1529, 0.3961, 1.0),
    (0.2, 0.6, 0.2, 1.0),
    (0.6, 0.4, 0.1, 1.0),
    (0.5, 0.2, 0.5, 1.0),
    (0.1, 0.5, 0.55, 1.0),
    (0.55, 0.35, 0.15, 1.0),
    (0.35, 0.35, 0.35, 1.0),
]


def multi_point_mjcf(num_agents: int) -> str:
    """Return mujoco XML string with ``num_agents`` point robots (2 actuators each: x vel, yaw vel)."""
    if num_agents < 1:
        raise ValueError('num_agents must be >= 1')

    bodies: list[str] = []
    sensor_lines: list[str] = []
    act_lines: list[str] = []

    for i in range(num_agents):
        if i == 0:
            bname = 'agent'
            jx, jy, jz = 'x', 'y', 'z'
            cam = 'vision'
            site = 'agent'
            act_x, act_z = 'x', 'z'
            arrow_name = 'pointarrow'
            sns = (
                ('accelerometer', 'accelerometer'),
                ('velocimeter', 'velocimeter'),
                ('gyro', 'gyro'),
                ('magnetometer', 'magnetometer'),
                ('subtreecom', 'subtreecom'),
                ('subtreelinvel', 'subtreelinvel'),
                ('subtreeangmom', 'subtreeangmom'),
            )
        else:
            suf = str(i)
            bname = f'agent{suf}'
            jx, jy, jz = f'x{suf}', f'y{suf}', f'z{suf}'
            cam = f'vision{suf}'
            site = bname
            act_x, act_z = f'x{suf}', f'z{suf}'
            arrow_name = f'pointarrow{suf}'
            sns = (
                ('accelerometer', f'accelerometer{suf}'),
                ('velocimeter', f'velocimeter{suf}'),
                ('gyro', f'gyro{suf}'),
                ('magnetometer', f'magnetometer{suf}'),
                ('subtreecom', f'subtreecom{suf}'),
                ('subtreelinvel', f'subtreelinvel{suf}'),
                ('subtreeangmom', f'subtreeangmom{suf}'),
            )

        r, g, b, a = _POINT_RGBA[i % len(_POINT_RGBA)]
        bodies.append(
            f"""
    <body name="{bname}" pos="0 0 .1">
      <camera name="{cam}" pos="0 0 .15" xyaxes="0 -1 0 .4 0 1" fovy="90"/>
      <joint type="slide" axis="1 0 0" name="{jx}" damping="0.01"/>
      <joint type="slide" axis="0 1 0" name="{jy}" damping="0.01"/>
      <joint type="hinge" axis="0 0 1" name="{jz}" damping="0.005"/>
      <geom name="{bname}" type="sphere" size=".1" friction="1 0.01 0.01"
          rgba="{r} {g} {b} {a}"/>
      <geom name="{arrow_name}" pos="0.1 0 0" size="0.05 0.05 0.05" type="box"
          rgba="{r} {g} {b} {a}"/>
      <site name="{site}" rgba="1 0 0 .1"/>
    </body>"""
        )

        for tag, sname in sns:
            if tag in ('subtreecom', 'subtreelinvel', 'subtreeangmom'):
                sensor_lines.append(f'    <{tag} body="{bname}" name="{sname}"/>')
            else:
                sensor_lines.append(f'    <{tag} site="{site}" name="{sname}"/>')

        act_lines.append(
            f'    <velocity gear="0.3 0 0 0 0 0" site="{site}" name="{act_x}"/>'
        )
        act_lines.append(f'    <velocity gear="0.3" jointinparent="{jz}" name="{act_z}"/>')

    return f"""<mujoco>
  <size njmax="3000" nconmax="1000"/>
  <option timestep="0.002"/>
  <default>
    <geom condim="6" density="1" rgba="0.7412 0.0431 0.1843 1"/>
    <joint damping=".001"/>
    <motor ctrlrange="-1 1" ctrllimited="true" forcerange="-.05 .05" forcelimited="true"/>
    <velocity ctrlrange="-1 1" ctrllimited="true" forcerange="-.05 .05" forcelimited="true"/>
    <site size="0.032" type="sphere"/>
  </default>
  <worldbody>
    <geom name="floor" size="5 5 0.1" type="plane" condim="6"/>
{''.join(bodies)}
  </worldbody>
  <sensor>
{chr(10).join(sensor_lines)}
  </sensor>
  <actuator>
{chr(10).join(act_lines)}
  </actuator>
</mujoco>"""

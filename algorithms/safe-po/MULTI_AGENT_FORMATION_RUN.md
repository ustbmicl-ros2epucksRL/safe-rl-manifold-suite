# 多智能体编队实验

在仓库根目录先安装一次包，然后用模块方式启动（工作目录可以是任意路径）：

```bash
pip install -e .
python -m safepo.multi_agent.mappo \
  --task SafetyPointMultiFormationGoal0-v0 \
  --num-envs 20 \
  --num_agents 3 \
  --formation-shape wedge
```

若用 RMP 版本，把模块名改成 `safepo.multi_agent.mappo_rmp`，其余参数可相同。

## 这几个参数的含义

| 参数 | 含义 |
|------|------|
| `--task`固定为编队任务 | `SafetyPointMultiFormationGoal0-v0` |
| `--num-envs` | 并行环境数量（rollout 线程数） |
| `--num_agents` | 编队里机器人个数 |
| `--formation-shape` | 编队形状：`mesh` / `line` / `wedge` / `circle` |

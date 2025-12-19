```

## 调用示例

```python
result = sampler.run_sampling_experiment(
    env=env,
    agent_configs=agent_configs,
    experiment_config={...},
    cam_id=0,
    cam_count=3,
    vehicle_zones=vehicle_zones,
    height=800,
    # === 新增参数 ===
    normal_variance_threshold=0.05,
    slope_threshold=0.866,
    safety_margin_cm=50  # 根据物体尺寸动态计算的安全边距
)
```

## 核心优势

✅ **物体特定性**：每个物体都有其独特的腐蚀半径  
✅ **旋转感知**：考虑物体的实际朝向  
✅ **动态约束**：已放置物体也被视为障碍物进行腐蚀  
✅ **逻辑闭环**：几何约束 → VLM决策 → 动态更新  
✅ **可视化**：生成带有自适应掩码的调试图像

这就是完整的自适应几何约束系统！
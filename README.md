# 集合通信模拟器

两个算子独立启动，各自包含模拟器、图形界面和测试，参考图片统一放在 `Docs/`

```text
AllGather/
├── main.py          # GUI / CLI 启动入口
├── simulator.py     # 单 channel、同步轮次的 Ring AllGather
├── gui.py           # 环形拓扑与逐轮数据块
├── tests/           # 算子测试
└── README.md
AllReduce/
├── main.py          # GUI / CLI 启动入口
├── simulator.py     # 多 channel、SimpleBuffer 事件模拟
├── gui.py           # 事件、链路与缓冲区视图
├── tests/           # 算子测试
└── README.md
Docs/                # 参考图片
```

根目录另保留 `README.md`、`.gitignore` 和 `.vscode/` 等仓库配置，各算子的 `__init__.py`、`__main__.py` 支持包导入和模块启动

## 启动

在仓库根目录运行，使用 Python 标准库，无需额外安装依赖，图形界面需要 Tkinter

| 操作 | AllGather | AllReduce |
| --- | --- | --- |
| 图形界面 | `python AllGather/main.py` | `python AllReduce/main.py` |
| 命令行 | `python AllGather/main.py --cli` | `python AllReduce/main.py --cli --quiet` |
| 模块启动 | `python -m AllGather` | `python -m AllReduce` |
| 单独测试 | `python -m unittest discover -s AllGather/tests -t . -v` | `python -m unittest discover -s AllReduce/tests -t . -v` |

也可进入相应算子目录执行 `python main.py`，全部测试在仓库根目录运行：

```powershell
python -m unittest discover -v
```

VS Code 的“运行任务”和调试配置可分别选择两个算子，`Ctrl+Shift+B` 默认启动 AllReduce，解释器路径在 `.vscode/` 中配置

详细说明见 [AllGather](AllGather/README.md) 和 [AllReduce](AllReduce/README.md)，参考图片见 [Docs](Docs/)

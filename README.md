# PCIe P2P RingAllReduce 四 GPU 模拟器

用 Python 容器类管理四张逻辑 GPU，模拟参考图中的 `A → D → C → B → A`。
只用 Python 标准库；本机已找到 Python 3.12.10 和 Tkinter，不需要 pip、CUDA 或真实 GPU。

## 在 VS Code 运行

1. 打开 `D:\RingAllReduce` 文件夹。
2. 按 **Ctrl+Shift+B**，启动图形模拟器。这条任务直接调用本机 Python，无需 Python 扩展。
3. 点击“下一步”查看六轮通信；“上一步”回看已保存状态；“重置”重新开始。
4. 如已安装 Python Debugger 扩展，也可按 **F5**，选择“RingAllReduce 图形模拟器”。

VS Code 集成终端（PowerShell）也可直接执行：

```powershell
& 'C:\Users\xiaoqiandai\AppData\Local\Programs\Python\Python312\python.exe' .\main.py
& 'C:\Users\xiaoqiandai\AppData\Local\Programs\Python\Python312\python.exe' .\main.py --cli --step
& 'C:\Users\xiaoqiandai\AppData\Local\Programs\Python\Python312\python.exe' .\main.py --cli --input examples/input.json --export traces/demo.json
& 'C:\Users\xiaoqiandai\AppData\Local\Programs\Python\Python312\python.exe' -m unittest discover -s tests -v
```

解释器路径是本机配置；换电脑时修改 `.vscode/settings.json`、`.vscode/tasks.json`。

## 容器类与数据流

```text
GPUContainer
├── gpus[rank]: GPU             原始块、工作块、NUMA 标签
├── links[source]: PCIeLink     后继 GPU、假设的路径、发送元素计数
│   └── buffer: RingBuffer      有限槽位、单调 head/tail
├── step()                     推进一个逻辑通信轮次
├── run()                      执行剩余轮次
├── history                    初始状态 + 每轮数据及传输记录
└── verify()                   与逐元素直接求和对比
```

`Chunk` 是不可变的数据块，`contributors` 记录哪些 Rank 已经参与求和。
`reduce_with()` 检查来源集合，防止重复加上同一张卡的数据。

```python
from simulator import GPUContainer, demo_inputs

cluster = GPUContainer(demo_inputs())
round_one = cluster.step()
for event in round_one['transfers']:
    print(event['source'], '->', event['target'], event['payload'])
results = cluster.run()
assert cluster.verify()
print(results[0])
```

默认各卡输入 8 个元素，每块 2 个元素。结果：
`[100, 104, 108, 112, 116, 120, 124, 128]`，四张卡都获得同一数组。

## 六轮调度

每一轮的四条边并行推进。下表标的是该边本轮发送的块编号。

| 轮次 | 阶段 | A→D | D→C | C→B | B→A |
|---|---|---|---|---|---|
| 1 | ReduceScatter | 1 | 0 | 3 | 2 |
| 2 | ReduceScatter | 2 | 1 | 0 | 3 |
| 3 | ReduceScatter | 3 | 2 | 1 | 0 |
| 4 | AllGather | 0 | 3 | 2 | 1 |
| 5 | AllGather | 1 | 0 | 3 | 2 |
| 6 | AllGather | 2 | 1 | 0 | 3 |

第 3 轮结束时 A/B/C/D 分别拥有完整的 0/1/2/3 号结果块。
后 3 轮只复制结果，不再做加法。

`step()` 先捕获四个发送载荷并写入各接收 FIFO，再统一消费并更新 GPU。
因此循环遍历 GPU 的先后顺序不会把本轮刚收到的数据再次作为本轮发送数据。
真实硬件会按更细粒度流水化；这里的轮次边界是为了讲解，并非硬件全局屏障。

## Buffer 与 NUMA

每条有向连接有自己的接收 FIFO，默认 2 个槽，每槽存一块模拟数据。
发送前要求 `tail - head < capacity`，槽位为 `tail % capacity`。
先写数据再推进 tail；接收要求 `head < tail`，消费后推进 head。
GUI 展示“发送前 → 写入后 → 消费后”的计数，能看到循环复用槽位。
主算法每轮将队列排空；满队列等待由 Buffer 的单元测试演示，不模拟线程竞争。

示例 NUMA 放置：A/B 在结点 0，C/D 在结点 1。A→D、C→B 跨 NUMA。
假设这四条边均支持 PCIe P2P；不读取本机拓扑，也不声称跨 NUMA 必然支持 P2P。
可以设置 `cluster.links[0].p2p_enabled = False` 查看明确错误，不会静默改用主机中转。

## 模型边界

- 支持四卡等长非空数值数组；非 4 的倍数会补零，最终结果裁去补零。
- 接受有限实数；整数结果精确比较，浮点结果按容差比较。
- 本模型不量化带宽、时延、PCIe 争用，不模拟真实 DMA、内存屏障或缓存一致性。
- 不实现 NCCL channel/LL128、多进程通信或 GPU kernel；`RingBuffer` 为教学模型。
- 图形界面可能缩略显示长数据块；命令行和 JSON 保留完整值。
- 不需要 GitHub 登录；项目不保存账号、密码或令牌。

## 文件

- `simulator.py`：容器、设备、链路、缓冲区和算法。
- `gui.py`：图形过程展示。
- `main.py`：GUI / CLI 入口。
- `tests/test_simulator.py`：算法、环排列、补零、浮点数、FIFO 和错误情况测试。
- `examples/input.json`：可修改的四卡输入。
- `.vscode/`：运行任务与调试配置。

项目目录原有图片作为参考保留，程序不修改它们。

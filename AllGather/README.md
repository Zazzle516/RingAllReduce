# Ring AllGather

四张 GPU，每卡两个元素，沿 `0 → 3 → 2 → 1 → 0` 传输，经过三轮后每张卡都得到 `[10, 11, 20, 21, 30, 31, 40, 41]`

这是单 channel 的同步轮次教学模型，只复制和转发，不做归约，也不模拟异步缓冲区和真实 GPU 执行

在仓库根目录运行：

```powershell
python AllGather/main.py
python AllGather/main.py --cli
python -m unittest discover -s AllGather/tests -t . -v
```

界面支持下一轮、上一轮和重置，箭头表示刚完成的传输，颜色表示来源 rank，虚线块表示尚未收到，粗边框表示本轮收到的数据块

`simulator.py` 负责逐轮状态与结果，`gui.py` 显示快照，`main.py` 选择 GUI 或 CLI，测试位于 `tests/`

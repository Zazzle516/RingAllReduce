"""Ring AllGather 的四卡逐轮界面。"""
from AllGather.simulator import DEMO_INPUTS, DEMO_RING, gather_rounds


class AllGatherWindow:
    """四卡教学视图：箭头表示本轮传输，卡片显示接收后的状态。"""

    COLORS = ("#dbeafe", "#dcfce7", "#ffedd5", "#ede9fe")

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        root.title("Ring AllGather · 四卡逐轮演示")
        root.geometry("1060x800")
        root.minsize(980, 760)
        toolbar = ttk.Frame(root, padding=12)
        toolbar.pack(fill="x")
        self.previous = ttk.Button(toolbar, text="上一轮", command=self.prev)
        self.previous.pack(side="left", padx=4)
        self.next_button = ttk.Button(toolbar, text="下一轮", command=self.next)
        self.next_button.pack(side="left", padx=4)
        ttk.Button(toolbar, text="重置", command=self.reset).pack(side="left", padx=4)
        self.status = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.status).pack(side="left", padx=20)
        ttk.Label(root, text="单 channel · 每卡两个元素 · 环路径 0 → 3 → 2 → 1 → 0",
                  padding=(16, 2)).pack(anchor="w")
        ttk.Label(root, text="颜色表示来源 rank；虚线块表示未收到；粗边框表示本轮新收到。只复制和转发，不求和。",
                  padding=(16, 4)).pack(anchor="w")
        self.canvas = tk.Canvas(root, background="#f6f8fc", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=8)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.detail = tk.StringVar()
        ttk.Label(root, textvariable=self.detail, padding=(16, 8),
                  justify="left", wraplength=940).pack(anchor="w")
        self.reset()

    def reset(self):
        self.rounds = gather_rounds(DEMO_INPUTS, DEMO_RING)
        self.history = [next(self.rounds)]
        self.view = 0
        self.refresh()

    def next(self):
        if self.view == 3:
            return
        if self.view + 1 == len(self.history):
            self.history.append(next(self.rounds))
        self.view += 1
        self.refresh()

    def prev(self):
        self.view = max(0, self.view - 1)
        self.refresh()

    def refresh(self):
        self.previous.configure(state="normal" if self.view else "disabled")
        self.next_button.configure(state="normal" if self.view < 3 else "disabled")
        self.status.set(f"第 {self.view} / 3 轮" + (" · 完成" if self.view == 3 else ""))
        if self.view == 0:
            self.detail.set("初始状态：每张 GPU 只有自己的数据块。\n点击“下一轮”，各卡同时把自己的块发送给下游。")
        elif self.view < 3:
            self.detail.set("箭头标注刚完成的传输；每个数据块在一轮内只走一跳。\n下一轮，各卡继续转发本轮从上游收到的块。")
        else:
            expected = tuple(value for block in DEMO_INPUTS for value in block)
            assert all(tuple(v for block in blocks for v in block) == expected
                       for blocks in self.history[self.view]["blocks"])
            self.detail.set(f"验证通过：四张 GPU 的结果均为 {list(expected)}\n输出按来源 rank 0、1、2、3 拼接，与环的传输顺序无关。")
        self.draw()

    def draw(self):
        if not hasattr(self, "history"):
            return
        c = self.canvas
        c.delete("all")
        width, height = max(c.winfo_width(), 950), max(c.winfo_height(), 480)
        # 四个位置依次沿顺时针放置，保持环的邻接关系。
        left, right = 192, width - 192
        top, bottom = 102, height - 102
        centers = dict(zip(DEMO_RING, ((left, top), (right, top),
                                      (right, bottom), (left, bottom))))
        state = self.history[self.view]
        transfers = {source: (owner, block) for source, _, owner, block in state["transfers"]}
        fresh = {(target, owner) for _, target, owner, _ in state["transfers"]}
        for position, source in enumerate(DEMO_RING):
            target = DEMO_RING[(position + 1) % 4]
            x1, y1 = centers[source]
            x2, y2 = centers[target]
            if y1 == y2:
                direction = 1 if x2 > x1 else -1
                x1 += direction * 164
                x2 -= direction * 164
                label_x, label_y = (x1 + x2) / 2, y1 - 30
            else:
                direction = 1 if y2 > y1 else -1
                y1 += direction * 81
                y2 -= direction * 81
                label_x, label_y = x1 + (82 if x1 == left else -82), (y1 + y2) / 2
            c.create_line(x1, y1, x2, y2, arrow="last", width=3, fill="#64748b",
                          arrowshape=(12, 14, 5))
            label = "等待首轮"
            if source in transfers:
                owner, block = transfers[source]
                label = f"来源 GPU {owner}\n{list(block)}"
            c.create_text(label_x, label_y, text=label, font=("Microsoft YaHei UI", 10),
                          fill="#334155")

        c.create_text(width / 2, height / 2, text=f"AllGather\n第 {self.view} / 3 轮",
                      font=("Microsoft YaHei UI", 17, "bold"), fill="#334155", justify="center")
        for rank, (x, y) in centers.items():
            c.create_rectangle(x - 164, y - 81, x + 164, y + 81,
                               fill="white", outline="#cbd5e1", width=2)
            c.create_text(x - 148, y - 59, text=f"GPU {rank}", anchor="w",
                          font=("Microsoft YaHei UI", 13, "bold"), fill="#0f172a")
            for owner, block in enumerate(state["blocks"][rank]):
                bx = x - 150 + owner * 76
                received = (rank, owner) in fresh
                c.create_rectangle(bx, y - 30, bx + 70, y + 48,
                                   fill=self.COLORS[owner] if block is not None else "#f8fafc",
                                   outline="#2563eb" if received else "#cbd5e1",
                                   width=3 if received else 1, dash=() if block is not None else (3, 3))
                c.create_text(bx + 35, y - 13, text=f"来源 {owner}",
                              font=("Microsoft YaHei UI", 9), fill="#475569")
                c.create_text(bx + 35, y + 19,
                              text=", ".join(map(str, block)) if block is not None else "未收到",
                              font=("Microsoft YaHei UI", 10), fill="#0f172a")

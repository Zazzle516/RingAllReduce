"""Tkinter visualization. View history without rewinding protocol counters."""
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from simulator import GPUContainer


class SimulatorWindow:
    def __init__(self, root: tk.Tk, container: GPUContainer):
        self.root, self.container = root, container
        self.view = 0
        root.title("PCIe P2P RingAllReduce · 四 GPU 模拟器")
        root.geometry("1120x880")
        root.minsize(920, 760)
        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill="x")
        self.previous = ttk.Button(toolbar, text="上一步", command=self.prev)
        self.previous.pack(side="left", padx=4)
        self.next_button = ttk.Button(toolbar, text="下一步", command=self.next)
        self.next_button.pack(side="left", padx=4)
        ttk.Button(toolbar, text="运行到结束", command=self.finish).pack(side="left", padx=4)
        ttk.Button(toolbar, text="重置", command=self.reset).pack(side="left", padx=4)
        ttk.Button(toolbar, text="导出 JSON", command=self.export).pack(side="left", padx=4)
        self.status = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.status).pack(side="left", padx=16)
        ttk.Label(root, padding=(14, 0), text="CPU 模拟 · 逻辑环 A → D → C → B → A · "
                  "假设所有边支持 P2P · 箭头表示本轮传输，节点表示本轮完成后的状态").pack(anchor="w")
        self.canvas = tk.Canvas(root, background="#f6f8fc", highlightthickness=0, height=456)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=8)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.detail = tk.StringVar()
        ttk.Label(root, textvariable=self.detail, padding=(14, 4)).pack(anchor="w")
        self.link_table = ttk.Treeview(root, columns=("edge", "route", "slot", "before", "write", "read"),
                                       show="headings", height=4)
        for col, title, width in [("edge", "连接", 80), ("route", "NUMA 路径（示意）", 330),
                                  ("slot", "槽位", 55), ("before", "发送前 H/T", 100),
                                  ("write", "写入后 H/T", 100), ("read", "消费后 H/T", 100)]:
            self.link_table.heading(col, text=title)
            self.link_table.column(col, width=width, minwidth=50, stretch=col == "route")
        self.link_table.pack(fill="x", padx=12)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="x")
        self.log = tk.Text(frame, height=6, wrap="word", font=("Consolas", 10))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="x", expand=True)
        scroll.pack(side="right", fill="y")
        self.draw()

    def prev(self):
        self.view = max(0, self.view - 1)
        self.draw()

    def next(self):
        if self.view < len(self.container.history) - 1:
            self.view += 1
        elif not self.container.done:
            self.container.step()
            self.view += 1
        self.draw()

    def finish(self):
        self.container.run()
        self.view = 6
        self.draw()

    def reset(self):
        self.container.reset()
        self.view = 0
        self.draw()

    def export(self):
        filename = filedialog.asksaveasfilename(parent=self.root, defaultextension=".json",
                                                initialfile="ring-trace.json",
                                                filetypes=[("JSON", "*.json")])
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as output:
                    json.dump(self.container.trace(), output, indent=2, ensure_ascii=False)
            except OSError as exc:
                messagebox.showerror("导出失败", str(exc), parent=self.root)

    def draw(self):
        state = self.container.history[self.view]
        self.status.set(f"第 {self.view}/6 轮 · {state['phase']}")
        self.previous.configure(state="normal" if self.view else "disabled")
        self.next_button.configure(state="normal" if self.view < 6 else "disabled")
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 880), max(canvas.winfo_height(), 430)
        bw, bh = min(365, (width - 200) / 2), 182
        left, right = 42, width - 42 - bw
        top, bottom = 18, height - bh - 18
        # The layout follows ring positions, so other valid permutations work too.
        positions = [(left, top), (right, top), (right, bottom), (left, bottom)]
        edge_data = [
            ((left+bw+8, top+bh/2, right-8, top+bh/2), (width/2, top+bh/2-24)),
            ((right+bw/2, top+bh+7, right+bw/2, bottom-7), (right+bw/2+125, (top+bh+bottom)/2)),
            ((right-8, bottom+bh/2, left+bw+8, bottom+bh/2), (width/2, bottom+bh/2-24)),
            ((left+bw/2, bottom-7, left+bw/2, top+bh+7), (left+bw/2-125, (top+bh+bottom)/2)),
        ]
        for pos, (line, label) in enumerate(edge_data):
            source = self.container.ring[pos]
            link = self.container.links[source]
            cross = self.container.numa_nodes[source] != self.container.numa_nodes[link.target]
            canvas.create_line(*line, arrow="last", width=2, fill="#a05b18" if cross else "#46637f")
            event = next((t for t in state['transfers'] if t['source'] == source), None)
            text = (f"块 {event['chunk']}\n" if event else "") + ("跨 NUMA" if cross else "同 NUMA")
            canvas.create_text(*label, text=text, fill="#344155", font=("Microsoft YaHei UI", 10))
        for pos, rank in enumerate(self.container.ring):
            gpu = state['gpus'][str(rank)]
            x, y = positions[pos]
            canvas.create_rectangle(x, y, x+bw, y+bh, fill="white", outline="#b8c4d5")
            canvas.create_text(x+bw/2, y+17,
                               text=f"GPU {gpu['name']} / Rank {rank} / NUMA {gpu['numa']}",
                               fill="#25344b", font=("Microsoft YaHei UI", 11, "bold"))
            for k, chunk in enumerate(gpu['chunks']):
                cy = y + 34 + k * 36
                full = len(chunk['contributors']) == 4
                changed = any(t['target'] == rank and t['chunk'] == k for t in state['transfers'])
                canvas.create_rectangle(x+6, cy, x+bw-6, cy+34,
                                         fill="#e4f4e9" if full else "#eef3fa",
                                         outline="#386ab4" if changed else "#eef3fa", width=2 if changed else 1)
                sources = "+".join(f"{chr(65+r)}{k}" for r in chunk['contributors'])
                values = str(chunk['values'])
                if len(values) > 34:
                    values = values[:31] + "..."
                canvas.create_text(x+14, cy+17, anchor="w", text=f"{sources} = {values}" + (" ✓" if full else ""),
                                   fill="#25344b", font=("Consolas", 10))
        if self.view == 0:
            detail = f"初始数据：每卡 {self.container.size} 个元素，分为 4 块；补零后 {self.container.padded_size} 个元素。"
        elif self.view <= 3:
            detail = "ReduceScatter：先捕获四条边的发送数据，再统一接收并加上本地同编号块。第 3 轮后各卡拥有一块完整结果。"
        else:
            detail = "AllGather：只复制完整结果，不再加法。蓝框 = 本轮更新；绿色 / ✓ = 已包含四卡贡献。"
        if self.view == 6:
            detail += "  校验：" + ("PASS" if self.container.verify() else "FAIL")
        self.detail.set(detail)
        for item in self.link_table.get_children():
            self.link_table.delete(item)
        for link in state['links']:
            event = next((t for t in state['transfers'] if t['source'] == link['source']), None)
            ht = lambda d: f"{d['head']} / {d['tail']}"
            self.link_table.insert("", "end", values=(f"{chr(65+link['source'])} → {chr(65+link['target'])}",
                                    link['route'], event['slot'] if event else "—",
                                    ht(event['before']) if event else "0 / 0",
                                    ht(event['published']) if event else "0 / 0",
                                    ht(event['consumed']) if event else "0 / 0"))
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for event in state['transfers']:
            self.log.insert("end", f"GPU {chr(65+event['source'])} → GPU {chr(65+event['target'])}  "
                                   f"块 {event['chunk']} | {event['operation']} | "
                                   f"{event['payload']['values']} → {event['result']['values']}\n")
        if self.view == 6:
            self.log.insert("end", f"最终结果（去掉补零）：{self.container.results}\n")
        self.log.configure(state="disabled")


def launch(container: GPUContainer) -> None:
    root = tk.Tk()
    SimulatorWindow(root, container)
    root.mainloop()

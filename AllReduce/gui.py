"""Tkinter views of independent channels, partial chunks and live FIFO slots."""
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from AllReduce.simulator import RingAllReduceSimulator
from AllReduce.main import format_event

LABELS = {'SEND': '写入并发布 tail', 'RECV_REDUCE': '接收＋归约，释放 head',
          'RECV_COPY': '接收＋复制，释放 head', 'WAIT_FULL': '发送等待：Buffer 满',
          'WAIT_EMPTY': '接收等待：Buffer 空', 'WAIT_DEPENDENCY': '发送等待：上游 slice 未就绪',
          'WAIT_DELAY': '接收延迟：尚未到消费时机'}


class SimulatorWindow:
    def __init__(self, root, simulator: RingAllReduceSimulator):
        self.root, self.simulator = root, simulator
        self.view = 0
        self.selected_channel = 0
        self.selected_source = simulator.ring[0]
        root.title('RingAllReduce · Channel / SimpleBuffer 事件模拟器')
        root.geometry('1200x980')
        root.minsize(1050, 900)
        settings = ttk.Frame(root, padding=(10, 8))
        settings.pack(fill='x')
        self.settings = {}
        for name, title, value, low, high in [('nchannels', 'Channels', simulator.nchannels, 1, 8),
                ('buffer_slots', '槽数', simulator.buffer_slots, 1, 16),
                ('slice_elements', '每 slice 元素', simulator.slice_elements, 1, 32),
                ('receive_period', '慢卡接收周期', simulator.receive_period, 1, 16)]:
            ttk.Label(settings, text=title).pack(side='left', padx=(7, 3))
            var = tk.StringVar(value=str(value))
            self.settings[name] = var
            ttk.Spinbox(settings, from_=low, to=high, textvariable=var, width=4).pack(side='left')
        ttk.Label(settings, text='慢卡').pack(side='left', padx=6)
        self.slow = tk.StringVar(value=str(simulator.slow_rank if simulator.slow_rank is not None else 3))
        ttk.Combobox(settings, textvariable=self.slow, values=('0', '1', '2', '3'), width=3,
                     state='readonly').pack(side='left')
        ttk.Button(settings, text='应用并重置', command=self.apply_settings).pack(side='left', padx=8)
        toolbar = ttk.Frame(root, padding=(10, 4))
        toolbar.pack(fill='x')
        self.previous = ttk.Button(toolbar, text='上一事件', command=self.prev)
        self.previous.pack(side='left', padx=3)
        self.next_button = ttk.Button(toolbar, text='下一事件', command=self.next)
        self.next_button.pack(side='left', padx=3)
        ttk.Button(toolbar, text='推进一轮', command=self.next_round).pack(side='left', padx=3)
        ttk.Button(toolbar, text='运行到结束', command=self.finish).pack(side='left', padx=3)
        ttk.Button(toolbar, text='重置', command=self.reset).pack(side='left', padx=3)
        ttk.Button(toolbar, text='导出 JSON', command=self.export).pack(side='left', padx=3)
        ttk.Label(toolbar, text='图中 Channel').pack(side='left', padx=(12, 3))
        self.channel_var = tk.StringVar(value='0')
        self.channel_picker = ttk.Combobox(toolbar, textvariable=self.channel_var,
                       values=tuple(str(i) for i in range(simulator.nchannels)), width=4, state='readonly')
        self.channel_picker.pack(side='left')
        self.channel_picker.bind('<<ComboboxSelected>>', self.change_channel)
        self.status, self.detail = tk.StringVar(), tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=(14, 2)).pack(anchor='w')
        self.canvas = tk.Canvas(root, height=380, background='#f6f8fc', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=12, pady=4)
        self.canvas.bind('<Configure>', lambda e: self.draw())
        ttk.Label(root, textvariable=self.detail, padding=(14, 3), wraplength=1000).pack(anchor='w')
        columns = ('channel', 'edge', 'progress', 'count', 'ht', 'period', 'state')
        self.link_table = ttk.Treeview(root, columns=columns, show='headings', height=8)
        for col, title, width in [('channel', 'Channel', 65), ('edge', '连接 / 接收 Buffer', 130),
                ('progress', '发送/接收轮次', 115), ('count', '占用/槽数', 85),
                ('ht', 'head / tail', 100), ('period', '接收周期', 75), ('state', '最近的发送尝试 | 接收尝试', 420)]:
            self.link_table.heading(col, text=title)
            self.link_table.column(col, width=width, minwidth=50, stretch=col == 'state')
        self.link_table.pack(fill='x', padx=12)
        self.link_table.bind('<<TreeviewSelect>>', self.select_link)
        self.buffer_label = tk.StringVar()
        ttk.Label(root, textvariable=self.buffer_label, padding=(14, 4)).pack(anchor='w')
        self.slots_canvas = tk.Canvas(root, height=72, background='#f6f8fc', highlightthickness=0)
        self.slots_canvas.pack(fill='x', padx=12)
        self.log = tk.Text(root, height=4, wrap='word', font=('Consolas', 10))
        self.log.pack(fill='x', padx=12, pady=8)
        self.refresh()

    def change_channel(self, event=None):
        self.selected_channel = int(self.channel_var.get())
        self.draw()
        self.draw_slots()

    def select_link(self, event=None):
        selected = self.link_table.selection()
        if selected:
            self.selected_channel, self.selected_source = map(int, selected[0].split(':'))
            self.channel_var.set(str(self.selected_channel))
            self.draw()
            self.draw_slots()

    def apply_settings(self):
        try:
            options = {k: int(v.get()) for k, v in self.settings.items()}
            if options['nchannels'] > 8 or options['buffer_slots'] > 16:
                raise ValueError('图形界面最多 8 个 Channel、16 个槽；更大配置请用命令行。')
            self.simulator = RingAllReduceSimulator(self.simulator.inputs, ring=self.simulator.ring,
                 numa_nodes=self.simulator.numa_nodes, slow_rank=int(self.slow.get()), **options)
        except (TypeError, ValueError) as exc:
            messagebox.showerror('配置无效', str(exc), parent=self.root)
            return
        self.channel_picker.configure(values=tuple(str(i) for i in range(self.simulator.nchannels)))
        self.selected_channel = 0
        self.channel_var.set('0')
        self.view = 0
        self.refresh()

    def prev(self):
        self.view = max(0, self.view-1)
        self.refresh()

    def next(self):
        if self.view < len(self.simulator.history)-1:
            self.view += 1
        elif not self.simulator.done:
            self.simulator.step_event()
            self.view += 1
        self.refresh()

    def next_round(self):
        target = self.simulator.history[self.view]['round']+1
        if target > 6:
            return
        while self.view < len(self.simulator.history)-1:
            self.view += 1
            if self.simulator.history[self.view]['round'] >= target:
                self.refresh()
                return
        while not self.simulator.done and self.simulator.round < target:
            self.simulator.step_event()
        self.view = len(self.simulator.history)-1
        self.refresh()

    def finish(self):
        self.simulator.run()
        self.view = len(self.simulator.history)-1
        self.refresh()

    def reset(self):
        self.simulator.reset()
        self.view = 0
        self.refresh()

    def export(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json',
                   initialfile='channel-trace.json', filetypes=[('JSON', '*.json')])
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as output:
                    trace = self.simulator.trace()
                    trace['viewed_event'] = self.view
                    json.dump(trace, output, indent=2, ensure_ascii=False)
            except OSError as exc:
                messagebox.showerror('导出失败', str(exc), parent=self.root)

    def refresh(self):
        state = self.simulator.history[self.view]
        progress = '  |  '.join(f"C{c['id']}: R{c['completed_round']}/6" for c in state['channels'])
        self.status.set(f"事件 {state['event_id']} · 调度 tick {state['tick']}  |  {progress}")
        event = state['event']
        if event:
            before, after = event['before'], event['after']
            self.detail.set(f"C{event['channel']} {chr(65+event['source'])}→{chr(65+event['target'])} "
                f"R{event['round']} chunk {event['chunk']} / slice {event['part']}：{LABELS[event['kind']]}；"
                f"H/T {before['head']}/{before['tail']} → {after['head']}/{after['tail']}" +
                ('；最终校验 PASS' if state['done'] and self.simulator.verify() else ''))
        else:
            self.detail.set('单线程交错调度；tick 不是物理时间。各 Channel 独立推进。点击下表选择一条连接查看槽位。')
        self.previous.configure(state='normal' if self.view else 'disabled')
        self.next_button.configure(state='disabled' if state['done'] else 'normal')
        for c in state['channels']:
            for link in c['links']:
                key = f"{c['id']}:{link['source']}"
                b = link['buffer']
                values = (c['id'], f"{chr(65+link['source'])} → {chr(65+link['target'])}",
                          f"{min(link['send_round'],7)} / {min(link['recv_round'],7)}", f"{b['occupied']}/{b['capacity']}",
                          f"{b['head']} / {b['tail']}", link['receive_period'],
                          f"{link['send_status']} | {link['recv_status']}")
                if self.link_table.exists(key):
                    self.link_table.item(key, values=values)
                else:
                    self.link_table.insert('', 'end', iid=key, values=values)
        valid = {f"{c['id']}:{l['source']}" for c in state['channels'] for l in c['links']}
        for key in self.link_table.get_children():
            if key not in valid:
                self.link_table.delete(key)
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        for old in self.simulator.history[max(1, self.view-3):self.view+1]:
            self.log.insert('end', format_event(old)+'\n')
        self.log.configure(state='disabled')
        self.draw()
        self.draw_slots()

    def draw(self):
        if not hasattr(self, 'slots_canvas'):
            return
        state = self.simulator.history[self.view]
        c = state['channels'][self.selected_channel]
        event = state['event']
        canvas = self.canvas
        canvas.delete('all')
        width, height = max(canvas.winfo_width(), 1000), max(canvas.winfo_height(), 370)
        bw, bh = min(370, (width-180)/2), 150
        left, right, top, bottom = 30, width-30-bw, 20, height-bh-20
        positions = [(left,top), (right,top), (right,bottom), (left,bottom)]
        edges = [(left+bw+7, top+75, right-7, top+75),
                 (right+bw/2, top+bh+5, right+bw/2, bottom-5),
                 (right-7, bottom+75, left+bw+7, bottom+75),
                 (left+bw/2, bottom-5, left+bw/2, top+bh+5)]
        for pos, source in enumerate(self.simulator.ring):
            active = bool(event and event['channel'] == c['id'] and event['source'] == source)
            canvas.create_line(*edges[pos], arrow='last', width=3 if active else 1,
                               fill='#c86817' if active else '#74889b')
            x,y = positions[pos]
            gpu = c['gpus'][str(source)]
            canvas.create_rectangle(x,y,x+bw,y+bh,fill='white',outline='#b8c4d5')
            canvas.create_text(x+bw/2,y+14,text=f"GPU {gpu['name']} · NUMA {gpu['numa']} · C{c['id']} [{c['start']}:{c['stop']})",
                               font=('Microsoft YaHei UI',10),fill='#26364b')
            for k, chunk in enumerate(gpu['chunks']):
                cy=y+28+k*29
                changed=bool(event and event['channel']==c['id'] and event['target']==source and
                             event['chunk']==k and event['kind'].startswith('RECV_'))
                canvas.create_rectangle(x+5,cy,x+bw-5,cy+27,fill='#e1f2e6' if chunk['complete'] else '#edf2f9',
                                         outline='#276cc2' if changed else '#edf2f9')
                values=str(chunk['values'])
                if len(values)>25:
                    values=values[:22]+'...'
                text=f"chunk{k}  {values}  完成 {chunk['completed_slices']}/{len(chunk['slices'])} slice"
                canvas.create_text(x+12,cy+13,anchor='w',text=text,font=('Consolas',10),fill='#26364b')

    def draw_slots(self):
        state=self.simulator.history[self.view]
        c=state['channels'][self.selected_channel]
        link=next(l for l in c['links'] if l['source']==self.selected_source)
        b=link['buffer']
        self.buffer_label.set(f"C{c['id']} {chr(65+link['source'])}→{chr(65+link['target'])} 的接收 SimpleBuffer | "
             f"head={b['head']}，tail={b['tail']}；读槽=head%{b['capacity']}，写槽=tail%{b['capacity']} | {link['route']}")
        canvas=self.slots_canvas
        canvas.delete('all')
        width=max(canvas.winfo_width(),1000)
        shown=min(b['capacity'],16)
        sw=(width-8)/shown
        for slot,packet in enumerate(b['slots'][:shown]):
            x=4+slot*sw
            canvas.create_rectangle(x,4,x+sw-5,66,fill='#ffe3bb' if packet else '#eaf0f6',outline='#aebed0')
            label=f"槽 {slot}\n"+(f"R{packet['round']} c{packet['chunk']} s{packet['part']}" if packet else '空闲')
            canvas.create_text(x+(sw-5)/2,24,text=label,font=('Microsoft YaHei UI',9),fill='#26364b')
            if packet and shown<=8:
                values=str(packet['values'])
                canvas.create_text(x+(sw-5)/2,53,text=values[:24],font=('Consolas',9),fill='#26364b')


def launch(simulator: RingAllReduceSimulator):
    root=tk.Tk()
    SimulatorWindow(root,simulator)
    root.mainloop()

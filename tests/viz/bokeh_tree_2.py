from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Plot, Circle, MultiLine, TapTool, Text
from bokeh.events import Tap
from bokeh.io import output_file, show

class Node:
    def __init__(self, name, children=None, is_expanded=True):
        self.name = name
        self.children = children if children else []
        self.is_expanded = is_expanded
        self.x = 0
        self.y = 0

# 构建示例树结构
root = Node("Root")
root.children = [
    Node("Child1", [Node("Grandchild1"), Node("Grandchild2")]),
    Node("Child2", [Node("Grandchild3")])
]

# 数据源
nodes_source = ColumnDataSource(data={'x': [], 'y': [], 'name': []})
edges_source = ColumnDataSource(data={'xs': [], 'ys': []})

# 创建绘图对象
plot = Plot(width=800, height=600, title="可展开树状图")
plot.add_tools(TapTool())

# 定义布局参数
H_SPACING = 1.5  # 水平间距
V_SPACING = 2.0   # 垂直间距

def compute_positions(node, x=0, y=0):
    """递归计算节点位置"""
    node.x = x
    node.y = y
    if node.is_expanded and node.children:
        num_children = len(node.children)
        total_width = (num_children - 1) * H_SPACING
        start_x = x - total_width / 2
        
        for i, child in enumerate(node.children):
            child_x = start_x + i * H_SPACING
            compute_positions(child, child_x, y - V_SPACING)

def update_sources():
    """更新数据源"""
    nodes = []
    edges = []
    
    def traverse(node, parent=None):
        nodes.append({'x': node.x, 'y': node.y, 'name': node.name})
        if parent:
            edges.append({'xs': [parent.x, node.x], 'ys': [parent.y, node.y]})
        if node.is_expanded:
            for child in node.children:
                traverse(child, node)
    
    traverse(root)
    nodes_source.data = {
        'x': [n['x'] for n in nodes],
        'y': [n['y'] for n in nodes],
        'name': [n['name'] for n in nodes]
    }
    edges_source.data = {
        'xs': [[e['xs'][0], e['xs'][1]] for e in edges],
        'ys': [[e['ys'][0], e['ys'][1]] for e in edges]
    }

# 初始化布局和绘图
compute_positions(root, 0, 0)
update_sources()

# 绘制节点和边
plot.add_glyph(nodes_source, Circle(x='x', y='y', radius=0.1, fill_color='skyblue'))
plot.add_glyph(nodes_source, Text(x='x', y='y', text='name', 
                                 text_baseline='middle', text_align='center'))
plot.add_glyph(edges_source, MultiLine(xs='xs', ys='ys', line_color='gray'))

def handle_tap(event):
    # 查找点击的节点
    min_distance = float('inf')
    tapped_index = -1
    
    for i, (nx, ny) in enumerate(zip(nodes_source.data['x'], nodes_source.data['y'])):
        distance = (nx - event.x)**2 + (ny - event.y)**2
        if distance < min_distance:
            min_distance = distance
            tapped_index = i
    
    if tapped_index != -1:
        # 递归查找节点对象
        target_name = nodes_source.data['name'][tapped_index]
        
        def find_node(node):
            if node.name == target_name:
                return node
            for child in node.children:
                result = find_node(child)
                if result:
                    return result
            return None
        
        node = find_node(root)
        if node:
            node.is_expanded = not node.is_expanded
            compute_positions(root, 0, 0)
            update_sources()

plot.on_event(Tap, handle_tap)

curdoc().add_root(plot)

output_file("tree_plot_2.html")
show(plot)
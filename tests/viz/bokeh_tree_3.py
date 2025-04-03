from bokeh.io import show, output_notebook, output_file
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, HoverTool, ColumnDataSource, CustomJS
from bokeh.palettes import Category20_20

# 准备数据：节点和边
nodes = {
    "index": [0, 1, 2, 3, 4],  # 节点索引
    "name": ["CEO", "Manager A", "Manager B", "Employee 1", "Employee 2"],
    "title": ["Chief Executive Officer", "Department Head", "Department Head", "Team Member", "Team Member"],
    "details": [
        "Responsible for overall company strategy.",
        "Manages Department A.",
        "Manages Department B.",
        "Works in Team A.",
        "Works in Team B."
    ]
}
edges = {
    "start": [0, 0, 1, 2],  # 边的起点
    "end": [1, 2, 3, 4]     # 边的终点
}

# 创建图形对象
plot = figure(title="Organization Chart", x_range=(-1.5, 1.5), y_range=(-1.5, 1.5),
              tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above")

# 图形渲染器
graph = GraphRenderer()

# 设置节点数据
graph.node_renderer.data_source.add(nodes["index"], 'index')
graph.node_renderer.data_source.add(Category20_20[:len(nodes["index"])], 'color')
graph.node_renderer.glyph = Circle(radius=0.05, fill_color='color')

# 设置边数据
graph.edge_renderer.data_source.data = edges

# 设置布局
layout = {
    0: (-0.5, 0),  # CEO
    1: (-1, -1),   # Manager A
    2: (0, -1),    # Manager B
    3: (-1, -2),   # Employee 1
    4: (0, -2)     # Employee 2
}
graph.layout_provider = StaticLayoutProvider(graph_layout=layout)

# 添加 HoverTool
hover = HoverTool(tooltips=[
    ("Name", "@name"),
    ("Title", "@title"),
    ("Details", "@details")
])
plot.add_tools(hover)

# 添加点击展开功能（示例：通过 JavaScript 控制）
callback = CustomJS(args=dict(renderer=graph.node_renderer.data_source), code="""
    const index = cb_data.index.indices[0];
    if (index !== undefined) {
        console.log(`Node ${index} clicked`);
        // 在这里实现动态加载子节点逻辑
    }
""")
plot.js_on_event('tap', callback)

# 将图形添加到画布
plot.renderers.append(graph)

# 显示图表
output_file("tree_plot_3.html")
show(plot)
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, HoverTool, CustomJS, ColumnDataSource
from bokeh.palettes import Category20_20
from bokeh.io import output_file, show
import networkx as nx

# 准备数据：创建一个树状结构
tree_data = {
    'root': ['child1', 'child2'],
    'child1': ['grandchild1', 'grandchild2'],
    'child2': ['grandchild3']
}

# 将树状结构转换为 NetworkX 图
G = nx.DiGraph()
for parent, children in tree_data.items():
    for child in children:
        G.add_edge(parent, child)

# 计算节点布局
pos = nx.spring_layout(G)  # 可以选择其他布局算法，例如树形布局
nodes = list(G.nodes)
edges = list(G.edges)

# 创建 Bokeh 数据源
node_indices = list(range(len(nodes)))
node_source = ColumnDataSource(data=dict(index=node_indices, name=nodes))
edge_source = ColumnDataSource(data=dict(start=[nodes.index(u) for u, v in edges],
                                         end=[nodes.index(v) for u, v in edges]))

# 创建图形渲染器
graph_renderer = GraphRenderer()
graph_renderer.node_renderer.data_source = node_source
graph_renderer.node_renderer.glyph = Circle(radius=0.05, fill_color=Category20_20[0])  # 使用 radius 替代 size
graph_renderer.edge_renderer.data_source = edge_source

# 设置布局
graph_layout = {i: pos[nodes[i]] for i in range(len(nodes))}
graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

# 创建绘图对象
plot = figure(title="Tree Structure with Bokeh", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
              tools="pan,wheel_zoom,reset,tap", toolbar_location="above")
plot.renderers.append(graph_renderer)

# 添加交互功能：点击节点展开/折叠子节点
callback_code = """
const selected_node = cb_data.source.selected.indices[0];
if (selected_node !== undefined) {
    const node_name = source.data['name'][selected_node];
    console.log(`Node clicked: ${node_name}`);
    // TODO: 动态更新数据源以显示或隐藏子节点
}
"""
callback = CustomJS(args=dict(source=node_source), code=callback_code)
node_source.selected.js_on_change('indices', callback)

# 显示图表
# output_notebook()
output_file("tree_plot.html")
show(plot)
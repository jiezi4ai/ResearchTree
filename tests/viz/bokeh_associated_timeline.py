from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Span, HoverTool, Range1d
import pandas as pd

# 数据准备
data = {
    "dates": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
    "events": ["Event A", "Event B", "Event C", "Event D"],
    "search_counts": [100, 250, 180, 300],
}
df = pd.DataFrame(data)
df["dates"] = pd.to_datetime(df["dates"])  # 转换日期为 datetime 类型

# 创建 ColumnDataSource
source = ColumnDataSource(df)

# 创建上方的时间轴 (显示事件)
p1 = figure(
    title="Important Events Timeline",
    x_axis_type="datetime",
    height=200,  # 使用 height 替代 plot_height
    tools="hover",
    tooltips=[("Event", "@events"), ("Date", "@dates{%F}")],
    x_range=Range1d(df["dates"].min(), df["dates"].max()),
)
p1.xaxis.axis_label = "Date"
p1.yaxis.visible = False  # 隐藏 y 轴
p1.toolbar.logo = None  # 隐藏 Bokeh logo

# 在时间轴上绘制事件点
p1.scatter(  # 使用 scatter 替代 circle
    x="dates",
    y=0,
    size=10,
    color="blue",
    alpha=0.8,
    marker="circle",  # 指定标记形状为圆形
    source=source,
)

# 添加垂直线标记事件
for date in df["dates"]:
    p1.add_layout(Span(location=date, dimension="height", line_color="gray", line_dash="dotted"))

# 创建下方的时间轴 (显示搜索次数波形)
p2 = figure(
    title="Search Counts Over Time",
    x_axis_type="datetime",
    height=200,  # 使用 height 替代 plot_height
    tools="hover",
    tooltips=[("Date", "@dates{%F}"), ("Search Count", "@search_counts")],
    x_range=p1.x_range,  # 共享 x 轴范围
)
p2.xaxis.axis_label = "Date"
p2.yaxis.axis_label = "Search Count"
p2.toolbar.logo = None  # 隐藏 Bokeh logo

# 绘制波形图
p2.line(
    x="dates",
    y="search_counts",
    line_width=2,
    color="green",
    alpha=0.8,
    source=source,
)
p2.scatter(  # 使用 scatter 替代 circle
    x="dates",
    y="search_counts",
    size=8,
    color="green",
    alpha=0.8,
    marker="circle",  # 指定标记形状为圆形
    source=source,
)

# 将两个图表垂直排列
layout = column(p1, p2)

# 输出文件并显示
output_file("timeline_with_search_counts.html")
show(layout)
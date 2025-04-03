from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, Div
from bokeh.layouts import column
from bokeh.io import curdoc, output_file
import pandas as pd
from bokeh.themes import Theme
from collections import defaultdict

# 数据
data = {
    'dates': pd.to_datetime([
        '2023-01-10', '2023-01-15',
        '2023-02-20',
        '2023-03-15', '2023-03-20',
        '2023-04-10'
    ]),
    'event_names': [
        '项目启动', '团队组建完成',
        '里程碑A',
        '重要决策', '战略调整',
        '发布版本1.0'
    ],
    'descriptions': [
        '项目正式启动，团队组建完成。',
        '团队组建完成，进入开发阶段。',
        '完成第一个重要里程碑，核心功能开发完毕。',
        '管理层做出关键战略决策。',
        '根据市场反馈调整发展方向。',
        '产品第一个正式版本对外发布。'
    ],
    'triggered_by': [
        '高层会议', '人力资源部',
        '技术突破',
        '市场分析', '用户反馈',
        '产品成熟'
    ],
    'impact': [
        '项目正式开始', '团队稳定',
        '验证技术可行性',
        '调整发展方向', '优化策略',
        '用户开始使用'
    ],
    'importance': [True, False, False, True, True, True]
}

# 动态生成 y 坐标
date_groups = defaultdict(list)
for i, date in enumerate(data['dates']):
    date_groups[date].append(i)

y_positions = []
for date in data['dates']:
    group = date_groups[date]
    index_in_group = group.index(data['dates'].tolist().index(date))
    y_positions.append(-0.5 + index_in_group * 0.5)
data['y'] = y_positions

# 动态生成点大小和颜色
data['point_size'] = [15 if imp else 8 for imp in data['importance']]
data['point_color'] = ['red' if imp else 'blue' for imp in data['importance']]

source = ColumnDataSource(data)

event_details_div = Div(
    text="""点击时间线上的事件点以查看详情。""",
    styles={
        'border': '1px solid black',
        'padding': '10px',
        'width': '100%',
        'box-sizing': 'border-box'
    }
)

# 定义主题
theme = Theme(json={
    'attrs': {
        'Figure': {
            'background_fill_color': '#fafafa',
            'border_fill_color': '#fafafa',
            'outline_line_color': 'transparent',
            'toolbar_location': 'above',
        },
        'Axis': {
            'axis_line_color': 'darkgray',
            'major_tick_line_color': 'darkgray',
            'major_label_text_color': 'darkgray',
            'major_label_text_font_size': '10pt',
        },
        'Grid': {
            'grid_line_color': 'lightgray',
            'grid_line_alpha': 0.3,
            'grid_line_dash': [4, 4],
        },
        'Title': {
            'text_color': 'darkslategray',
            'text_font_size': '14pt',
            'text_font_style': 'bold',
        },
        'Tooltip': {
            'background_fill_color': 'white',
            'border_line_color': 'black',
            'text_font_size': '10pt',
        },
    }
})
curdoc().theme = theme

# 创建时间线图形
p = figure(
    title="交互式项目时间线",
    x_axis_type="datetime",
    height=600,
    sizing_mode='stretch_width',
    x_range=(data['dates'].min() - pd.Timedelta(days=10), data['dates'].max() + pd.Timedelta(days=10)),
    tools="xpan,xwheel_zoom,reset,save"
)

# 时间段背景颜色
time_segments = [
    {'start': pd.to_datetime('2023-01-01'), 'end': pd.to_datetime('2023-02-01'), 'color': 'lightskyblue'},
    {'start': pd.to_datetime('2023-02-01'), 'end': pd.to_datetime('2023-03-01'), 'color': 'lightcoral'},
    {'start': pd.to_datetime('2023-03-01'), 'end': pd.to_datetime('2023-04-01'), 'color': 'lightgreen'},
    {'start': pd.to_datetime('2023-04-01'), 'end': pd.to_datetime('2023-04-25'), 'color': 'lightsalmon'}
]

for segment in time_segments:
    p.patch(
        x=[segment['start'], segment['end'], segment['end'], segment['start']],
        y=[-1, -1, 1, 1],  # 覆盖整个 Y 轴范围
        color=segment['color'],
        alpha=0.2,  # 设置透明度
        line_width=0  # 不显示边框
    )

# 绘制时间线
timeline = p.line(x='dates', y=0, source=source, line_width=2, color="gray")

# 绘制事件点
event_points = p.scatter(
    x='dates',
    y='y',
    source=source,
    size='point_size',
    fill_color='point_color',
    line_color='black',
    line_width=1
)

# 添加悬停工具
hover = HoverTool()
hover.tooltips = [
    ("事件名称", "@event_names"),
    ("描述", "@descriptions"),
    ("触发因素", "@triggered_by"),
    ("影响", "@impact"),
    ("日期", "@dates{%F}")
]
hover.formatters = {'@dates': 'datetime'}
p.add_tools(hover)

# 添加点击工具
tap_callback = CustomJS(args=dict(source=source, event_details_div=event_details_div), code="""
    const index = cb_data.source.selected.indices[0];
    if (index != null) {
        const eventName = source.data['event_names'][index];
        const description = source.data['descriptions'][index];
        const triggeredBy = source.data['triggered_by'][index];
        const impact = source.data['impact'][index];
        const eventDate = source.data['dates'][index];

        const detailsText = `
            <b>事件名称:</b> ${eventName}<br>
            <b>描述:</b> ${description}<br>
            <b>触发因素:</b> ${triggeredBy}<br>
            <b>影响:</b> ${impact}<br>
            <b>日期:</b> ${new Date(eventDate).toLocaleDateString()}
        `;
        event_details_div.text = detailsText;
    } else {
        event_details_div.text = "点击时间线上的事件点以查看详情。";
    }
""")
tap = TapTool(callback=tap_callback, renderers=[event_points])
p.add_tools(tap)

# 隐藏 Y 轴和网格线
p.yaxis.major_tick_line_color = None
p.yaxis.major_label_standoff = 0
p.yaxis.visible = False
p.grid.grid_line_color = None
p.outline_line_color = None

# 调整 Y 轴范围
p.y_range.start = -1
p.y_range.end = 1

# 布局
layout = column(p, event_details_div, sizing_mode='stretch_width')

output_file("timeline.html", title="交互式项目时间线")
show(layout)
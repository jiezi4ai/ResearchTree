from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, Div
from bokeh.layouts import column
from bokeh.io import curdoc, output_file  # 导入 curdoc 和 output_file
import pandas as pd
from bokeh.themes import Theme  # 导入主题

# 数据
time_segments = [
    {'start': pd.to_datetime('2023-01-01'), 'end': pd.to_datetime('2023-02-01'), 'color': 'lightskyblue'},
    {'start': pd.to_datetime('2023-02-01'), 'end': pd.to_datetime('2023-03-01'), 'color': 'lightcoral'},
    {'start': pd.to_datetime('2023-03-01'), 'end': pd.to_datetime('2023-04-01'), 'color': 'lightgreen'},
    {'start': pd.to_datetime('2023-04-01'), 'end': pd.to_datetime('2023-04-25'), 'color': 'lightsalmon'}
]

data = {
    'dates': pd.to_datetime(['2023-01-10', '2023-02-20', '2023-03-15', '2023-04-10']),
    'event_names': ['项目启动', '里程碑A', '重要决策', '发布版本1.0'],
    'descriptions': [
        '项目正式启动，团队组建完成。',
        '完成第一个重要里程碑，核心功能开发完毕。',
        '管理层做出关键战略决策。',
        '产品第一个正式版本对外发布。'
    ],
    'triggered_by': ['高层会议', '技术突破', '市场分析', '产品成熟'],
    'impact': ['项目正式开始', '验证技术可行性', '调整发展方向', '用户开始使用'],
    'importance': [True, False, True, True]
}

data['point_size'] = [15 if imp else 8 for imp in data['importance']]
data['point_color'] = ['red' if imp else 'blue' for imp in data['importance']]

source = ColumnDataSource(data)

event_details_div = Div(
    text="""点击时间线上的事件点以查看详情。""",
    styles={
        'border': '1px solid black',
        'padding': '10px',
        'width': '100%',  # 确保 Div 宽度随页面调整
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

# 应用主题
curdoc().theme = theme

# 创建时间线图形
p = figure(
    title="交互式项目时间线",
    x_axis_type="datetime",
    height=600,  # 固定高度
    sizing_mode='stretch_width',  # 宽度随页面自动调整
    x_range=(data['dates'].min() - pd.Timedelta(days=10), data['dates'].max() + pd.Timedelta(days=10)),
    tools="xpan,xwheel_zoom,reset,save"
)

# 绘制时间区间背景色
for segment in time_segments:
    p.patch(
        x=[segment['start'], segment['end'], segment['end'], segment['start']],
        y=[-1, -1, 1, 1],
        color=segment['color'],
        alpha=0.2,
        line_width=0
    )

# 绘制时间线
timeline = p.line(x='dates', y=0, source=source, line_width=2, color="gray")

# 绘制事件点
event_points = p.scatter(
    x='dates',
    y=0,
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

# 布局：时间线图形和事件详情 Div 垂直排列
layout = column(
    p,
    event_details_div,
    sizing_mode='stretch_width'  # 确保布局宽度随页面自动调整
)

# 初始化输出文件
output_file("timeline.html", title="交互式项目时间线")

show(layout)
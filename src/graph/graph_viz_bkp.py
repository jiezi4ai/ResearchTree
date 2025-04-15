# graph_viz.py
import networkx as nx
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

from bokeh.plotting import figure, show
from bokeh.models import (Scatter, MultiLine, Div, CustomJS, TapTool,
                          HoverTool, LabelSet, ColumnDataSource, Plot,
                          GraphRenderer, StaticLayoutProvider, Paragraph) # Added Paragraph
from bokeh.layouts import column, row # Added column, row
from bokeh.palettes import brewer, Category10, Pastel1 # Example palettes

# =============================================================================
# Graph Viz Preprocessing Functions
# =============================================================================

def assign_node_size(
        G,
        sig_nid_lst: Optional[List[str]] = None,
        min_node_size: Optional[int] = 10,
        max_node_size: Optional[int] = 50,
        ):
    """assign node size (Corrected Logic)"""
    paper_cites_ref, author_writes_ref = {}, {}

    # First pass: Calculate counts for all relevant nodes
    for nid, node_data in G.nodes(data=True):
        node_type = node_data.get('nodeType')
        if node_type == 'Paper':
            in_edges_info = G.in_edges(nid, data=True)
            cites_cnt = sum(1 for u, v, data in in_edges_info if data.get('relationshipType') == 'CITES')
            paper_cites_ref[nid] = cites_cnt
        elif node_type == 'Author':
            out_edges_info = G.out_edges(nid, data=True)
            writes_cnt = sum(1 for u, v, data in out_edges_info if data.get('relationshipType') == 'WRITES')
            author_writes_ref[nid] = writes_cnt

    max_cites_cnt = max(paper_cites_ref.values()) if paper_cites_ref else 0
    min_cites_cnt = min(paper_cites_ref.values()) if paper_cites_ref else 0
    max_writes_cnt = max(author_writes_ref.values()) if author_writes_ref else 0
    min_writes_cnt = min(author_writes_ref.values()) if author_writes_ref else 0

    cites_range = max_cites_cnt - min_cites_cnt if max_cites_cnt > min_cites_cnt else 1
    writes_range = max_writes_cnt - min_writes_cnt if max_writes_cnt > min_writes_cnt else 1

    # Second pass: Assign sizes
    for nid, node_data in G.nodes(data=True):
        node_data['vizSize'] = min_node_size # Default size

        if sig_nid_lst is not None and nid in sig_nid_lst:
            node_data['vizSize'] = max_node_size
            continue

        node_type = node_data.get('nodeType')
        if node_type == 'Paper' and nid in paper_cites_ref:
            value = paper_cites_ref[nid]
            node_size = min_node_size + ((max_node_size - min_node_size) * (value - min_cites_cnt)) / cites_range
            node_data['vizSize'] = max(min_node_size, min(max_node_size, node_size))
        elif node_type == 'Author' and nid in author_writes_ref:
            value = author_writes_ref[nid]
            node_size = min_node_size + ((max_node_size - min_node_size) * (value - min_writes_cnt)) / writes_range
            node_data['vizSize'] = max(min_node_size, min(max_node_size, node_size))


def assign_edge_weight(
        G,
        edge_type_weight_ref,
        default_weight: Optional[float] = 0.1
        ):
    """assign edge weight and vizWidth"""
    for u, v, data in G.edges(data=True):
        weight = data.get('weight')
        if weight is None:
            rel_type = data.get('relationshipType') # Renamed to avoid conflict
            weight = edge_type_weight_ref.get(rel_type, default_weight) # Use default if type not found
        data['weight'] = weight # Assign calculated or existing weight back

        # Ensure vizWidth is always set based on the final weight
        # Scale weight for better visibility if needed, adjust multiplier as necessary
        data['vizWidth'] = max(0.5, data['weight'] * 5) # Ensure minimum width


def assign_node_color(
        G,
        sig_nid_lst: Optional[List[str]] = None,
        default_colormap_name: Optional[str] = 'Category10', # Using Bokeh palette name
        default_color_cnt: Optional[int] = 10
        ) -> Dict[str, str]: # Return the color mapping
    """Assign color to node and return the type-to-color mapping."""
    highlight_border_color = '#FFD700' # Gold/Yellow
    highlight_border_width = 4
    normal_border_width = 1
    default_node_color = '#CCCCCC'   # Default color if type is missing or unmapped
    default_border_color = '#888888' # Default border color

    node_types_lst = [G.nodes[nid].get('nodeType') for nid in G.nodes]
    node_types_cnt = Counter(node_types_lst)
    unique_node_types = sorted([t for t in node_types_cnt if t is not None])
    unique_node_cnt = len(unique_node_types)

    # Use Bokeh palettes directly if possible, fallback to seaborn if needed
    try:
        # Try getting palette from Bokeh
        palette_key = f"{default_colormap_name}{max(3, min(default_color_cnt, unique_node_cnt)) if unique_node_cnt > 0 else 3}"
        if unique_node_cnt > 0 and unique_node_cnt <= default_color_cnt and hasattr(brewer, default_colormap_name):
             colors_hex = brewer[default_colormap_name][max(3, unique_node_cnt)] # Brewer needs at least 3
        elif unique_node_cnt > 0:
             # Fallback or handle more colors
             base_palette = Category10.get(default_color_cnt) # Use a known Bokeh palette
             if unique_node_cnt <= default_color_cnt:
                 colors_hex = base_palette[:unique_node_cnt]
             else:
                 colors_hex = list(base_palette) + ['#808080'] * (unique_node_cnt - default_color_cnt)
        else:
            colors_hex = []
    except ImportError:
         # Fallback to Seaborn if Bokeh palettes fail or aren't sufficient
         print("Bokeh palettes not found or sufficient, falling back to Seaborn.")
         if unique_node_cnt == 0:
            colors_hex = []
         elif unique_node_cnt <= default_color_cnt:
            colors_hex = sns.color_palette(default_colormap_name, n_colors=unique_node_cnt).as_hex()
         else:
            colors_hex = sns.color_palette(default_colormap_name, n_colors=default_color_cnt).as_hex()
            colors_hex.extend(['#808080']*(unique_node_cnt - default_color_cnt))


    # Create a mapping from node type to its assigned color
    type_to_color = dict(zip(unique_node_types, colors_hex))
    # Add a mapping for the default color for potential display in legend
    type_to_color['Unknown/Default'] = default_node_color

    # Assign colors and border properties to nodes in the graph
    for nid, node_data in G.nodes(data=True):
        node_type = node_data.get('nodeType') # Use .get() for safety
        original_color = type_to_color.get(node_type, default_node_color) # Use default if type is None or not mapped
        node_data['vizColor'] = original_color

        if sig_nid_lst is not None and nid in sig_nid_lst:
            node_data['vizBorderColor'] = highlight_border_color
            node_data['vizBorderWidth'] = highlight_border_width
        else:
            node_data['vizBorderColor'] = default_border_color
            node_data['vizBorderWidth'] = normal_border_width

    return type_to_color # Return the map

def assign_edge_color(
        G,
        default_colormap_name: Optional[str] = 'Pastel1', # Use a different palette for edges
        default_color_cnt: Optional[int] = 9 # Pastel1 has 9 colors in Bokeh
        ) -> Dict[str, str]: # Return the color mapping
    """assign color to edge and return the type-to-color mapping"""
    default_edge_color = '#AAAAAA' # Default color for unmapped or None types

    edge_types_lst = [d.get('relationshipType') for u, v, d in G.edges(data=True)]
    edge_types_cnt = Counter(edge_types_lst)
    unique_edge_types = sorted([t for t in edge_types_cnt if t is not None])
    unique_edge_cnt = len(unique_edge_types)

    # Use Bokeh palettes
    try:
        base_palette = Pastel1.get(default_color_cnt) # Use a known Bokeh palette
        if unique_edge_cnt == 0:
            colors_hex = []
        elif unique_edge_cnt <= default_color_cnt:
            colors_hex = base_palette[:unique_edge_cnt]
        else:
            colors_hex = list(base_palette) + ['#D3D3D3'] * (unique_edge_cnt - default_color_cnt) # Light grey for extras
    except ImportError:
         # Fallback to Seaborn
         print("Bokeh palettes not found or sufficient, falling back to Seaborn for edges.")
         if unique_edge_cnt == 0:
             colors_hex = []
         elif unique_edge_cnt <= default_color_cnt:
             colors_hex = sns.color_palette(default_colormap_name, n_colors=unique_edge_cnt).as_hex()
         else:
             colors_hex = sns.color_palette(default_colormap_name, n_colors=default_color_cnt).as_hex()
             colors_hex.extend(['#D3D3D3']*(unique_edge_cnt - default_color_cnt))

    # Create a mapping from edge type to its assigned color
    type_to_color = dict(zip(unique_edge_types, colors_hex))
    # Add default mapping
    type_to_color['Unknown/Default'] = default_edge_color


    # Assign colors to edges in the graph
    for u, v, edge_data in G.edges(data=True):
        edge_type = edge_data.get('relationshipType')
        edge_color = type_to_color.get(edge_type, default_edge_color)
        edge_data['vizColor'] = edge_color

    return type_to_color # Return the map


def update_node_keys(G, node_key_ref):
    # node_key_ref = {'Paper': 'title', 'Author': 'name', 'Affiliation': '', 'Journal': '', 'Venue': ''}
    for nid, node_data in G.nodes(data=True):
        node_type = node_data.get('nodeType')
        label_key = node_key_ref.get(node_type)

        if label_key and label_key in node_data:
            node_data['vizLabel'] = node_data.get(label_key, f"ID: {nid}")
        elif 'nodeType' in node_data:
            node_data['vizLabel'] = f"{node_data['nodeType']} ID: {nid}"
        else:
            node_data['vizLabel'] = f"ID: {nid}"

        # Extract non-viz keys for detail display
        keys = [key for key in node_data.keys() if not key.startswith('viz')]
        # Convert detail to string representation for Div display if it's complex
        # For simplicity here, keep as dict; JS will handle formatting.
        node_data['nodeDetail'] = {key: node_data[key] for key in keys}

        # Keep brief info if needed for hover, otherwise nodeDetail is primary
        # node_data['brief'] = ... (removed for brevity unless specifically needed)


def update_edges_keys(G):
    """Update edge labels and extract details."""
    for u, v, data in G.edges(data=True): # Iterate through all edges
        u_label = G.nodes[u].get('vizLabel', f"ID: {u}")
        v_label = G.nodes[v].get('vizLabel', f"ID: {v}")
        rel_type = data.get('relationshipType', 'UNKNOWN')

        # Construct a meaningful label
        data['vizLabel'] = f"{rel_type}: {u_label} -> {v_label}"

        # Extract non-viz keys for detail display
        keys = [key for key in data.keys() if not key.startswith('viz')]
        # Corrected: Store the relevant details in edgeDetail
        data['edgeDetail'] = {key: data[key] for key in keys}


# =============================================================================
# Graph Visulization Using Bokeh
# =============================================================================
class GraphViz:
    def __init__(self, graph, title):
        self.graph = graph
        self.title = title
        self.node_color_map = {} # To store node color mapping
        self.edge_color_map = {} # To store edge color mapping

    def preprocessing(self):
        """Preprocessing Paper Graph and store color maps"""
        print("Running preprocessing...")
        node_key_ref = {'Paper': 'title', 'Author': 'name', 'Venue': 'name', 'Journal': 'name'}
        edge_type_weight_ref = {'CITES':0.5, 'DISCUSS':0.4, 'WRITES':0.3, 'WORKS_IN':0.2, 'PRINTS_ON':0.1, 'RELEASES_IN':0.1, 'REVIEWS': 0.05}

        assign_node_size(self.graph)
        assign_edge_weight(self.graph, edge_type_weight_ref)
        # Store the returned color maps
        self.node_color_map = assign_node_color(self.graph)
        self.edge_color_map = assign_edge_color(self.graph)
        update_node_keys(self.graph, node_key_ref)
        update_edges_keys(self.graph) # Ensure this is called
        print("Preprocessing complete.")

    def _create_legend(self, color_map: Dict[str, str], title: str) -> Div:
        """Helper function to create an HTML legend Div."""
        legend_html = f"<b>{title}</b><br>"
        item_style = "margin-bottom: 5px; display: flex; align-items: center;"
        color_box_style = "width: 15px; height: 15px; margin-right: 5px; border: 1px solid #ccc; display: inline-block;"
        text_style = "font-size: 0.8em;"

        # Sort items for consistent legend order
        sorted_items = sorted(color_map.items())

        for item_type, color in sorted_items:
            legend_html += f'<div style="{item_style}">'
            legend_html += f'<span style="{color_box_style} background-color: {color};"></span>'
            legend_html += f'<span style="{text_style}">{item_type}</span>'
            legend_html += '</div>'

        return Div(text=legend_html, width=150, height=len(color_map) * 25 + 30) # Adjust height as needed

    def visulization(self):
        """Visualize the graph with legends and interactive detail display."""
        # --- 1. Calculate Layout ---
        print("Calculating layout...")
        try:
            pos = nx.spring_layout(self.graph, k=0.5, iterations=100, seed=42, weight='weight') # Use weights
            print("Layout calculated using spring_layout.")
        except Exception as e:
            print(f"Spring layout failed ({e}), trying Kamada-Kawai layout.")
            try:
                pos = nx.kamada_kawai_layout(self.graph, weight='weight') # Use weights
                print("Layout calculated using kamada_kawai_layout.")
            except Exception as e2:
                print(f"Kamada-Kawai layout also failed ({e2}), using random layout.")
                pos = nx.random_layout(self.graph, seed=42)
                print("Layout calculated using random_layout.")

        # --- 2. Prepare Data Sources ---
        print("Preparing data sources...")
        node_ids = list(self.graph.nodes())
        node_viz_info = dict(
            index=node_ids,
            node_id=node_ids,
            x=[pos[nid][0] for nid in node_ids],
            y=[pos[nid][1] for nid in node_ids],
            nodeType=[node_data.get('nodeType', 'Unknown/Default') for nid, node_data in self.graph.nodes(data=True)],
            # Ensure nodeDetail is serializable (it should be if dict of primitives)
            nodeDetail=[node_data.get('nodeDetail', {}) for nid, node_data in self.graph.nodes(data=True)],
            vizSize=[node_data.get('vizSize', 10) for nid, node_data in self.graph.nodes(data=True)],
            vizColor=[node_data.get('vizColor', '#CCCCCC') for nid, node_data in self.graph.nodes(data=True)],
            vizBorderColor=[node_data.get('vizBorderColor', '#888888') for nid, node_data in self.graph.nodes(data=True)],
            vizBorderWidth=[node_data.get('vizBorderWidth', 1) for nid, node_data in self.graph.nodes(data=True)],
            vizLabel=[node_data.get('vizLabel', str(nid)) for nid, node_data in self.graph.nodes(data=True)],
        )

        edge_starts = [u for u, v, k in self.graph.edges(keys=True)] # Handle MultiDiGraph keys
        edge_ends = [v for u, v, k in self.graph.edges(keys=True)]
        edge_ids = list(range(self.graph.number_of_edges()))
        edge_data_list = [data for u, v, data in self.graph.edges(data=True)] # Get edge data

        edge_viz_info = dict(
            start=edge_starts,
            end=edge_ends,
            edge_id=edge_ids,
            edgeType=[data.get('relationshipType', 'Unknown/Default') for data in edge_data_list],
             # Ensure edgeDetail is serializable
            edgeDetail=[data.get('edgeDetail', {}) for data in edge_data_list],
            vizWidth=[data.get('vizWidth', 1) for data in edge_data_list],
            vizColor=[data.get('vizColor', '#AAAAAA') for data in edge_data_list],
            vizLabel=[data.get('vizLabel', '') for data in edge_data_list]
        )

        node_source = ColumnDataSource(data=node_viz_info, name="Node Source") # Add names for clarity
        edge_source = ColumnDataSource(data=edge_viz_info, name="Edge Source")

        # --- 3. Create Bokeh Plot ---
        print("Creating plot...")
        plot_width = 800
        plot_height = 600
        # Dynamically calculate range or set fixed range
        x_coords = [d[0] for d in pos.values()]
        y_coords = [d[1] for d in pos.values()]
        x_range = (min(x_coords)*1.1 - 0.1, max(x_coords)*1.1 + 0.1) # Add buffer
        y_range = (min(y_coords)*1.1 - 0.1, max(y_coords)*1.1 + 0.1) # Add buffer

        plot = figure(
            title=self.title,
            width=plot_width, height=plot_height,
            x_range=x_range, y_range=y_range,
            tools="pan,wheel_zoom,save,reset", # TapTool added below
            toolbar_location="below",
            x_axis_location=None, y_axis_location=None
            )
        plot.grid.grid_line_color = None

        # --- 4. Setup GraphRenderer ---
        graph_renderer = GraphRenderer()

        graph_renderer.node_renderer.data_source = node_source
        graph_renderer.node_renderer.glyph = Scatter(
            size='vizSize', fill_color='vizColor',
            line_color="vizBorderColor", line_width="vizBorderWidth"
        )
        # Make selected nodes more obvious
        graph_renderer.node_renderer.selection_glyph = Scatter(
            size='vizSize', fill_color='vizColor', line_color='firebrick', line_width=4
        )
        graph_renderer.node_renderer.hover_glyph = Scatter(
            size='vizSize', fill_color='vizColor', line_color='deepskyblue', line_width=3
        )

        graph_renderer.edge_renderer.data_source = edge_source
        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="vizColor", line_alpha=0.6, line_width='vizWidth' # Slightly transparent
        )
        graph_renderer.edge_renderer.selection_glyph = MultiLine(
            line_color='firebrick', line_width=5, line_alpha=1.0
        )
        graph_renderer.edge_renderer.hover_glyph = MultiLine(
            line_color='deepskyblue', line_width=4, line_alpha=1.0
        )

        graph_layout = StaticLayoutProvider(graph_layout=pos)
        graph_renderer.layout_provider = graph_layout

        # Define inspection and selection policies (optional but can be useful)
        # graph_renderer.selection_policy = None # Use TapTool for selection logic via callback
        # graph_renderer.inspection_policy = None # Use HoverTool for inspection

        plot.renderers.append(graph_renderer)

        # --- 5. Configure HoverTool ---
        node_hover_tooltips = [
            ("Label", "@vizLabel"),
            ("Type", "@nodeType"),
            ("Node ID", "@node_id"), # Use node_id field
        ]
        # Filter tooltips
        valid_node_tooltips = [(label, field) for label, field in node_hover_tooltips
                               if field.split('{')[0].lstrip('@') in node_viz_info]
        node_hover = HoverTool(tooltips=valid_node_tooltips, renderers=[graph_renderer.node_renderer])

        edge_hover_tooltips = [
            ("Label", "@vizLabel{safe}"),
            ("Type", "@edgeType"),
            ("Weight", "@weight"), # Show original weight if available
        ]
        # Add weight to edge_viz_info if not already present for hover
        if 'weight' not in edge_viz_info:
             edge_viz_info['weight'] = [data.get('weight', 'N/A') for data in edge_data_list]
             edge_source.data['weight'] = edge_viz_info['weight'] # Update source if added

        valid_edge_tooltips = [(label, field) for label, field in edge_hover_tooltips
                               if field.split('{')[0].lstrip('@') in edge_viz_info]
        edge_hover = HoverTool(tooltips=valid_edge_tooltips, renderers=[graph_renderer.edge_renderer])

        # Add Hover tools BEFORE Tap tool might be better
        plot.add_tools(node_hover, edge_hover)

        # --- 6. Setup TapTool and Details Div ---
        detail_div = Div(text="<p>Click on a node or edge to see details here.</p>", width=300, height=plot_height - 100) # Adjust size

        # CustomJS Callback for Tap interaction
        # This JS code runs in the browser when a node or edge selection changes
        callback_code = """
            const node_indices = node_source.selected.indices;
            const edge_indices = edge_source.selected.indices;
            let details_html = "<p>Click on a node or edge to see details here.</p>";

            // Function to format dictionary details into HTML list
            const formatDetails = (details) => {
                let html = "<ul>";
                for (const [key, value] of Object.entries(details)) {
                    // Basic formatting, handle objects/arrays if necessary
                    let displayValue = value;
                    if (typeof value === 'object' && value !== null) {
                        try {
                            displayValue = JSON.stringify(value, null, 2); // Pretty print JSON
                             displayValue = '<pre style="white-space: pre-wrap; word-break: break-all;">' + displayValue + '</pre>'; // Use pre for formatting
                        } catch (e) {
                            displayValue = '[Object]'; // Fallback
                        }
                    } else if (value === null) {
                         displayValue = '<i>null</i>';
                    } else if (value === undefined) {
                         displayValue = '<i>undefined</i>';
                    } else {
                        // Escape HTML to prevent injection issues if data isn't trusted
                        const escapeHtml = (unsafe) => {
                             if (typeof unsafe !== 'string') return unsafe;
                             return unsafe
                                 .replace(/&/g, "&amp;")
                                 .replace(/</g, "&lt;")
                                 .replace(/>/g, "&gt;")
                                 .replace(/"/g, "&quot;")
                                 .replace(/'/g, "&#039;");
                        }
                        displayValue = escapeHtml(String(value));
                    }

                    // Escape key as well, though less likely to contain HTML
                     const escapedKey = String(key)
                         .replace(/&/g, "&amp;")
                         .replace(/</g, "&lt;")
                         .replace(/>/g, "&gt;");


                    html += `<li><b>${escapedKey}:</b> ${displayValue}</li>`;
                }
                html += "</ul>";
                return html;
            };

            if (node_indices.length > 0) {
                const index = node_indices[0]; // Get the first selected node index
                const node_data = node_source.data;
                const node_label = node_data['vizLabel'][index];
                const node_details = node_data['nodeDetail'][index]; // Get the details dict
                details_html = `<h3>Node Details: ${node_label} (ID: ${node_data['node_id'][index]})</h3>`;
                if (node_details && typeof node_details === 'object' && Object.keys(node_details).length > 0) {
                    details_html += formatDetails(node_details);
                } else {
                    details_html += "<p>No additional details available.</p>";
                }

            } else if (edge_indices.length > 0) {
                const index = edge_indices[0]; // Get the first selected edge index
                const edge_data = edge_source.data;
                 const edge_label = edge_data['vizLabel'][index]; // Use pre-formatted label
                const edge_details = edge_data['edgeDetail'][index]; // Get the details dict
                details_html = `<h3>Edge Details: ${edge_label}</h3>`;
                 if (edge_details && typeof edge_details === 'object' && Object.keys(edge_details).length > 0) {
                    details_html += formatDetails(edge_details);
                } else {
                    details_html += "<p>No additional details available.</p>";
                }
            }

            detail_div.text = details_html; // Update the Div content
        """

        callback = CustomJS(args=dict(node_source=node_source, edge_source=edge_source, detail_div=detail_div),
                            code=callback_code)

        # Trigger callback when selection indices change for either nodes or edges
        node_source.selected.js_on_change('indices', callback)
        edge_source.selected.js_on_change('indices', callback)

        # Add TapTool - essential for selection via clicking
        # Configure TapTool to affect both node and edge renderers.
        # 'select' mode automatically updates the source's selected indices.
        tap_tool = TapTool(renderers=[graph_renderer.node_renderer, graph_renderer.edge_renderer], behavior="select")
        plot.add_tools(tap_tool)

        # --- 7. Create Legends ---
        print("Creating legends...")
        node_legend_div = self._create_legend(self.node_color_map, "Node Types")
        edge_legend_div = self._create_legend(self.edge_color_map, "Edge Types")

        # --- 8. Arrange Layout ---
        print("Arranging layout...")
        # Combine legends and details vertically
        sidebar = column(node_legend_div, edge_legend_div, Div(height=20), detail_div, width=320) # Add spacer

        # Combine plot and sidebar horizontally
        layout = row(plot, sidebar)

        # --- 9. Show Plot ---
        print("Displaying plot.")
        show(layout)
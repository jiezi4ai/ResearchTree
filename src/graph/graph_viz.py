# graph_viz.py
import math
import networkx as nx
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional, Tuple

from bokeh.plotting import figure, show
from bokeh.models import (Scatter, MultiLine, Div, CustomJS, TapTool,
                          HoverTool, LabelSet, ColumnDataSource, Plot, TextInput,
                          GraphRenderer, StaticLayoutProvider, Paragraph)
from bokeh.layouts import column, row
from bokeh.palettes import Category10, Pastel1, brewer, Set3

# =============================================================================
# Graph Viz Preprocessing Functions (Keep as they are from previous version)
# =============================================================================
def assign_node_size(
        G,
        sig_nid_lst: Optional[List[str]] = None,
        min_node_size: Optional[int] = 100,
        max_node_size: Optional[int] = 900,
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
            node_data['vizSize'] = math.ceil(math.sqrt(max(min_node_size, min(max_node_size, node_size))))
        elif node_type == 'Author' and nid in author_writes_ref:
            value = author_writes_ref[nid]
            node_size = min_node_size + ((max_node_size - min_node_size) * (value - min_writes_cnt)) / writes_range
            node_data['vizSize'] = math.ceil(math.sqrt(max(min_node_size, min(max_node_size, node_size))))


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
        default_colormap_name: Optional[str] = 'Set3', # Using Bokeh palette name
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
    colors_hex = []
    try:
        # Try getting palette from Bokeh palettes module
        if default_colormap_name == 'Set3':
             base_palette = Set3[max(3, default_color_cnt)] # Set3 has 3-10
        elif default_colormap_name in brewer:
             # Brewer palettes often need a specific number (e.g., Paired9)
             # Find the largest available size <= default_color_cnt
             available_sizes = list(brewer[default_colormap_name].keys())
             use_size = max(s for s in available_sizes if s <= default_color_cnt) if any(s <= default_color_cnt for s in available_sizes) else min(available_sizes)
             base_palette = brewer[default_colormap_name][max(3, use_size)]
        else:
             # Fallback or other palettes
             base_palette = Set3[max(3, default_color_cnt)] # Default fallback

        if unique_node_cnt > 0:
             if unique_node_cnt <= len(base_palette):
                 colors_hex = list(base_palette)[:unique_node_cnt]
             else:
                 colors_hex = list(base_palette) + ['#808080'] * (unique_node_cnt - len(base_palette))

    except Exception as e: # Catch potential errors like missing palette name
         print(f"Warning: Could not load Bokeh palette '{default_colormap_name}'. Error: {e}. Falling back.")
         # Fallback logic (e.g., use Set3 or manual list)
         base_palette = Set3[max(3, default_color_cnt)]
         if unique_node_cnt > 0:
             if unique_node_cnt <= len(base_palette):
                 colors_hex = list(base_palette)[:unique_node_cnt]
             else:
                 colors_hex = list(base_palette) + ['#808080'] * (unique_node_cnt - len(base_palette))

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
    colors_hex = []
    try:
         if default_colormap_name == 'Pastel1':
              base_palette = Pastel1[max(3, default_color_cnt)] # Pastel1 has 3-9
         elif default_colormap_name in brewer:
              available_sizes = list(brewer[default_colormap_name].keys())
              use_size = max(s for s in available_sizes if s <= default_color_cnt) if any(s <= default_color_cnt for s in available_sizes) else min(available_sizes)
              base_palette = brewer[default_colormap_name][max(3, use_size)]
         else:
              base_palette = Pastel1[max(3, default_color_cnt)] # Default fallback

         if unique_edge_cnt == 0:
             colors_hex = []
         elif unique_edge_cnt <= len(base_palette):
             colors_hex = list(base_palette)[:unique_edge_cnt]
         else:
             colors_hex = list(base_palette) + ['#D3D3D3'] * (unique_edge_cnt - len(base_palette)) # Light grey for extras

    except Exception as e:
         print(f"Warning: Could not load Bokeh palette '{default_colormap_name}' for edges. Error: {e}. Falling back.")
         # Fallback logic
         base_palette = Pastel1[max(3, default_color_cnt)]
         if unique_edge_cnt == 0:
             colors_hex = []
         elif unique_edge_cnt <= len(base_palette):
             colors_hex = list(base_palette)[:unique_edge_cnt]
         else:
             colors_hex = list(base_palette) + ['#D3D3D3'] * (unique_edge_cnt - len(base_palette))


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
        node_data['nodeDetail'] = {key: node_data[key] for key in keys}


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
        self.node_color_map = assign_node_color(self.graph)
        self.edge_color_map = assign_edge_color(self.graph)
        update_node_keys(self.graph, node_key_ref)
        update_edges_keys(self.graph)
        print("Preprocessing complete.")

    def _create_legend(self, color_map: Dict[str, str], title: str) -> Div:
        """Helper function to create an HTML legend Div."""
        legend_html = f"<b>{title}</b><br>"
        item_style = "margin-bottom: 5px; display: flex; align-items: center;"
        color_box_style = "width: 15px; height: 15px; margin-right: 5px; border: 1px solid #ccc; display: inline-block;"
        text_style = "font-size: 0.8em;"
        sorted_items = sorted(color_map.items())
        for item_type, color in sorted_items:
            # Escape HTML in item_type to prevent XSS if type names are unsafe
            escaped_item_type = item_type.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            legend_html += f'<div style="{item_style}">'
            legend_html += f'<span style="{color_box_style} background-color: {color};"></span>'
            legend_html += f'<span style="{text_style}">{escaped_item_type}</span>'
            legend_html += '</div>'
        # Calculate height dynamically, with a min/max if needed
        calculated_height = len(color_map) * 22 + 40 # Approx height per item + title/padding
        return Div(text=legend_html, width=150, height=min(300, calculated_height), # Max height
                   styles={'overflow-y': 'auto' if calculated_height > 300 else 'visible'}) # Add scroll if too tall


    def visulization(self):
        """Visualize the graph with legends, interactive details, and search widgets."""
        # --- 1. Calculate Layout ---
        print("Calculating layout...")
        pos = {} # Define pos outside try block
        try:
            # Check graph size, Kamada-Kawai can be slow for large graphs
            if self.graph.number_of_nodes() < 500:
                 pos = nx.kamada_kawai_layout(self.graph, weight='weight')
                 print("Layout calculated using kamada_kawai_layout.")
            else:
                 print("Graph is large, using spring_layout (may take time)...")
                 pos = nx.spring_layout(self.graph, k=0.3, iterations=50, seed=42, weight='weight')
                 print("Layout calculated using spring_layout.")
        except Exception as e_kk:
            print(f"Kamada-Kawai failed or skipped ({e_kk}), trying spring_layout.")
            try:
                pos = nx.spring_layout(self.graph, k=0.5, iterations=100, seed=42, weight='weight')
                print("Layout calculated using spring_layout.")
            except Exception as e_spring:
                print(f"Spring layout also failed ({e_spring}), using random layout.")
                pos = nx.random_layout(self.graph, seed=42)
                print("Layout calculated using random_layout.")

        # --- 2. Prepare Data Sources ---
        print("Preparing data sources...")
        node_ids = list(self.graph.nodes())
        node_viz_info = dict(
            index=node_ids, # IMPORTANT: GraphRenderer uses 'index' for mapping node IDs
            node_id=node_ids,# Keep separate 'node_id' column for explicit reference if needed
            x=[pos[nid][0] for nid in node_ids],
            y=[pos[nid][1] for nid in node_ids],
            nodeType=[node_data.get('nodeType', 'Unknown/Default') for nid, node_data in self.graph.nodes(data=True)],
            nodeDetail=[node_data.get('nodeDetail', {}) for nid, node_data in self.graph.nodes(data=True)],
            vizSize=[node_data.get('vizSize', 10) for nid, node_data in self.graph.nodes(data=True)],
            vizColor=[node_data.get('vizColor', '#CCCCCC') for nid, node_data in self.graph.nodes(data=True)],
            vizBorderColor=[node_data.get('vizBorderColor', '#888888') for nid, node_data in self.graph.nodes(data=True)],
            vizBorderWidth=[node_data.get('vizBorderWidth', 1) for nid, node_data in self.graph.nodes(data=True)],
            vizLabel=[str(node_data.get('vizLabel', nid)) for nid, node_data in self.graph.nodes(data=True)], # Ensure label is string
        )

        edge_starts = []
        edge_ends = []
        edge_data_list = []
        for i, (u, v, data) in enumerate(self.graph.edges(data=True)):
            edge_starts.append(u)
            edge_ends.append(v)
            # Add the original weight back if needed for hover/search
            data['weight'] = data.get('weight', 'N/A')
             # Ensure edgeDetail exists
            if 'edgeDetail' not in data: data['edgeDetail'] = {}
            # Ensure vizLabel exists and is string
            if 'vizLabel' not in data: data['vizLabel'] = f"{u}->{v}"
            else: data['vizLabel'] = str(data['vizLabel'])

            edge_data_list.append(data)


        edge_ids = list(range(len(edge_starts))) # Index based ID

        edge_viz_info = dict(
            start=edge_starts,
            end=edge_ends,
            edge_id=edge_ids, # Keep simple index for internal reference
            edgeType=[data.get('relationshipType', 'Unknown/Default') for data in edge_data_list],
            edgeDetail=[data.get('edgeDetail', {}) for data in edge_data_list],
            vizWidth=[data.get('vizWidth', 1) for data in edge_data_list],
            vizColor=[data.get('vizColor', '#AAAAAA') for data in edge_data_list],
            vizLabel=[data.get('vizLabel', '') for data in edge_data_list],
            weight=[data.get('weight', 'N/A') for data in edge_data_list] # Add weight here
        )

        node_source = ColumnDataSource(data=node_viz_info, name="Node Source")
        edge_source = ColumnDataSource(data=edge_viz_info, name="Edge Source")

        # --- 3. Create Bokeh Plot ---
        print("Creating plot...")
        plot_width = 800
        plot_height = 700 # Increased height slightly for widgets maybe
        x_coords = [d[0] for d in pos.values()]
        y_coords = [d[1] for d in pos.values()]
        x_range = (min(x_coords)*1.1 - 0.1, max(x_coords)*1.1 + 0.1)
        y_range = (min(y_coords)*1.1 - 0.1, max(y_coords)*1.1 + 0.1)

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
        graph_renderer.node_renderer.selection_glyph = Scatter(
            size='vizSize', fill_color='vizColor', line_color='firebrick', line_width=4
        )
        graph_renderer.node_renderer.hover_glyph = Scatter(
            size='vizSize', fill_color='vizColor', line_color='deepskyblue', line_width=3
        )

        graph_renderer.edge_renderer.data_source = edge_source
        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="vizColor", line_alpha=0.6, line_width='vizWidth'
        )
        graph_renderer.edge_renderer.selection_glyph = MultiLine(
            line_color='firebrick', line_width=5, line_alpha=1.0
        )
        graph_renderer.edge_renderer.hover_glyph = MultiLine(
            line_color='deepskyblue', line_width=4, line_alpha=1.0
        )

        graph_layout = StaticLayoutProvider(graph_layout=pos)
        graph_renderer.layout_provider = graph_layout
        plot.renderers.append(graph_renderer)

        # --- 5. Configure HoverTool ---
        node_hover_tooltips = [("Label", "@vizLabel"), ("Type", "@nodeType"), ("Node ID", "@node_id")]
        valid_node_tooltips = [(label, field) for label, field in node_hover_tooltips
                               if field.split('{')[0].lstrip('@') in node_viz_info]
        node_hover = HoverTool(tooltips=valid_node_tooltips, renderers=[graph_renderer.node_renderer])

        edge_hover_tooltips = [("Label", "@vizLabel{safe}"), ("Type", "@edgeType"), ("Weight", "@weight")]
        valid_edge_tooltips = [(label, field) for label, field in edge_hover_tooltips
                               if field.split('{')[0].lstrip('@') in edge_viz_info]
        edge_hover = HoverTool(tooltips=valid_edge_tooltips, renderers=[graph_renderer.edge_renderer])

        plot.add_tools(node_hover, edge_hover)

        # --- 6. Setup TapTool and Details Div ---
        detail_div_height = 200
        detail_div = Div(text="<p>Click on a node or edge to see details.</p>",
                         width=300, height=detail_div_height, sizing_mode="stretch_width",
                         styles={'overflow-y': 'auto', 'border-top': '1px solid #ccc', 'padding-top': '5px'})

        # CustomJS Callback for Tap interaction - REVISED WITH DEBUGGING
        tap_callback_code = """
            console.log("Tap callback triggered."); // DEBUG: Check if callback runs

            // --- DEFINE escapeHtml function in the TOP scope ---
            const escapeHtml = (unsafe) => {
                 if (typeof unsafe !== 'string') {
                    try {
                       // Attempt to stringify non-strings before escaping
                       unsafe = JSON.stringify(unsafe);
                    } catch (e) {
                       return '[Unescapable Value]'; // Fallback if stringify fails
                    }
                 }
                 return unsafe
                     .replace(/&/g, "&amp;")
                     .replace(/</g, "&lt;")
                     .replace(/>/g, "&gt;")
                     .replace(/"/g, "&quot;")
                     .replace(/'/g, "&#039;");
             };
            // --- END escapeHtml definition ---

            const node_indices = node_source.selected.indices;
            const edge_indices = edge_source.selected.indices;
            console.log("Node indices:", node_indices, "Edge indices:", edge_indices); // DEBUG: Check selected indices

            let details_html = "<p>Click on a node or edge to see details.</p>"; // Default text

            // Function to format dictionary details into HTML list
            // It will now use the globally defined escapeHtml
            const formatDetails = (details) => {
                let html = "<ul style='margin:0; padding-left: 20px; list-style-type: disc;'>";
                if (typeof details !== 'object' || details === null) {
                    console.error("formatDetails expected an object, but received:", details);
                    return "<li>Error: Invalid details format.</li>";
                }

                for (const [key, value] of Object.entries(details)) {
                    let displayValue;
                    if (value === null) {
                         displayValue = '<i>null</i>';
                    } else if (value === undefined) {
                         displayValue = '<i>undefined</i>';
                    } else if (typeof value === 'object') {
                        try {
                            const jsonString = JSON.stringify(value, null, 2);
                            // Use the globally defined escapeHtml here
                            const escapedJson = escapeHtml(jsonString);
                            displayValue = '<pre style="white-space: pre-wrap; word-break: break-all; margin:0; padding: 2px; background-color: #f8f8f8; border: 1px solid #eee; font-size: 0.9em;">' + escapedJson + '</pre>';
                        } catch (e) {
                            console.error("Error stringifying object:", value, e);
                            displayValue = '[Object - Error Displaying]';
                        }
                    } else {
                        // Use the globally defined escapeHtml here
                        displayValue = escapeHtml(String(value));
                    }
                     // Use the globally defined escapeHtml here
                     const escapedKey = escapeHtml(String(key));
                    html += `<li style='margin-bottom: 3px;'><b>${escapedKey}:</b> ${displayValue}</li>`;
                }
                html += "</ul>";
                return html;
            };

            try { // Add a try-catch block for the main logic
                if (node_indices.length > 0) {
                    const index = node_indices[0];
                    console.log("Processing selected node index:", index);
                    const node_data = node_source.data;
                    if (index < 0 || index >= node_data['node_id'].length) {
                         console.error("Selected node index out of bounds:", index);
                         throw new Error("Selected node index out of bounds.");
                    }

                    // Use the globally defined escapeHtml here
                    const node_label = escapeHtml(String(node_data['vizLabel'][index] || 'N/A'));
                    const node_id = escapeHtml(String(node_data['node_id'][index] || 'N/A'));
                    const node_details = node_data['nodeDetail'][index];

                    console.log("Node Details object:", node_details);

                    details_html = `<h6>Node: ${node_label} (ID: ${node_id})</h6>`;
                    if (node_details && typeof node_details === 'object' && Object.keys(node_details).length > 0) {
                        details_html += formatDetails(node_details);
                    } else {
                        details_html += "<p style='font-style: italic; font-size: 0.9em;'>No additional details.</p>";
                    }

                } else if (edge_indices.length > 0) {
                    const index = edge_indices[0];
                    console.log("Processing selected edge index:", index);
                    const edge_data = edge_source.data;
                    if (index < 0 || index >= edge_data['start'].length) {
                         console.error("Selected edge index out of bounds:", index);
                         throw new Error("Selected edge index out of bounds.");
                    }

                    // Use the globally defined escapeHtml here
                    const edge_label = escapeHtml(String(edge_data['vizLabel'][index] || 'N/A'));
                    const edge_details = edge_data['edgeDetail'][index];

                    console.log("Edge Details object:", edge_details);

                    details_html = `<h6>Edge: ${edge_label}</h6>`;
                    if (edge_details && typeof edge_details === 'object' && Object.keys(edge_details).length > 0) {
                        details_html += formatDetails(edge_details);
                    } else {
                        details_html += "<p style='font-style: italic; font-size: 0.9em;'>No additional details.</p>";
                    }
                } else {
                   console.log("No node or edge selected.");
                   details_html = "<p>Click on a node or edge to see details.</p>";
                }

                console.log("Updating detail_div.text with HTML:", details_html);
                detail_div.text = details_html;

            } catch (error) {
                console.error("Error in tap_callback:", error);
                detail_div.text = "<p style='color: red;'><b>Error displaying details.</b><br>Check browser console (F12) for more info.</p>";
            }
        """

        # Create the CustomJS object (args remain the same)
        tap_callback = CustomJS(args=dict(node_source=node_source, edge_source=edge_source, detail_div=detail_div),
                                code=tap_callback_code)

        # Attach the callback (remains the same)
        node_source.selected.js_on_change('indices', tap_callback)
        edge_source.selected.js_on_change('indices', tap_callback)

        # Add TapTool (remains the same)
        tap_tool = TapTool(renderers=[graph_renderer.node_renderer, graph_renderer.edge_renderer], behavior="select")
        plot.add_tools(tap_tool)

        # --- 7. Setup Search Widgets and Callbacks ---
        print("Creating search widgets...")
        node_search_input = TextInput(title="Search Nodes (ID or Label):", placeholder="Enter Node ID or part of Label")
        edge_search_input = TextInput(title="Search Edges (Label):", placeholder="Enter part of Edge Label")

        # JS code for node search
        node_search_code = """
            const search_term = node_search_input.value.trim().toLowerCase();
            const node_data = node_source.data;
            const node_ids = node_data['node_id']; // Use node_id column
            const node_labels = node_data['vizLabel'];
            let selected_indices = [];

            // Clear previous search selection
            edge_source.selected.indices = [];

            if (search_term === "") {
                // If search is empty, clear selection
                node_source.selected.indices = [];
                // Optionally trigger the detail update to show default message
                // detail_div.text = "<p>Click on a node or edge to see details.</p>"; // Or trigger tap_callback?
            } else {
                // 1. Try exact match on node_id first
                const exact_id_index = node_ids.indexOf(node_search_input.value.trim()); // Use original case for ID match if IDs are case-sensitive

                if (exact_id_index !== -1) {
                    selected_indices.push(exact_id_index);
                } else {
                    // 2. If no exact ID match, perform case-insensitive substring match on label
                    for (let i = 0; i < node_labels.length; i++) {
                        if (node_labels[i] && node_labels[i].toLowerCase().includes(search_term)) {
                            selected_indices.push(i);
                        }
                    }
                }
                 // Remove duplicates just in case, though logic should prevent it
                // selected_indices = [...new Set(selected_indices)];
                node_source.selected.indices = selected_indices;
            }
            // Trigger update on source to reflect changes (might not be strictly needed if selection_glyph works)
            node_source.change.emit();
        """

        # JS code for edge search
        edge_search_code = """
            const search_term = edge_search_input.value.trim().toLowerCase();
            const edge_data = edge_source.data;
            const edge_labels = edge_data['vizLabel'];
            let selected_indices = [];

            // Clear previous search selection
            node_source.selected.indices = [];

            if (search_term === "") {
                edge_source.selected.indices = [];
                // Optionally trigger the detail update
                // detail_div.text = "<p>Click on a node or edge to see details.</p>";
            } else {
                // Perform case-insensitive substring match on label
                for (let i = 0; i < edge_labels.length; i++) {
                    // Check if label exists before calling toLowerCase
                    if (edge_labels[i] && edge_labels[i].toLowerCase().includes(search_term)) {
                        selected_indices.push(i);
                    }
                }
                edge_source.selected.indices = selected_indices;
            }
            edge_source.change.emit();
        """

        node_search_callback = CustomJS(args=dict(node_search_input=node_search_input,
                                                  node_source=node_source,
                                                  edge_source=edge_source, # Need to clear edge selection
                                                  detail_div=detail_div), # Pass if needed to update details
                                        code=node_search_code)

        edge_search_callback = CustomJS(args=dict(edge_search_input=edge_search_input,
                                                  edge_source=edge_source,
                                                  node_source=node_source, # Need to clear node selection
                                                  detail_div=detail_div), # Pass if needed to update details
                                        code=edge_search_code)

        # Trigger search on input value change
        node_search_input.js_on_change('value', node_search_callback)
        edge_search_input.js_on_change('value', edge_search_callback)

        # --- 8. Create Legends ---
        print("Creating legends...")
        node_legend_div = self._create_legend(self.node_color_map, "Node Types")
        edge_legend_div = self._create_legend(self.edge_color_map, "Edge Types")

        # --- 9. Arrange Layout ---
        print("Arranging layout...")
        # Calculate heights for sidebar components
        legends_height = node_legend_div.height + edge_legend_div.height + 20 # Legends + spacer
        inputs_height = 120 # Approximate height for two text inputs with titles
        details_max_height = plot_height - legends_height - inputs_height - 20 # Remaining height for details, with padding
        detail_div.height = max(100, details_max_height) # Ensure a minimum height for details

        sidebar = column(
            node_search_input,
            edge_search_input,
            Div(height=10, styles={'border-top': '1px solid #eee', 'margin-top': '10px'}), # Separator
            node_legend_div,
            edge_legend_div,
            # Div(height=10, styles={'border-top': '1px solid #eee', 'margin-top': '10px'}), # Separator
            detail_div, # Details at the bottom
            width=320,
            height=plot_height, # Match plot height
            sizing_mode="fixed" # Fixed width for sidebar
        )

        layout = row(plot, sidebar)

        # --- 10. Show Plot ---
        print("Displaying plot.")
        show(layout)
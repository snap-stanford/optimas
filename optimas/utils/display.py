import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path 
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.tree import Tree
from rich import box

def display_system_overview(system) -> None:
    """Standalone visualization for a CompoundAISystem instance."""
    console = Console()

    def truncate_text(text: str, max_chars: int = 100) -> str:
        return (text[:max_chars] + " ...") if len(text) > max_chars else text

    def print_logo():
        console.print(f'\n{"Welcome to".center(65)}', style="bold cyan")
        logo = r"""
       ██████  ██████  ████████ ██ ███    ███  █████  ███████ 
      ██    ██ ██   ██    ██    ██ ████  ████ ██   ██ ██      
      ██    ██ ██████     ██    ██ ██ ████ ██ ███████ ███████ 
      ██    ██ ██         ██    ██ ██  ██  ██ ██   ██      ██ 
       ██████  ██         ██    ██ ██      ██ ██   ██ ███████ 
    """
        console.print(logo, style="bold blue")

    def print_system_config():
        config_items = [
            ("🔍 Required Inputs", ', '.join(system.required_input_fields) or 'None'),
            ("🎯 Final Outputs", ', '.join(system.final_output_fields)),
            ("📏 Ground Fields", ', '.join(system.ground_fields) or 'None'),
        ]
        if system.eval_func:
            config_items.append(("📊 Evaluation Function", getattr(system.eval_func, '__name__', str(system.eval_func))))
        config_items.extend([
            ("👥 Max Sample Workers", str(system.max_sample_workers)),
            ("⚖️ Max Eval Workers", str(system.max_eval_workers)),
        ])

        config_text = Text()
        for label, value in config_items:
            config_text.append(f"{label}: ", style="bold cyan")
            config_text.append(f"{value}\n", style="white")

        console.print(Panel(config_text, title="System Config", border_style="blue"))

    def display_visual_graph():
        if not system.execution_order:
            return

        G = nx.DiGraph()
        for name in system.execution_order:
            G.add_node(name, optimizable=getattr(system.components[name], 'optimizable', False))
        for comp_name, predecessors in system.predecessor_map.items():
            for pred in predecessors:
                G.add_edge(pred, comp_name)

        plt.figure(figsize=(14, 10))
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=2, iterations=100)

        opt_nodes = [n for n, d in G.nodes(data=True) if d.get('optimizable')]
        fixed_nodes = [n for n, d in G.nodes(data=True) if not d.get('optimizable')]

        nx.draw_networkx_edges(
            G, pos,
            edge_color='#444444',
            arrows=True,
            arrowsize=20,
            width=2,
            connectionstyle='arc3,rad=0.1'
        )
        nx.draw_networkx_nodes(G, pos, nodelist=opt_nodes, node_color='#4CAF50', node_size=2500, alpha=0.95)
        nx.draw_networkx_nodes(G, pos, nodelist=fixed_nodes, node_color='#9E9E9E', node_size=2500, alpha=0.95)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white')

        plt.title('🤖 Component Dependency Graph', fontsize=16, fontweight='bold', pad=20)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='🔧 Optimizable', markerfacecolor='#4CAF50', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='⚙️ Fixed', markerfacecolor='#9E9E9E', markersize=15)
        ]
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True)
        plt.axis('off')
        plt.tight_layout()
        plt.figtext(0.5, 0.02, f"Execution Order: {' → '.join(system.execution_order)}", ha='center', fontsize=12, style='italic')
        plt.show()

    def format_search_space(comp) -> str:
        if not getattr(comp, 'variable_search_space', None):
            return "None"
        parts = []
        for var, values in comp.variable_search_space.items():
            if isinstance(values, list):
                parts.append(f"{var}: {values}" if len(values) <= 5 else f"{var}: [{len(values)} options]")
            elif isinstance(values, range):
                parts.append(f"{var}: range({values.start}, {values.stop})")
            else:
                parts.append(f"{var}: {len(values)} items")
        return "\n".join(parts)

    def format_config(comp) -> str:
        if not getattr(comp, 'config', None):
            return "None"
        items = []
        for attr in dir(comp.config):
            if not attr.startswith('_'):
                try:
                    val = getattr(comp.config, attr)
                    if val is not None and not callable(val):
                        items.append((attr, str(val)))
                except:
                    continue
        priority = ['model', 'temperature', 'max_tokens', 'top_p', 'top_k']
        pri_items = [(k, v) for k, v in items if k in priority]
        other_items = [(k, v) for k, v in items if k not in priority]
        shown = pri_items + other_items[:3]
        if len(other_items) > 3:
            shown.append(("...", f"({len(other_items) - 3} more)"))
        return "\n".join([f"{k}: {v}" for k, v in shown]) or "None"

    def get_component_row(i: int, name: str, comp) -> tuple:
        is_opt = "🔧 Yes" if getattr(comp, 'optimizable', False) else "⚙️ No"
        var = getattr(comp, 'variable', None)
        if not getattr(comp, 'optimizable', False):
            return str(i), name, is_opt, "N/A", "N/A", "N/A", format_config(comp)
        
        if isinstance(var, Path): 
            vtype = 'Local Model'
            vdisp = truncate_text(str(var), 100)
            sspace = "None"
        elif isinstance(var, str):
            vtype = "Prompt"
            vdisp = truncate_text(var, 100)
            sspace = "None"
        elif isinstance(var, dict):
            vtype = "Hyperparameter / Model Selection"
            vdisp = truncate_text("\n".join([f"{k}: {v}" for k, v in var.items()]), 100)
            raw_sspace = format_search_space(comp)
            sspace = truncate_text(raw_sspace, 300)
        else:
            vtype, vdisp, sspace = "Unknown", truncate_text(str(var) or "None", 100), "None"

        return str(i), name, is_opt, vtype, vdisp, sspace, format_config(comp)

    def print_components_table():
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, show_lines=True)
        headers = ["Step", "Component", "Is Optimizable", "Variable Type", "Initial Variable", "Search Space", "Other Configs"]
        styles = ["cyan", "green", "yellow", "magenta", "blue", "white", "red"]
        widths = [4, 20, 12, 15, 25, 20, 35]
        for h, s, w in zip(headers, styles, widths):
            table.add_column(h, style=s, width=w)
        for i, name in enumerate(system.execution_order):
            table.add_row(*get_component_row(i+1, name, system.components[name]))
        console.print(table)

    def print_reward_model_info():
        if not getattr(system, 'rm', None):
            return
        items = ["🏆 REWARD MODEL ACTIVE\n"]
        if getattr(system, 'sample_size', 1) > 1:
            items.append(f"📊 Sample Size: {system.sample_size}")
        if getattr(system, 'sample_temperature', None):
            items.append(f"🌡️ Sample Temperature: {system.sample_temperature}")
        targets = ', '.join(system.components_to_apply) if system.components_to_apply else 'None'
        items.append(f"🎯 Applied to: {targets}")
        console.print(Panel(Text("\n".join(items), style="yellow"), title="Reward Model", border_style="yellow"))

    def create_dependency_tree() -> Tree:
        tree = Tree("🌳 [bold cyan]Component Dependencies[/bold cyan]")
        
        # Build a mapping of output -> component that produces it
        output_to_component = {}
        for comp_name, comp in system.components.items():
            outputs = getattr(comp, 'output_fields', [])
            for output in outputs:
                output_to_component[output] = comp_name
        
        # Find components that can be placed under each component
        def get_dependent_components(comp_name):
            """Get components that directly depend on this component's outputs"""
            comp_outputs = getattr(system.components[comp_name], 'output_fields', [])
            dependents = []
            
            for other_name, other_comp in system.components.items():
                if other_name == comp_name:
                    continue
                other_inputs = getattr(other_comp, 'input_fields', [])
                # Check if this component uses any outputs from comp_name
                if any(inp in comp_outputs for inp in other_inputs):
                    dependents.append(other_name)
            
            return dependents
        
        # Build tree based on data flow dependencies
        added = set()
        
        def build_node(parent, comp_name, depth=0):
            if comp_name in added or depth > 10:  # Prevent infinite loops
                return
            
            comp = system.components[comp_name]
            icon = "🔧" if getattr(comp, 'optimizable', False) else "⚙️"
            style = "bold green" if getattr(comp, 'optimizable', False) else "dim white"
            label = f"{icon} [{style}]{comp_name}[/{style}]"
            
            inputs = getattr(comp, 'input_fields', [])
            outputs = getattr(comp, 'output_fields', [])
            
            # Show input dependencies
            if inputs:
                input_sources = []
                for inp in inputs:
                    if inp in output_to_component:
                        input_sources.append(f"{inp} ← {output_to_component[inp]}")
                    else:
                        input_sources.append(f"{inp} ← [external]")
                label += f"\n    [dim cyan]📥 Inputs: {'; '.join(input_sources)}[/dim cyan]"
            
            if outputs:
                label += f"\n    [dim red]📤 Outputs: {', '.join(outputs)}[/dim red]"
            
            node = parent.add(label)
            added.add(comp_name)
            
            # Find components that depend on this one and add them as children
            dependents = get_dependent_components(comp_name)
            # Sort by execution order to maintain some structure
            dependents.sort(key=lambda x: system.execution_order.index(x) if x in system.execution_order else float('inf'))
            
            for dependent in dependents:
                if dependent not in added:
                    build_node(node, dependent, depth + 1)
        
        # Start with root components (those with no dependencies on other components)
        roots = []
        for comp_name in system.execution_order:
            comp = system.components[comp_name]
            inputs = getattr(comp, 'input_fields', [])
            # Check if all inputs come from external sources
            has_internal_deps = any(inp in output_to_component for inp in inputs)
            if not has_internal_deps:
                roots.append(comp_name)
        
        # If no clear roots found, start with the first component
        if not roots:
            roots = [system.execution_order[0]]
        
        for root in roots:
            build_node(tree, root)
        
        return tree

    # === Final Display ===
    print_logo()
    print_system_config()
    print_components_table()
    print_reward_model_info()
    if len(system.execution_order) > 1:
        console.print(create_dependency_tree())
    console.print("\n✅ [bold green]System ready![/bold green]")

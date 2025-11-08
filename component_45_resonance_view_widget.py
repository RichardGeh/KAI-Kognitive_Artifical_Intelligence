"""
component_45_resonance_view_widget.py

Interactive Resonance Visualization Widget

Features:
- Activation Heatmap: Color-coded nodes (red=high activation, blue=low)
- Wave Animation: Step-by-step spreading activation visualization
- Resonance Points: Highlighted nodes with multiple convergent paths
- Interactive: Click on concept to see detailed explanation
- Graph Layout: NetworkX spring/hierarchical layout

Technology:
- PySide6 for UI
- NetworkX for graph layout
- Matplotlib for visualization

Part of Phase 2.2: Cognitive Resonance Core - Visualization

Author: KAI Development Team
Created: 2025-11-07
"""

import logging
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Import matplotlib for embedding in Qt
try:
    import matplotlib

    matplotlib.use("Qt5Agg")  # Use Qt5 backend for PySide6 compatibility
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, ResonanceView will be limited")

# Import NetworkX for graph layout
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available, using fallback layout")

# Import ResonanceEngine types
try:
    from component_44_resonance_engine import (
        ActivationMap,
    )

    RESONANCE_ENGINE_AVAILABLE = True
except ImportError:
    RESONANCE_ENGINE_AVAILABLE = False
    logger.warning("ResonanceEngine not available")


class ResonanceViewWidget(QWidget):
    """
    Interactive visualization widget for Resonance Engine activation maps.

    Features:
    - Activation heatmap with color-coded nodes
    - Wave-by-wave animation
    - Resonance point highlighting
    - Interactive click-to-explain
    - Multiple layout algorithms
    """

    # Signals
    concept_clicked = Signal(str)  # Emitted when concept node is clicked

    def __init__(self, parent=None):
        """
        Initialize Resonance View Widget

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)

        self.activation_map: Optional[ActivationMap] = None
        self.current_wave = -1  # -1 = show all, 0+ = show specific wave
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._advance_animation)
        self.animation_running = False

        # Graph data
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.node_positions = {}
        self.node_colors = {}
        self.node_sizes = {}

        # Layout settings
        self.layout_algorithm = "spring"  # spring, hierarchical, circular
        self.show_labels = True
        self.show_edge_labels = False
        self.highlight_resonance = True

        # Initialize UI
        self._init_ui()

        logger.info("ResonanceViewWidget initialized")

    def _init_ui(self):
        """Initialize user interface"""
        main_layout = QVBoxLayout(self)

        # Controls at top
        controls_layout = self._create_controls()
        main_layout.addLayout(controls_layout)

        # Matplotlib canvas
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(10, 8), facecolor="#2b2b2b")
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor("#2b2b2b")

            # Make canvas interactive
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

            # Add matplotlib toolbar
            self.toolbar = NavigationToolbar(self.canvas, self)

            main_layout.addWidget(self.toolbar)
            main_layout.addWidget(self.canvas)
        else:
            # Fallback: show message
            placeholder = QLabel(
                "Matplotlib nicht verfügbar. Bitte installieren: pip install matplotlib"
            )
            placeholder.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(placeholder)

        # Info panel at bottom
        info_panel = self._create_info_panel()
        main_layout.addWidget(info_panel)

    def _create_controls(self) -> QHBoxLayout:
        """Create control panel with buttons and sliders"""
        layout = QHBoxLayout()

        # Layout algorithm selector
        layout_group = QGroupBox("Layout")
        layout_layout = QHBoxLayout()
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(
            ["Spring", "Hierarchical", "Circular", "Kamada-Kawai"]
        )
        self.layout_combo.currentTextChanged.connect(self._on_layout_changed)
        layout_layout.addWidget(QLabel("Algorithmus:"))
        layout_layout.addWidget(self.layout_combo)
        layout_group.setLayout(layout_layout)
        layout.addWidget(layout_group)

        # Wave controls
        wave_group = QGroupBox("Wave Animation")
        wave_layout = QHBoxLayout()

        self.wave_slider = QSlider(Qt.Horizontal)
        self.wave_slider.setMinimum(-1)  # -1 = all waves
        self.wave_slider.setMaximum(5)
        self.wave_slider.setValue(-1)
        self.wave_slider.setTickPosition(QSlider.TicksBelow)
        self.wave_slider.valueChanged.connect(self._on_wave_changed)

        self.wave_label = QLabel("Alle Waves")
        self.wave_label.setMinimumWidth(100)

        self.play_button = QPushButton("▶ Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_animation)

        wave_layout.addWidget(QLabel("Wave:"))
        wave_layout.addWidget(self.wave_slider)
        wave_layout.addWidget(self.wave_label)
        wave_layout.addWidget(self.play_button)

        wave_group.setLayout(wave_layout)
        layout.addWidget(wave_group)

        # Display options
        options_group = QGroupBox("Anzeige")
        options_layout = QHBoxLayout()

        self.labels_checkbox = QCheckBox("Labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_options_changed)

        self.edge_labels_checkbox = QCheckBox("Kanten-Labels")
        self.edge_labels_checkbox.setChecked(False)
        self.edge_labels_checkbox.stateChanged.connect(self._on_options_changed)

        self.resonance_checkbox = QCheckBox("Resonanz-Punkte")
        self.resonance_checkbox.setChecked(True)
        self.resonance_checkbox.stateChanged.connect(self._on_options_changed)

        options_layout.addWidget(self.labels_checkbox)
        options_layout.addWidget(self.edge_labels_checkbox)
        options_layout.addWidget(self.resonance_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Reset button
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self._reset_view)
        layout.addWidget(reset_button)

        layout.addStretch()

        return layout

    def _create_info_panel(self) -> QGroupBox:
        """Create information panel at bottom"""
        info_group = QGroupBox("Informationen")
        info_layout = QVBoxLayout()

        self.info_label = QLabel("Keine Aktivierungsdaten geladen")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { color: #cccccc; }")

        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)

        return info_group

    def set_activation_map(self, activation_map: ActivationMap):
        """
        Set activation map to visualize

        Args:
            activation_map: ActivationMap from ResonanceEngine
        """
        if not RESONANCE_ENGINE_AVAILABLE:
            logger.warning("ResonanceEngine not available, cannot set activation map")
            return

        self.activation_map = activation_map
        self.current_wave = -1  # Reset to show all

        # Update wave slider range
        max_waves = activation_map.waves_executed
        self.wave_slider.setMaximum(max_waves)
        self.wave_slider.setValue(-1)

        # Build graph
        self._build_graph()

        # Update visualization
        self._update_visualization()

        # Update info
        self._update_info()

        logger.info(
            f"Activation map set: {activation_map.concepts_activated} concepts, "
            f"{activation_map.waves_executed} waves"
        )

    def _build_graph(self):
        """Build NetworkX graph from activation map"""
        if not NETWORKX_AVAILABLE or not self.activation_map:
            return

        self.graph.clear()

        # Add nodes with activation levels
        for concept, activation in self.activation_map.activations.items():
            self.graph.add_node(concept, activation=activation)

        # Add edges from reasoning paths
        for path in self.activation_map.reasoning_paths:
            relation = path.relations[0] if path.relations else "REL"
            self.graph.add_edge(
                path.source,
                path.target,
                relation=relation,
                confidence=path.confidence_product,
                wave=path.wave_depth,
            )

        logger.debug(
            f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )

    def _calculate_layout(self):
        """Calculate node positions using selected layout algorithm"""
        if not NETWORKX_AVAILABLE or not self.graph:
            return

        algorithm = self.layout_combo.currentText().lower()

        try:
            if algorithm == "spring":
                self.node_positions = nx.spring_layout(
                    self.graph, k=1.5, iterations=50, seed=42
                )
            elif algorithm == "hierarchical":
                # Try to use hierarchical layout (requires graphviz)
                try:
                    self.node_positions = nx.nx_agraph.graphviz_layout(
                        self.graph, prog="dot"
                    )
                except Exception:
                    # Fallback to kamada-kawai
                    logger.warning("Graphviz not available, using Kamada-Kawai")
                    self.node_positions = nx.kamada_kawai_layout(self.graph)
            elif algorithm == "circular":
                self.node_positions = nx.circular_layout(self.graph)
            elif algorithm == "kamada-kawai":
                self.node_positions = nx.kamada_kawai_layout(self.graph)
            else:
                # Default to spring
                self.node_positions = nx.spring_layout(
                    self.graph, k=1.5, iterations=50, seed=42
                )

            logger.debug(f"Layout calculated using {algorithm}")

        except Exception as e:
            logger.warning(f"Layout calculation failed: {e}, using fallback")
            self.node_positions = nx.spring_layout(self.graph, seed=42)

    def _calculate_node_colors(self) -> List[str]:
        """
        Calculate node colors based on activation levels

        Returns:
            List of color strings (hex)
        """
        if not self.graph or not self.activation_map:
            return []

        colors = []
        for node in self.graph.nodes():
            activation = self.activation_map.activations.get(node, 0.0)

            # Filter by current wave if needed
            if self.current_wave >= 0:
                # Check if node was activated in this wave
                if self.current_wave < len(self.activation_map.wave_history):
                    wave_activations = self.activation_map.wave_history[
                        self.current_wave
                    ]
                    if node not in wave_activations:
                        # Not activated in this wave, use gray
                        colors.append("#404040")
                        continue
                    activation = wave_activations[node]
                else:
                    colors.append("#404040")
                    continue

            # Color based on activation level
            # High = red, low = blue
            color = self._activation_to_color(activation)
            colors.append(color)

        return colors

    def _activation_to_color(self, activation: float) -> str:
        """
        Convert activation level to color (blue → red)

        Args:
            activation: Activation level (0.0 to 1.0+)

        Returns:
            Hex color string
        """
        # Normalize to 0-1
        normalized = min(max(activation, 0.0), 1.0)

        # Blue (low) → Yellow (medium) → Red (high)
        if normalized < 0.5:
            # Blue → Yellow
            r = int(normalized * 2 * 255)
            g = int(normalized * 2 * 255)
            b = int((1.0 - normalized * 2) * 255)
        else:
            # Yellow → Red
            r = 255
            g = int((1.0 - (normalized - 0.5) * 2) * 255)
            b = 0

        return f"#{r:02x}{g:02x}{b:02x}"

    def _calculate_node_sizes(self) -> List[float]:
        """
        Calculate node sizes based on activation and resonance

        Returns:
            List of node sizes
        """
        if not self.graph or not self.activation_map:
            return []

        sizes = []
        for node in self.graph.nodes():
            activation = self.activation_map.activations.get(node, 0.0)

            # Base size from activation
            size = 300 + activation * 700  # 300-1000

            # Boost for resonance points
            if self.highlight_resonance and self.activation_map.is_resonance_point(
                node
            ):
                size *= 1.5

            sizes.append(size)

        return sizes

    def _update_visualization(self):
        """Update the visualization with current settings"""
        if not MATPLOTLIB_AVAILABLE or not self.graph:
            return

        # Clear axis
        self.ax.clear()
        self.ax.set_facecolor("#2b2b2b")
        self.ax.axis("off")

        # Calculate layout if needed
        if not self.node_positions:
            self._calculate_layout()

        # Calculate visual properties
        node_colors = self._calculate_node_colors()
        node_sizes = self._calculate_node_sizes()

        # Get edges to draw (filter by wave if needed)
        edges_to_draw = self._get_visible_edges()

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=edges_to_draw,
            edge_color="#666666",
            arrows=True,
            arrowstyle="->",
            arrowsize=10,
            width=1.5,
            alpha=0.6,
            ax=self.ax,
        )

        # Draw edge labels if enabled
        if self.show_edge_labels and edges_to_draw:
            edge_labels = {
                (u, v): self.graph[u][v].get("relation", "") for u, v in edges_to_draw
            }
            nx.draw_networkx_edge_labels(
                self.graph,
                self.node_positions,
                edge_labels,
                font_size=8,
                font_color="#aaaaaa",
                ax=self.ax,
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            self.node_positions,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=self.ax,
        )

        # Draw labels if enabled
        if self.show_labels:
            nx.draw_networkx_labels(
                self.graph,
                self.node_positions,
                font_size=9,
                font_color="white",
                font_weight="bold",
                ax=self.ax,
            )

        # Highlight resonance points with special markers
        if self.highlight_resonance:
            self._draw_resonance_highlights()

        # Set title
        title = self._get_title()
        self.ax.set_title(title, color="white", fontsize=12, pad=10)

        # Refresh canvas
        self.canvas.draw()

    def _get_visible_edges(self) -> List[Tuple[str, str]]:
        """
        Get edges to display based on current wave

        Returns:
            List of (source, target) tuples
        """
        if not self.graph:
            return []

        if self.current_wave < 0:
            # Show all edges
            return list(self.graph.edges())

        # Filter edges by wave
        visible_edges = []
        for u, v, data in self.graph.edges(data=True):
            wave = data.get("wave", 0)
            if wave <= self.current_wave:
                visible_edges.append((u, v))

        return visible_edges

    def _draw_resonance_highlights(self):
        """Draw special highlights for resonance points"""
        if not self.activation_map or not self.node_positions:
            return

        for rp in self.activation_map.resonance_points:
            if rp.concept not in self.node_positions:
                continue

            pos = self.node_positions[rp.concept]
            x, y = pos

            # Draw pulsing circle around resonance point
            circle = Circle(
                (x, y),
                0.05,
                fill=False,
                edgecolor="yellow",
                linewidth=3,
                linestyle="--",
                alpha=0.8,
            )
            self.ax.add_patch(circle)

    def _get_title(self) -> str:
        """Get title for visualization"""
        if not self.activation_map:
            return "Resonance Activation Map"

        if self.current_wave < 0:
            return (
                f"Resonance Map: {self.activation_map.concepts_activated} Konzepte, "
                f"{self.activation_map.waves_executed} Waves, "
                f"{len(self.activation_map.resonance_points)} Resonanz-Punkte"
            )
        else:
            wave_concepts = (
                len(self.activation_map.wave_history[self.current_wave])
                if self.current_wave < len(self.activation_map.wave_history)
                else 0
            )
            return f"Wave {self.current_wave}: {wave_concepts} neue Aktivierungen"

    def _update_info(self):
        """Update information panel"""
        if not self.activation_map:
            self.info_label.setText("Keine Aktivierungsdaten geladen")
            return

        info_lines = []
        info_lines.append(
            f"<b>Gesamt:</b> {self.activation_map.concepts_activated} Konzepte aktiviert"
        )
        info_lines.append(f"<b>Waves:</b> {self.activation_map.waves_executed}")
        info_lines.append(
            f"<b>Resonanz-Punkte:</b> {len(self.activation_map.resonance_points)}"
        )
        info_lines.append(
            f"<b>Max. Aktivierung:</b> {self.activation_map.max_activation:.3f}"
        )
        info_lines.append(f"<b>Pfade:</b> {len(self.activation_map.reasoning_paths)}")

        if self.current_wave >= 0 and self.current_wave < len(
            self.activation_map.wave_history
        ):
            wave_data = self.activation_map.wave_history[self.current_wave]
            info_lines.append(
                f"<b>Wave {self.current_wave}:</b> {len(wave_data)} neue Konzepte"
            )

        self.info_label.setText("<br>".join(info_lines))

    def _on_layout_changed(self, layout_name: str):
        """Handle layout algorithm change"""
        self.layout_algorithm = layout_name.lower()
        self.node_positions = {}  # Force recalculation
        self._calculate_layout()
        self._update_visualization()
        logger.debug(f"Layout changed to {layout_name}")

    def _on_wave_changed(self, wave: int):
        """Handle wave slider change"""
        self.current_wave = wave

        if wave < 0:
            self.wave_label.setText("Alle Waves")
        else:
            self.wave_label.setText(f"Wave {wave}")

        self._update_visualization()
        self._update_info()

    def _on_options_changed(self):
        """Handle display options change"""
        self.show_labels = self.labels_checkbox.isChecked()
        self.show_edge_labels = self.edge_labels_checkbox.isChecked()
        self.highlight_resonance = self.resonance_checkbox.isChecked()

        self._update_visualization()

    def _toggle_animation(self, checked: bool):
        """Toggle wave animation"""
        if checked:
            # Start animation
            self.animation_running = True
            self.play_button.setText("⏸ Pause")
            self.current_wave = 0
            self.wave_slider.setValue(0)
            self.animation_timer.start(1000)  # 1 second per wave
            logger.debug("Animation started")
        else:
            # Stop animation
            self.animation_running = False
            self.play_button.setText("▶ Play")
            self.animation_timer.stop()
            logger.debug("Animation stopped")

    def _advance_animation(self):
        """Advance to next wave in animation"""
        if not self.activation_map:
            self._toggle_animation(False)
            return

        self.current_wave += 1

        if self.current_wave > self.activation_map.waves_executed:
            # Finished, loop or stop
            self.current_wave = 0

        self.wave_slider.setValue(self.current_wave)

    def _reset_view(self):
        """Reset view to default"""
        self.current_wave = -1
        self.wave_slider.setValue(-1)

        if self.animation_running:
            self.play_button.setChecked(False)
            self._toggle_animation(False)

        self._update_visualization()
        logger.debug("View reset")

    def _on_canvas_click(self, event):
        """Handle click on canvas to show concept explanation"""
        if not event.inaxes or not self.activation_map or not self.node_positions:
            return

        # Find closest node to click
        click_x, click_y = event.xdata, event.ydata
        closest_node = None
        min_distance = float("inf")

        for node, (x, y) in self.node_positions.items():
            distance = ((x - click_x) ** 2 + (y - click_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_node = node

        # If close enough, emit signal and show explanation
        if closest_node and min_distance < 0.1:
            logger.info(f"Concept clicked: {closest_node}")
            self.concept_clicked.emit(closest_node)
            self._show_concept_explanation(closest_node)

    def _show_concept_explanation(self, concept: str):
        """
        Show explanation for clicked concept

        Args:
            concept: The concept to explain
        """
        if not self.activation_map:
            return

        # Try to get engine for explanation
        try:
            from component_44_resonance_engine import ResonanceEngine

            # Create temporary engine (we need it for explanation method)
            # In production, this should be passed from outside
            engine = ResonanceEngine(None)
            explanation = engine.explain_activation(
                concept, self.activation_map, max_paths=5
            )

            # Show in info panel
            self.info_label.setText(f"<pre>{explanation}</pre>")

            logger.debug(f"Showing explanation for {concept}")

        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
            # Fallback: show basic info
            activation = self.activation_map.activations.get(concept, 0.0)
            is_resonance = self.activation_map.is_resonance_point(concept)
            paths = self.activation_map.get_paths_to(concept)

            info = f"<b>{concept}</b><br>"
            info += f"Aktivierung: {activation:.3f}<br>"
            info += f"Resonanz-Punkt: {'Ja' if is_resonance else 'Nein'}<br>"
            info += f"Pfade: {len(paths)}"

            self.info_label.setText(info)

    def clear(self):
        """Clear all visualization data"""
        self.activation_map = None
        self.current_wave = -1

        if NETWORKX_AVAILABLE and self.graph:
            self.graph.clear()

        self.node_positions = {}

        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self.ax.set_facecolor("#2b2b2b")
            self.ax.axis("off")
            self.canvas.draw()

        self.info_label.setText("Keine Aktivierungsdaten geladen")

        logger.debug("Visualization cleared")

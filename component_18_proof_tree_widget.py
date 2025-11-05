"""
component_18_proof_tree_widget.py

Interactive Proof Tree Visualization Widget for KAI

Provides a PySide6-based graphical tree visualization for proof explanations
with support for interactive exploration, export, and detailed inspection.

Features:
- QGraphicsView-based tree rendering with custom node shapes
- Hierarchical layout algorithm (top-down tree)
- Color-coded confidence levels
- Interactive expand/collapse
- Tooltips with full explanations
- Path highlighting from root to selected node
- Export to JSON and image
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsLineItem,
    QFileDialog,
    QLabel,
    QSlider,
    QCheckBox,
    QMenu,
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QTimer
from PySide6.QtGui import (
    QPen,
    QBrush,
    QColor,
    QPainter,
    QPolygonF,
    QFont,
    QPainterPath,
    QPixmap,
)
from typing import Optional, List, Dict

try:
    from component_17_proof_explanation import (
        ProofTree,
        ProofTreeNode,
        StepType,
        export_proof_to_json,
    )

    PROOF_SYSTEM_AVAILABLE = True
except ImportError:
    PROOF_SYSTEM_AVAILABLE = False


# ==================== Custom Graphics Items ====================


class ProofNodeItem(QGraphicsItem):
    """
    Custom graphics item for proof tree nodes.

    Supports different shapes based on step type:
    - Rectangle: Facts, inferences
    - Diamond: Rules, rule applications
    - Circle: Hypotheses, conclusions
    """

    def __init__(self, tree_node: "ProofTreeNode", parent=None):
        super().__init__(parent)
        self.tree_node = tree_node
        self.node_width = 150
        self.node_height = 60
        self.is_highlighted = False
        self.is_selected_item = False

        # Make item interactive
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

        # Set tooltip
        self._update_tooltip()

    def _update_tooltip(self):
        """Generate tooltip with full explanation"""
        step = self.tree_node.step
        tooltip_lines = [
            f"<b>Schritt:</b> {step.step_type.value}",
            (
                f"<b>Ausgabe:</b> {step.output[:100]}..."
                if len(step.output) > 100
                else f"<b>Ausgabe:</b> {step.output}"
            ),
            f"<b>Konfidenz:</b> {step.confidence:.2f}",
            "",
        ]

        if step.explanation_text:
            tooltip_lines.append(f"<b>Erklärung:</b><br>{step.explanation_text}")

        if step.rule_name:
            tooltip_lines.append(f"<b>Regel:</b> {step.rule_name}")

        if step.inputs:
            tooltip_lines.append(f"<b>Eingaben:</b> {len(step.inputs)}")

        # Enhanced: Source component
        tooltip_lines.append("")
        tooltip_lines.append(f"<b>Quelle:</b> {step.source_component}")

        # Enhanced: Timestamp (formatted)
        timestamp_str = step.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        tooltip_lines.append(f"<b>Zeitstempel:</b> {timestamp_str}")

        # Enhanced: Metadata (formatted if present)
        if step.metadata:
            tooltip_lines.append("")
            tooltip_lines.append("<b>Metadata:</b>")
            for key, value in list(step.metadata.items())[:5]:  # Limit to 5 entries
                value_str = str(value)[:50]  # Truncate long values
                if len(str(value)) > 50:
                    value_str += "..."
                tooltip_lines.append(f"  * {key}: {value_str}")
            if len(step.metadata) > 5:
                tooltip_lines.append(f"  ... (+{len(step.metadata) - 5} weitere)")

        self.setToolTip("<br>".join(tooltip_lines))

    def boundingRect(self) -> QRectF:
        """Define bounding rectangle for the item"""
        return QRectF(
            -self.node_width / 2,
            -self.node_height / 2,
            self.node_width,
            self.node_height,
        )

    def shape(self) -> QPainterPath:
        """Define the shape for collision detection"""
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def paint(self, painter: QPainter, option, widget=None):
        """Paint the node with appropriate shape and color"""
        step = self.tree_node.step

        # Determine shape based on step type
        shape_type = self._get_shape_type(step.step_type)

        # Determine color based on confidence
        fill_color = self._get_confidence_color(step.confidence)

        # Adjust color if highlighted or selected
        if self.is_selected_item:
            pen = QPen(QColor("#3498db"), 3)  # Blue border for selection
        elif self.is_highlighted:
            pen = QPen(QColor("#f39c12"), 3)  # Orange border for highlight
        else:
            pen = QPen(QColor("#2c3e50"), 2)  # Dark border

        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        # Draw shape
        if shape_type == "rectangle":
            painter.drawRect(self.boundingRect())
        elif shape_type == "diamond":
            self._draw_diamond(painter)
        elif shape_type == "circle":
            self._draw_circle(painter)

        # Draw text (abbreviated)
        painter.setPen(QPen(QColor("#ecf0f1")))
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))

        # Draw step type icon
        icon = self._get_step_icon(step.step_type)
        painter.drawText(
            QRectF(-self.node_width / 2 + 5, -self.node_height / 2 + 5, 30, 20),
            Qt.AlignmentFlag.AlignLeft,
            icon,
        )

        # Draw truncated output
        output_text = step.output[:30] + "..." if len(step.output) > 30 else step.output
        painter.setFont(QFont("Arial", 8))
        painter.drawText(
            QRectF(
                -self.node_width / 2 + 5,
                -self.node_height / 2 + 25,
                self.node_width - 10,
                self.node_height - 30,
            ),
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
            | Qt.AlignmentFlag.AlignHCenter,
            output_text,
        )

        # Draw confidence indicator (small bar at bottom) - IMPROVED
        # Verwende Confidence-Level-Farbe statt immer Grün
        conf_bar_width = (self.node_width - 20) * step.confidence
        conf_bar_color = self._get_confidence_bar_color(step.confidence)
        painter.setBrush(QBrush(conf_bar_color))
        painter.drawRect(
            QRectF(
                -self.node_width / 2 + 10, self.node_height / 2 - 10, conf_bar_width, 5
            )
        )

        # Draw confidence percentage text
        painter.setPen(QPen(QColor("#ecf0f1")))
        painter.setFont(QFont("Arial", 7, QFont.Weight.Bold))
        conf_text = f"{step.confidence:.0%}"
        painter.drawText(
            QRectF(self.node_width / 2 - 35, self.node_height / 2 - 25, 30, 15),
            Qt.AlignmentFlag.AlignRight,
            conf_text,
        )

        # Draw confidence level icon
        conf_icon = self._get_confidence_icon(step.confidence)
        painter.setFont(QFont("Arial", 10))
        painter.drawText(
            QRectF(self.node_width / 2 - 20, -self.node_height / 2 + 5, 15, 15),
            Qt.AlignmentFlag.AlignCenter,
            conf_icon,
        )

    def _get_shape_type(self, step_type: "StepType") -> str:
        """Determine shape based on step type"""
        if step_type in [StepType.FACT_MATCH, StepType.INFERENCE]:
            return "rectangle"
        elif step_type in [StepType.RULE_APPLICATION, StepType.DECOMPOSITION]:
            return "diamond"
        else:  # HYPOTHESIS, PROBABILISTIC, GRAPH_TRAVERSAL, UNIFICATION
            return "circle"

    def _get_confidence_color(self, confidence: float) -> QColor:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return QColor("#27ae60")  # Green (high confidence)
        elif confidence >= 0.5:
            return QColor("#f39c12")  # Yellow/Orange (medium)
        else:
            return QColor("#e74c3c")  # Red (low confidence)

    def _get_confidence_bar_color(self, confidence: float) -> QColor:
        """
        Get color for confidence bar (brighter variants for better visibility).

        Integriert mit ConfidenceManager-Thresholds:
        - >= 0.8: GREEN (High confidence)
        - 0.5-0.8: ORANGE (Medium confidence)
        - 0.3-0.5: RED (Low confidence)
        - < 0.3: DARK RED (Unknown/Very low)
        """
        if confidence >= 0.8:
            return QColor("#2ecc71")  # Bright green
        elif confidence >= 0.5:
            return QColor("#f39c12")  # Orange
        elif confidence >= 0.3:
            return QColor("#e74c3c")  # Red
        else:
            return QColor("#c0392b")  # Dark red

    def _get_confidence_icon(self, confidence: float) -> str:
        """
        Get icon representing confidence level.

        Integriert mit ConfidenceManager-Levels:
        - >= 0.8: [OK] (High)
        - 0.5-0.8: ~ (Medium)
        - 0.3-0.5: ! (Low)
        - < 0.3: ? (Unknown)
        """
        if confidence >= 0.8:
            return "[OK]"  # Check mark for high confidence
        elif confidence >= 0.5:
            return "~"  # Tilde for medium confidence
        elif confidence >= 0.3:
            return "!"  # Exclamation for low confidence
        else:
            return "?"  # Question mark for unknown

    def _get_step_icon(self, step_type: "StepType") -> str:
        """Get icon for step type"""
        icons = {
            StepType.FACT_MATCH: "[INFO]",
            StepType.RULE_APPLICATION: "⚙️",
            StepType.INFERENCE: "💡",
            StepType.HYPOTHESIS: "🔬",
            StepType.GRAPH_TRAVERSAL: "🗺️",
            StepType.PROBABILISTIC: "🎲",
            StepType.DECOMPOSITION: "🔀",
            StepType.UNIFICATION: "🔗",
        }
        return icons.get(step_type, "*")

    def _draw_diamond(self, painter: QPainter):
        """Draw diamond shape"""
        w, h = self.node_width / 2, self.node_height / 2
        points = [
            QPointF(0, -h),  # Top
            QPointF(w, 0),  # Right
            QPointF(0, h),  # Bottom
            QPointF(-w, 0),  # Left
        ]
        painter.drawPolygon(QPolygonF(points))

    def _draw_circle(self, painter: QPainter):
        """Draw circular shape"""
        radius = min(self.node_width, self.node_height) / 2
        painter.drawEllipse(QPointF(0, 0), radius, radius)

    def set_highlighted(self, highlighted: bool):
        """Set highlight state"""
        self.is_highlighted = highlighted
        self.update()

    def set_selected_state(self, selected: bool):
        """Set selection state"""
        self.is_selected_item = selected
        self.update()

    def hoverEnterEvent(self, event):
        """Handle mouse hover"""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Handle mouse leave"""
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)


class ProofEdgeItem(QGraphicsLineItem):
    """
    Custom graphics item for edges between proof nodes.
    """

    def __init__(self, parent_item: ProofNodeItem, child_item: ProofNodeItem):
        super().__init__()
        self.parent_item = parent_item
        self.child_item = child_item
        self.is_highlighted = False

        self._update_line()

        # Set pen
        self.setPen(QPen(QColor("#7f8c8d"), 2))

    def _update_line(self):
        """Update line position based on parent/child positions"""
        parent_pos = self.parent_item.scenePos()
        child_pos = self.child_item.scenePos()

        # Start from bottom of parent
        start = QPointF(
            parent_pos.x(), parent_pos.y() + self.parent_item.node_height / 2
        )
        # End at top of child
        end = QPointF(child_pos.x(), child_pos.y() - self.child_item.node_height / 2)

        self.setLine(start.x(), start.y(), end.x(), end.y())

    def set_highlighted(self, highlighted: bool):
        """Set highlight state"""
        self.is_highlighted = highlighted
        if highlighted:
            self.setPen(QPen(QColor("#f39c12"), 3))
        else:
            self.setPen(QPen(QColor("#7f8c8d"), 2))
        self.update()


# ==================== Main Widget ====================


class ProofTreeWidget(QWidget):
    """
    Interactive proof tree visualization widget.

    Displays proof trees with hierarchical layout, interactive features,
    and export capabilities.

    Signals:
        node_selected: Emitted when a node is clicked (ProofStep)
    """

    node_selected = Signal(object)  # ProofStep

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_tree: Optional[ProofTree] = None
        self.tree_nodes: List[ProofTreeNode] = []
        self.node_items: Dict[str, ProofNodeItem] = {}  # step_id -> ProofNodeItem
        self.edge_items: List[ProofEdgeItem] = []
        self.selected_node: Optional[ProofTreeNode] = None

        # Filter state
        self.min_confidence: float = 0.0  # Filter threshold
        self.filter_enabled: bool = False
        self.enabled_step_types: set = set(StepType)  # All types enabled by default

        # Performance settings
        self.max_nodes_threshold = 100  # Auto-collapse if tree exceeds this
        self.rendered_node_count = 0

        # LAZY LOADING: Progressive Rendering Settings
        self.progressive_rendering_enabled: bool = True  # Toggle für Lazy Loading
        self.render_batch_size: int = 50  # Wie viele Nodes pro Batch rendern
        self.current_render_index: int = 0  # Tracker für progressive Rendering
        self.pending_nodes: List[ProofTreeNode] = (
            []
        )  # Nodes die noch gerendert werden müssen

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Graphics view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setBackgroundBrush(QBrush(QColor("#1e1e1e")))

        layout.addWidget(self.view)

        # Status bar
        self.status_label = QLabel("Kein Beweisbaum geladen")
        self.status_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        layout.addWidget(self.status_label)

    def _create_toolbar(self) -> QWidget:
        """Create toolbar with controls"""
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        # Expand/Collapse All
        btn_expand_all = QPushButton("Alles Aufklappen")
        btn_expand_all.clicked.connect(self.expand_all)
        toolbar_layout.addWidget(btn_expand_all)

        btn_collapse_all = QPushButton("Alles Zuklappen")
        btn_collapse_all.clicked.connect(self.collapse_all)
        toolbar_layout.addWidget(btn_collapse_all)

        toolbar_layout.addStretch()

        # Zoom controls
        toolbar_layout.addWidget(QLabel("Zoom:"))

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.clicked.connect(lambda: self.view.scale(1.2, 1.2))
        btn_zoom_in.setFixedWidth(30)
        toolbar_layout.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("-")
        btn_zoom_out.clicked.connect(lambda: self.view.scale(0.8, 0.8))
        btn_zoom_out.setFixedWidth(30)
        toolbar_layout.addWidget(btn_zoom_out)

        btn_zoom_fit = QPushButton("Fit")
        btn_zoom_fit.clicked.connect(self.fit_to_view)
        toolbar_layout.addWidget(btn_zoom_fit)

        toolbar_layout.addStretch()

        # Filter controls
        toolbar_layout.addWidget(QLabel("Filter:"))

        self.filter_checkbox = QCheckBox("Konfidenz ≥")
        self.filter_checkbox.setChecked(False)
        self.filter_checkbox.toggled.connect(self._on_filter_toggled)
        toolbar_layout.addWidget(self.filter_checkbox)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(0)
        self.confidence_slider.setFixedWidth(100)
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        toolbar_layout.addWidget(self.confidence_slider)

        self.confidence_label = QLabel("0.00")
        self.confidence_label.setFixedWidth(35)
        toolbar_layout.addWidget(self.confidence_label)

        # StepType filter button
        btn_steptype_filter = QPushButton("StepType-Filter")
        btn_steptype_filter.clicked.connect(self._show_steptype_filter_menu)
        toolbar_layout.addWidget(btn_steptype_filter)

        toolbar_layout.addStretch()

        # Filter presets
        toolbar_layout.addWidget(QLabel("Presets:"))

        btn_preset_high_conf = QPushButton("High Conf")
        btn_preset_high_conf.setToolTip("Nur Schritte mit Konfidenz ≥ 0.8")
        btn_preset_high_conf.clicked.connect(self._apply_preset_high_confidence)
        toolbar_layout.addWidget(btn_preset_high_conf)

        btn_preset_rules = QPushButton("Rules")
        btn_preset_rules.setToolTip("Nur Rule Applications & Inferences")
        btn_preset_rules.clicked.connect(self._apply_preset_rules_only)
        toolbar_layout.addWidget(btn_preset_rules)

        btn_reset_filters = QPushButton("Reset")
        btn_reset_filters.setToolTip("Alle Filter zurücksetzen")
        btn_reset_filters.clicked.connect(self._reset_all_filters)
        toolbar_layout.addWidget(btn_reset_filters)

        toolbar_layout.addStretch()

        # Progressive Rendering Toggle
        self.progressive_checkbox = QCheckBox("Progressive Rendering")
        self.progressive_checkbox.setChecked(self.progressive_rendering_enabled)
        self.progressive_checkbox.setToolTip(
            "Rendert große Bäume schrittweise für bessere Performance"
        )
        self.progressive_checkbox.toggled.connect(self._on_progressive_toggled)
        toolbar_layout.addWidget(self.progressive_checkbox)

        toolbar_layout.addStretch()

        # Export buttons
        btn_export_json = QPushButton("Export JSON")
        btn_export_json.clicked.connect(self.export_to_json)
        toolbar_layout.addWidget(btn_export_json)

        btn_export_image = QPushButton("Export Bild")
        btn_export_image.clicked.connect(self.export_to_image)
        toolbar_layout.addWidget(btn_export_image)

        # Style toolbar
        toolbar.setStyleSheet(
            """
            QWidget {
                background-color: #2c3e50;
            }
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QLabel {
                color: #ecf0f1;
            }
        """
        )

        return toolbar

    def set_proof_tree(self, tree: ProofTree):
        """
        Set and display a proof tree.

        Args:
            tree: ProofTree to visualize
        """
        self.current_tree = tree
        self.tree_nodes = tree.to_tree_nodes()
        self.selected_node = None

        # Render tree
        self._render_tree()

        # Update status (only if not already set by performance warning)
        if "Auto-Collapse" not in self.status_label.text():
            self._update_status_with_counter()

    def _render_tree(self):
        """Render the current tree to the scene"""
        # Clear scene
        self.scene.clear()
        self.node_items.clear()
        self.edge_items.clear()

        if not self.tree_nodes:
            return

        # Performance optimization: Check total node count
        total_nodes = sum(self._count_nodes(root) for root in self.tree_nodes)

        # Auto-collapse for large trees (>100 nodes)
        if total_nodes > self.max_nodes_threshold:
            self._auto_collapse_large_tree()
            self.status_label.setText(
                f"[WARNING] Großer Baum ({total_nodes} Knoten) - Auto-Collapse aktiviert"
            )

        # LAZY LOADING: Entscheide zwischen Progressive und Full Rendering
        if self.progressive_rendering_enabled and total_nodes > 50:
            self._render_tree_progressive()
        else:
            self._render_tree_full()

    def _render_tree_full(self):
        """Full tree rendering (alte Methode)"""
        # Layout parameters
        horizontal_spacing = 200
        vertical_spacing = 120

        # Calculate layout positions
        all_root_nodes = []
        for root_node in self.tree_nodes:
            all_root_nodes.extend(self._flatten_tree(root_node))

        # Assign positions using hierarchical layout
        self._layout_tree(self.tree_nodes, horizontal_spacing, vertical_spacing)

        # Create graphics items
        self._create_graphics_items()

        # Track rendered node count
        self.rendered_node_count = len(self.node_items)

        # Update status with counter (unless performance warning is active)
        if "Auto-Collapse" not in self.status_label.text():
            self._update_status_with_counter()

        # Fit to view
        QTimer.singleShot(100, self.fit_to_view)

    def _render_tree_progressive(self):
        """
        LAZY LOADING: Progressive Tree Rendering.

        Rendert Baum in Batches für bessere Performance bei großen Bäumen.
        Rendert initial nur ersten Batch, weitere Batches on-demand.
        """
        # Layout parameters
        horizontal_spacing = 200
        vertical_spacing = 120

        # Calculate layout positions
        self._layout_tree(self.tree_nodes, horizontal_spacing, vertical_spacing)

        # Sammle alle zu rendernden Nodes (flatten)
        all_nodes = []
        for root in self.tree_nodes:
            all_nodes.extend(self._flatten_tree(root))

        # Filtere Nodes
        filtered_nodes = [node for node in all_nodes if self._should_display_node(node)]

        # Speichere pending nodes
        self.pending_nodes = filtered_nodes
        self.current_render_index = 0

        # Rendere ersten Batch
        self._render_next_batch()

        # Update status
        self.status_label.setText(
            f"🔄 Progressive Rendering: {self.rendered_node_count} von {len(filtered_nodes)} Knoten geladen"
        )

        # Fit to view nach initialem Rendering
        QTimer.singleShot(100, self.fit_to_view)

    def _render_next_batch(self):
        """Rendert den nächsten Batch von Nodes"""
        if self.current_render_index >= len(self.pending_nodes):
            # Alle Nodes gerendert
            self._update_status_with_counter()
            return

        # Bestimme Batch
        batch_end = min(
            self.current_render_index + self.render_batch_size, len(self.pending_nodes)
        )

        batch_nodes = self.pending_nodes[self.current_render_index : batch_end]

        # Rendere Batch
        for node in batch_nodes:
            # Create item for this node
            item = ProofNodeItem(node)
            item.setPos(node.position[0], node.position[1])
            self.scene.addItem(item)
            self.node_items[node.step.step_id] = item

            # Connect click handler
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

        # Update Index
        self.current_render_index = batch_end

        # Track rendered node count
        self.rendered_node_count = len(self.node_items)

        # Update status
        if self.current_render_index < len(self.pending_nodes):
            self.status_label.setText(
                f"🔄 Progressive Rendering: {self.rendered_node_count} von {len(self.pending_nodes)} Knoten"
            )

            # Schedule nächsten Batch (mit kleinem Delay für Responsiveness)
            QTimer.singleShot(50, self._render_next_batch)
        else:
            # Rendering abgeschlossen - rendere jetzt Edges
            self._render_edges_for_loaded_nodes()
            self._update_status_with_counter()

    def _render_edges_for_loaded_nodes(self):
        """Rendert Edges zwischen allen geladenen Nodes"""
        for root in self.tree_nodes:
            self._create_edge_items_recursive(root)

    def _flatten_tree(self, node: ProofTreeNode) -> List[ProofTreeNode]:
        """Flatten tree to list of all nodes"""
        nodes = [node]
        if node.expanded:
            for child in node.children:
                nodes.extend(self._flatten_tree(child))
        return nodes

    def _layout_tree(
        self, roots: List[ProofTreeNode], h_spacing: float, v_spacing: float
    ):
        """
        Calculate positions for all nodes using hierarchical layout.

        Uses a top-down tree layout algorithm that positions nodes
        to minimize edge crossings and maintain visual clarity.
        """
        # Track positions per level
        level_positions: Dict[int, float] = {}

        # Process each root separately
        x_offset = 0
        for root in roots:
            x_offset = self._layout_subtree(root, x_offset, 0, h_spacing, v_spacing)
            x_offset += h_spacing  # Space between separate trees

    def _layout_subtree(
        self,
        node: ProofTreeNode,
        x_start: float,
        depth: int,
        h_spacing: float,
        v_spacing: float,
    ) -> float:
        """
        Layout a subtree recursively.

        Returns:
            The next available x position after this subtree
        """
        y = depth * v_spacing

        if not node.expanded or not node.children:
            # Leaf node or collapsed
            node.position = (x_start, y)
            return x_start + h_spacing

        # Layout children first
        child_x = x_start
        child_positions = []
        for child in node.children:
            child_x_start = child_x
            child_x = self._layout_subtree(
                child, child_x, depth + 1, h_spacing, v_spacing
            )
            child_positions.append(child_x_start)

        # Position parent at midpoint of children
        if child_positions:
            first_child_x = child_positions[0]
            last_child_x = child_positions[-1]
            parent_x = (first_child_x + last_child_x) / 2
        else:
            parent_x = x_start

        node.position = (parent_x, y)
        return child_x

    def _create_graphics_items(self):
        """Create QGraphicsItems for all nodes and edges"""
        # Create node items
        for root in self.tree_nodes:
            self._create_node_items_recursive(root)

        # Create edge items
        for root in self.tree_nodes:
            self._create_edge_items_recursive(root)

    def _create_node_items_recursive(self, node: ProofTreeNode):
        """Recursively create graphics items for nodes"""
        # Apply filters
        if not self._should_display_node(node):
            return  # Skip this node and its children

        # Create item for this node
        item = ProofNodeItem(node)
        item.setPos(node.position[0], node.position[1])
        self.scene.addItem(item)
        self.node_items[node.step.step_id] = item

        # Connect click handler
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

        # Recurse for children (if expanded)
        if node.expanded:
            for child in node.children:
                self._create_node_items_recursive(child)

    def _create_edge_items_recursive(self, node: ProofTreeNode):
        """Recursively create edge items"""
        # Skip if node is filtered out
        if not self._should_display_node(node):
            return

        if node.expanded:
            parent_item = self.node_items.get(node.step.step_id)
            for child in node.children:
                # Only create edge if child is also displayed
                if not self._should_display_node(child):
                    continue

                child_item = self.node_items.get(child.step.step_id)
                if parent_item and child_item:
                    edge = ProofEdgeItem(parent_item, child_item)
                    self.scene.addItem(edge)
                    edge.setZValue(-1)  # Behind nodes
                    self.edge_items.append(edge)

                # Recurse
                self._create_edge_items_recursive(child)

    def mousePressEvent(self, event):
        """Handle mouse press for node selection"""
        # Find clicked item
        scene_pos = self.view.mapToScene(
            self.view.mapFromGlobal(event.globalPosition().toPoint())
        )
        item = self.scene.itemAt(scene_pos, self.view.transform())

        if isinstance(item, ProofNodeItem):
            self._select_node(item.tree_node)

        super().mousePressEvent(event)

    def _select_node(self, node: ProofTreeNode):
        """
        Select a node and highlight path to root.

        Args:
            node: The ProofTreeNode to select
        """
        # Clear previous selection
        if self.selected_node:
            for step_id, item in self.node_items.items():
                item.set_selected_state(False)
                item.set_highlighted(False)
            for edge in self.edge_items:
                edge.set_highlighted(False)

        # Set new selection
        self.selected_node = node

        # Highlight selected node
        selected_item = self.node_items.get(node.step.step_id)
        if selected_item:
            selected_item.set_selected_state(True)

        # Highlight path to root
        path = node.get_path_to_root()
        for path_node in path:
            item = self.node_items.get(path_node.step.step_id)
            if item:
                item.set_highlighted(True)

        # Highlight edges in path
        for i in range(len(path) - 1):
            parent_item = self.node_items.get(path[i].step.step_id)
            child_item = self.node_items.get(path[i + 1].step.step_id)

            for edge in self.edge_items:
                if edge.parent_item == parent_item and edge.child_item == child_item:
                    edge.set_highlighted(True)

        # Emit signal
        self.node_selected.emit(node.step)

    def expand_all(self):
        """Expand all nodes"""
        for root in self.tree_nodes:
            self._expand_recursive(root)
        self._render_tree()

    def collapse_all(self):
        """Collapse all nodes"""
        for root in self.tree_nodes:
            self._collapse_recursive(root)
        self._render_tree()

    def _expand_recursive(self, node: ProofTreeNode):
        """Recursively expand node and children"""
        node.expand()
        for child in node.children:
            self._expand_recursive(child)

    def _collapse_recursive(self, node: ProofTreeNode):
        """Recursively collapse node and children"""
        node.collapse()
        for child in node.children:
            self._collapse_recursive(child)

    def fit_to_view(self):
        """Fit the entire tree to the view"""
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def export_to_json(self):
        """Export proof tree to JSON file"""
        if not self.current_tree:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Beweisbaum", "", "JSON Files (*.json)"
        )

        if filename:
            export_proof_to_json(self.current_tree, filename)
            self.status_label.setText(f"Exportiert nach: {filename}")

    def export_to_image(self):
        """Export proof tree to image file"""
        if not self.current_tree:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Beweisbaum als Bild",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf)",
        )

        if filename:
            # Render scene to pixmap
            rect = self.scene.sceneRect()
            pixmap = QPixmap(int(rect.width()), int(rect.height()))
            pixmap.fill(QColor("#1e1e1e"))

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.scene.render(painter)
            painter.end()

            pixmap.save(filename)
            self.status_label.setText(f"Bild exportiert nach: {filename}")

    def clear(self):
        """Clear the current proof tree"""
        self.current_tree = None
        self.tree_nodes = []
        self.selected_node = None
        self.scene.clear()
        self.node_items.clear()
        self.edge_items.clear()
        self.status_label.setText("Kein Beweisbaum geladen")

    # ==================== Filter Methods ====================

    def _on_filter_toggled(self, checked: bool):
        """Handle filter checkbox toggle"""
        self.filter_enabled = checked
        if self.current_tree:
            self._render_tree()

    def _on_confidence_changed(self, value: int):
        """Handle confidence slider change"""
        self.min_confidence = value / 100.0  # Convert 0-100 to 0.0-1.0
        self.confidence_label.setText(f"{self.min_confidence:.2f}")

        # Re-render if filter is enabled
        if self.filter_enabled and self.current_tree:
            self._render_tree()

    def _on_progressive_toggled(self, checked: bool):
        """Handle progressive rendering toggle"""
        self.progressive_rendering_enabled = checked
        if self.current_tree:
            self._render_tree()

    def _should_display_node(self, node: ProofTreeNode) -> bool:
        """
        Determine if a node should be displayed based on active filters.

        Args:
            node: ProofTreeNode to check

        Returns:
            True if node passes all active filters
        """
        # Confidence filter
        if self.filter_enabled:
            if node.step.confidence < self.min_confidence:
                return False

        # StepType filter
        if node.step.step_type not in self.enabled_step_types:
            return False

        return True

    def _show_steptype_filter_menu(self):
        """Show popup menu for StepType filter selection"""
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
            }
            QMenu::item:selected {
                background-color: #3498db;
            }
        """
        )

        # Add "All" option
        all_action = menu.addAction(
            "[OK] Alle aktivieren"
            if len(self.enabled_step_types) == len(StepType)
            else "Alle aktivieren"
        )
        all_action.triggered.connect(self._enable_all_steptypes)

        none_action = menu.addAction("Alle deaktivieren")
        none_action.triggered.connect(self._disable_all_steptypes)

        menu.addSeparator()

        # Add checkbox for each StepType
        self.steptype_actions = {}
        for step_type in StepType:
            icon = self._get_steptype_icon(step_type)
            is_enabled = step_type in self.enabled_step_types
            action = menu.addAction(
                f"{'[OK]' if is_enabled else '  '} {icon} {step_type.value}"
            )
            action.setCheckable(True)
            action.setChecked(is_enabled)
            action.triggered.connect(
                lambda checked, st=step_type: self._toggle_steptype(st, checked)
            )
            self.steptype_actions[step_type] = action

        # Show menu below button
        btn = self.sender()
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _get_steptype_icon(self, step_type: StepType) -> str:
        """Get icon for step type (reuse from ProofNodeItem)"""
        icons = {
            StepType.FACT_MATCH: "[INFO]",
            StepType.RULE_APPLICATION: "⚙️",
            StepType.INFERENCE: "💡",
            StepType.HYPOTHESIS: "🔬",
            StepType.GRAPH_TRAVERSAL: "🗺️",
            StepType.PROBABILISTIC: "🎲",
            StepType.DECOMPOSITION: "🔀",
            StepType.UNIFICATION: "🔗",
        }
        return icons.get(step_type, "*")

    def _toggle_steptype(self, step_type: StepType, enabled: bool):
        """Toggle a specific StepType filter"""
        if enabled:
            self.enabled_step_types.add(step_type)
        else:
            self.enabled_step_types.discard(step_type)

        # Re-render tree
        if self.current_tree:
            self._render_tree()

    def _enable_all_steptypes(self):
        """Enable all StepTypes"""
        self.enabled_step_types = set(StepType)
        if self.current_tree:
            self._render_tree()

    def _disable_all_steptypes(self):
        """Disable all StepTypes"""
        self.enabled_step_types.clear()
        if self.current_tree:
            self._render_tree()

    # ==================== Filter Presets ====================

    def _apply_preset_high_confidence(self):
        """Apply preset: Only high confidence steps (≥ 0.8)"""
        self.filter_enabled = True
        self.filter_checkbox.setChecked(True)
        self.min_confidence = 0.8
        self.confidence_slider.setValue(80)
        self.confidence_label.setText("0.80")

        if self.current_tree:
            self._render_tree()

    def _apply_preset_rules_only(self):
        """Apply preset: Only rule applications and inferences"""
        self.enabled_step_types = {StepType.RULE_APPLICATION, StepType.INFERENCE}

        if self.current_tree:
            self._render_tree()

    def _reset_all_filters(self):
        """Reset all filters to default (show everything)"""
        # Reset confidence filter
        self.filter_enabled = False
        self.filter_checkbox.setChecked(False)
        self.min_confidence = 0.0
        self.confidence_slider.setValue(0)
        self.confidence_label.setText("0.00")

        # Reset StepType filter
        self.enabled_step_types = set(StepType)

        if self.current_tree:
            self._render_tree()

    def _update_status_with_counter(self):
        """Update status bar with node counter"""
        if not self.current_tree:
            self.status_label.setText("Kein Beweisbaum geladen")
            return

        total_steps = len(self.current_tree.get_all_steps())
        visible_steps = self.rendered_node_count

        if visible_steps < total_steps:
            # Filters active
            self.status_label.setText(
                f"Beweisbaum: {visible_steps} von {total_steps} Knoten sichtbar "
                f"({len(self.tree_nodes)} Wurzeln) - Filter aktiv"
            )
        else:
            # No filters
            self.status_label.setText(
                f"Beweisbaum: {total_steps} Schritte, {len(self.tree_nodes)} Wurzeln"
            )

    # ==================== Performance Methods ====================

    def _count_nodes(self, node: ProofTreeNode) -> int:
        """
        Count total nodes in subtree (including collapsed).

        Args:
            node: Root of subtree

        Returns:
            Total number of nodes
        """
        count = 1  # Count this node
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _auto_collapse_large_tree(self):
        """
        Auto-collapse large trees for performance.

        Collapses all nodes beyond depth 2 to reduce rendering load.
        """
        for root in self.tree_nodes:
            self._collapse_beyond_depth(root, current_depth=0, max_depth=2)

    def _collapse_beyond_depth(
        self, node: ProofTreeNode, current_depth: int, max_depth: int
    ):
        """
        Recursively collapse nodes beyond a certain depth.

        Args:
            node: Current node
            current_depth: Current depth in tree (0 = root)
            max_depth: Maximum depth to keep expanded
        """
        if current_depth >= max_depth:
            node.collapse()
        else:
            node.expand()

        # Recurse for children
        for child in node.children:
            self._collapse_beyond_depth(child, current_depth + 1, max_depth)


# ==================== Comparison Mode Widget ====================


class ComparisonProofTreeWidget(QWidget):
    """
    Side-by-side comparison widget for two proof trees.

    Displays two ProofTreeWidget instances in a horizontal splitter
    for comparing different reasoning strategies or proof variants.

    Features:
    - Independent tree viewing
    - Synchronized zoom controls (optional)
    - Labels for Tree A and Tree B
    - Export both trees
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize comparison UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)

        control_layout.addWidget(QLabel("Vergleichsmodus"))
        control_layout.addStretch()

        # Synchronized controls checkbox
        self.sync_checkbox = QCheckBox("Zoom synchronisieren")
        self.sync_checkbox.setChecked(False)
        self.sync_checkbox.toggled.connect(self._on_sync_toggled)
        control_layout.addWidget(self.sync_checkbox)

        # Export both button
        btn_export_both = QPushButton("Beide exportieren")
        btn_export_both.clicked.connect(self._export_both_trees)
        control_layout.addWidget(btn_export_both)

        control_bar.setStyleSheet(
            """
            QWidget {
                background-color: #34495e;
                color: #ecf0f1;
                padding: 5px;
            }
            QPushButton {
                background-color: #2c3e50;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """
        )

        layout.addWidget(control_bar)

        # Splitter with two tree widgets
        from PySide6.QtWidgets import QSplitter

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left tree (Tree A)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_label = QLabel("Baum A")
        left_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; padding: 5px; font-weight: bold;"
        )
        left_layout.addWidget(left_label)

        self.tree_widget_a = ProofTreeWidget()
        left_layout.addWidget(self.tree_widget_a)

        # Right tree (Tree B)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        right_label = QLabel("Baum B")
        right_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; padding: 5px; font-weight: bold;"
        )
        right_layout.addWidget(right_label)

        self.tree_widget_b = ProofTreeWidget()
        right_layout.addWidget(self.tree_widget_b)

        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right_container)
        self.splitter.setSizes([500, 500])  # Equal split

        layout.addWidget(self.splitter)

    def set_trees(self, tree_a: ProofTree, tree_b: ProofTree):
        """
        Set both proof trees for comparison.

        Args:
            tree_a: First proof tree
            tree_b: Second proof tree
        """
        self.tree_widget_a.set_proof_tree(tree_a)
        self.tree_widget_b.set_proof_tree(tree_b)

    def set_tree_a(self, tree: ProofTree):
        """Set Tree A"""
        self.tree_widget_a.set_proof_tree(tree)

    def set_tree_b(self, tree: ProofTree):
        """Set Tree B"""
        self.tree_widget_b.set_proof_tree(tree)

    def _on_sync_toggled(self, checked: bool):
        """Handle zoom synchronization toggle"""
        if checked:
            # Connect zoom signals (simplified - just fit both to view)
            self.tree_widget_a.fit_to_view()
            self.tree_widget_b.fit_to_view()

    def _export_both_trees(self):
        """Export both trees side-by-side as single image"""
        from PySide6.QtGui import QPainter

        # Get scenes
        scene_a = self.tree_widget_a.scene
        scene_b = self.tree_widget_b.scene

        rect_a = scene_a.sceneRect()
        rect_b = scene_b.sceneRect()

        # Calculate combined dimensions
        combined_width = int(rect_a.width() + rect_b.width() + 20)  # 20px gap
        combined_height = int(max(rect_a.height(), rect_b.height()))

        # Create combined pixmap
        pixmap = QPixmap(combined_width, combined_height)
        pixmap.fill(QColor("#1e1e1e"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Render Tree A (left)
        scene_a.render(painter, target=QRectF(0, 0, rect_a.width(), rect_a.height()))

        # Render Tree B (right)
        scene_b.render(
            painter,
            target=QRectF(rect_a.width() + 20, 0, rect_b.width(), rect_b.height()),
        )

        painter.end()

        # Save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Vergleich", "", "PNG Files (*.png)"
        )

        if filename:
            pixmap.save(filename)
            print(f"Comparison exported to: {filename}")

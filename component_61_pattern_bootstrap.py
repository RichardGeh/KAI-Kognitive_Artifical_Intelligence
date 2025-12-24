# component_61_pattern_bootstrap.py
"""
Pattern Bootstrap System for Pattern Discovery.

This module provides functionality to load seed templates from YAML files
and create initial Pattern structures in Neo4j. These seed patterns represent
"2-3 years of linguistic knowledge" and bootstrap the pattern learning system.

Responsibilities:
- Load seed templates from YAML configuration
- Create Pattern, PatternItem, and Slot nodes
- Validate template structure
- Handle UTF-8 encoding properly (CLAUDE.md requirement)

Part of Pattern Discovery System - Part 2: Utterance Storage & Pattern Matching.
"""

import threading
from pathlib import Path
from typing import Any, Dict

import yaml

from component_15_logging_config import get_logger
from kai_exceptions import ConfigurationException, DatabaseException, wrap_exception

logger = get_logger(__name__)

# Thread lock for bootstrap operations
_bootstrap_lock = threading.RLock()


def load_seed_templates(
    netzwerk, yaml_path: str = "seed_templates.yml"
) -> Dict[str, int]:
    """
    Loads seed templates from YAML and creates Pattern nodes in Neo4j.

    This function is idempotent - can be called multiple times safely.
    Executed once on first startup or via UI button "Bootstrap Patterns".

    CRITICAL: Uses UTF-8 encoding to prevent Windows cp1252 issues (CLAUDE.md requirement).

    Args:
        netzwerk: KonzeptNetzwerk instance for database access
        yaml_path: Path to YAML file with seed templates (default: "seed_templates.yml")

    Returns:
        Dict with statistics:
        - patterns_created: Number of patterns created
        - items_created: Number of pattern items created
        - slots_created: Number of slots created

    Raises:
        ConfigurationException: If YAML is invalid or missing required fields
        DatabaseException: If Neo4j operations fail

    Example:
        >>> stats = load_seed_templates(netzwerk, "seed_templates.yml")
        >>> print(f"Created {stats['patterns_created']} patterns")
    """
    with _bootstrap_lock:
        logger.info(f"Loading seed templates from: {yaml_path}")

        # Validate file exists
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise ConfigurationException(
                f"Seed templates file not found: {yaml_path}",
                context={"yaml_path": yaml_path},
            )

        # Load YAML with UTF-8 encoding (CRITICAL for Windows cp1252 compatibility)
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationException(
                f"Invalid YAML syntax in seed templates: {e}",
                context={"yaml_path": yaml_path, "error": str(e)},
            )
        except Exception as e:
            raise wrap_exception(
                e,
                ConfigurationException,
                "Failed to read seed templates file",
                yaml_path=yaml_path,
            )

        # Validate YAML structure
        if not data or "templates" not in data:
            raise ConfigurationException(
                "Seed templates YAML must contain 'templates' key",
                context={"yaml_path": yaml_path},
            )

        templates = data["templates"]
        if not isinstance(templates, list):
            raise ConfigurationException(
                "'templates' must be a list",
                context={"yaml_path": yaml_path, "type": type(templates).__name__},
            )

        # Statistics
        stats = {"patterns_created": 0, "items_created": 0, "slots_created": 0}

        # Process each template
        for idx, template in enumerate(templates):
            try:
                logger.debug(
                    f"Starting template {idx}/{len(templates)}: {template.get('name', 'unknown')}"
                )
                _process_template(netzwerk, template, idx, stats)
                logger.debug(
                    f"Successfully processed template {idx}: {template.get('name', 'unknown')}"
                )
            except Exception as e:
                # Log error with full traceback and continue with next template
                logger.error(
                    f"Failed to process template {idx} ('{template.get('name', 'unknown')}'): {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={
                        "template_name": template.get("name", "unknown"),
                        "template_id": template.get("id", "unknown"),
                        "error_type": type(e).__name__,
                    },
                )
                # Re-raise if it's a critical database error (not configuration)
                if isinstance(e, DatabaseException):
                    logger.critical("Database error during bootstrap - stopping")
                    raise

        logger.info(
            f"Seed templates loaded successfully: "
            f"{stats['patterns_created']} patterns, "
            f"{stats['items_created']} items, "
            f"{stats['slots_created']} slots"
        )

        return stats


def _process_template(
    netzwerk, template: Dict[str, Any], template_idx: int, stats: Dict[str, int]
):
    """
    Process a single template and create Pattern structure.

    Args:
        netzwerk: KonzeptNetzwerk instance
        template: Template dict from YAML
        template_idx: Template index (for error reporting)
        stats: Statistics dict to update

    Raises:
        ConfigurationException: If template structure is invalid
        DatabaseException: If database operations fail
    """
    logger.debug(
        f"Processing template {template_idx}: {template.get('name', 'UNNAMED')}"
    )

    # Validate required fields
    required_fields = ["id", "name", "category", "pattern"]
    for field in required_fields:
        if field not in template:
            raise ConfigurationException(
                f"Template {template_idx} missing required field: {field}",
                context={"template": template},
            )

    # Validate pattern structure
    pattern = template["pattern"]
    if not isinstance(pattern, list) or len(pattern) == 0:
        raise ConfigurationException(
            f"Template {template_idx} pattern must be non-empty list",
            context={"template_name": template["name"]},
        )

    # Create Pattern node
    logger.debug(f"Creating pattern node for template: {template['name']}")
    pattern_id = netzwerk.create_pattern(
        name=template["name"],
        pattern_type="seed",  # Mark as manually created
        metadata={"category": template["category"], "template_id": template["id"]},
    )
    logger.debug(f"Pattern created with ID: {pattern_id}")
    stats["patterns_created"] += 1

    # Create PatternItems and Slots
    items_data = []
    for idx, item in enumerate(pattern):
        item_data = _process_pattern_item(netzwerk, item, idx, stats)
        items_data.append(item_data)

    # Batch create PatternItems (efficient UNWIND operation)
    items_created = netzwerk.batch_create_pattern_items(pattern_id, items_data)
    stats["items_created"] += items_created

    # Initialize pattern statistics
    netzwerk.update_pattern_stats(
        pattern_id=pattern_id,
        support_increment=0,
        new_precision=0.5,  # Initial uncertainty
    )

    logger.debug(
        f"Template processed: {template['name']} (items={items_created})",
        extra={"pattern_id": pattern_id},
    )


def _process_pattern_item(
    netzwerk, item: Dict[str, Any], idx: int, stats: Dict[str, int]
) -> Dict[str, Any]:
    """
    Process a single pattern item (LITERAL or SLOT).

    Args:
        netzwerk: KonzeptNetzwerk instance
        item: Item dict from YAML pattern
        idx: Item index in pattern
        stats: Statistics dict to update

    Returns:
        Dict with item data for batch creation

    Raises:
        ConfigurationException: If item structure is invalid
    """
    logger.debug(f"Processing pattern item {idx}: kind={item.get('kind', 'MISSING')}")

    # Validate item structure
    if "kind" not in item:
        raise ConfigurationException(
            f"Pattern item {idx} missing 'kind' field", context={"item": item}
        )

    kind = item["kind"]
    if kind not in ["LITERAL", "SLOT"]:
        raise ConfigurationException(
            f"Pattern item {idx} has invalid kind: {kind}",
            context={"item": item, "valid_kinds": ["LITERAL", "SLOT"]},
        )

    import uuid

    item_id = str(uuid.uuid4())

    if kind == "LITERAL":
        # LITERAL item: Direct value match
        if "value" not in item:
            raise ConfigurationException(
                f"LITERAL item {idx} missing 'value' field", context={"item": item}
            )

        return {
            "id": item_id,
            "idx": idx,
            "kind": "LITERAL",
            "literalValue": item["value"],
        }

    else:  # SLOT
        # SLOT item: Variable with constraints
        if "slot_type" not in item:
            raise ConfigurationException(
                f"SLOT item {idx} missing 'slot_type' field", context={"item": item}
            )

        # Create Slot node
        logger.debug(
            f"Creating slot for item {idx}: type={item['slot_type']}, allowed={len(item.get('allowed', []))}"
        )
        slot_id = netzwerk.create_slot(
            slot_type=item["slot_type"],
            allowed_values=item.get("allowed", []),
            min_count=1,
            max_count=1,
        )
        logger.debug(f"Slot created with ID: {slot_id}")
        stats["slots_created"] += 1

        return {"id": item_id, "idx": idx, "kind": "SLOT", "slotId": slot_id}


def check_bootstrap_status(netzwerk) -> Dict[str, Any]:
    """
    Check if seed templates have been loaded.

    Args:
        netzwerk: KonzeptNetzwerk instance

    Returns:
        Dict with:
        - bootstrapped: True if seed patterns exist
        - seed_pattern_count: Number of seed patterns
        - total_pattern_count: Total patterns in graph

    Example:
        >>> status = check_bootstrap_status(netzwerk)
        >>> if not status["bootstrapped"]:
        ...     load_seed_templates(netzwerk)
    """
    try:
        # Count seed patterns
        seed_patterns = netzwerk.get_all_patterns(type_filter="seed", limit=100)
        all_patterns = netzwerk.get_all_patterns(limit=1000)

        return {
            "bootstrapped": len(seed_patterns) > 0,
            "seed_pattern_count": len(seed_patterns),
            "total_pattern_count": len(all_patterns),
        }
    except Exception as e:
        logger.error(f"Failed to check bootstrap status: {e}", exc_info=True)
        return {
            "bootstrapped": False,
            "seed_pattern_count": 0,
            "total_pattern_count": 0,
            "error": str(e),
        }

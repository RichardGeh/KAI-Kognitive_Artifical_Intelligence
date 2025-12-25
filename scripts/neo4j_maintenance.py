"""
neo4j_maintenance.py

Neo4j database maintenance utility for the KAI project.

This module provides functions for Neo4j database maintenance operations including:
- Connection testing
- Database clearing (full or test data only)
- Statistics retrieval
- Memory information (edition-dependent)
- Restart request via PowerShell script

Usage as CLI:
    python scripts/neo4j_maintenance.py --check          # Check connection
    python scripts/neo4j_maintenance.py --stats          # Show database stats
    python scripts/neo4j_maintenance.py --clear-test     # Clear test data only
    python scripts/neo4j_maintenance.py --clear-all      # Clear entire database
    python scripts/neo4j_maintenance.py --restart        # Request restart

Usage as module:
    from scripts.neo4j_maintenance import (
        check_connection,
        get_database_stats,
        clear_test_data,
        clear_database,
        request_restart
    )

    if check_connection():
        stats = get_database_stats(driver)
        print(f"Total nodes: {stats['node_count']}")

Note: This module follows CLAUDE.md guidelines:
    - NO Unicode chars that cause cp1252 issues
    - Proper error handling with kai_exceptions
    - Type hints throughout
    - Parameterized Neo4j queries only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports when run as script
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import (
    AuthError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)

from component_15_logging_config import get_logger
from kai_exceptions import (
    Neo4jConnectionError,
    Neo4jQueryError,
    Neo4jWriteError,
)

logger = get_logger(__name__)

# Default connection constants
DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "password"

# Path to restart script (relative to this file)
RESTART_SCRIPT_PATH = _script_dir / "restart_neo4j.ps1"


def check_connection(
    uri: str = DEFAULT_URI,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    timeout: int = 5,
) -> bool:
    """
    Test if Neo4j database is responsive.

    Args:
        uri: Neo4j bolt URI (default: bolt://127.0.0.1:7687)
        user: Database username (default: neo4j)
        password: Database password (default: password)
        timeout: Connection timeout in seconds (default: 5)

    Returns:
        True if connection successful, False otherwise

    Example:
        if check_connection():
            print("[OK] Neo4j is available")
        else:
            print("[ERROR] Neo4j is not responding")
    """
    driver: Optional[Driver] = None
    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=timeout,
            max_connection_lifetime=timeout,
        )
        # Verify connectivity
        driver.verify_connectivity()

        # Run a simple query to confirm database is fully operational
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            if record and record["test"] == 1:
                logger.info(
                    "Neo4j connection check successful",
                    extra={"uri": uri, "user": user},
                )
                return True

        logger.warning("Neo4j connection established but query failed")
        return False

    except AuthError as e:
        logger.error(
            "Neo4j authentication failed",
            extra={"uri": uri, "user": user, "error": str(e)},
        )
        return False

    except ServiceUnavailable as e:
        logger.error(
            "Neo4j service unavailable",
            extra={"uri": uri, "error": str(e)},
        )
        return False

    except Exception as e:
        logger.error(
            "Neo4j connection check failed",
            extra={"uri": uri, "error": str(e), "error_type": type(e).__name__},
        )
        return False

    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass  # Ignore close errors


def get_driver(
    uri: str = DEFAULT_URI,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
) -> Driver:
    """
    Create and return a Neo4j driver instance.

    Args:
        uri: Neo4j bolt URI
        user: Database username
        password: Database password

    Returns:
        Neo4j Driver instance

    Raises:
        Neo4jConnectionError: If connection cannot be established
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except AuthError as e:
        raise Neo4jConnectionError(
            f"Authentication failed for user '{user}'",
            context={"uri": uri, "user": user},
            original_exception=e,
        )
    except ServiceUnavailable as e:
        raise Neo4jConnectionError(
            f"Neo4j service not available at {uri}",
            context={"uri": uri},
            original_exception=e,
        )
    except Exception as e:
        raise Neo4jConnectionError(
            f"Failed to connect to Neo4j: {e}",
            context={"uri": uri},
            original_exception=e,
        )


def clear_database(driver: Driver, batch_size: int = 1000) -> int:
    """
    Delete all nodes and relationships from the database in batches.

    Uses CALL { ... } IN TRANSACTIONS pattern for efficient large-scale deletion
    without running out of memory.

    Args:
        driver: Neo4j driver instance
        batch_size: Number of nodes to delete per batch (default: 1000)

    Returns:
        Total number of nodes deleted

    Raises:
        Neo4jWriteError: If deletion fails

    Warning:
        This operation is DESTRUCTIVE and cannot be undone!
        All data in the database will be permanently deleted.

    Example:
        driver = get_driver()
        deleted = clear_database(driver)
        print(f"Deleted {deleted} nodes")
    """
    total_deleted = 0

    try:
        with driver.session() as session:
            # First, get count for logging
            count_result = session.run("MATCH (n) RETURN count(n) AS count")
            record = count_result.single()
            initial_count = record["count"] if record else 0

            if initial_count == 0:
                logger.info("Database is already empty, nothing to delete")
                return 0

            logger.info(
                f"Starting database clear operation",
                extra={"initial_node_count": initial_count, "batch_size": batch_size},
            )

            # Delete in batches using CALL { } IN TRANSACTIONS
            # This pattern handles large datasets efficiently
            while True:
                # Delete a batch of nodes (relationships are deleted automatically
                # when nodes are deleted via DETACH DELETE)
                delete_query = """
                MATCH (n)
                WITH n LIMIT $batch_size
                DETACH DELETE n
                RETURN count(*) AS deleted
                """
                result = session.run(delete_query, {"batch_size": batch_size})
                record = result.single()
                deleted_count = record["deleted"] if record else 0

                if deleted_count == 0:
                    break

                total_deleted += deleted_count
                logger.debug(
                    f"Deleted batch",
                    extra={"batch_deleted": deleted_count, "total_deleted": total_deleted},
                )

            logger.info(
                "Database clear completed",
                extra={"total_deleted": total_deleted},
            )
            return total_deleted

    except (ServiceUnavailable, SessionExpired, TransientError) as e:
        raise Neo4jWriteError(
            f"Database connection lost during clear operation",
            context={"deleted_so_far": total_deleted},
            original_exception=e,
        )
    except Exception as e:
        raise Neo4jWriteError(
            f"Failed to clear database: {e}",
            context={"deleted_so_far": total_deleted},
            original_exception=e,
        )


def clear_test_data(driver: Driver, prefix: str = "test_") -> int:
    """
    Delete only test data nodes (nodes with lemma starting with prefix).

    Uses the indexed 'lemma' property for efficient lookups.
    This is a safer alternative to clear_database() for development workflows
    where you want to clean up test artifacts without affecting real data.

    Args:
        driver: Neo4j driver instance
        prefix: Prefix to match for test nodes (default: "test_")

    Returns:
        Number of test nodes deleted

    Raises:
        Neo4jWriteError: If deletion fails

    Example:
        driver = get_driver()
        deleted = clear_test_data(driver, prefix="test_")
        print(f"Deleted {deleted} test nodes")
    """
    total_deleted = 0

    try:
        with driver.session() as session:
            # Count test nodes first (use only lemma - it has an index)
            count_query = """
            MATCH (n)
            WHERE n.lemma STARTS WITH $prefix
            RETURN count(n) AS count
            """
            count_result = session.run(count_query, {"prefix": prefix})
            record = count_result.single()
            initial_count = record["count"] if record else 0

            if initial_count == 0:
                logger.info(
                    f"No test data found with prefix '{prefix}'",
                    extra={"prefix": prefix},
                )
                return 0

            logger.info(
                f"Found {initial_count} test nodes to delete",
                extra={"prefix": prefix, "count": initial_count},
            )

            # Delete test nodes in batches (use only lemma - it has an index)
            batch_size = 500
            while True:
                delete_query = """
                MATCH (n)
                WHERE n.lemma STARTS WITH $prefix
                WITH n LIMIT $batch_size
                DETACH DELETE n
                RETURN count(*) AS deleted
                """
                result = session.run(
                    delete_query, {"prefix": prefix, "batch_size": batch_size}
                )
                record = result.single()
                deleted_count = record["deleted"] if record else 0

                if deleted_count == 0:
                    break

                total_deleted += deleted_count
                logger.debug(
                    f"Deleted test data batch",
                    extra={
                        "batch_deleted": deleted_count,
                        "total_deleted": total_deleted,
                        "prefix": prefix,
                    },
                )

            logger.info(
                f"Test data clear completed",
                extra={"prefix": prefix, "total_deleted": total_deleted},
            )
            return total_deleted

    except (ServiceUnavailable, SessionExpired, TransientError) as e:
        raise Neo4jWriteError(
            f"Database connection lost during test data clear",
            context={"prefix": prefix, "deleted_so_far": total_deleted},
            original_exception=e,
        )
    except Exception as e:
        raise Neo4jWriteError(
            f"Failed to clear test data: {e}",
            context={"prefix": prefix, "deleted_so_far": total_deleted},
            original_exception=e,
        )


def get_database_stats(driver: Driver) -> Dict[str, Any]:
    """
    Get database statistics including node and relationship counts.

    Args:
        driver: Neo4j driver instance

    Returns:
        Dictionary with database statistics:
        - node_count: Total number of nodes
        - relationship_count: Total number of relationships
        - labels: Dict of label -> count
        - relationship_types: Dict of type -> count

    Raises:
        Neo4jQueryError: If statistics query fails

    Example:
        driver = get_driver()
        stats = get_database_stats(driver)
        print(f"Nodes: {stats['node_count']}")
        print(f"Relationships: {stats['relationship_count']}")
    """
    try:
        with driver.session() as session:
            # Get total node count
            node_result = session.run("MATCH (n) RETURN count(n) AS count")
            node_record = node_result.single()
            node_count = node_record["count"] if node_record else 0

            # Get total relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            rel_record = rel_result.single()
            rel_count = rel_record["count"] if rel_record else 0

            # Get label counts (Neo4j 5.x compatible syntax)
            label_query = """
            CALL db.labels() YIELD label
            CALL (label) {
                MATCH (n)
                WHERE label IN labels(n)
                RETURN count(n) AS count
            }
            RETURN label, count
            ORDER BY count DESC
            """
            try:
                label_result = session.run(label_query)
                labels = {record["label"]: record["count"] for record in label_result}
            except Exception:
                # Fallback for older Neo4j versions
                labels = {}

            # Get relationship type counts (Neo4j 5.x compatible syntax)
            rel_type_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            CALL (relationshipType) {
                MATCH ()-[r]->()
                WHERE type(r) = relationshipType
                RETURN count(r) AS count
            }
            RETURN relationshipType, count
            ORDER BY count DESC
            """
            try:
                rel_type_result = session.run(rel_type_query)
                rel_types = {
                    record["relationshipType"]: record["count"]
                    for record in rel_type_result
                }
            except Exception:
                # Fallback for older Neo4j versions
                rel_types = {}

            stats = {
                "node_count": node_count,
                "relationship_count": rel_count,
                "labels": labels,
                "relationship_types": rel_types,
            }

            logger.info(
                "Database stats retrieved",
                extra={
                    "node_count": node_count,
                    "rel_count": rel_count,
                    "label_count": len(labels),
                    "rel_type_count": len(rel_types),
                },
            )

            return stats

    except Exception as e:
        raise Neo4jQueryError(
            f"Failed to retrieve database statistics: {e}",
            original_exception=e,
        )


def get_memory_info(driver: Driver) -> Dict[str, Any]:
    """
    Attempt to retrieve Neo4j memory metrics.

    Note: Memory information may not be available in all Neo4j editions.
    Community edition has limited access to these metrics.

    Args:
        driver: Neo4j driver instance

    Returns:
        Dictionary with memory information:
        - available: Whether memory info is available
        - heap_used: Heap memory used (if available)
        - heap_max: Maximum heap size (if available)
        - page_cache_used: Page cache used (if available)
        - page_cache_max: Maximum page cache (if available)
        - message: Status message

    Example:
        driver = get_driver()
        mem = get_memory_info(driver)
        if mem['available']:
            print(f"Heap: {mem['heap_used']} / {mem['heap_max']}")
    """
    try:
        with driver.session() as session:
            # Try to get memory metrics (Enterprise/some Community versions)
            try:
                # This query works in Neo4j 4.x+ with appropriate permissions
                mem_query = """
                CALL dbms.queryJmx('org.neo4j:name=Primitive count,*')
                YIELD name, attributes
                RETURN name, attributes
                """
                result = session.run(mem_query)
                records = list(result)

                if records:
                    # Parse JMX output
                    return {
                        "available": True,
                        "message": "Memory metrics retrieved via JMX",
                        "raw_data": [
                            {"name": r["name"], "attributes": r["attributes"]}
                            for r in records
                        ],
                    }

            except Exception:
                pass

            # Try alternative approach - database store size
            try:
                store_query = """
                CALL dbms.listConfig() YIELD name, value
                WHERE name CONTAINS 'memory' OR name CONTAINS 'heap'
                RETURN name, value
                """
                result = session.run(store_query)
                config = {record["name"]: record["value"] for record in result}

                if config:
                    return {
                        "available": True,
                        "message": "Memory configuration retrieved",
                        "config": config,
                    }
            except Exception:
                pass

            # Memory info not available
            return {
                "available": False,
                "message": "Memory information not available (may require Enterprise edition or elevated permissions)",
            }

    except Exception as e:
        logger.warning(
            f"Could not retrieve memory information: {e}",
            extra={"error_type": type(e).__name__},
        )
        return {
            "available": False,
            "message": f"Error retrieving memory info: {e}",
        }


def request_restart() -> bool:
    """
    Request Neo4j restart via PowerShell script.

    This function calls the restart_neo4j.ps1 script to stop the Neo4j Desktop
    process. The user will need to manually start the database in Neo4j Desktop
    GUI after the restart.

    Returns:
        True if restart script executed successfully, False otherwise

    IMPORTANT - CLAUDE CODE NOTICE:
        Neo4j Desktop is a GUI application. After calling this function,
        you MUST inform the user that they need to manually:
        1. Open Neo4j Desktop (if not already open)
        2. Click "Start" on their database instance
        3. Wait for the database to show "Active" status

        Claude Code CANNOT start the database automatically!
        Always ask the user to confirm when the database is ready.

    Example:
        if request_restart():
            print("Neo4j restart requested.")
            print("Please start the database in Neo4j Desktop GUI.")
            # WAIT for user confirmation before continuing!
    """
    if not RESTART_SCRIPT_PATH.exists():
        logger.error(
            f"Restart script not found",
            extra={"path": str(RESTART_SCRIPT_PATH)},
        )
        print(f"[ERROR] Restart script not found at: {RESTART_SCRIPT_PATH}")
        return False

    try:
        print("[INFO] Requesting Neo4j restart...")
        print(f"[INFO] Running: {RESTART_SCRIPT_PATH}")

        # Run PowerShell script
        result = subprocess.run(
            [
                "powershell.exe",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(RESTART_SCRIPT_PATH),
            ],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        # Print script output
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")

        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                print(f"  [STDERR] {line}")

        if result.returncode == 0:
            print("")
            print("[OK] Neo4j restart script completed successfully.")
            print("")
            print("IMPORTANT: You may need to manually start the database")
            print("           in the Neo4j Desktop GUI.")
            logger.info("Neo4j restart requested successfully")
            return True
        else:
            print(f"[ERROR] Restart script exited with code: {result.returncode}")
            print("")
            print("Please manually restart Neo4j Desktop:")
            print("  1. Close Neo4j Desktop if running")
            print("  2. Open Neo4j Desktop")
            print("  3. Start your database instance")
            logger.error(
                "Neo4j restart script failed",
                extra={"return_code": result.returncode},
            )
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] Restart script timed out after 120 seconds")
        logger.error("Neo4j restart script timed out")
        return False
    except FileNotFoundError:
        print("[ERROR] PowerShell not found. Please restart Neo4j manually.")
        logger.error("PowerShell not found for restart")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run restart script: {e}")
        logger.error(f"Failed to run restart script: {e}")
        return False


def _format_stats(stats: Dict[str, Any]) -> str:
    """Format database stats for display."""
    lines = []
    lines.append("=" * 50)
    lines.append("Neo4j Database Statistics")
    lines.append("=" * 50)
    lines.append(f"Total Nodes:         {stats['node_count']:,}")
    lines.append(f"Total Relationships: {stats['relationship_count']:,}")

    if stats.get("labels"):
        lines.append("")
        lines.append("Node Labels:")
        for label, count in sorted(
            stats["labels"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  - {label}: {count:,}")

    if stats.get("relationship_types"):
        lines.append("")
        lines.append("Relationship Types:")
        for rel_type, count in sorted(
            stats["relationship_types"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  - {rel_type}: {count:,}")

    lines.append("=" * 50)
    return "\n".join(lines)


def _format_memory_info(mem_info: Dict[str, Any]) -> str:
    """Format memory info for display."""
    lines = []
    lines.append("=" * 50)
    lines.append("Neo4j Memory Information")
    lines.append("=" * 50)

    if mem_info.get("available"):
        lines.append(f"Status: {mem_info.get('message', 'Available')}")

        if "config" in mem_info:
            lines.append("")
            lines.append("Memory Configuration:")
            for key, value in sorted(mem_info["config"].items()):
                lines.append(f"  - {key}: {value}")
    else:
        lines.append(f"Status: Not Available")
        lines.append(f"Reason: {mem_info.get('message', 'Unknown')}")

    lines.append("=" * 50)
    return "\n".join(lines)


def main() -> int:
    """
    CLI entry point for neo4j_maintenance.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Neo4j database maintenance utility for KAI project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python neo4j_maintenance.py --check          # Check connection
  python neo4j_maintenance.py --stats          # Show database stats
  python neo4j_maintenance.py --memory         # Show memory info
  python neo4j_maintenance.py --clear-test     # Clear test data only
  python neo4j_maintenance.py --clear-all      # Clear entire database
  python neo4j_maintenance.py --restart        # Request restart
        """,
    )

    # Connection options
    parser.add_argument(
        "--uri",
        default=DEFAULT_URI,
        help=f"Neo4j bolt URI (default: {DEFAULT_URI})",
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER,
        help=f"Database username (default: {DEFAULT_USER})",
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help=f"Database password (default: {DEFAULT_PASSWORD})",
    )

    # Operations (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Check if Neo4j is responsive",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics",
    )
    group.add_argument(
        "--memory",
        action="store_true",
        help="Show memory information",
    )
    group.add_argument(
        "--clear-test",
        action="store_true",
        help="Clear test data only (nodes with test_ prefix)",
    )
    group.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear entire database (DESTRUCTIVE!)",
    )
    group.add_argument(
        "--restart",
        action="store_true",
        help="Request Neo4j restart via PowerShell script",
    )

    # Optional arguments
    parser.add_argument(
        "--prefix",
        default="test_",
        help="Prefix for test data nodes (default: test_)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for delete operations (default: 1000)",
    )

    args = parser.parse_args()

    # Handle restart (doesn't need connection)
    if args.restart:
        success = request_restart()
        return 0 if success else 1

    # Handle connection check
    if args.check:
        print(f"Checking connection to {args.uri}...")
        if check_connection(args.uri, args.user, args.password):
            print(f"[OK] Neo4j is responsive at {args.uri}")
            return 0
        else:
            print(f"[ERROR] Neo4j is not responding at {args.uri}")
            return 1

    # All other operations need a driver
    try:
        driver = get_driver(args.uri, args.user, args.password)
    except Neo4jConnectionError as e:
        print(f"[ERROR] {e.message}")
        return 1

    try:
        if args.stats:
            stats = get_database_stats(driver)
            print(_format_stats(stats))
            return 0

        elif args.memory:
            mem_info = get_memory_info(driver)
            print(_format_memory_info(mem_info))
            return 0

        elif args.clear_test:
            if not args.yes:
                confirm = input(
                    f"Delete all nodes with prefix '{args.prefix}'? [y/N]: "
                )
                if confirm.lower() != "y":
                    print("Operation cancelled.")
                    return 0

            deleted = clear_test_data(driver, prefix=args.prefix)
            print(f"[OK] Deleted {deleted:,} test nodes with prefix '{args.prefix}'")
            return 0

        elif args.clear_all:
            if not args.yes:
                print("WARNING: This will delete ALL data in the database!")
                print("This operation cannot be undone.")
                confirm = input("Type 'DELETE ALL' to confirm: ")
                if confirm != "DELETE ALL":
                    print("Operation cancelled.")
                    return 0

            deleted = clear_database(driver, batch_size=args.batch_size)
            print(f"[OK] Deleted {deleted:,} nodes from database")
            return 0

    except (Neo4jQueryError, Neo4jWriteError) as e:
        print(f"[ERROR] {e.message}")
        return 1

    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

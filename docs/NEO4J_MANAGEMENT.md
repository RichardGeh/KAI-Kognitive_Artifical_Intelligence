# Neo4j Self-Management for Claude Code

## Status: IMPLEMENTED

Implementation completed on 2025-12-25.

---

## Problem (Original)

During integration test runs, Neo4j frequently becomes unresponsive due to:
- Memory pool exhaustion (`MemoryPoolOutOfMemoryError`)
- Java heap space errors (`OutOfMemoryError`)
- Connection timeouts after heavy load

---

## Solution Implemented

### Installation Type: Neo4j Desktop

Neo4j is installed as **Neo4j Desktop** at:
```
C:\Users\gehri\AppData\Local\Programs\neo4j-desktop
```

**Important Limitation**: Neo4j Desktop is a GUI application. The database must be started manually through the Neo4j Desktop interface.

---

## Components Created

### 1. PowerShell Restart Script

**File**: `scripts/restart_neo4j.ps1`

Stops and restarts Neo4j Desktop application. After restart, user must manually start the database in the GUI.

```powershell
# Usage
.\scripts\restart_neo4j.ps1

# With custom timeout
.\scripts\restart_neo4j.ps1 -MaxWaitSeconds 120
```

### 2. Python Maintenance Utility

**File**: `scripts/neo4j_maintenance.py`

Provides database maintenance functions and CLI interface.

**CLI Usage**:
```bash
# Check connection
python scripts/neo4j_maintenance.py --check

# Show database statistics
python scripts/neo4j_maintenance.py --stats

# Show memory info (if available)
python scripts/neo4j_maintenance.py --memory

# Clear test data only (safe - only deletes test_ prefixed nodes)
python scripts/neo4j_maintenance.py --clear-test

# Clear entire database (requires confirmation)
python scripts/neo4j_maintenance.py --clear-all

# Request restart (calls PowerShell script)
python scripts/neo4j_maintenance.py --restart
```

**Module Usage**:
```python
from scripts.neo4j_maintenance import (
    check_connection,
    get_database_stats,
    clear_test_data,
    clear_database,
    get_memory_info,
    request_restart
)

# Check if Neo4j is available
if check_connection():
    print("[OK] Neo4j is responding")

# Get statistics
stats = get_database_stats(driver)
print(f"Nodes: {stats['node_count']}")

# Clear only test data (safe for development)
deleted = clear_test_data(driver, prefix="test_")
print(f"Deleted {deleted} test nodes")
```

### 3. Pytest Fixtures

**File**: `tests/conftest.py` (lines 530-574)

Two new fixtures added:

```python
# Cleanup test data before/after each test
@pytest.fixture(scope="function")
def neo4j_cleanup(netzwerk_session):
    ...

# Verify Neo4j connection at session start
@pytest.fixture(scope="session")
def neo4j_health_check():
    ...
```

**Usage in tests**:
```python
def test_with_cleanup(netzwerk_session, neo4j_cleanup):
    # Test data with 'test_' prefix will be auto-cleaned
    ...

def test_requires_neo4j(neo4j_health_check, netzwerk_session):
    # Fails early with helpful message if Neo4j unavailable
    ...
```

---

## Claude Code Notice

**IMPORTANT FOR CLAUDE CODE:**

Neo4j Desktop is a GUI application. Claude Code CANNOT start the database automatically!

When using `request_restart()` or `--restart`, you MUST:
1. Inform the user that manual action is required
2. Ask the user to start the database in Neo4j Desktop GUI
3. Wait for user confirmation before continuing with tests/operations
4. Use `check_connection()` to verify the database is ready

Example interaction:
```
Claude: "Neo4j restart requested. Please start the database in Neo4j Desktop:
         1. Open Neo4j Desktop
         2. Click 'Start' on your database
         3. Let me know when it shows 'Active'"
User: "Done, it's running"
Claude: [runs check_connection() to verify, then continues]
```

---

## Manual Restart Procedure

When Neo4j becomes unresponsive, follow these steps:

1. **Try the restart script first**:
   ```bash
   python scripts/neo4j_maintenance.py --restart
   ```

2. **If that fails, manually restart**:
   - Close Neo4j Desktop (if frozen, use Task Manager)
   - Open Neo4j Desktop
   - Select your database project
   - Click "Start" on the database instance
   - Wait for "Active" status

3. **Verify connection**:
   ```bash
   python scripts/neo4j_maintenance.py --check
   ```

---

## Best Practices for Test Runs

1. **Use test data prefixes**: All test nodes should use `test_` prefix for names/lemmas
2. **Run cleanup between test sessions**:
   ```bash
   python scripts/neo4j_maintenance.py --clear-test
   ```
3. **Monitor database size** before long test runs:
   ```bash
   python scripts/neo4j_maintenance.py --stats
   ```
4. **Use the cleanup fixture** in tests that create data

---

## Memory Recommendations

For heavy test workloads, consider adjusting Neo4j Desktop settings:

1. Open Neo4j Desktop -> Database Settings
2. Adjust memory settings:
   - `dbms.memory.heap.initial_size`: 512m
   - `dbms.memory.heap.max_size`: 1G
   - `dbms.memory.transaction.total.max`: 512m

---

## Files Overview

| File | Purpose |
|------|---------|
| `scripts/restart_neo4j.ps1` | PowerShell script to restart Neo4j Desktop |
| `scripts/neo4j_maintenance.py` | Python maintenance utility (CLI + module) |
| `tests/conftest.py` | Pytest fixtures for cleanup and health check |

---

*Completed: 2025-12-25*

# debug_orchestrator.py - Check extraction rules
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "password"))

print("=== EXTRACTION RULES ===")
with driver.session(database="neo4j") as session:
    result = session.run("MATCH (r:ExtractionRule) RETURN r.relation_type, r.regex_pattern")
    rules = list(result)
    for r in rules:
        print(f"  {r['r.relation_type']}: {r['r.regex_pattern']}")
    if not rules:
        print("  (keine Regeln gefunden)")

print("\n=== PRODUCTION RULES COUNT ===")
with driver.session(database="neo4j") as session:
    result = session.run("MATCH (r:ProductionRule) RETURN count(r) as count")
    count = result.single()["count"]
    print(f"  ProductionRules: {count}")

driver.close()
print("\n[DONE]")

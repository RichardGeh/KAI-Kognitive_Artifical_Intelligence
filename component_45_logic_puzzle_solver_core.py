"""
component_45_logic_puzzle_solver_core.py
========================================
SAT-based logic puzzle solver with CNF conversion and answer formatting.

This module handles:
- Converting logical conditions to CNF formulas
- Solving puzzles using SAT solver (DPLL algorithm)
- Formatting solutions as natural language answers
- Generating proof trees for transparent reasoning
- Routing to numerical CSP solver for constraint-based puzzles

Logic transformations:
- IMPLICATION (X -> Y):  NOT X OR Y
- XOR (X XOR Y):         (X OR Y) AND (NOT X OR NOT Y)
- NEVER_BOTH NOT(X AND Y): NOT X OR NOT Y
- CONJUNCTION (X AND Y): X, Y (separate clauses)
- DISJUNCTION (X OR Y):  X OR Y
- NEGATION (NOT X):      NOT X

Author: KAI Development Team
Date: 2025-11-29 (Split from component_45)
Modified: 2025-12-05 (Added puzzle type routing - PHASE 4)
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from component_15_logging_config import get_logger
from component_30_sat_solver import Clause, CNFFormula, Literal, SATSolver
from component_45_logic_puzzle_parser import LogicCondition, LogicConditionParser
from kai_exceptions import ConstraintReasoningError, ParsingError, SpaCyModelError

logger = get_logger(__name__)


class PuzzleType(Enum):
    """Puzzle Type für Solver-Routing."""

    ENTITY_SAT = "entity_sat"  # Entitaeten-basierte Raetsel (Leo/Mark/Nick)
    NUMERICAL_CSP = "numerical_csp"  # Zahlen-basierte Constraint-Raetsel
    ARITHMETIC = "arithmetic"  # Arithmetik-Raetsel (Alter, Summen, Gleichungen)
    CIRCULAR_SEATING = "circular_seating"  # Runder Tisch / Kreisanordnung
    LINEAR_ORDERING = "linear_ordering"  # Lineare Reihenfolge-Raetsel
    HYBRID = "hybrid"  # Kombination aus beiden
    UNKNOWN = "unknown"  # Unbestimmt


class LogicPuzzleSolver:
    """
    Solves logic puzzles using SAT solver.

    Workflow:
    1. Parse conditions -> LogicCondition list
    2. Convert to CNF -> CNFFormula
    3. Solve with SAT solver -> Model
    4. Format answer with proof tree
    """

    def __init__(self):
        self.parser = LogicConditionParser()
        self.solver = SATSolver()
        self.solver.enable_proof = True  # Enable ProofTree generation

        # Lazy import für numerical solver (vermeidet zirkuläre Imports)
        self._numerical_solver = None

    def _get_numerical_solver(self):
        """Lazy loading des numerical puzzle solvers."""
        if self._numerical_solver is None:
            from component_45_numerical_puzzle_solver import NumericalPuzzleSolver

            self._numerical_solver = NumericalPuzzleSolver()
        return self._numerical_solver

    def detect_puzzle_type(self, text: str, entities: List[str]) -> PuzzleType:
        """
        Klassifiziert Puzzle-Typ fuer Solver-Routing.

        Args:
            text: Der Puzzle-Text
            entities: Liste erkannter Entitaeten

        Returns:
            PuzzleType enum
        """
        import re

        text_lower = text.lower()

        # =================================================================
        # PRIORITY 1: Check for ARITHMETIC puzzles (age puzzles, equations)
        # =================================================================
        arithmetic_patterns = [
            r"\bjahre?\s+(?:alt|aelter|juenger)\b",  # "Jahre alt", "aelter"
            r"\b(?:aelter|juenger)\s+als\b",  # "aelter als", "juenger als"
            r"\bdoppelt\s+so\s+(?:alt|gross|schwer)\b",  # "doppelt so alt"
            r"\bhalb\s+so\s+(?:alt|gross)\b",  # "halb so alt"
            r"\bsumme\s+(?:aller\s+)?(?:alter|jahre)\b",  # "Summe aller Alter"
            r"\bdie\s+summe\s+.{0,30}\s+ist\s+\d+",  # "Die Summe ... ist 55"
            r"\b\d+\s+jahre?\s+(?:aelter|juenger)\b",  # "5 Jahre aelter"
            r"\bwie\s+alt\b",  # "wie alt"
            r"\balter\s+(?:von|ist)\b",  # "Alter von", "Alter ist"
        ]

        arithmetic_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE)) for p in arithmetic_patterns
        )

        if arithmetic_count >= 2:
            logger.info(
                f"Puzzle-Typ: ARITHMETIC ({arithmetic_count} arithmetic patterns)"
            )
            return PuzzleType.ARITHMETIC

        # =================================================================
        # PRIORITY 2: Check for CIRCULAR_SEATING puzzles (round table)
        # =================================================================
        circular_seating_patterns = [
            r"\brunde[rn]?\s+tisch\b",  # "runder Tisch"
            r"\bim\s+kreis\b",  # "im Kreis"
            r"\bgegenueber\s+(?:von)?\b",  # "gegenueber von"
            r"\bsitzen?\s+.{0,20}gegenueber\b",  # "sitzt gegenueber"
            r"\bsitzordnung\b",  # "Sitzordnung"
            r"\buhrzeigersinn\b",  # "Uhrzeigersinn"
            r"\bkreisfoermig\b",  # "kreisfoermig"
        ]

        circular_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE))
            for p in circular_seating_patterns
        )

        if circular_count >= 1:
            logger.info(
                f"Puzzle-Typ: CIRCULAR_SEATING ({circular_count} circular patterns)"
            )
            return PuzzleType.CIRCULAR_SEATING

        # =================================================================
        # PRIORITY 3: Check for LINEAR_ORDERING puzzles (line arrangement)
        # =================================================================
        linear_ordering_patterns = [
            r"\bsteh(?:t|en)\s+(?:links|rechts)\s+von\b",  # "steht links von"
            r"\bsteh(?:t|en)\s+.{0,10}in\s+(?:einer\s+)?reihe\b",  # "stehen in einer Reihe"
            r"\breihenfolge\b",  # "Reihenfolge"
            r"\blinks\s+von\b",  # "links von"
            r"\brechts\s+von\b",  # "rechts von"
            r"\bnicht\s+am\s+ende\b",  # "nicht am Ende"
            r"\bnebeneinander\b",  # "nebeneinander"
            r"\bdirekt\s+neben\b",  # "direkt neben"
        ]

        linear_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE))
            for p in linear_ordering_patterns
        )

        # Strong indicator: multiple linear patterns
        if linear_count >= 2:
            logger.info(f"Puzzle-Typ: LINEAR_ORDERING ({linear_count} linear patterns)")
            return PuzzleType.LINEAR_ORDERING

        # =================================================================
        # PRIORITY 4: Original detection logic for other puzzle types
        # =================================================================

        # Patterns fuer numerische Puzzles (ohne generisches numbered list pattern)
        numerical_patterns = [
            r"\bzahl\b",
            r"\bteilbar\s+durch\b",
            r"\bsumme\s+der\b",
            r"\bdifferenz\b",
            r"\banzahl\s+der\b",
            r"\bteiler\b",
        ]

        # Meta-constraint patterns (numbered statements mit meta-keywords)
        # Nur wenn "richtig", "falsch", "wahr", "behauptung" folgen
        meta_constraint_patterns = [
            r"\b\d+\.\s+.*(richtig|falsch|wahr|behauptung)",  # "1. ist richtig"
            r"(erste|letzte|n-te)\s+(richtig|falsch)",  # "erste richtige"
            r"anzahl\s+der\s+(richtigen|falschen)",  # "Anzahl der richtigen"
        ]

        # Patterns fuer Entitaeten-basierte Puzzles (erweitert)
        entity_patterns = [
            r"\b[A-Z][a-z]+\s+(?:trinkt|mag|isst|bestellt|ist|hat)\b",
            r"\b(?:wer|was|welche[rs]?)\s+(?:trinkt|mag|isst|hat)\b",
            r"\b(?:ist\s+kein|ist\s+nicht|hat\s+kein)\b",  # Negative constraints
        ]

        # Zaehle Matches mit Logging
        numerical_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE)) for p in numerical_patterns
        )
        meta_count = sum(
            len(re.findall(p, text_lower, re.IGNORECASE))
            for p in meta_constraint_patterns
        )
        entity_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in entity_patterns
        )

        # Beruecksichtige erkannte Entitaeten (nur wenn mindestens 2 vorhanden)
        entity_bonus = 2 if len(entities) >= 2 else 0
        total_entity_signals = entity_count + entity_bonus

        total_numerical = numerical_count + meta_count

        logger.debug(
            f"Puzzle type detection: numerical={numerical_count}, "
            f"meta={meta_count}, entity_patterns={entity_count}, "
            f"entities={len(entities)}, total_entity_signals={total_entity_signals}, "
            f"arithmetic={arithmetic_count}, circular={circular_count}, linear={linear_count}"
        )

        # ENTITY_SAT: Entities + entity patterns + keine/wenig numerical
        # Prioritized classification: Entities are strong indicators
        if len(entities) >= 2 and entity_count >= 2 and total_numerical == 0:
            logger.info(
                f"Puzzle-Typ: ENTITY_SAT ({len(entities)} entities, "
                f"{entity_count} entity patterns)"
            )
            return PuzzleType.ENTITY_SAT

        # NUMERICAL_CSP: Viele numerical/meta patterns + keine entities
        if total_numerical >= 2 and total_entity_signals == 0:
            logger.info(
                f"Puzzle-Typ: NUMERICAL_CSP ({total_numerical} numerical patterns)"
            )
            return PuzzleType.NUMERICAL_CSP

        # HYBRID: Beide vorhanden
        if total_numerical >= 1 and total_entity_signals >= 1:
            logger.info(
                f"Puzzle-Typ: HYBRID ({total_numerical} numerical, "
                f"{total_entity_signals} entity)"
            )
            return PuzzleType.HYBRID

        # DEFAULT: ENTITY_SAT wenn Entitaeten vorhanden (auch ohne entity patterns)
        if len(entities) >= 2:
            logger.info(
                f"Puzzle-Typ: ENTITY_SAT (default - {len(entities)} entities detected)"
            )
            return PuzzleType.ENTITY_SAT

        logger.debug("Puzzle-Typ: UNKNOWN")
        return PuzzleType.UNKNOWN

    def solve(
        self, conditions_text: str, entities: List[str], question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solves a logic puzzle (with routing to appropriate solver).

        Args:
            conditions_text: Text with logical conditions
            entities: List of entities (e.g., ["Leo", "Mark", "Nick"])
            question: Optional question (e.g., "Wer trinkt Brandy?")

        Returns:
            Dictionary with:
            - solution: Dict[var_name, bool/int] - Variable assignments
            - proof_tree: ProofTree - Solution path
            - answer: str - Formatted answer
            - result: str - "SATISFIABLE" | "UNSATISFIABLE"

        Raises:
            ParsingError: If conditions cannot be parsed
            ConstraintReasoningError: If solver fails
        """
        try:
            # STEP 0: Detect puzzle type and route to appropriate solver
            puzzle_type = self.detect_puzzle_type(conditions_text, entities)

            # ================================================================
            # NEW: Route ARITHMETIC puzzles to specialized solver
            # ================================================================
            if puzzle_type == PuzzleType.ARITHMETIC:
                logger.info("Routing to ARITHMETIC solver")
                return self._solve_arithmetic_puzzle(
                    conditions_text, entities, question
                )

            # ================================================================
            # NEW: Route CIRCULAR_SEATING puzzles to specialized solver
            # ================================================================
            if puzzle_type == PuzzleType.CIRCULAR_SEATING:
                logger.info("Routing to CIRCULAR_SEATING solver")
                return self._solve_circular_seating(conditions_text, entities, question)

            # ================================================================
            # NEW: Route LINEAR_ORDERING puzzles to specialized solver
            # ================================================================
            if puzzle_type == PuzzleType.LINEAR_ORDERING:
                logger.info("Routing to LINEAR_ORDERING solver")
                return self._solve_linear_ordering(conditions_text, entities, question)

            # Validate classification against puzzle structure
            if puzzle_type == PuzzleType.NUMERICAL_CSP:
                if len(entities) >= 2:
                    logger.warning(
                        f"Puzzle classified as NUMERICAL_CSP but has {len(entities)} entities. "
                        f"This might be an ENTITY_SAT puzzle with numbered constraints. "
                        f"Entities: {entities}"
                    )
                    # Override classification if entities present
                    if len(entities) >= 3:
                        logger.info("Overriding to ENTITY_SAT due to multiple entities")
                        puzzle_type = PuzzleType.ENTITY_SAT

                if puzzle_type == PuzzleType.NUMERICAL_CSP:
                    logger.info("Routing to NUMERICAL CSP solver")
                    return self._solve_numerical_csp(conditions_text, question)

            elif puzzle_type == PuzzleType.ENTITY_SAT:
                if len(entities) < 2:
                    raise ParsingError(
                        f"Puzzle classified as ENTITY_SAT but only {len(entities)} entities found",
                        context={
                            "entities": entities,
                            "text_length": len(conditions_text),
                        },
                    )
                logger.info("Routing to ENTITY SAT solver")
                return self._solve_entity_sat(conditions_text, entities, question)

            elif puzzle_type == PuzzleType.HYBRID:
                logger.warning(
                    "HYBRID puzzles not yet fully supported - trying ENTITY SAT"
                )
                return self._solve_entity_sat(conditions_text, entities, question)
            else:
                logger.warning("UNKNOWN puzzle type - trying ENTITY SAT as fallback")
                return self._solve_entity_sat(conditions_text, entities, question)

        except (ParsingError, SpaCyModelError, ConstraintReasoningError):
            raise  # Re-raise known exceptions
        except Exception as e:
            logger.error(
                f"Unexpected error in LogicPuzzleSolver.solve(): {e}", exc_info=True
            )
            raise ConstraintReasoningError(
                "Unexpected error solving logic puzzle",
                context={
                    "num_entities": len(entities),
                    "text_length": len(conditions_text),
                },
                original_exception=e,
            )

    def _solve_numerical_csp(
        self, conditions_text: str, question: Optional[str]
    ) -> Dict[str, Any]:
        """Routes to numerical CSP solver."""
        numerical_solver = self._get_numerical_solver()
        return numerical_solver.solve_puzzle(conditions_text, question)

    def _solve_entity_sat(
        self, conditions_text: str, entities: List[str], question: Optional[str]
    ) -> Dict[str, Any]:
        """
        Solves entity-based logic puzzle using SAT solver.

        This is the original solve() method logic.
        """
        try:
            logger.info(f"Solving entity SAT puzzle with {len(entities)} entities")

            # STEP 1: Parse conditions
            try:
                conditions = self.parser.parse_conditions(conditions_text, entities)
                logger.info(f"Parsed: {len(conditions)} conditions")
            except (ParsingError, SpaCyModelError) as e:
                # Re-raise with additional context
                logger.error(f"Parsing failed: {e}")
                raise

            if not conditions:
                logger.warning("No logical conditions found")
                return {
                    "result": "UNSATISFIABLE",
                    "solution": {},
                    "proof_tree": None,
                    "answer": "No logical conditions found.",
                    "puzzle_type": "entity_sat",
                }

            # STEP 2: Convert to CNF
            try:
                cnf = self._build_cnf(conditions)
                logger.info(f"CNF created: {len(cnf.clauses)} clauses")
            except ConstraintReasoningError:
                raise  # Re-raise ConstraintReasoningError from _build_cnf
            except Exception as e:
                raise ConstraintReasoningError(
                    "Error converting to CNF formula",
                    context={"num_conditions": len(conditions)},
                    original_exception=e,
                )

            # STEP 3: Solve with SAT solver
            try:
                # SATSolver.solve() returns None if UNSAT, else model
                model = self.solver.solve(cnf)
            except Exception as e:
                raise ConstraintReasoningError(
                    "SAT solver failed",
                    context={"num_clauses": len(cnf.clauses)},
                    original_exception=e,
                )

            if model is not None:
                logger.info(f"[OK] Solution found: {model}")

                # STEP 4: Format answer
                try:
                    answer = self._format_answer(model, question)
                except ConstraintReasoningError:
                    raise  # Re-raise ConstraintReasoningError from _format_answer
                except Exception as e:
                    raise ConstraintReasoningError(
                        "Error formatting answer",
                        context={"model_size": len(model)},
                        original_exception=e,
                    )

                # STEP 5: Extract ProofTree
                proof_tree = self.solver.get_proof_tree(
                    query=question or "Logic Puzzle Solution"
                )

                return {
                    "solution": model,
                    "proof_tree": proof_tree,
                    "answer": answer,
                    "result": "SATISFIABLE",
                    "puzzle_type": "entity_sat",
                }
            else:
                # UNSAT - puzzle has no solution
                logger.error(
                    f"Puzzle is UNSATISFIABLE: {len(cnf.clauses)} clauses with "
                    f"{len(self.parser.variables)} variables. This indicates either "
                    f"(1) contradictory input, or (2) incorrect parsing/constraints."
                )
                # Log constraint details for debugging
                for i, cond in enumerate(conditions[:10]):  # First 10
                    logger.debug(f"Condition {i}: {cond.condition_type} - {cond.text}")

                # IMPROVED: Return structured result instead of raising exception
                # This allows better user feedback and test debugging
                diagnostic_info = {
                    "num_clauses": len(cnf.clauses),
                    "num_variables": len(self.parser.variables),
                    "entities": entities,
                    "detected_objects": list(self.parser._detected_objects),
                    "uniqueness_constraints_count": sum(
                        1 for c in conditions if "uniqueness" in c.text.lower()
                    ),
                    "conditions_summary": [
                        {"type": c.condition_type, "text": c.text[:80]}
                        for c in conditions[:5]  # First 5 conditions
                    ],
                }

                # Construct helpful error message
                answer = (
                    "Das Rätsel hat keine Lösung. Die gegebenen Bedingungen widersprechen sich. "
                    f"({len(cnf.clauses)} Constraints, {len(self.parser.variables)} Variablen)"
                )

                return {
                    "result": "UNSATISFIABLE",
                    "answer": answer,
                    "diagnostic": diagnostic_info,
                    "proof_tree": None,
                    "puzzle_type": "entity_sat",
                }

        except (ParsingError, SpaCyModelError, ConstraintReasoningError):
            raise  # Re-raise known exceptions
        except Exception as e:
            logger.error(
                f"Unexpected error in LogicPuzzleSolver.solve(): {e}", exc_info=True
            )
            raise ConstraintReasoningError(
                "Unexpected error solving logic puzzle",
                context={
                    "num_entities": len(entities),
                    "text_length": len(conditions_text),
                },
                original_exception=e,
            )

    def _build_cnf(self, conditions: List[LogicCondition]) -> CNFFormula:
        """
        Converts LogicCondition list to CNF formula.

        Logic transformations:
        - SIMPLE_FACT (X): X (positive literal)
        - IMPLICATION (X -> Y):  NOT X OR Y
        - XOR (X XOR Y):         (X OR Y) AND (NOT X OR NOT Y)
        - NEVER_BOTH NOT(X AND Y): NOT X OR NOT Y
        - CONJUNCTION (X AND Y): X, Y (separate clauses)
        - DISJUNCTION (X OR Y OR Z...):  X OR Y OR Z... (multi-operand)
        - NEGATION (NOT X):      NOT X
        - ORDERING: Position-based constraints (see _convert_ordering_to_cnf)

        Raises:
            ConstraintReasoningError: If CNF conversion fails
        """
        try:
            clauses: List[Clause] = []

            # Detect if this is an ordering puzzle
            ordering_conditions = [
                c for c in conditions if c.condition_type == "ORDERING"
            ]

            if ordering_conditions:
                # This is an ordering puzzle - extract entities from operands
                entities = set()
                for cond in ordering_conditions:
                    entities.update(cond.operands)
                entities = sorted(list(entities))  # Deterministic ordering

                logger.info(
                    f"Detected ordering puzzle with {len(entities)} entities: {entities}"
                )

                # Add structural position constraints
                position_clauses = self._add_position_constraints(entities)
                clauses.extend(position_clauses)
                logger.debug(
                    f"Added {len(position_clauses)} position constraint clauses"
                )

            for cond in conditions:
                if cond.condition_type == "SIMPLE_FACT":
                    # X (positive literal - fact is true)
                    x = cond.operands[0]
                    clauses.append(Clause({Literal(x, negated=False)}))

                elif cond.condition_type == "IMPLICATION":
                    # X -> Y = NOT X OR Y
                    x, y = cond.operands
                    clauses.append(
                        Clause({Literal(x, negated=True), Literal(y, negated=False)})
                    )

                elif cond.condition_type == "XOR":
                    # X XOR Y = (X OR Y) AND (NOT X OR NOT Y)
                    x, y = cond.operands
                    clauses.append(Clause({Literal(x), Literal(y)}))  # X OR Y
                    clauses.append(
                        Clause({Literal(x, negated=True), Literal(y, negated=True)})
                    )  # NOT X OR NOT Y

                elif cond.condition_type == "NEVER_BOTH":
                    # NOT(X AND Y) = NOT X OR NOT Y
                    x, y = cond.operands
                    clauses.append(
                        Clause({Literal(x, negated=True), Literal(y, negated=True)})
                    )

                elif cond.condition_type == "CONJUNCTION":
                    # X AND Y = X, Y (two separate clauses)
                    x, y = cond.operands
                    clauses.append(Clause({Literal(x)}))
                    clauses.append(Clause({Literal(y)}))

                elif cond.condition_type == "DISJUNCTION":
                    # X OR Y OR Z OR ... (multi-operand support)
                    # Create single clause with all literals
                    literals = {Literal(var) for var in cond.operands}
                    clauses.append(Clause(literals))

                elif cond.condition_type == "NEGATION":
                    # NOT X
                    x = cond.operands[0]
                    clauses.append(Clause({Literal(x, negated=True)}))

                elif cond.condition_type == "ORDERING":
                    # Convert ordering constraint to position-based CNF
                    # Re-extract entities (already done above)
                    entities = set()
                    for c in ordering_conditions:
                        entities.update(c.operands)
                    entities = sorted(list(entities))

                    ordering_clauses = self._convert_ordering_to_cnf(cond, entities)
                    clauses.extend(ordering_clauses)
                    logger.debug(
                        f"Added {len(ordering_clauses)} clauses for ordering: {cond.text}"
                    )

            return CNFFormula(clauses)
        except Exception as e:
            raise ConstraintReasoningError(
                "Error converting to CNF formula",
                context={"num_conditions": len(conditions)},
                original_exception=e,
            )

    def _add_position_constraints(self, entities: List[str]) -> List[Clause]:
        """
        Add structural constraints for position assignment in ordering puzzles.

        Constraints:
        1. Each entity has exactly one position (at-least-one + at-most-one)
        2. Each position has exactly one entity

        Args:
            entities: List of entity names (e.g., ["A", "B", "C", "D", "E"])

        Returns:
            List of CNF clauses encoding position constraints
        """
        clauses = []
        n = len(entities)

        # 1. Each entity has exactly one position
        for entity in entities:
            # At-least-one: entity must be at SOME position
            # pos_A_1 OR pos_A_2 OR ... OR pos_A_n
            clauses.append(
                Clause(
                    {
                        Literal(f"pos_{entity}_{i}", negated=False)
                        for i in range(1, n + 1)
                    }
                )
            )

            # At-most-one: entity cannot be at two positions
            # For all i < j: NOT pos_A_i OR NOT pos_A_j
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    clauses.append(
                        Clause(
                            {
                                Literal(f"pos_{entity}_{i}", negated=True),
                                Literal(f"pos_{entity}_{j}", negated=True),
                            }
                        )
                    )

        # 2. Each position has exactly one entity
        for pos in range(1, n + 1):
            # At-least-one: some entity at position pos
            # pos_A_1 OR pos_B_1 OR pos_C_1 OR ...
            clauses.append(
                Clause(
                    {
                        Literal(f"pos_{entity}_{pos}", negated=False)
                        for entity in entities
                    }
                )
            )

            # At-most-one: not two entities at same position
            # For all entities i < j: NOT pos_i_pos OR NOT pos_j_pos
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    clauses.append(
                        Clause(
                            {
                                Literal(f"pos_{entities[i]}_{pos}", negated=True),
                                Literal(f"pos_{entities[j]}_{pos}", negated=True),
                            }
                        )
                    )

        return clauses

    def _convert_ordering_to_cnf(
        self, condition: LogicCondition, entities: List[str]
    ) -> List[Clause]:
        """
        Convert ordering constraint to CNF clauses.

        Strategy:
        - Create position variables: pos_ENTITY_1, pos_ENTITY_2, ..., pos_ENTITY_N
        - Each ordering constraint adds clauses restricting valid position assignments

        Args:
            condition: ORDERING type LogicCondition with metadata["relation"]
            entities: List of all entities in puzzle

        Returns:
            List of CNF clauses encoding the ordering constraint

        Raises:
            ConstraintReasoningError: If relation type is unsupported
        """
        clauses = []
        n = len(entities)

        if not condition.metadata or "relation" not in condition.metadata:
            raise ConstraintReasoningError(
                "ORDERING condition missing metadata['relation']",
                context={"condition": condition.text},
            )

        relation = condition.metadata["relation"]

        if relation == "left_of":
            # A left_of B means pos(A) < pos(B)
            # For all positions i, j: pos_A_i AND pos_B_j -> i < j
            # Encoded as: NOT pos_A_i OR NOT pos_B_j (for i >= j)
            entity1, entity2 = condition.operands
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i >= j:  # Violates A < B
                        clauses.append(
                            Clause(
                                {
                                    Literal(f"pos_{entity1}_{i}", negated=True),
                                    Literal(f"pos_{entity2}_{j}", negated=True),
                                }
                            )
                        )

        elif relation == "right_of":
            # A right_of B means pos(A) > pos(B)
            # For all positions i, j: pos_A_i AND pos_B_j -> i > j
            # Encoded as: NOT pos_A_i OR NOT pos_B_j (for i <= j)
            entity1, entity2 = condition.operands
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i <= j:  # Violates A > B
                        clauses.append(
                            Clause(
                                {
                                    Literal(f"pos_{entity1}_{i}", negated=True),
                                    Literal(f"pos_{entity2}_{j}", negated=True),
                                }
                            )
                        )

        elif relation == "not_adjacent":
            # A not_adjacent B means |pos(A) - pos(B)| != 1
            # For each position i: if A at i, B cannot be at i-1 or i+1
            entity1, entity2 = condition.operands
            for i in range(1, n + 1):
                # If A at position i, B cannot be at i-1
                if i > 1:
                    clauses.append(
                        Clause(
                            {
                                Literal(f"pos_{entity1}_{i}", negated=True),
                                Literal(f"pos_{entity2}_{i-1}", negated=True),
                            }
                        )
                    )
                # If A at position i, B cannot be at i+1
                if i < n:
                    clauses.append(
                        Clause(
                            {
                                Literal(f"pos_{entity1}_{i}", negated=True),
                                Literal(f"pos_{entity2}_{i+1}", negated=True),
                            }
                        )
                    )

        elif relation == "not_at_end":
            # E not_at_end means pos(E) != 1 AND pos(E) != N
            entity = condition.operands[0]
            # Not at position 1
            clauses.append(Clause({Literal(f"pos_{entity}_1", negated=True)}))
            # Not at position N
            clauses.append(Clause({Literal(f"pos_{entity}_{n}", negated=True)}))

        elif relation == "adjacent":
            # A adjacent B means |pos(A) - pos(B)| == 1
            # (pos_A_1 AND pos_B_2) OR (pos_A_2 AND pos_B_1) OR ... OR (pos_A_2 AND pos_B_3) ...
            # This is complex - use disjunction of all valid adjacent pairs
            entity1, entity2 = condition.operands
            adjacent_pairs = []
            for i in range(1, n + 1):
                # A at i, B at i-1
                if i > 1:
                    adjacent_pairs.append(
                        (f"pos_{entity1}_{i}", f"pos_{entity2}_{i-1}")
                    )
                # A at i, B at i+1
                if i < n:
                    adjacent_pairs.append(
                        (f"pos_{entity1}_{i}", f"pos_{entity2}_{i+1}")
                    )

            # Convert to CNF: at least one adjacent pair must be true
            # This is a disjunction of conjunctions - not in CNF!
            # Use Tseitin transformation or approximation
            # For simplicity, enforce: if A at i, then B at i-1 OR i+1
            for i in range(1, n + 1):
                possible_b_positions = []
                if i > 1:
                    possible_b_positions.append(
                        Literal(f"pos_{entity2}_{i-1}", negated=False)
                    )
                if i < n:
                    possible_b_positions.append(
                        Literal(f"pos_{entity2}_{i+1}", negated=False)
                    )

                if possible_b_positions:
                    # If A at i, then B must be at one of the adjacent positions
                    # NOT pos_A_i OR pos_B_(i-1) OR pos_B_(i+1)
                    clauses.append(
                        Clause(
                            {Literal(f"pos_{entity1}_{i}", negated=True)}
                            | set(possible_b_positions)
                        )
                    )

            # Symmetric: if B at j, then A at j-1 OR j+1
            for j in range(1, n + 1):
                possible_a_positions = []
                if j > 1:
                    possible_a_positions.append(
                        Literal(f"pos_{entity1}_{j-1}", negated=False)
                    )
                if j < n:
                    possible_a_positions.append(
                        Literal(f"pos_{entity1}_{j+1}", negated=False)
                    )

                if possible_a_positions:
                    clauses.append(
                        Clause(
                            {Literal(f"pos_{entity2}_{j}", negated=True)}
                            | set(possible_a_positions)
                        )
                    )

        else:
            raise ConstraintReasoningError(
                f"Unsupported ordering relation: {relation}",
                context={"condition": condition.text, "relation": relation},
            )

        return clauses

    def _extract_verb_from_question(self, question: Optional[str]) -> Optional[str]:
        """
        Extracts the main verb phrase from a question.

        Examples:
            "Wer trinkt gerne Brandy?" -> "trinkt gerne"
            "Wer von den dreien bestellt einen Kaffee?" -> "bestellt"
            "Wer mag Schokolade?" -> "mag"

        Args:
            question: The question text

        Returns:
            Verb phrase or None if not found
        """
        if not question:
            return None

        try:
            import spacy

            nlp = spacy.load("de_core_news_sm")
            doc = nlp(question)

            # Find main verb (VERB or AUX)
            verbs = []
            for token in doc:
                if token.pos_ in ("VERB", "AUX") and token.dep_ in ("ROOT", "aux"):
                    verbs.append(token)

            if not verbs:
                return None

            # Get main verb and check for adverbs like "gerne"
            main_verb = verbs[0]
            verb_phrase = [main_verb.text]

            # Check for adverbs modifying the verb (like "gerne")
            for child in main_verb.children:
                if child.pos_ == "ADV" and child.dep_ == "mo":
                    verb_phrase.append(child.text)

            return " ".join(verb_phrase) if verb_phrase else None

        except Exception as e:
            logger.warning(f"Could not extract verb from question: {e}")
            return None

    def _format_answer(self, model: Dict[str, bool], question: Optional[str]) -> str:
        """
        Formats the solution as a natural language answer.

        Args:
            model: Variable assignments (var_name -> bool)
            question: Optional question (for contextual answer)

        Returns:
            Formatted answer

        Raises:
            ConstraintReasoningError: If answer formatting fails
        """
        try:
            # Find all TRUE variables
            true_vars = [var for var, value in model.items() if value]

            if not true_vars:
                return "None of the conditions are satisfied."

            # Detect ordering puzzle: Variables follow pattern "pos_ENTITY_N"
            position_vars = [v for v in true_vars if v.startswith("pos_")]

            if position_vars:
                # This is an ordering puzzle - extract and format positions
                ordering = self._extract_ordering_from_positions(position_vars)
                if ordering:
                    # Check if question asks for ordering/sequence
                    if question and any(
                        keyword in question.lower()
                        for keyword in ["reihenfolge", "welcher", "ordnung", "sequenz"]
                    ):
                        return f"Die Reihenfolge ist: {', '.join(ordering)}"
                    else:
                        # Default formatting for ordering
                        return ", ".join(ordering)

            # Assignment puzzle - existing logic
            # Extract verb from question if available
            question_verb = self._extract_verb_from_question(question)

            # Format variables as sentences
            statements = []
            for var_name in true_vars:
                var = self.parser.get_variable(var_name)
                if var:
                    # Safety: Skip malformed variables with excessively long property names
                    # Real properties are short (e.g., "hat_rot", "trinkt_tee")
                    # Malformed variables from parsing artifacts can have 40+ char properties
                    if len(var.property) > 25:
                        logger.debug(f"Skipping malformed variable: {var_name}")
                        continue

                    # Use question verb if available and property contains underscore
                    if question_verb and "_" in var.property:
                        # Extract object from property (e.g., "hat_brandy" -> "brandy")
                        property_parts = var.property.split("_", 1)
                        if len(property_parts) > 1:
                            obj = property_parts[1].replace("_", " ")
                            statements.append(f"{var.entity} {question_verb} {obj}")
                        else:
                            # Fallback to default formatting
                            property_text = var.property.replace("_", " ")
                            statements.append(f"{var.entity} {property_text}")
                    else:
                        # Default: Convert property back to natural language
                        property_text = var.property.replace("_", " ")
                        statements.append(f"{var.entity} {property_text}")

            if len(statements) == 1:
                return statements[0].capitalize()
            else:
                return ", ".join(statements[:-1]) + " und " + statements[-1]
        except Exception as e:
            raise ConstraintReasoningError(
                "Error formatting answer",
                context={
                    "num_true_vars": len(true_vars) if "true_vars" in locals() else 0
                },
                original_exception=e,
            )

    def _extract_ordering_from_positions(self, position_vars: List[str]) -> List[str]:
        """
        Extract ordering from position variables.

        Args:
            position_vars: List of TRUE position variables (e.g., ["pos_A_1", "pos_B_2", ...])

        Returns:
            Ordered list of entities (e.g., ["A", "B", "E", "C", "D"])

        Example:
            Input: ["pos_A_1", "pos_B_2", "pos_E_3", "pos_C_4", "pos_D_5"]
            Output: ["A", "B", "E", "C", "D"]
        """
        import re

        positions = {}
        for var in position_vars:
            # Parse "pos_ENTITY_N"
            match = re.match(r"pos_([A-Z]+)_(\d+)", var)
            if match:
                entity = match.group(1)
                pos = int(match.group(2))
                positions[pos] = entity

        # Sort by position
        sorted_positions = sorted(positions.keys())
        ordering = [positions[pos] for pos in sorted_positions]

        logger.info(f"Extracted ordering: {ordering}")
        return ordering

    def _is_assignment_answer(self, true_vars: List[str]) -> bool:
        """
        Detect if answer represents an assignment puzzle.

        Heuristic: >50% of variables match pattern "entity_hat_object"

        Args:
            true_vars: List of TRUE variable names

        Returns:
            True if assignment puzzle detected
        """
        if not true_vars:
            return False

        assignment_pattern_count = sum(1 for v in true_vars if "_hat_" in v)
        return assignment_pattern_count / len(true_vars) > 0.5

    # =========================================================================
    # ARITHMETIC PUZZLE SOLVER
    # =========================================================================

    def _solve_arithmetic_puzzle(
        self, conditions_text: str, entities: List[str], question: Optional[str]
    ) -> Dict[str, Any]:
        """
        Solves arithmetic puzzles (age problems, linear equations).

        Strategy:
        1. Parse text for arithmetic constraints
        2. Build system of linear equations
        3. Solve using Gaussian elimination / substitution
        4. Generate ProofTree

        Supported patterns:
        - "X ist N Jahre aelter als Y" -> X = Y + N
        - "X ist doppelt so alt wie Y" -> X = 2 * Y
        - "Die Summe aller Alter ist N" -> X + Y + Z = N
        - "X ist juenger als N Jahre" -> X < N

        Args:
            conditions_text: Text with arithmetic constraints
            entities: List of entity names (e.g., ["Anna", "Ben", "Clara"])
            question: Optional question

        Returns:
            Solution dict with ages and ProofTree
        """
        import re

        from component_17_proof_explanation import ProofStep, ProofTree
        from component_17_proof_explanation import StepType as ProofStepType

        logger.info(f"Solving ARITHMETIC puzzle with {len(entities)} entities")

        # Initialize proof tree steps
        proof_steps = []

        # Create proof step for puzzle detection
        proof_steps.append(
            ProofStep(
                step_id="arithmetic_detection",
                step_type=ProofStepType.PREMISE,
                output="Arithmetic/Age puzzle detected",
                explanation_text=f"Puzzle with {len(entities)} persons and age constraints",
                confidence=0.95,
                metadata={"entities": entities},
                source_component="component_45_logic_puzzle_solver_core",
            )
        )

        # Parse entities from text if not provided
        if not entities:
            # Extract names from text
            name_pattern = r"\b([A-Z][a-z]+)\b"
            potential_names = set(re.findall(name_pattern, conditions_text))
            # Filter out common German words
            stop_words = {
                "Die",
                "Der",
                "Das",
                "Eine",
                "Einer",
                "Eines",
                "Sind",
                "Ist",
                "Hat",
                "Haben",
                "Wie",
                "Alt",
                "Jahre",
                "Alter",
                "Summe",
                "Hinweise",
                "Frage",
                "Personen",
                "Drei",
                "Vier",
                "Verschiedene",
                "Alle",
                "Aller",
                "Doppelt",
                "Halb",
            }
            entities = [n for n in potential_names if n not in stop_words]
            logger.info(f"Extracted entities from text: {entities}")

        # Normalize entity names
        entity_names = [e.lower() for e in entities]

        # =====================================================================
        # STEP 1: Parse constraints
        # =====================================================================
        constraints = []
        text_lower = conditions_text.lower()

        # Pattern 1: "X ist N Jahre aelter als Y"
        older_pattern = r"(\w+)\s+ist\s+(\d+)\s+jahre?\s+aelter\s+als\s+(\w+)"
        for match in re.finditer(older_pattern, text_lower):
            person1 = match.group(1)
            diff = int(match.group(2))
            person2 = match.group(3)
            if person1 in entity_names and person2 in entity_names:
                constraints.append(
                    {
                        "type": "older_by",
                        "person1": person1,
                        "person2": person2,
                        "diff": diff,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {person1} = {person2} + {diff}")

        # Pattern 2: "X ist N Jahre juenger als Y"
        younger_pattern = r"(\w+)\s+ist\s+(\d+)\s+jahre?\s+juenger\s+als\s+(\w+)"
        for match in re.finditer(younger_pattern, text_lower):
            person1 = match.group(1)
            diff = int(match.group(2))
            person2 = match.group(3)
            if person1 in entity_names and person2 in entity_names:
                constraints.append(
                    {
                        "type": "younger_by",
                        "person1": person1,
                        "person2": person2,
                        "diff": diff,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {person1} = {person2} - {diff}")

        # Pattern 3: "X ist doppelt so alt wie Y"
        double_pattern = r"(\w+)\s+ist\s+doppelt\s+so\s+alt\s+wie\s+(\w+)"
        for match in re.finditer(double_pattern, text_lower):
            person1 = match.group(1)
            person2 = match.group(2)
            if person1 in entity_names and person2 in entity_names:
                constraints.append(
                    {
                        "type": "double",
                        "person1": person1,
                        "person2": person2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {person1} = 2 * {person2}")

        # Pattern 4: "X ist halb so alt wie Y"
        half_pattern = r"(\w+)\s+ist\s+halb\s+so\s+alt\s+wie\s+(\w+)"
        for match in re.finditer(half_pattern, text_lower):
            person1 = match.group(1)
            person2 = match.group(2)
            if person1 in entity_names and person2 in entity_names:
                constraints.append(
                    {
                        "type": "half",
                        "person1": person1,
                        "person2": person2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {person1} = {person2} / 2")

        # Pattern 5: "Die Summe aller Alter ist N"
        sum_pattern = r"(?:die\s+)?summe\s+(?:aller\s+)?(?:alter|jahre)\s+(?:ist|betraegt)\s+(\d+)"
        for match in re.finditer(sum_pattern, text_lower):
            total = int(match.group(1))
            constraints.append(
                {
                    "type": "sum",
                    "total": total,
                    "text": match.group(0),
                }
            )
            logger.debug(f"Constraint: SUM = {total}")

        # Pattern 6: "X ist juenger als N Jahre"
        less_than_pattern = r"(\w+)\s+ist\s+(?:juenger\s+als|unter)\s+(\d+)\s+jahre?"
        for match in re.finditer(less_than_pattern, text_lower):
            person = match.group(1)
            threshold = int(match.group(2))
            if person in entity_names:
                constraints.append(
                    {
                        "type": "less_than",
                        "person": person,
                        "threshold": threshold,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {person} < {threshold}")

        # Pattern 7: "X ist aelter als N Jahre"
        greater_than_pattern = r"(\w+)\s+ist\s+(?:aelter\s+als|ueber)\s+(\d+)\s+jahre?"
        for match in re.finditer(greater_than_pattern, text_lower):
            person = match.group(1)
            threshold = int(match.group(2))
            if person in entity_names:
                constraints.append(
                    {
                        "type": "greater_than",
                        "person": person,
                        "threshold": threshold,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {person} > {threshold}")

        logger.info(f"Parsed {len(constraints)} arithmetic constraints")

        # Add constraint parsing to proof tree
        for i, constraint in enumerate(constraints):
            proof_steps.append(
                ProofStep(
                    step_id=f"constraint_{i+1}",
                    step_type=ProofStepType.RULE_APPLICATION,
                    output=f"Constraint {i+1}: {constraint['text']}",
                    explanation_text=f"Parsed arithmetic constraint: {constraint['type']}",
                    confidence=0.90,
                    metadata=constraint,
                    source_component="component_45_logic_puzzle_solver_core",
                )
            )

        if not constraints:
            return {
                "solution": {},
                "proof_tree": None,
                "answer": "Keine arithmetischen Bedingungen erkannt.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "arithmetic",
            }

        # =====================================================================
        # STEP 2: Solve system of equations using substitution
        # =====================================================================
        solution = self._solve_linear_system(entity_names, constraints)

        if solution is None:
            logger.warning("No solution found for arithmetic puzzle")
            return {
                "solution": {},
                "proof_tree": ProofTree(
                    query=question or "Arithmetic puzzle",
                    root_steps=proof_steps,
                    metadata={"confidence": 0.3},
                ),
                "answer": "Keine Loesung gefunden - die Bedingungen widersprechen sich.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "arithmetic",
            }

        # =====================================================================
        # STEP 3: Format answer
        # =====================================================================
        # Add solution step to proof tree
        proof_steps.append(
            ProofStep(
                step_id="solution_found",
                step_type=ProofStepType.CONCLUSION,
                output=f"Loesung gefunden: {solution}",
                explanation_text="System of equations solved",
                confidence=0.95,
                metadata={"solution": solution},
                source_component="component_45_logic_puzzle_solver_core",
            )
        )

        # Create proof tree
        proof_tree = ProofTree(
            query=question or "Arithmetic puzzle",
            root_steps=proof_steps,
            metadata={"confidence": 0.90, "num_steps": len(proof_steps)},
        )

        # Format human-readable answer
        answer_parts = []
        for name in sorted(solution.keys()):
            age = solution[name]
            # Capitalize first letter
            name_cap = name.capitalize()
            answer_parts.append(f"{name_cap} ist {age} Jahre alt")

        answer = ". ".join(answer_parts) + "."

        return {
            "solution": solution,
            "proof_tree": proof_tree,
            "answer": answer,
            "result": "SATISFIABLE",
            "puzzle_type": "arithmetic",
            "confidence": 0.90,
        }

    def _solve_linear_system(
        self, entities: List[str], constraints: List[Dict]
    ) -> Optional[Dict[str, int]]:
        """
        Solve system of linear equations via substitution.

        For age puzzles, we typically have:
        - N entities (variables)
        - Linear relationships between them
        - One sum constraint

        Strategy:
        1. Express all variables in terms of one base variable
        2. Use sum constraint to solve for base variable
        3. Substitute back to get all values
        4. Verify inequality constraints

        Args:
            entities: List of variable names
            constraints: List of constraint dicts

        Returns:
            Dict mapping entity name to age, or None if unsolvable
        """
        logger.debug(f"Solving linear system for {len(entities)} variables")

        # Build expressions: each variable as (coefficient, variable, constant)
        # e.g., Anna = 2*Clara + 5 is stored as {"anna": (2, "clara", 5)}
        # Base variables have coefficient 1 and are expressed as themselves

        expressions = {}  # var_name -> (coef_of_base, base_var, constant)
        sum_total = None
        inequality_constraints = []

        # Pass 1: Extract sum and inequality constraints
        for c in constraints:
            if c["type"] == "sum":
                sum_total = c["total"]
            elif c["type"] == "less_than":
                inequality_constraints.append((c["person"], "<", c["threshold"]))
            elif c["type"] == "greater_than":
                inequality_constraints.append((c["person"], ">", c["threshold"]))

        # Pass 2: Build expression graph from relationship constraints
        # Find a base variable (one that others are expressed relative to)
        relation_graph = {}  # source -> [(target, multiplier, addend)]

        for c in constraints:
            if c["type"] == "older_by":
                # person1 = person2 + diff
                p1, p2, diff = c["person1"], c["person2"], c["diff"]
                if p2 not in relation_graph:
                    relation_graph[p2] = []
                relation_graph[p2].append((p1, 1, diff))
                # Also add reverse: person2 = person1 - diff
                if p1 not in relation_graph:
                    relation_graph[p1] = []
                relation_graph[p1].append((p2, 1, -diff))

            elif c["type"] == "younger_by":
                # person1 = person2 - diff
                p1, p2, diff = c["person1"], c["person2"], c["diff"]
                if p2 not in relation_graph:
                    relation_graph[p2] = []
                relation_graph[p2].append((p1, 1, -diff))
                if p1 not in relation_graph:
                    relation_graph[p1] = []
                relation_graph[p1].append((p2, 1, diff))

            elif c["type"] == "double":
                # person1 = 2 * person2
                p1, p2 = c["person1"], c["person2"]
                if p2 not in relation_graph:
                    relation_graph[p2] = []
                relation_graph[p2].append((p1, 2, 0))
                # Reverse: person2 = person1 / 2 (will handle as fraction)
                if p1 not in relation_graph:
                    relation_graph[p1] = []
                relation_graph[p1].append((p2, 0.5, 0))

            elif c["type"] == "half":
                # person1 = person2 / 2
                p1, p2 = c["person1"], c["person2"]
                if p2 not in relation_graph:
                    relation_graph[p2] = []
                relation_graph[p2].append((p1, 0.5, 0))
                if p1 not in relation_graph:
                    relation_graph[p1] = []
                relation_graph[p1].append((p2, 2, 0))

        # Find connected component containing all entities
        # Choose a base variable (prefer one with low "degree" of expression)
        # For typical age puzzles, choose the "youngest" or "simplest" variable

        # Use BFS to express all variables in terms of one base
        base_var = entities[0] if entities else None

        # Try to find a variable that appears in a "less_than" constraint (often the base)
        for ineq in inequality_constraints:
            if ineq[0] in entities:
                base_var = ineq[0]
                break

        if not base_var:
            logger.warning("No base variable found")
            return None

        # BFS to express all variables in terms of base_var
        # Expression: var = coef * base_var + const
        expressions[base_var] = (1.0, base_var, 0.0)  # base = 1*base + 0

        visited = {base_var}
        queue = [base_var]

        while queue:
            current = queue.pop(0)
            curr_coef, _, curr_const = expressions[current]

            if current in relation_graph:
                for target, mult, add in relation_graph[current]:
                    if target not in visited:
                        # target = mult * current + add
                        # current = curr_coef * base + curr_const
                        # target = mult * (curr_coef * base + curr_const) + add
                        #        = (mult * curr_coef) * base + (mult * curr_const + add)
                        new_coef = mult * curr_coef
                        new_const = mult * curr_const + add
                        expressions[target] = (new_coef, base_var, new_const)
                        visited.add(target)
                        queue.append(target)

        # Check if all entities are covered
        missing = [e for e in entities if e not in expressions]
        if missing:
            logger.warning(f"Could not express all variables: missing {missing}")
            # Try to include missing entities with default expression
            for m in missing:
                expressions[m] = (1.0, m, 0.0)  # Assume independent

        # Now solve using sum constraint if available
        if sum_total is not None:
            # Sum of all entities = sum_total
            # Sum of (coef_i * base + const_i) = sum_total
            # (Sum of coef_i) * base + (Sum of const_i) = sum_total
            # base = (sum_total - Sum of const_i) / (Sum of coef_i)

            total_coef = 0.0
            total_const = 0.0

            for entity in entities:
                if entity in expressions:
                    coef, _, const = expressions[entity]
                    total_coef += coef
                    total_const += const

            if abs(total_coef) < 0.0001:
                logger.warning("Degenerate system: coefficient sum is zero")
                return None

            base_value = (sum_total - total_const) / total_coef

            # Check if base_value is a valid integer
            if abs(base_value - round(base_value)) > 0.001:
                # Try nearby integer values
                candidates = [int(base_value), int(base_value) + 1, int(base_value) - 1]
            else:
                candidates = [int(round(base_value))]

            # Try each candidate and check inequality constraints
            for base_val in candidates:
                solution = {}
                valid = True

                for entity in entities:
                    if entity in expressions:
                        coef, _, const = expressions[entity]
                        value = coef * base_val + const
                        # Check for integer value
                        if abs(value - round(value)) > 0.001:
                            valid = False
                            break
                        solution[entity] = int(round(value))
                    else:
                        valid = False
                        break

                if not valid:
                    continue

                # Verify inequality constraints
                for person, op, threshold in inequality_constraints:
                    if person not in solution:
                        valid = False
                        break
                    if op == "<" and solution[person] >= threshold:
                        valid = False
                        break
                    if op == ">" and solution[person] <= threshold:
                        valid = False
                        break

                if valid:
                    # Verify sum
                    actual_sum = sum(solution.values())
                    if actual_sum == sum_total:
                        logger.info(f"Found valid solution: {solution}")
                        return solution

            # If strict constraints failed, try relaxed (<=, >=)
            for base_val in candidates:
                solution = {}
                valid = True

                for entity in entities:
                    if entity in expressions:
                        coef, _, const = expressions[entity]
                        value = coef * base_val + const
                        if abs(value - round(value)) > 0.001:
                            valid = False
                            break
                        solution[entity] = int(round(value))

                if not valid:
                    continue

                # Relaxed check for inequalities (allow equality)
                for person, op, threshold in inequality_constraints:
                    if person not in solution:
                        valid = False
                        break
                    if op == "<" and solution[person] > threshold:
                        valid = False
                        break
                    if op == ">" and solution[person] < threshold:
                        valid = False
                        break

                if valid:
                    actual_sum = sum(solution.values())
                    if actual_sum == sum_total:
                        logger.info(
                            f"Found solution with relaxed constraints: {solution}"
                        )
                        return solution

            logger.warning("No valid solution satisfies all constraints")
            return None

        else:
            # No sum constraint - try to find solution via enumeration
            # Use inequality constraints to bound the search
            logger.warning("No sum constraint - attempting bounded enumeration")

            # Find bounds from inequality constraints
            bounds = {e: (1, 100) for e in entities}  # Default bounds
            for person, op, threshold in inequality_constraints:
                if person in bounds:
                    low, high = bounds[person]
                    if op == "<":
                        bounds[person] = (low, min(high, threshold - 1))
                    elif op == ">":
                        bounds[person] = (max(low, threshold + 1), high)

            # Try base variable values within bounds
            if base_var in bounds:
                low, high = bounds[base_var]
                for base_val in range(low, min(high + 1, low + 100)):  # Limit search
                    solution = {}
                    valid = True

                    for entity in entities:
                        if entity in expressions:
                            coef, _, const = expressions[entity]
                            value = coef * base_val + const
                            if abs(value - round(value)) > 0.001 or value < 0:
                                valid = False
                                break
                            solution[entity] = int(round(value))

                    if not valid:
                        continue

                    # Verify all constraints
                    for person, op, threshold in inequality_constraints:
                        if person not in solution:
                            valid = False
                            break
                        if op == "<" and solution[person] >= threshold:
                            valid = False
                            break
                        if op == ">" and solution[person] <= threshold:
                            valid = False
                            break

                    if valid:
                        logger.info(f"Found solution via enumeration: {solution}")
                        return solution

            logger.warning("Enumeration failed to find solution")
            return None

    # =========================================================================
    # CIRCULAR SEATING PUZZLE SOLVER
    # =========================================================================

    def _solve_circular_seating(
        self, conditions_text: str, entities: List[str], question: Optional[str]
    ) -> Dict[str, Any]:
        """
        Solves circular seating puzzles (round table arrangements).

        Strategy:
        1. Parse spatial constraints (opposite, left/right, not next to)
        2. Model as constraint satisfaction on circular positions
        3. Solve via backtracking with constraint propagation
        4. Generate ProofTree

        Circular position semantics (for N people):
        - Positions: 0, 1, 2, ..., N-1 (clockwise)
        - "gegenueber" (opposite): position differs by N/2 (mod N)
        - "rechts von" (right of): position + 1 (mod N)
        - "links von" (left of): position - 1 (mod N)
        - "nicht neben" (not next to): position differs by != 1 (mod N)

        Args:
            conditions_text: Text with seating constraints
            entities: List of entity names
            question: Optional question

        Returns:
            Solution dict with positions and ProofTree
        """
        import re

        from component_17_proof_explanation import ProofStep, ProofTree
        from component_17_proof_explanation import StepType as ProofStepType

        logger.info(f"Solving CIRCULAR_SEATING puzzle with {len(entities)} entities")

        # Initialize proof tree steps
        proof_steps = []

        proof_steps.append(
            ProofStep(
                step_id="circular_detection",
                step_type=ProofStepType.PREMISE,
                output="Circular seating puzzle detected",
                explanation_text=f"Round table puzzle with {len(entities)} persons",
                confidence=0.95,
                metadata={"entities": entities},
                source_component="component_45_logic_puzzle_solver_core",
            )
        )

        # Parse entities from text if needed
        if not entities:
            name_pattern = r"\b([A-Z][a-z]+)\b"
            potential_names = set(re.findall(name_pattern, conditions_text))
            stop_words = {
                "Die",
                "Der",
                "Das",
                "Eine",
                "Einer",
                "Eines",
                "Sind",
                "Ist",
                "Hat",
                "Haben",
                "Wie",
                "Vier",
                "Drei",
                "Fuenf",
                "Runde",
                "Runden",
                "Runder",
                "Tisch",
                "Sitzen",
                "Sitzt",
                "Gegenueber",
                "Rechts",
                "Links",
                "Neben",
                "Direkt",
                "Hinweise",
                "Frage",
                "Personen",
                "Sitzordnung",
                "Uhrzeigersinn",
            }
            entities = [n for n in potential_names if n not in stop_words]
            logger.info(f"Extracted entities: {entities}")

        n = len(entities)
        if n < 2:
            return {
                "solution": {},
                "proof_tree": None,
                "answer": "Mindestens 2 Personen benoetigt.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "circular_seating",
            }

        entity_lower = {e.lower(): e for e in entities}
        entity_names = list(entity_lower.keys())
        text_lower = conditions_text.lower()

        # =====================================================================
        # STEP 1: Parse circular seating constraints
        # =====================================================================
        constraints = []

        # Pattern 1: "X sitzt gegenueber von Y" or "X sitzt direkt gegenueber von Y"
        opposite_pattern = (
            r"(\w+)\s+sitzt\s+(?:direkt\s+)?gegenueber\s+(?:von\s+)?(\w+)"
        )
        for match in re.finditer(opposite_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_names and p2 in entity_names:
                constraints.append(
                    {
                        "type": "opposite",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} opposite {p2}")

        # Pattern 2: "X sitzt rechts von Y"
        right_pattern = r"(\w+)\s+sitzt\s+rechts\s+von\s+(\w+)"
        for match in re.finditer(right_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_names and p2 in entity_names:
                constraints.append(
                    {
                        "type": "right_of",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} right of {p2}")

        # Pattern 3: "X sitzt links von Y"
        left_pattern = r"(\w+)\s+sitzt\s+links\s+von\s+(\w+)"
        for match in re.finditer(left_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_names and p2 in entity_names:
                constraints.append(
                    {
                        "type": "left_of",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} left of {p2}")

        # Pattern 4: "X sitzt nicht neben Y"
        not_next_pattern = r"(\w+)\s+sitzt\s+nicht\s+(?:direkt\s+)?neben\s+(\w+)"
        for match in re.finditer(not_next_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_names and p2 in entity_names:
                constraints.append(
                    {
                        "type": "not_adjacent",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} not adjacent {p2}")

        # Pattern 5: "X sitzt neben Y" (positiv)
        adjacent_pattern = r"(\w+)\s+sitzt\s+(?:direkt\s+)?neben\s+(\w+)"
        for match in re.finditer(adjacent_pattern, text_lower):
            # Skip if this was already matched as "nicht neben"
            if (
                "nicht"
                in conditions_text[max(0, match.start() - 10) : match.start()].lower()
            ):
                continue
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_names and p2 in entity_names:
                constraints.append(
                    {
                        "type": "adjacent",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} adjacent {p2}")

        logger.info(f"Parsed {len(constraints)} circular seating constraints")

        # Add constraints to proof tree
        for i, c in enumerate(constraints):
            proof_steps.append(
                ProofStep(
                    step_id=f"constraint_{i+1}",
                    step_type=ProofStepType.RULE_APPLICATION,
                    output=f"Constraint {i+1}: {c['text']}",
                    explanation_text=f"Spatial constraint: {c['type']}",
                    confidence=0.90,
                    metadata=c,
                    source_component="component_45_logic_puzzle_solver_core",
                )
            )

        if not constraints:
            return {
                "solution": {},
                "proof_tree": None,
                "answer": "Keine Sitzordnungs-Bedingungen erkannt.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "circular_seating",
            }

        # =====================================================================
        # STEP 2: Solve via constraint backtracking
        # =====================================================================
        solution = self._solve_circular_constraints(entity_names, n, constraints)

        if solution is None:
            logger.warning("No solution found for circular seating puzzle")
            return {
                "solution": {},
                "proof_tree": ProofTree(
                    query=question or "Circular seating puzzle",
                    root_steps=proof_steps,
                    metadata={"confidence": 0.3},
                ),
                "answer": "Keine gueltige Sitzordnung gefunden.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "circular_seating",
            }

        # =====================================================================
        # STEP 3: Format answer
        # =====================================================================
        proof_steps.append(
            ProofStep(
                step_id="solution_found",
                step_type=ProofStepType.CONCLUSION,
                output=f"Sitzordnung gefunden: {solution}",
                explanation_text="Constraint satisfaction solved",
                confidence=0.95,
                metadata={"solution": solution},
                source_component="component_45_logic_puzzle_solver_core",
            )
        )

        proof_tree = ProofTree(
            query=question or "Circular seating puzzle",
            root_steps=proof_steps,
            metadata={"confidence": 0.90, "num_steps": len(proof_steps)},
        )

        # Sort by position and format as clockwise order
        sorted_seats = sorted(solution.items(), key=lambda x: x[1])
        order = [entity_lower.get(name, name.capitalize()) for name, _ in sorted_seats]
        answer = f"Die Sitzordnung im Uhrzeigersinn: {', '.join(order)}"

        return {
            "solution": {entity_lower.get(k, k): v for k, v in solution.items()},
            "proof_tree": proof_tree,
            "answer": answer,
            "result": "SATISFIABLE",
            "puzzle_type": "circular_seating",
            "confidence": 0.90,
            "ordering": order,
        }

    def _solve_circular_constraints(
        self, entities: List[str], n: int, constraints: List[Dict]
    ) -> Optional[Dict[str, int]]:
        """
        Solve circular seating via backtracking with constraint checking.

        Args:
            entities: List of entity names (lowercase)
            n: Number of positions (same as number of entities)
            constraints: List of constraint dicts

        Returns:
            Dict mapping entity name to position (0-indexed), or None
        """

        def check_constraint(assignment: Dict[str, int], c: Dict) -> bool:
            """Check if a constraint is satisfied by the current assignment."""
            ctype = c["type"]
            p1 = c.get("person1")
            p2 = c.get("person2")

            if p1 not in assignment or (p2 and p2 not in assignment):
                return True  # Not yet assigned, assume OK

            pos1 = assignment[p1]
            pos2 = assignment.get(p2, -1)

            if ctype == "opposite":
                # Opposite = exactly n/2 apart (mod n)
                diff = abs(pos1 - pos2)
                return diff == n // 2 or diff == n - n // 2

            elif ctype == "right_of":
                # p1 is right of p2 = p1 is at position (p2 + 1) mod n
                return pos1 == (pos2 + 1) % n

            elif ctype == "left_of":
                # p1 is left of p2 = p1 is at position (p2 - 1) mod n
                return pos1 == (pos2 - 1) % n

            elif ctype == "adjacent":
                # Adjacent = exactly 1 apart (mod n)
                diff = abs(pos1 - pos2)
                return diff == 1 or diff == n - 1

            elif ctype == "not_adjacent":
                # Not adjacent = not exactly 1 apart
                diff = abs(pos1 - pos2)
                return diff != 1 and diff != n - 1

            return True

        def backtrack(
            assignment: Dict[str, int], remaining: List[str]
        ) -> Optional[Dict[str, int]]:
            """Recursive backtracking search."""
            if not remaining:
                # All assigned - verify all constraints
                for c in constraints:
                    if not check_constraint(assignment, c):
                        return None
                return assignment.copy()

            entity = remaining[0]
            used_positions = set(assignment.values())

            for pos in range(n):
                if pos in used_positions:
                    continue

                assignment[entity] = pos

                # Check constraints involving this entity
                valid = True
                for c in constraints:
                    if c.get("person1") == entity or c.get("person2") == entity:
                        if not check_constraint(assignment, c):
                            valid = False
                            break

                if valid:
                    result = backtrack(assignment, remaining[1:])
                    if result is not None:
                        return result

                del assignment[entity]

            return None

        # Try with the first entity fixed at position 0 (reduces symmetry)
        if entities:
            initial = {entities[0]: 0}
            return backtrack(initial, entities[1:])

        return None

    # =========================================================================
    # LINEAR ORDERING PUZZLE SOLVER
    # =========================================================================

    def _solve_linear_ordering(
        self, conditions_text: str, entities: List[str], question: Optional[str]
    ) -> Dict[str, Any]:
        """
        Solves linear ordering puzzles (people in a line/row).

        Strategy:
        1. Parse ordering constraints (left of, right of, not at end, not adjacent)
        2. Model as constraint satisfaction on linear positions
        3. Solve via backtracking with constraint propagation
        4. Generate ProofTree

        Linear position semantics (for N people):
        - Positions: 0, 1, 2, ..., N-1 (left to right)
        - "links von" (left of): position < other position
        - "rechts von" (right of): position > other position
        - "nicht am Ende": position not in {0, N-1}
        - "direkt neben" (directly next to): |position - other| == 1
        - "nicht direkt neben" (not directly next to): |position - other| != 1

        Args:
            conditions_text: Text with ordering constraints
            entities: List of entity names
            question: Optional question

        Returns:
            Solution dict with positions and ProofTree
        """
        import re

        from component_17_proof_explanation import ProofStep, ProofTree
        from component_17_proof_explanation import StepType as ProofStepType

        logger.info(f"Solving LINEAR_ORDERING puzzle with {len(entities)} entities")

        proof_steps = []

        proof_steps.append(
            ProofStep(
                step_id="linear_detection",
                step_type=ProofStepType.PREMISE,
                output="Linear ordering puzzle detected",
                explanation_text=f"Line arrangement puzzle with {len(entities)} elements",
                confidence=0.95,
                metadata={"entities": entities},
                source_component="component_45_logic_puzzle_solver_core",
            )
        )

        # Parse entities from text if needed
        if not entities:
            name_pattern = r"\b([A-Z])\b"  # Single capital letters for this puzzle type
            potential_names = set(re.findall(name_pattern, conditions_text))
            entities = sorted(list(potential_names))
            logger.info(f"Extracted entities: {entities}")

        n = len(entities)
        if n < 2:
            return {
                "solution": {},
                "proof_tree": None,
                "answer": "Mindestens 2 Elemente benoetigt.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "linear_ordering",
            }

        entity_set = set(e.lower() for e in entities)
        text_lower = conditions_text.lower()

        # =====================================================================
        # STEP 1: Parse linear ordering constraints
        # =====================================================================
        constraints = []

        # Pattern 1: "X steht links von Y" or "X links von Y"
        left_pattern = r"(\w+)\s+(?:steht\s+)?links\s+von\s+(\w+)"
        for match in re.finditer(left_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_set and p2 in entity_set:
                constraints.append(
                    {
                        "type": "left_of",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} left of {p2}")

        # Pattern 2: "X steht rechts von Y" or "X rechts von Y"
        right_pattern = r"(\w+)\s+(?:steht\s+)?rechts\s+von\s+(\w+)"
        for match in re.finditer(right_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_set and p2 in entity_set:
                constraints.append(
                    {
                        "type": "right_of",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} right of {p2}")

        # Pattern 3: "X steht nicht am Ende"
        not_end_pattern = r"(\w+)\s+(?:steht\s+)?nicht\s+am\s+ende"
        for match in re.finditer(not_end_pattern, text_lower):
            p1 = match.group(1)
            if p1 in entity_set:
                constraints.append(
                    {
                        "type": "not_at_end",
                        "person1": p1,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} not at end")

        # Pattern 4: "X steht nicht direkt neben Y"
        not_adjacent_pattern = (
            r"(\w+)\s+(?:steht\s+)?nicht\s+(?:direkt\s+)?neben\s+(\w+)"
        )
        for match in re.finditer(not_adjacent_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_set and p2 in entity_set:
                constraints.append(
                    {
                        "type": "not_adjacent",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} not adjacent {p2}")

        # Pattern 5: "X steht direkt neben Y" (positive adjacent)
        adjacent_pattern = r"(\w+)\s+(?:steht\s+)?direkt\s+neben\s+(\w+)"
        for match in re.finditer(adjacent_pattern, text_lower):
            # Skip if "nicht" appears before
            if (
                "nicht"
                in conditions_text[max(0, match.start() - 10) : match.start()].lower()
            ):
                continue
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_set and p2 in entity_set:
                constraints.append(
                    {
                        "type": "adjacent",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} adjacent {p2}")

        # =====================================================================
        # Pattern 6: "X ist groesser als Y" (comparison - greater than)
        # In ordering: position 1 = greatest, position N = smallest
        # "groesser" means lower position number
        # =====================================================================
        greater_pattern = r"(\w+)\s+ist\s+(?:groesser|schneller|aelter|schwerer|staerker|hoeher|laenger)\s+als\s+(\w+)"
        for match in re.finditer(greater_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_set and p2 in entity_set:
                constraints.append(
                    {
                        "type": "greater_than",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} greater than {p2}")

        # =====================================================================
        # Pattern 7: "X ist kleiner als Y" (comparison - less than)
        # "kleiner" means higher position number
        # =====================================================================
        less_pattern = r"(\w+)\s+ist\s+(?:kleiner|langsamer|juenger|leichter|schwaecker|niedriger|kuerzer)\s+als\s+(\w+)"
        for match in re.finditer(less_pattern, text_lower):
            p1, p2 = match.group(1), match.group(2)
            if p1 in entity_set and p2 in entity_set:
                constraints.append(
                    {
                        "type": "less_than",
                        "person1": p1,
                        "person2": p2,
                        "text": match.group(0),
                    }
                )
                logger.debug(f"Constraint: {p1} less than {p2}")

        # =====================================================================
        # Pattern 8: "X ist kleiner als A, aber groesser als B" (compound)
        # Extracts TWO constraints from a single statement
        # =====================================================================
        compound_pattern = r"(\w+)\s+ist\s+(groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter)\s+als\s+(\w+),?\s+aber\s+(groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter)\s+als\s+(\w+)"
        for match in re.finditer(compound_pattern, text_lower):
            subj = match.group(1)
            comp1 = match.group(2)
            obj1 = match.group(3)
            comp2 = match.group(4)
            obj2 = match.group(5)

            # First comparison
            if subj in entity_set and obj1 in entity_set:
                if comp1 in (
                    "groesser",
                    "schneller",
                    "aelter",
                    "schwerer",
                    "staerker",
                    "hoeher",
                    "laenger",
                ):
                    constraints.append(
                        {
                            "type": "greater_than",
                            "person1": subj,
                            "person2": obj1,
                            "text": f"{subj} ist {comp1} als {obj1}",
                        }
                    )
                    logger.debug(f"Compound constraint 1: {subj} greater than {obj1}")
                else:
                    constraints.append(
                        {
                            "type": "less_than",
                            "person1": subj,
                            "person2": obj1,
                            "text": f"{subj} ist {comp1} als {obj1}",
                        }
                    )
                    logger.debug(f"Compound constraint 1: {subj} less than {obj1}")

            # Second comparison
            if subj in entity_set and obj2 in entity_set:
                if comp2 in (
                    "groesser",
                    "schneller",
                    "aelter",
                    "schwerer",
                    "staerker",
                    "hoeher",
                    "laenger",
                ):
                    constraints.append(
                        {
                            "type": "greater_than",
                            "person1": subj,
                            "person2": obj2,
                            "text": f"{subj} ist {comp2} als {obj2}",
                        }
                    )
                    logger.debug(f"Compound constraint 2: {subj} greater than {obj2}")
                else:
                    constraints.append(
                        {
                            "type": "less_than",
                            "person1": subj,
                            "person2": obj2,
                            "text": f"{subj} ist {comp2} als {obj2}",
                        }
                    )
                    logger.debug(f"Compound constraint 2: {subj} less than {obj2}")

        logger.info(f"Parsed {len(constraints)} linear ordering constraints")

        # Add constraints to proof tree
        for i, c in enumerate(constraints):
            proof_steps.append(
                ProofStep(
                    step_id=f"constraint_{i+1}",
                    step_type=ProofStepType.RULE_APPLICATION,
                    output=f"Constraint {i+1}: {c['text']}",
                    explanation_text=f"Ordering constraint: {c['type']}",
                    confidence=0.90,
                    metadata=c,
                    source_component="component_45_logic_puzzle_solver_core",
                )
            )

        if not constraints:
            return {
                "solution": {},
                "proof_tree": None,
                "answer": "Keine Ordnungs-Bedingungen erkannt.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "linear_ordering",
            }

        # =====================================================================
        # STEP 2: Solve via constraint backtracking
        # =====================================================================
        entity_list = [e.lower() for e in entities]
        solution = self._solve_linear_constraints(entity_list, n, constraints)

        if solution is None:
            logger.warning("No solution found for linear ordering puzzle")
            return {
                "solution": {},
                "proof_tree": ProofTree(
                    query=question or "Linear ordering puzzle",
                    root_steps=proof_steps,
                    metadata={"confidence": 0.3},
                ),
                "answer": "Keine gueltige Reihenfolge gefunden.",
                "result": "UNSATISFIABLE",
                "puzzle_type": "linear_ordering",
            }

        # =====================================================================
        # STEP 3: Format answer
        # =====================================================================
        proof_steps.append(
            ProofStep(
                step_id="solution_found",
                step_type=ProofStepType.CONCLUSION,
                output=f"Reihenfolge gefunden: {solution}",
                explanation_text="Constraint satisfaction solved",
                confidence=0.95,
                metadata={"solution": solution},
                source_component="component_45_logic_puzzle_solver_core",
            )
        )

        proof_tree = ProofTree(
            query=question or "Linear ordering puzzle",
            root_steps=proof_steps,
            metadata={"confidence": 0.90, "num_steps": len(proof_steps)},
        )

        # Build original case mapping
        entity_case = {e.lower(): e for e in entities}

        # Sort by position and format as left-to-right order
        sorted_positions = sorted(solution.items(), key=lambda x: x[1])
        order = [
            entity_case.get(name, name.capitalize()) for name, _ in sorted_positions
        ]

        # Detect comparison constraints to use ">" format
        has_comparison = any(
            c["type"] in ("greater_than", "less_than") for c in constraints
        )

        if has_comparison:
            # Use ">" format for comparison puzzles
            answer = f"Reihenfolge: {' > '.join(order)}"
        else:
            # Use comma format for positional puzzles
            answer = f"Die Reihenfolge ist: {', '.join(order)}"

        # Build metadata for questions about groesste/kleinste
        groesste = order[0] if order else None
        kleinste = order[-1] if order else None

        return {
            "solution": {entity_case.get(k, k): v for k, v in solution.items()},
            "proof_tree": proof_tree,
            "answer": answer,
            "result": "SATISFIABLE",
            "puzzle_type": "linear_ordering",
            "confidence": 0.90,
            "ordering": order,
            "groesste": groesste,
            "kleinste": kleinste,
        }

    def _solve_linear_constraints(
        self, entities: List[str], n: int, constraints: List[Dict]
    ) -> Optional[Dict[str, int]]:
        """
        Solve linear ordering via backtracking with constraint checking.

        Args:
            entities: List of entity names (lowercase)
            n: Number of positions
            constraints: List of constraint dicts

        Returns:
            Dict mapping entity name to position (0-indexed), or None
        """

        def check_constraint(assignment: Dict[str, int], c: Dict) -> bool:
            """Check if a constraint is satisfied by the current assignment."""
            ctype = c["type"]
            p1 = c.get("person1")
            p2 = c.get("person2")

            if p1 not in assignment:
                return True  # Not yet assigned

            pos1 = assignment[p1]

            if ctype == "left_of":
                if p2 not in assignment:
                    return True
                pos2 = assignment[p2]
                return pos1 < pos2

            elif ctype == "right_of":
                if p2 not in assignment:
                    return True
                pos2 = assignment[p2]
                return pos1 > pos2

            elif ctype == "not_at_end":
                return pos1 not in {0, n - 1}

            elif ctype == "adjacent":
                if p2 not in assignment:
                    return True
                pos2 = assignment[p2]
                return abs(pos1 - pos2) == 1

            elif ctype == "not_adjacent":
                if p2 not in assignment:
                    return True
                pos2 = assignment[p2]
                return abs(pos1 - pos2) != 1

            elif ctype == "greater_than":
                # A > B means pos(A) < pos(B) (position 0 = greatest)
                if p2 not in assignment:
                    return True
                pos2 = assignment[p2]
                return pos1 < pos2

            elif ctype == "less_than":
                # A < B means pos(A) > pos(B) (position N-1 = smallest)
                if p2 not in assignment:
                    return True
                pos2 = assignment[p2]
                return pos1 > pos2

            return True

        def backtrack(
            assignment: Dict[str, int], remaining: List[str]
        ) -> Optional[Dict[str, int]]:
            """Recursive backtracking search."""
            if not remaining:
                # All assigned - verify all constraints
                for c in constraints:
                    if not check_constraint(assignment, c):
                        return None
                return assignment.copy()

            entity = remaining[0]
            used_positions = set(assignment.values())

            for pos in range(n):
                if pos in used_positions:
                    continue

                assignment[entity] = pos

                # Check constraints involving this entity
                valid = True
                for c in constraints:
                    if c.get("person1") == entity or c.get("person2") == entity:
                        if not check_constraint(assignment, c):
                            valid = False
                            break

                if valid:
                    result = backtrack(assignment, remaining[1:])
                    if result is not None:
                        return result

                del assignment[entity]

            return None

        return backtrack({}, entities)

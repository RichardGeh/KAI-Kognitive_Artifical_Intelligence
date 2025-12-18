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

    ENTITY_SAT = "entity_sat"  # Entitäten-basierte Rätsel (Leo/Mark/Nick)
    NUMERICAL_CSP = "numerical_csp"  # Zahlen-basierte Constraint-Rätsel
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
        Klassifiziert Puzzle-Typ für Solver-Routing.

        Args:
            text: Der Puzzle-Text
            entities: Liste erkannter Entitäten

        Returns:
            PuzzleType enum
        """
        import re

        text_lower = text.lower()

        # Patterns für numerische Puzzles (ohne generisches numbered list pattern)
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

        # Patterns für Entitäten-basierte Puzzles (erweitert)
        entity_patterns = [
            r"\b[A-Z][a-z]+\s+(?:trinkt|mag|isst|bestellt|ist|hat)\b",
            r"\b(?:wer|was|welche[rs]?)\s+(?:trinkt|mag|isst|hat)\b",
            r"\b(?:ist\s+kein|ist\s+nicht|hat\s+kein)\b",  # Negative constraints
        ]

        # Zähle Matches mit Logging
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

        # Berücksichtige erkannte Entitäten (nur wenn mindestens 2 vorhanden)
        entity_bonus = 2 if len(entities) >= 2 else 0
        total_entity_signals = entity_count + entity_bonus

        total_numerical = numerical_count + meta_count

        logger.debug(
            f"Puzzle type detection: numerical={numerical_count}, "
            f"meta={meta_count}, entity_patterns={entity_count}, "
            f"entities={len(entities)}, total_entity_signals={total_entity_signals}"
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

        # DEFAULT: ENTITY_SAT wenn Entitäten vorhanden (auch ohne entity patterns)
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

            if puzzle_type == PuzzleType.NUMERICAL_CSP:
                logger.info("Routing to NUMERICAL CSP solver")
                return self._solve_numerical_csp(conditions_text, question)
            elif puzzle_type == PuzzleType.ENTITY_SAT:
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
                logger.warning("Puzzle is unsolvable (UNSAT)")
                return {
                    "solution": {},
                    "proof_tree": None,
                    "answer": "The puzzle has no solution (contradiction in conditions).",
                    "result": "UNSATISFIABLE",
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

        Raises:
            ConstraintReasoningError: If CNF conversion fails
        """
        try:
            clauses: List[Clause] = []

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

            return CNFFormula(clauses)
        except Exception as e:
            raise ConstraintReasoningError(
                "Error converting to CNF formula",
                context={"num_conditions": len(conditions)},
                original_exception=e,
            )

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

            # Extract verb from question if available
            question_verb = self._extract_verb_from_question(question)

            # Detect assignment puzzle pattern
            is_assignment = self._is_assignment_answer(true_vars)

            # Format variables as sentences
            statements = []
            for var_name in true_vars:
                var = self.parser.get_variable(var_name)
                if var:
                    # For assignment puzzles, format as "Entity ist Object"
                    if is_assignment and "_hat_" in var_name:
                        # Extract object from variable: "bob_hat_lehrer" -> "lehrer"
                        parts = var_name.split("_hat_")
                        if len(parts) == 2:
                            entity = parts[0].capitalize()
                            obj = parts[1].capitalize()
                            statements.append(f"{entity} ist {obj}")
                            continue

                    # Default formatting (existing logic)
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

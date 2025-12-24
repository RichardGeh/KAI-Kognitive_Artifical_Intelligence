"""
component_45_zebra_puzzle_solver.py
===================================
CSP-based solver for Zebra (Einstein) puzzles.

This module provides a solver that:
1. Parses German text into ZebraPuzzle structure
2. Converts to CSP problem with position variables
3. Solves using AC-3 + backtracking with MRV/LCV heuristics
4. Generates ProofTree for transparent reasoning
5. Formats answers to specific questions

The solver uses a position-based CSP model where each value gets a position variable
with domain [1..N]. Constraints restrict which positions are valid.

Performance target: < 30 seconds for classic 5-person Zebra puzzle with 15+ constraints.

Author: KAI Development Team
Date: 2025-12-24
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from component_15_logging_config import get_logger
from component_17_proof_explanation import ProofStep, ProofTree, StepType
from component_29_constraint_reasoning import (
    Constraint,
    ConstraintProblem,
    ConstraintSolver,
    Variable,
    adjacent_to_constraint,
    all_different_constraint,
    at_position_constraint,
    different_position_constraint,
    directly_left_of_constraint,
    greater_than_constraint,
    less_than_constraint,
    same_position_constraint,
)
from component_45_zebra_puzzle_model import (
    ZebraConstraint,
    ZebraConstraintType,
    ZebraPuzzle,
    ZebraSolution,
)

logger = get_logger(__name__)


class ZebraPuzzleSolver:
    """
    Solver for Zebra (Einstein) puzzles using CSP.

    Workflow:
    1. Parse text -> ZebraPuzzle (or receive pre-built puzzle)
    2. Build CSP problem with position variables
    3. Add uniqueness constraints (all_different per category)
    4. Add puzzle constraints (same_entity, adjacent, etc.)
    5. Solve with AC-3 + backtracking + MRV + LCV
    6. Format solution and generate ProofTree
    """

    def __init__(self):
        """Initialize solver with CSP backend."""
        self.csp_solver = ConstraintSolver(use_ac3=True, use_mrv=True, use_lcv=True)

    def solve(self, puzzle: ZebraPuzzle) -> Dict[str, Any]:
        """
        Solve a Zebra puzzle.

        Args:
            puzzle: ZebraPuzzle instance with categories and constraints

        Returns:
            Dictionary with:
            - result: "SATISFIABLE" | "UNSATISFIABLE"
            - solution: ZebraSolution with position assignments
            - answer: Formatted answer string
            - proof_tree: ProofTree for reasoning transparency
            - confidence: 1.0 for CSP solutions
            - puzzle_type: "zebra_csp"
        """
        logger.info(
            f"Solving Zebra puzzle: {puzzle.num_entities} entities, "
            f"{len(puzzle.categories)} categories, {len(puzzle.constraints)} constraints"
        )

        # Step 1: Build CSP problem
        csp_problem, var_mapping = self._build_csp_problem(puzzle)

        logger.info(
            f"CSP problem built: {len(csp_problem.variables)} variables, "
            f"{len(csp_problem.constraints)} constraints"
        )

        # Step 2: Solve
        solution, proof_tree = self.csp_solver.solve(csp_problem, track_proof=True)

        if solution is None:
            logger.warning("Zebra puzzle is UNSATISFIABLE")
            return {
                "result": "UNSATISFIABLE",
                "solution": None,
                "answer": "Das Raetsel hat keine Loesung (Widerspruch in den Bedingungen).",
                "proof_tree": proof_tree,
                "confidence": 0.0,
                "puzzle_type": "zebra_csp",
            }

        # Step 3: Convert CSP solution to ZebraSolution
        zebra_solution = self._convert_solution(solution, puzzle, var_mapping)

        logger.info(
            f"[OK] Zebra puzzle solved: {len(zebra_solution.assignments)} positions"
        )

        # Step 4: Format answer
        answer = self._format_answer(zebra_solution, puzzle.question, puzzle)

        # Step 5: Enhance proof tree with solution details
        if proof_tree:
            self._enhance_proof_tree(proof_tree, zebra_solution, puzzle)

        return {
            "result": "SATISFIABLE",
            "solution": zebra_solution,
            "answer": answer,
            "proof_tree": proof_tree,
            "confidence": 1.0,
            "puzzle_type": "zebra_csp",
            "metadata": {
                "num_variables": len(csp_problem.variables),
                "num_constraints": len(csp_problem.constraints),
                "backtracks": self.csp_solver.backtrack_count,
                "constraint_checks": self.csp_solver.constraint_checks,
            },
        }

    def solve_from_text(
        self,
        text: str,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse German text and solve the Zebra puzzle.

        Args:
            text: German puzzle text with hints
            question: Optional question to answer

        Returns:
            Solution dictionary (see solve())
        """
        logger.info(f"Parsing Zebra puzzle from text ({len(text)} chars)")

        # Parse text to ZebraPuzzle
        puzzle = self._parse_puzzle_text(text, question)

        if puzzle is None:
            return {
                "result": "PARSE_ERROR",
                "solution": None,
                "answer": "Konnte das Raetsel nicht parsen.",
                "proof_tree": None,
                "confidence": 0.0,
                "puzzle_type": "zebra_csp",
            }

        return self.solve(puzzle)

    def _build_csp_problem(
        self, puzzle: ZebraPuzzle
    ) -> Tuple[ConstraintProblem, Dict[str, str]]:
        """
        Build CSP problem from ZebraPuzzle.

        Creates position variables for each value in each category.
        Variable names: "pos_{category}_{value}" with domain [1..N]

        Returns:
            Tuple of (ConstraintProblem, var_mapping)
            var_mapping maps value -> variable name
        """
        variables = {}
        var_mapping = {}  # value -> var_name
        positions = set(puzzle.get_positions())

        # Create position variable for each value
        for category, values in puzzle.categories.items():
            for value in values:
                var_name = f"pos_{category}_{value.lower()}"
                variables[var_name] = Variable(
                    name=var_name,
                    domain=positions.copy(),
                )
                var_mapping[value.lower()] = var_name

        # Build constraints
        constraints = []

        # 1. All-different constraints per category
        for category, values in puzzle.categories.items():
            cat_vars = [var_mapping[v.lower()] for v in values]
            constraints.append(all_different_constraint(cat_vars))

        # 2. Puzzle-specific constraints
        for zc in puzzle.constraints:
            constraint = self._convert_zebra_constraint(zc, var_mapping, puzzle)
            if constraint:
                constraints.append(constraint)

        return (
            ConstraintProblem(
                variables=variables,
                constraints=constraints,
                name="ZebraPuzzle",
            ),
            var_mapping,
        )

    def _convert_zebra_constraint(
        self,
        zc: ZebraConstraint,
        var_mapping: Dict[str, str],
        puzzle: ZebraPuzzle,
    ) -> Optional[Constraint]:
        """Convert ZebraConstraint to CSP Constraint."""
        try:
            if zc.constraint_type == ZebraConstraintType.SAME_ENTITY:
                # Two values at same position
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    return same_position_constraint(var1, var2)

            elif zc.constraint_type == ZebraConstraintType.DIRECTLY_LEFT_OF:
                # val1 directly left of val2: pos(val1) + 1 == pos(val2)
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    return directly_left_of_constraint(var1, var2)

            elif zc.constraint_type == ZebraConstraintType.DIRECTLY_RIGHT_OF:
                # val1 directly right of val2: pos(val2) + 1 == pos(val1)
                # Equivalent to: directly_left_of(val2, val1)
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    return directly_left_of_constraint(var2, var1)

            elif zc.constraint_type == ZebraConstraintType.ADJACENT_TO:
                # val1 adjacent to val2: |pos(val1) - pos(val2)| == 1
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    return adjacent_to_constraint(var1, var2)

            elif zc.constraint_type == ZebraConstraintType.AT_POSITION:
                # val at specific position
                val = zc.values[0]
                var = var_mapping.get(val.lower())
                if var and zc.position:
                    return at_position_constraint(var, zc.position)

            elif zc.constraint_type == ZebraConstraintType.LEFT_OF:
                # val1 somewhere left of val2: pos(val1) < pos(val2)
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    from component_29_constraint_reasoning import left_of_constraint

                    return left_of_constraint(var1, var2)

            elif zc.constraint_type == ZebraConstraintType.DIFFERENT_ENTITY:
                # Two values must be at different positions (negation)
                # "Clara ist nicht die Anwaeltin" -> pos(clara) != pos(anwaeltin)
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    return different_position_constraint(var1, var2)

            elif zc.constraint_type == ZebraConstraintType.GREATER_THAN:
                # Ordering: A > B means pos(A) < pos(B)
                # "Anna ist groesser als Ben" -> pos(anna) < pos(ben)
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    return greater_than_constraint(var1, var2)

            elif zc.constraint_type == ZebraConstraintType.LESS_THAN:
                # Ordering: A < B means pos(A) > pos(B)
                # "Ben ist kleiner als Anna" -> pos(ben) > pos(anna)
                val1, val2 = zc.values
                var1 = var_mapping.get(val1.lower())
                var2 = var_mapping.get(val2.lower())
                if var1 and var2:
                    return less_than_constraint(var1, var2)

            logger.warning(
                f"Could not convert constraint: {zc} - missing variable mapping"
            )
            return None

        except Exception as e:
            logger.error(f"Error converting constraint {zc}: {e}")
            return None

    def _convert_solution(
        self,
        csp_solution: Dict[str, int],
        puzzle: ZebraPuzzle,
        var_mapping: Dict[str, str],
    ) -> ZebraSolution:
        """Convert CSP solution (var -> position) to ZebraSolution."""
        # Invert var_mapping: var_name -> value
        inv_mapping = {v: k for k, v in var_mapping.items()}

        # Build position -> category -> value assignments
        assignments: Dict[int, Dict[str, str]] = {}

        for var_name, position in csp_solution.items():
            if not var_name.startswith("pos_"):
                continue

            # Parse var_name: "pos_{category}_{value}"
            parts = var_name[4:].split("_", 1)  # Remove "pos_" prefix
            if len(parts) != 2:
                continue

            category, value = parts

            if position not in assignments:
                assignments[position] = {}
            assignments[position][category] = value

        return ZebraSolution(
            assignments=assignments,
            confidence=1.0,
            proof_steps=[f"CSP solution with {len(csp_solution)} variables"],
        )

    def _format_answer(
        self,
        solution: ZebraSolution,
        question: Optional[str],
        puzzle: ZebraPuzzle,
    ) -> str:
        """Format answer to the puzzle question."""
        # Check if this is an ordering puzzle (has only "person" category)
        is_ordering = len(puzzle.categories) == 1 and "person" in puzzle.categories

        # Use ordering-specific answer formatting
        if is_ordering:
            return self._format_ordering_answer(solution, question, puzzle)

        # Check if this is a multi-attribute puzzle (has "name" category)
        is_multi_attr = "name" in puzzle.categories

        if not question:
            # Default: describe solution based on puzzle type
            return self._format_full_solution(solution, puzzle, is_multi_attr)

        question_lower = question.lower()

        # Check for multi-attribute full assignment question
        # "Wer hat welchen Beruf, wohnt wo, hat welches Haustier und welches Hobby?"
        if is_multi_attr and (
            "welchen beruf" in question_lower
            or "wer hat welchen" in question_lower
            or "wohnt wo" in question_lower
        ):
            return self._format_full_solution(solution, puzzle, is_multi_attr)

        # Check for combined question FIRST (most specific)
        # "Wer hat das Zebra und wer trinkt Wasser?"
        has_zebra_q = "zebra" in question_lower
        has_water_q = "wasser" in question_lower

        if has_zebra_q and has_water_q:
            zebra_owner = solution.get_entity_with_value("zebra", "nationality")
            water_drinker = solution.get_entity_with_value("wasser", "nationality")

            parts = []
            if zebra_owner:
                parts.append(f"Der {zebra_owner.capitalize()} hat das Zebra")
            if water_drinker:
                parts.append(f"der {water_drinker.capitalize()} trinkt Wasser")

            if parts:
                return " und ".join(parts) + "."
            return "Konnte die Antwort nicht bestimmen."

        # Pattern: "Wer hat das Zebra" / "Wer hat ein Zebra" (single question)
        if has_zebra_q:
            owner = solution.get_entity_with_value("zebra", "nationality")
            if owner:
                return f"Der {owner.capitalize()} hat das Zebra."
            return "Konnte den Zebra-Besitzer nicht bestimmen."

        # Pattern: "Wer trinkt Wasser" / "wer trinkt wasser" (single question)
        if has_water_q:
            drinker = solution.get_entity_with_value("wasser", "nationality")
            if drinker:
                return f"Der {drinker.capitalize()} trinkt Wasser."
            return "Konnte den Wassertrinker nicht bestimmen."

        # Generic: try to find what's being asked
        return self._format_full_solution(solution, puzzle, is_multi_attr)

    def _format_full_solution(
        self,
        solution: ZebraSolution,
        puzzle: ZebraPuzzle,
        is_multi_attr: bool,
    ) -> str:
        """
        Format complete solution for all entities.

        Args:
            solution: The solution assignments
            puzzle: The puzzle definition
            is_multi_attr: Whether this is a multi-attribute puzzle

        Returns:
            Formatted solution string
        """
        lines = []

        if is_multi_attr:
            # Multi-attribute format: group by person name
            # Find the name category and use it as the key
            puzzle.categories.get("name", [])

            for pos in sorted(solution.assignments.keys()):
                cats = solution.assignments[pos]
                # Find the name in this position's assignments
                name = cats.get("name", f"Person {pos}")
                # Build attribute list
                attrs = []
                for cat, val in cats.items():
                    if cat != "name":
                        # Format category name nicely
                        cat_display = {
                            "beruf": "Beruf",
                            "stadt": "Stadt",
                            "haustier": "Haustier",
                            "hobby": "Hobby",
                            "farbe": "Farbe",
                            "getraenk": "Getraenk",
                        }.get(cat, cat.capitalize())
                        attrs.append(f"{cat_display}: {val.capitalize()}")

                if attrs:
                    lines.append(f"{name.capitalize()}: " + ", ".join(attrs))
        else:
            # Classic Zebra format: by position (Haus 1, Haus 2, etc.)
            for pos in sorted(solution.assignments.keys()):
                cats = solution.assignments[pos]
                cat_strs = []
                for k, v in cats.items():
                    cat_strs.append(f"{k.capitalize()}={v.capitalize()}")
                lines.append(f"Haus {pos}: " + ", ".join(cat_strs))

        return "\n".join(lines) if lines else "Keine Loesung gefunden."

    def _enhance_proof_tree(
        self,
        proof_tree: ProofTree,
        solution: ZebraSolution,
        puzzle: ZebraPuzzle,
    ):
        """Add solution details to proof tree."""
        # Add solution steps
        for pos in sorted(solution.assignments.keys()):
            cats = solution.assignments[pos]
            step = ProofStep(
                step_id=f"solution_pos_{pos}",
                step_type=StepType.CONCLUSION,
                inputs=[f"Position {pos}"],
                output=", ".join(f"{k}={v}" for k, v in cats.items()),
                confidence=1.0,
                explanation_text=f"Haus {pos}: "
                + ", ".join(
                    f"{k.capitalize()}={v.capitalize()}" for k, v in cats.items()
                ),
                source_component="zebra_puzzle_solver",
            )
            proof_tree.add_root_step(step)

    def _detect_ordering_puzzle(self, text: str) -> bool:
        """
        Detect if puzzle is a transitive ordering puzzle.

        Ordering puzzles have:
        - Comparison patterns: "A ist groesser/kleiner/schneller/aelter als B"
        - Questions about extremes: "groesste/kleinste Person"
        - Questions about order: "Reihenfolge"

        Works for any transitive comparison (groesser, schneller, aelter, etc.)

        Args:
            text: Puzzle text

        Returns:
            True if this is an ordering puzzle
        """
        text_lower = text.lower()

        # Strong indicators for ordering puzzle
        ordering_indicators = 0

        # Check for comparison patterns (any transitive property)
        comparison_patterns = [
            r"\bist\s+(?:groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter|staerker|schwaecker|hoeher|niedriger|laenger|kuerzer)\s+als\b",
        ]
        comparison_matches = sum(
            len(re.findall(p, text_lower)) for p in comparison_patterns
        )
        if comparison_matches >= 3:
            ordering_indicators += 3
        elif comparison_matches >= 1:
            ordering_indicators += 1

        # Check for superlative questions
        superlative_patterns = [
            r"(?:groesste|kleinste|schnellste|langsamste|aelteste|juengste)",
            r"(?:schwerste|leichteste|staerkste|schwaechste)",
            r"(?:hoechste|niedrigste|laengste|kuerzeste)",
        ]
        superlative_matches = sum(
            1 for p in superlative_patterns if re.search(p, text_lower)
        )
        if superlative_matches >= 1:
            ordering_indicators += 2

        # Check for order/ranking questions
        order_patterns = [
            r"reihenfolge",
            r"sortier",
            r"ordne",
            r"rang",
        ]
        order_matches = sum(1 for p in order_patterns if re.search(p, text_lower))
        if order_matches >= 1:
            ordering_indicators += 2

        # Check for "unterschiedliche" + measurable attribute
        if re.search(
            r"unterschiedliche\s+(?:koerpergroessen|groessen|alter|geschwindigkeiten)",
            text_lower,
        ):
            ordering_indicators += 2

        is_ordering = ordering_indicators >= 4

        if is_ordering:
            logger.info(
                f"[Ordering Puzzle] Detected ordering puzzle "
                f"(score={ordering_indicators}): "
                f"{comparison_matches} comparisons, "
                f"{superlative_matches} superlatives"
            )

        return is_ordering

    def _parse_ordering_entities(self, text: str) -> List[str]:
        """
        Extract entity names from ordering puzzle text.

        Looks for capitalized names in comparison patterns.

        Args:
            text: Puzzle text

        Returns:
            List of unique entity names
        """
        entities = set()

        # Pattern: "X ist groesser/kleiner als Y"
        comparison_pattern = r"(\b[A-Z][a-z]+\b)\s+ist\s+(?:groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter|staerker|schwaecker|hoeher|niedriger|laenger|kuerzer)\s+als\s+(\b[A-Z][a-z]+\b)"
        for match in re.finditer(comparison_pattern, text):
            entities.add(match.group(1))
            entities.add(match.group(2))

        # Also extract from compound patterns: "X ist kleiner als A, aber groesser als B"
        compound_pattern = r"(\b[A-Z][a-z]+\b)\s+ist\s+(?:groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter)\s+als\s+(\b[A-Z][a-z]+\b),?\s+aber\s+(?:groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter)\s+als\s+(\b[A-Z][a-z]+\b)"
        for match in re.finditer(compound_pattern, text):
            entities.add(match.group(1))
            entities.add(match.group(2))
            entities.add(match.group(3))

        # Filter out common German words that might be capitalized
        excluded = {
            "Person",
            "Personen",
            "Frage",
            "Fragen",
            "Hinweis",
            "Hinweise",
            "Gegeben",
            "Zusaetzliche",
            "Reihenfolge",
            "Koerpergroessen",
        }
        entities = {e for e in entities if e not in excluded}

        logger.debug(
            f"[Ordering Puzzle] Extracted {len(entities)} entities: {entities}"
        )
        return list(entities)

    def _parse_ordering_constraints(
        self, text: str, entities: List[str]
    ) -> List[ZebraConstraint]:
        """
        Parse ordering/comparison constraints from puzzle text.

        Handles patterns like:
        - "X ist groesser als Y" -> GREATER_THAN(X, Y)
        - "X ist kleiner als Y" -> LESS_THAN(X, Y)
        - "X ist kleiner als A, aber groesser als B" -> two constraints

        Works for any transitive comparison attribute.

        Args:
            text: Puzzle text
            entities: List of entity names

        Returns:
            List of ZebraConstraint objects
        """
        constraints = []
        entities_lower = {e.lower() for e in entities}

        # Split into lines for processing
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove leading numbers (e.g., "1. ", "12. ")
            line_clean = re.sub(r"^\d+\.\s*", "", line)
            line_lower = line_clean.lower()

            # Pattern 1: Simple comparison "X ist groesser/kleiner als Y"
            match = re.search(
                r"(\b\w+\b)\s+ist\s+(groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter|staerker|schwaecker|hoeher|niedriger|laenger|kuerzer)\s+als\s+(\b\w+\b)",
                line_lower,
            )
            if match:
                subj = match.group(1)
                comparison = match.group(2)
                obj = match.group(3)

                # Validate entities
                if subj in entities_lower and obj in entities_lower:
                    # Determine constraint type based on comparison word
                    if comparison in (
                        "groesser",
                        "schneller",
                        "aelter",
                        "schwerer",
                        "staerker",
                        "hoeher",
                        "laenger",
                    ):
                        constraint_type = ZebraConstraintType.GREATER_THAN
                    else:  # kleiner, langsamer, juenger, leichter, etc.
                        constraint_type = ZebraConstraintType.LESS_THAN

                    constraints.append(
                        ZebraConstraint(
                            constraint_type=constraint_type,
                            values=[subj, obj],
                            original_text=line_clean,
                        )
                    )

            # Pattern 2: Compound comparison "X ist kleiner als A, aber groesser als B"
            compound_match = re.search(
                r"(\b\w+\b)\s+ist\s+(groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter)\s+als\s+(\b\w+\b),?\s+aber\s+(groesser|kleiner|schneller|langsamer|aelter|juenger|schwerer|leichter)\s+als\s+(\b\w+\b)",
                line_lower,
            )
            if compound_match:
                subj = compound_match.group(1)
                comp1 = compound_match.group(2)
                obj1 = compound_match.group(3)
                comp2 = compound_match.group(4)
                obj2 = compound_match.group(5)

                # First comparison
                if subj in entities_lower and obj1 in entities_lower:
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
                            ZebraConstraint(
                                constraint_type=ZebraConstraintType.GREATER_THAN,
                                values=[subj, obj1],
                                original_text=line_clean,
                            )
                        )
                    else:
                        constraints.append(
                            ZebraConstraint(
                                constraint_type=ZebraConstraintType.LESS_THAN,
                                values=[subj, obj1],
                                original_text=line_clean,
                            )
                        )

                # Second comparison
                if subj in entities_lower and obj2 in entities_lower:
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
                            ZebraConstraint(
                                constraint_type=ZebraConstraintType.GREATER_THAN,
                                values=[subj, obj2],
                                original_text=line_clean,
                            )
                        )
                    else:
                        constraints.append(
                            ZebraConstraint(
                                constraint_type=ZebraConstraintType.LESS_THAN,
                                values=[subj, obj2],
                                original_text=line_clean,
                            )
                        )

        logger.info(f"[Ordering Puzzle] Parsed {len(constraints)} ordering constraints")
        return constraints

    def _format_ordering_answer(
        self,
        solution: ZebraSolution,
        question: Optional[str],
        puzzle: ZebraPuzzle,
    ) -> str:
        """
        Format answer for ordering puzzle questions.

        Handles:
        - "Wer ist die groesste Person?" -> entity at position 1
        - "Wer ist die kleinste Person?" -> entity at position N
        - "Ist A groesser als B?" -> compare positions
        - "Reihenfolge" -> "A > B > C > ..."

        Args:
            solution: CSP solution
            question: Original question
            puzzle: Puzzle definition

        Returns:
            Formatted answer string
        """
        if not question:
            # Default: show complete ordering
            return self._format_complete_ordering(solution, puzzle)

        question_lower = question.lower()

        # Build entity -> position mapping from solution
        entity_positions = {}
        for pos, cats in solution.assignments.items():
            if "person" in cats:
                entity_positions[cats["person"]] = pos

        # Check for specific question types
        # "Wer ist die groesste Person?"
        if re.search(
            r"(?:groesste|schnellste|aelteste|schwerste|staerkste)", question_lower
        ):
            # Position 1 = greatest
            for entity, pos in entity_positions.items():
                if pos == 1:
                    return f"Die groesste Person ist {entity.capitalize()}."
            return "Konnte die groesste Person nicht bestimmen."

        # "Wer ist die kleinste Person?"
        if re.search(
            r"(?:kleinste|langsamste|juengste|leichteste|schwaechste)", question_lower
        ):
            # Position N = smallest
            max_pos = max(entity_positions.values()) if entity_positions else 0
            for entity, pos in entity_positions.items():
                if pos == max_pos:
                    return f"Die kleinste Person ist {entity.capitalize()}."
            return "Konnte die kleinste Person nicht bestimmen."

        # "Ist A groesser als B?"
        comparison_match = re.search(
            r"ist\s+(\w+)\s+(?:groesser|kleiner|schneller|langsamer|aelter|juenger)\s+als\s+(\w+)",
            question_lower,
        )
        if comparison_match:
            entity_a = comparison_match.group(1)
            entity_b = comparison_match.group(2)
            pos_a = None
            pos_b = None

            for entity, pos in entity_positions.items():
                if entity.lower() == entity_a:
                    pos_a = pos
                if entity.lower() == entity_b:
                    pos_b = pos

            if pos_a is not None and pos_b is not None:
                is_greater = pos_a < pos_b  # Lower position = greater
                if (
                    "groesser" in question_lower
                    or "schneller" in question_lower
                    or "aelter" in question_lower
                ):
                    if is_greater:
                        return f"Ja, {entity_a.capitalize()} ist groesser als {entity_b.capitalize()}."
                    else:
                        return f"Nein, {entity_a.capitalize()} ist nicht groesser als {entity_b.capitalize()}."
                else:  # kleiner, langsamer, juenger
                    if not is_greater:
                        return f"Ja, {entity_a.capitalize()} ist kleiner als {entity_b.capitalize()}."
                    else:
                        return f"Nein, {entity_a.capitalize()} ist nicht kleiner als {entity_b.capitalize()}."

        # "Reihenfolge" or "vollstaendige Reihenfolge"
        if "reihenfolge" in question_lower or "ordnung" in question_lower:
            return self._format_complete_ordering(solution, puzzle)

        # Default: show complete ordering
        return self._format_complete_ordering(solution, puzzle)

    def _format_complete_ordering(
        self,
        solution: ZebraSolution,
        puzzle: ZebraPuzzle,
    ) -> str:
        """
        Format complete ordering from solution.

        Args:
            solution: CSP solution
            puzzle: Puzzle definition

        Returns:
            String like "A > B > C > D > E"
        """
        # Build position -> entity mapping
        position_entities = {}
        for pos, cats in solution.assignments.items():
            if "person" in cats:
                position_entities[pos] = cats["person"]

        if not position_entities:
            return "Keine Reihenfolge gefunden."

        # Sort by position and create chain
        ordered = [
            position_entities[pos].capitalize()
            for pos in sorted(position_entities.keys())
        ]

        return "Reihenfolge: " + " > ".join(ordered)

    def _detect_multi_attribute_format(self, text: str) -> bool:
        """
        Detect if puzzle uses multi-attribute format (Namen:, Berufe:, etc.)

        Multi-attribute format has explicit category declarations like:
        - "Namen: Anna, Ben, Clara, David, Emma"
        - "Berufe: Lehrer, Arzt, Ingenieur, Koch, Anwalt"

        Args:
            text: Puzzle text

        Returns:
            True if multi-attribute format detected
        """
        text_lower = text.lower()

        # Check for explicit category declarations
        multi_attr_patterns = [
            r"namen\s*:\s*\w+",
            r"berufe?\s*:\s*\w+",
            r"staedte\s*:\s*\w+",
            r"haustiere?\s*:\s*\w+",
            r"hobbys?\s*:\s*\w+",
        ]

        matches = sum(1 for p in multi_attr_patterns if re.search(p, text_lower))
        return matches >= 3  # At least 3 category declarations

    def _parse_multi_attribute_domains(
        self, text: str
    ) -> Tuple[int, Dict[str, List[str]]]:
        """
        Parse multi-attribute format domains.

        Extracts categories from patterns like:
        "Namen: Anna, Ben, Clara, David, Emma"
        "Berufe: Lehrer, Arzt, Ingenieur, Koch, Anwalt"

        Args:
            text: Puzzle text

        Returns:
            Tuple of (num_entities, categories_dict)
        """
        categories: Dict[str, List[str]] = {}

        # Category patterns with their normalized names
        category_patterns = [
            (r"namen\s*:\s*([^\n]+)", "name"),
            (r"berufe?\s*:\s*([^\n]+)", "beruf"),
            (r"staedte\s*:\s*([^\n]+)", "stadt"),
            (r"haustiere?\s*:\s*([^\n]+)", "haustier"),
            (r"hobbys?\s*:\s*([^\n]+)", "hobby"),
            # Additional categories for flexibility
            (r"farben?\s*:\s*([^\n]+)", "farbe"),
            (r"getraenke?\s*:\s*([^\n]+)", "getraenk"),
        ]

        for pattern, cat_name in category_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values_str = match.group(1).strip()
                # Split on comma, clean up whitespace
                values = [v.strip() for v in values_str.split(",")]
                # Remove any empty values
                values = [v for v in values if v]
                if values:
                    categories[cat_name] = values
                    logger.debug(f"Parsed category '{cat_name}': {len(values)} values")

        # Determine number of entities from the primary category (name or first found)
        num_entities = 5  # Default
        if "name" in categories:
            num_entities = len(categories["name"])
        elif categories:
            # Use first category's size
            first_cat = list(categories.keys())[0]
            num_entities = len(categories[first_cat])

        logger.info(
            f"Parsed multi-attribute domains: {num_entities} entities, "
            f"{len(categories)} categories: {list(categories.keys())}"
        )

        return num_entities, categories

    def _parse_multi_attribute_constraints(
        self,
        text: str,
        categories: Dict[str, List[str]],
    ) -> List[ZebraConstraint]:
        """
        Parse constraints for multi-attribute puzzle format.

        Handles patterns like:
        - "Anna ist Lehrerin" -> SAME_ENTITY(anna, lehrerin)
        - "Die Person aus Berlin hat einen Hund" -> SAME_ENTITY(berlin, hund)
        - "Clara ist nicht die Anwaeltin" -> DIFFERENT_ENTITY(clara, anwaeltin)
        - "Der Arzt wohnt in Hamburg" -> SAME_ENTITY(arzt, hamburg)
        - "Die Person mit dem Vogel treibt Sport" -> SAME_ENTITY(vogel, sport)

        Args:
            text: Puzzle text
            categories: Dictionary of category -> values

        Returns:
            List of ZebraConstraint objects
        """
        constraints = []

        # Build value lookup: value (lowercased) -> category
        value_to_category: Dict[str, str] = {}
        for cat_name, values in categories.items():
            for val in values:
                value_to_category[val.lower()] = cat_name
                # Also add common variants
                if val.lower().endswith("in"):
                    # Lehrerin -> Lehrer
                    value_to_category[val.lower()[:-2]] = cat_name
                if val.lower().endswith("er"):
                    # Lehrer -> Lehrerin
                    value_to_category[val.lower() + "in"] = cat_name

        # Create normalized value lookup (handles German declension)
        # Maps declined forms to base forms
        self._value_normalize_map = self._build_value_normalize_map(categories)

        # Split into hints/lines
        lines = re.split(r"[\n]", text)

        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Remove hint numbers at start (e.g., "1. ", "12. ")
            line = re.sub(r"^\d+\.\s*", "", line)
            line_lower = line.lower()

            # Try each pattern
            constraint = self._try_multi_attr_patterns(
                line, line_lower, categories, value_to_category
            )
            if constraint:
                constraints.append(constraint)

        logger.info(f"Parsed {len(constraints)} multi-attribute constraints")
        return constraints

    def _build_value_normalize_map(
        self, categories: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """
        Build normalization map for German word variants.

        Handles:
        - Feminine forms: Lehrerin -> lehrer
        - Plural forms: Hunde -> hund
        - Case endings: einen, einem, einer -> base

        Returns:
            Dict mapping variants to normalized base forms
        """
        normalize_map = {}

        for cat_name, values in categories.items():
            for val in values:
                base = val.lower()
                normalize_map[base] = base

                # Add common German variants
                # Feminine forms (-in suffix)
                if base.endswith("in"):
                    normalize_map[base[:-2]] = base  # Lehrer -> Lehrerin
                else:
                    normalize_map[base + "in"] = base  # Arzt -> Aerztin (maps to arzt)

                # Accusative/Dative article forms
                if base.startswith("d"):
                    continue  # Skip articles

                # Common plural endings
                if base.endswith("e"):
                    normalize_map[base + "n"] = base  # Katze -> Katzen
                elif not base.endswith("s"):
                    normalize_map[base + "e"] = base  # Hund -> Hunde

        return normalize_map

    def _try_multi_attr_patterns(
        self,
        line: str,
        line_lower: str,
        categories: Dict[str, List[str]],
        value_to_category: Dict[str, str],
    ) -> Optional[ZebraConstraint]:
        """
        Try matching multi-attribute constraint patterns.

        Args:
            line: Original line text
            line_lower: Lowercased line text
            categories: Category definitions
            value_to_category: Value to category mapping

        Returns:
            ZebraConstraint if pattern matched, None otherwise
        """
        # Pattern 1: "X ist nicht Y" (negation) - CHECK FIRST before positive pattern!
        # "Clara ist nicht die Anwaeltin" -> DIFFERENT_ENTITY(clara, anwaeltin)
        match = re.search(
            r"(\w+)\s+ist\s+nicht\s+(?:der\s+|die\s+|das\s+)?(\w+)",
            line_lower,
        )
        if match:
            subj = match.group(1)
            obj = match.group(2)
            val1 = self._normalize_multi_attr_value(subj, categories)
            val2 = self._normalize_multi_attr_value(obj, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.DIFFERENT_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 2: "X ist Y" / "X ist Yin" (direct assignment)
        # "Anna ist Lehrerin" -> SAME_ENTITY(anna, lehrerin)
        match = re.search(
            r"(\w+)\s+ist\s+(?:der\s+|die\s+|das\s+)?(\w+)",
            line_lower,
        )
        if match:
            subj = match.group(1)
            obj = match.group(2)
            val1 = self._normalize_multi_attr_value(subj, categories)
            val2 = self._normalize_multi_attr_value(obj, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.SAME_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 3: "Die Person aus X hat Y" (location -> attribute)
        # "Die Person aus Berlin hat einen Hund" -> SAME_ENTITY(berlin, hund)
        match = re.search(
            r"(?:die\s+)?person\s+aus\s+(\w+)\s+hat\s+(?:einen?\s+|eine\s+)?(\w+)",
            line_lower,
        )
        if match:
            location = match.group(1)
            attr = match.group(2)
            val1 = self._normalize_multi_attr_value(location, categories)
            val2 = self._normalize_multi_attr_value(attr, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.SAME_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 4: "Der/Die X wohnt in Y" (profession/role -> location)
        # "Der Arzt wohnt in Hamburg" -> SAME_ENTITY(arzt, hamburg)
        match = re.search(
            r"(?:der|die)\s+(\w+)\s+wohnt\s+in\s+(\w+)",
            line_lower,
        )
        if match:
            role = match.group(1)
            location = match.group(2)
            val1 = self._normalize_multi_attr_value(role, categories)
            val2 = self._normalize_multi_attr_value(location, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.SAME_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 5: "Die Person mit X macht/treibt Y" (attribute -> hobby)
        # "Die Person mit dem Vogel treibt Sport" -> SAME_ENTITY(vogel, sport)
        match = re.search(
            r"(?:die\s+)?person\s+mit\s+(?:dem\s+|der\s+)?(\w+)\s+(?:treibt|macht|mag)\s+(\w+)",
            line_lower,
        )
        if match:
            attr = match.group(1)
            hobby = match.group(2)
            val1 = self._normalize_multi_attr_value(attr, categories)
            val2 = self._normalize_multi_attr_value(hobby, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.SAME_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 6: "X wohnt nicht in Y" (negation for location)
        # "Emma wohnt nicht in Frankfurt" -> DIFFERENT_ENTITY(emma, frankfurt)
        match = re.search(
            r"(\w+)\s+wohnt\s+nicht\s+in\s+(\w+)",
            line_lower,
        )
        if match:
            name = match.group(1)
            location = match.group(2)
            val1 = self._normalize_multi_attr_value(name, categories)
            val2 = self._normalize_multi_attr_value(location, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.DIFFERENT_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 7: "X mag Y als Hobby" (name -> hobby)
        # "Ben mag Lesen als Hobby" -> SAME_ENTITY(ben, lesen)
        match = re.search(
            r"(\w+)\s+mag\s+(\w+)\s+(?:als\s+hobby)?",
            line_lower,
        )
        if match:
            name = match.group(1)
            hobby = match.group(2)
            val1 = self._normalize_multi_attr_value(name, categories)
            val2 = self._normalize_multi_attr_value(hobby, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.SAME_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 8: "Die Person aus X Y gerne" (location -> hobby with verb)
        # "Die Person aus Muenchen malt gerne" -> SAME_ENTITY(muenchen, malen)
        match = re.search(
            r"(?:die\s+)?person\s+aus\s+(\w+)\s+(\w+)\s+gerne",
            line_lower,
        )
        if match:
            location = match.group(1)
            hobby_verb = match.group(2)
            val1 = self._normalize_multi_attr_value(location, categories)
            # Convert verb to noun form for hobby
            val2 = self._normalize_multi_attr_value(hobby_verb, categories)
            # Try infinitive form if verb form not found
            if not val2 and hobby_verb.endswith("t"):
                val2 = self._normalize_multi_attr_value(
                    hobby_verb[:-1] + "en", categories
                )
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.SAME_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        # Pattern 9: "Der X hat Y" (profession -> attribute)
        # "Der Ingenieur hat eine Katze" -> SAME_ENTITY(ingenieur, katze)
        match = re.search(
            r"(?:der|die)\s+(\w+)\s+hat\s+(?:einen?\s+|eine\s+)?(\w+)",
            line_lower,
        )
        if match:
            role = match.group(1)
            attr = match.group(2)
            val1 = self._normalize_multi_attr_value(role, categories)
            val2 = self._normalize_multi_attr_value(attr, categories)
            if val1 and val2:
                return ZebraConstraint(
                    constraint_type=ZebraConstraintType.SAME_ENTITY,
                    values=[val1, val2],
                    original_text=line,
                )

        return None

    def _normalize_multi_attr_value(
        self,
        value: str,
        categories: Dict[str, List[str]],
    ) -> Optional[str]:
        """
        Normalize a value to match a category value.

        Handles German word variants (feminine forms, plurals, umlauts, etc.)

        Args:
            value: Value to normalize
            categories: Category definitions

        Returns:
            Normalized value (lowercase) or None if not found
        """
        value_lower = value.lower().strip()

        # Normalize umlauts: ae -> a, oe -> o, ue -> u (and vice versa)
        # Also handle real umlauts to ae/oe/ue
        umlaut_variants = self._get_umlaut_variants(value_lower)

        # Direct match check (including umlaut variants)
        for val_variant in [value_lower] + umlaut_variants:
            for cat_values in categories.values():
                for cat_value in cat_values:
                    cv_lower = cat_value.lower()
                    if cv_lower == val_variant:
                        return cv_lower

        # Try with normalization map
        if hasattr(self, "_value_normalize_map"):
            for val_variant in [value_lower] + umlaut_variants:
                if val_variant in self._value_normalize_map:
                    normalized = self._value_normalize_map[val_variant]
                    # Verify normalized value exists in categories
                    for cat_values in categories.values():
                        if normalized in [v.lower() for v in cat_values]:
                            return normalized

        # Try prefix/suffix matching for variants (including umlaut variants)
        for val_variant in [value_lower] + umlaut_variants:
            for cat_values in categories.values():
                for cat_value in cat_values:
                    cv_lower = cat_value.lower()

                    # Check for feminine form (-in suffix)
                    if val_variant.endswith("in") and cv_lower == val_variant[:-2]:
                        return cv_lower
                    if cv_lower.endswith("in") and val_variant == cv_lower[:-2]:
                        return cv_lower
                    # Same base with -in
                    if val_variant + "in" == cv_lower or cv_lower + "in" == val_variant:
                        return cv_lower

                    # Check for verb -> noun (malt -> malen)
                    if (
                        val_variant.endswith("t")
                        and cv_lower == val_variant[:-1] + "en"
                    ):
                        return cv_lower
                    if cv_lower.endswith("en") and val_variant == cv_lower[:-2] + "t":
                        return cv_lower

                    # Check for plural forms
                    if val_variant + "e" == cv_lower or cv_lower + "e" == val_variant:
                        return cv_lower
                    if val_variant + "n" == cv_lower or cv_lower + "n" == val_variant:
                        return cv_lower

        return None

    def _get_umlaut_variants(self, value: str) -> List[str]:
        """
        Generate umlaut variants for a German word.

        Converts between:
        - ae <-> a (Anwaeltin -> Anwaltin)
        - oe <-> o (Koeln -> Koln)
        - ue <-> u (Muenchen -> Munchen)

        Args:
            value: Lowercase value

        Returns:
            List of variant forms
        """
        variants = []

        # ae -> a variant
        if "ae" in value:
            variants.append(value.replace("ae", "a"))
        # a -> ae variant (less common but possible)
        if "a" in value and "ae" not in value:
            variants.append(value.replace("a", "ae"))

        # oe -> o variant
        if "oe" in value:
            variants.append(value.replace("oe", "o"))

        # ue -> u variant
        if "ue" in value:
            variants.append(value.replace("ue", "u"))

        return variants

    def _parse_puzzle_text(
        self, text: str, question: Optional[str] = None
    ) -> Optional[ZebraPuzzle]:
        """
        Parse German Zebra puzzle text into ZebraPuzzle structure.

        Supports three formats:
        1. Ordering Puzzle: Transitive comparison (A > B > C)
        2. Classic Zebra: Nationalities, house colors, drinks, pets, cigarettes
        3. Multi-attribute: Namen:, Berufe:, Staedte:, etc.

        Extracts:
        - Number of entities (houses/people)
        - Categories and their values
        - Constraints from hints

        Returns:
            ZebraPuzzle instance or None if parsing fails
        """
        logger.info("Parsing Zebra puzzle text")

        # Check for ordering puzzle FIRST (highest priority)
        if self._detect_ordering_puzzle(text):
            logger.info("Detected ORDERING puzzle format")

            # Extract entities from comparison patterns
            entities = self._parse_ordering_entities(text)

            if len(entities) < 2:
                logger.warning(
                    f"Ordering puzzle detected but only {len(entities)} "
                    "entities found - insufficient"
                )
                return None

            num_entities = len(entities)

            # Create single "person" category with all entities
            categories: Dict[str, List[str]] = {"person": [e.lower() for e in entities]}

            # Parse ordering constraints
            constraints = self._parse_ordering_constraints(text, entities)

            logger.info(
                f"Parsed ordering puzzle: {num_entities} entities, "
                f"{len(constraints)} constraints"
            )

            return ZebraPuzzle(
                num_entities=num_entities,
                categories=categories,
                constraints=constraints,
                question=question,
                original_text=text,
            )

        # Check for multi-attribute format second
        if self._detect_multi_attribute_format(text):
            logger.info("Detected multi-attribute puzzle format")
            num_entities, categories = self._parse_multi_attribute_domains(text)

            if len(categories) < 2:
                logger.warning(
                    f"Multi-attribute format detected but only {len(categories)} "
                    "categories found - insufficient"
                )
                return None

            # Parse multi-attribute constraints
            constraints = self._parse_multi_attribute_constraints(text, categories)

            logger.info(
                f"Parsed multi-attribute puzzle: {num_entities} entities, "
                f"{len(categories)} categories, {len(constraints)} constraints"
            )

            return ZebraPuzzle(
                num_entities=num_entities,
                categories=categories,
                constraints=constraints,
                question=question,
                original_text=text,
            )

        # Classic Zebra puzzle format follows...
        # Detect number of entities (default: 5 for classic Zebra)
        num_match = re.search(
            r"(?:fuenf|fünf|5)\s+(?:haeuser|häuser|personen|leute)",
            text.lower(),
        )
        num_entities = 5  # Default

        # Define categories based on classic Zebra puzzle structure
        # The parser looks for mentions in the text to build value lists
        categories: Dict[str, List[str]] = {}

        # Nationality patterns (German Zebra puzzle)
        nationalities = self._extract_category_values(
            text,
            [
                "brite",
                "brit",
                "englaender",
                "schwede",
                "schwedisch",
                "daene",
                "dänisch",
                "daenisch",
                "norweger",
                "norwegisch",
                "deutsche",
                "deutscher",
                "deutsch",
            ],
            normalize_map={
                "brite": "brit",
                "brit": "brit",
                "englaender": "brit",
                "schwede": "schwede",
                "schwedisch": "schwede",
                "daene": "daene",
                "dänisch": "daene",
                "daenisch": "daene",
                "norweger": "norweger",
                "norwegisch": "norweger",
                "deutsche": "deutscher",
                "deutscher": "deutscher",
                "deutsch": "deutscher",
            },
        )
        if len(nationalities) >= 3:
            categories["nationality"] = list(nationalities)[:num_entities]

        # House colors
        colors = self._extract_category_values(
            text,
            [
                "rot",
                "roten",
                "rotes",
                "gruen",
                "gruenen",
                "grünen",
                "grün",
                "weiss",
                "weissen",
                "weißen",
                "weiß",
                "gelb",
                "gelben",
                "gelbes",
                "blau",
                "blauen",
                "blaues",
            ],
            normalize_map={
                "rot": "rot",
                "roten": "rot",
                "rotes": "rot",
                "gruen": "gruen",
                "gruenen": "gruen",
                "grünen": "gruen",
                "grün": "gruen",
                "weiss": "weiss",
                "weissen": "weiss",
                "weißen": "weiss",
                "weiß": "weiss",
                "gelb": "gelb",
                "gelben": "gelb",
                "gelbes": "gelb",
                "blau": "blau",
                "blauen": "blau",
                "blaues": "blau",
            },
        )
        if len(colors) >= 3:
            categories["color"] = list(colors)[:num_entities]

        # Drinks
        drinks = self._extract_category_values(
            text,
            [
                "tee",
                "kaffee",
                "milch",
                "bier",
                "wasser",
            ],
        )
        if len(drinks) >= 3:
            categories["drink"] = list(drinks)[:num_entities]

        # Pets
        pets = self._extract_category_values(
            text,
            [
                "hund",
                "hunde",
                "vogel",
                "voegel",
                "katze",
                "katzen",
                "pferd",
                "pferde",
                "zebra",
                "zebras",
            ],
            normalize_map={
                "hund": "hund",
                "hunde": "hund",
                "vogel": "vogel",
                "voegel": "vogel",
                "katze": "katze",
                "katzen": "katze",
                "pferd": "pferd",
                "pferde": "pferd",
                "zebra": "zebra",
                "zebras": "zebra",
            },
        )
        if len(pets) >= 3:
            categories["pet"] = list(pets)[:num_entities]

        # Cigarettes/Brands
        cigarettes = self._extract_category_values(
            text,
            [
                "pall mall",
                "pallmall",
                "dunhill",
                "blend",
                "blue master",
                "bluemaster",
                "prince",
            ],
            normalize_map={
                "pall mall": "pallmall",
                "pallmall": "pallmall",
                "dunhill": "dunhill",
                "blend": "blend",
                "blue master": "bluemaster",
                "bluemaster": "bluemaster",
                "prince": "prince",
            },
        )
        if len(cigarettes) >= 3:
            categories["cigarette"] = list(cigarettes)[:num_entities]

        # Ensure we have at least 2 categories
        if len(categories) < 2:
            logger.warning(f"Only found {len(categories)} categories - insufficient")
            return None

        # Pad categories to num_entities if needed (with placeholders)
        for cat_name, values in categories.items():
            while len(values) < num_entities:
                values.append(f"unknown_{cat_name}_{len(values)+1}")

        # Parse constraints from hints
        constraints = self._parse_constraints(text, categories)

        logger.info(
            f"Parsed puzzle: {num_entities} entities, "
            f"{len(categories)} categories, {len(constraints)} constraints"
        )

        return ZebraPuzzle(
            num_entities=num_entities,
            categories=categories,
            constraints=constraints,
            question=question,
            original_text=text,
        )

    def _extract_category_values(
        self,
        text: str,
        patterns: List[str],
        normalize_map: Optional[Dict[str, str]] = None,
    ) -> Set[str]:
        """Extract unique values for a category from text."""
        text_lower = text.lower()
        found = set()

        for pattern in patterns:
            if pattern in text_lower:
                normalized = pattern
                if normalize_map and pattern in normalize_map:
                    normalized = normalize_map[pattern]
                found.add(normalized)

        return found

    def _parse_constraints(
        self,
        text: str,
        categories: Dict[str, List[str]],
    ) -> List[ZebraConstraint]:
        """Parse constraint hints from puzzle text."""
        constraints = []
        all_values = set()
        for values in categories.values():
            all_values.update(v.lower() for v in values)

        # Split into lines/sentences for hint processing
        lines = re.split(r"[.\n]", text)

        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            line_lower = line.lower()

            # Pattern 1: "Der X lebt im Y Haus" -> SAME_ENTITY(X, Y)
            # "Der Brite lebt im roten Haus"
            match = re.search(
                r"der\s+(\w+)\s+lebt\s+im\s+(\w+)(?:en)?\s+haus",
                line_lower,
            )
            if match:
                nationality = self._normalize_value(match.group(1), categories)
                color = self._normalize_value(match.group(2), categories)
                if nationality and color:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[nationality, color],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 2: "Der X hat einen Y" -> SAME_ENTITY(X, Y)
            # "Der Schwede hat einen Hund"
            match = re.search(
                r"der\s+(\w+)\s+hat\s+(?:einen?|ein)\s+(\w+)",
                line_lower,
            )
            if match:
                nationality = self._normalize_value(match.group(1), categories)
                pet = self._normalize_value(match.group(2), categories)
                if nationality and pet:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[nationality, pet],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 3: "Der X trinkt Y" -> SAME_ENTITY(X, Y)
            # "Der Daene trinkt Tee"
            match = re.search(
                r"der\s+(\w+)\s+trinkt\s+(\w+)",
                line_lower,
            )
            if match:
                nationality = self._normalize_value(match.group(1), categories)
                drink = self._normalize_value(match.group(2), categories)
                if nationality and drink:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[nationality, drink],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 4: "Das X Haus steht direkt links vom Y Haus"
            # -> DIRECTLY_LEFT_OF(X, Y)
            match = re.search(
                r"das\s+(\w+)e?\s+haus\s+steht\s+direkt\s+links\s+vom?\s+(\w+)(?:en)?\s+haus",
                line_lower,
            )
            if match:
                color1 = self._normalize_value(match.group(1), categories)
                color2 = self._normalize_value(match.group(2), categories)
                if color1 and color2:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.DIRECTLY_LEFT_OF,
                            values=[color1, color2],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 5: "Der Besitzer des X Hauses trinkt/raucht Y"
            # -> SAME_ENTITY(X, Y)
            match = re.search(
                r"(?:der\s+)?besitzer\s+des\s+(\w+)(?:en)?\s+hauses\s+(?:trinkt|raucht)\s+(\w+)",
                line_lower,
            )
            if match:
                color = self._normalize_value(match.group(1), categories)
                item = self._normalize_value(match.group(2), categories)
                if color and item:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[color, item],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 6: "Die Person, die X raucht, hat einen Y"
            # -> SAME_ENTITY(X, Y)
            match = re.search(
                r"die\s+person,?\s+die\s+(\w+(?:\s+\w+)?)\s+raucht,?\s+hat\s+(?:einen?|ein)\s+(\w+)",
                line_lower,
            )
            if match:
                cigarette = self._normalize_value(match.group(1), categories)
                pet = self._normalize_value(match.group(2), categories)
                if cigarette and pet:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[cigarette, pet],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 7: "Der Besitzer des X Hauses raucht Y"
            # -> SAME_ENTITY(X, Y)
            match = re.search(
                r"(?:der\s+)?besitzer\s+des\s+(\w+)(?:en)?\s+hauses\s+raucht\s+(\w+)",
                line_lower,
            )
            if match:
                color = self._normalize_value(match.group(1), categories)
                cigarette = self._normalize_value(match.group(2), categories)
                if color and cigarette:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[color, cigarette],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 8: "Die Person im mittleren Haus trinkt X"
            # -> AT_POSITION(X, 3)  (middle of 5)
            match = re.search(
                r"(?:die\s+)?person\s+im\s+mittleren\s+haus\s+trinkt\s+(\w+)",
                line_lower,
            )
            if match:
                drink = self._normalize_value(match.group(1), categories)
                if drink:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.AT_POSITION,
                            values=[drink],
                            position=3,  # Middle of 5
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 9: "Der X lebt im ersten Haus"
            # -> AT_POSITION(X, 1)
            match = re.search(
                r"der\s+(\w+)\s+lebt\s+im\s+ersten\s+haus",
                line_lower,
            )
            if match:
                nationality = self._normalize_value(match.group(1), categories)
                if nationality:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.AT_POSITION,
                            values=[nationality],
                            position=1,
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 10: "Die Person, die X raucht, lebt neben der Person mit Y"
            # -> ADJACENT_TO(X, Y)
            match = re.search(
                r"die\s+person,?\s+die\s+(\w+(?:\s+\w+)?)\s+raucht,?\s+lebt\s+neben\s+(?:der\s+person\s+mit\s+(?:der|dem)?\s*)?(\w+)",
                line_lower,
            )
            if match:
                cigarette = self._normalize_value(match.group(1), categories)
                item = self._normalize_value(match.group(2), categories)
                if cigarette and item:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.ADJACENT_TO,
                            values=[cigarette, item],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 11: "Die Person mit dem X lebt neben der Person, die Y raucht"
            # -> ADJACENT_TO(X, Y)
            match = re.search(
                r"die\s+person\s+mit\s+(?:dem|der)?\s*(\w+)\s+lebt\s+neben\s+(?:der\s+person,?\s+die\s+)?(\w+(?:\s+\w+)?)\s+raucht",
                line_lower,
            )
            if match:
                pet = self._normalize_value(match.group(1), categories)
                cigarette = self._normalize_value(match.group(2), categories)
                if pet and cigarette:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.ADJACENT_TO,
                            values=[pet, cigarette],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 12: "Die Person, die X raucht, trinkt Y"
            # -> SAME_ENTITY(X, Y)
            match = re.search(
                r"die\s+person,?\s+die\s+(\w+(?:\s+\w+)?)\s+raucht,?\s+trinkt\s+(\w+)",
                line_lower,
            )
            if match:
                cigarette = self._normalize_value(match.group(1), categories)
                drink = self._normalize_value(match.group(2), categories)
                if cigarette and drink:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[cigarette, drink],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 13: "Der X raucht Y"
            # -> SAME_ENTITY(X, Y)
            match = re.search(
                r"der\s+(\w+)\s+raucht\s+(\w+(?:\s+\w+)?)",
                line_lower,
            )
            if match:
                nationality = self._normalize_value(match.group(1), categories)
                cigarette = self._normalize_value(match.group(2), categories)
                if nationality and cigarette:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.SAME_ENTITY,
                            values=[nationality, cigarette],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 14: "Der X lebt neben dem Y Haus"
            # -> ADJACENT_TO(X, Y)
            match = re.search(
                r"der\s+(\w+)\s+lebt\s+neben\s+dem\s+(\w+)(?:en)?\s+haus",
                line_lower,
            )
            if match:
                nationality = self._normalize_value(match.group(1), categories)
                color = self._normalize_value(match.group(2), categories)
                if nationality and color:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.ADJACENT_TO,
                            values=[nationality, color],
                            original_text=line,
                        )
                    )
                    continue

            # Pattern 15: "Die Person, die X raucht, hat einen Nachbarn, der Y trinkt"
            # -> ADJACENT_TO(X, Y)
            match = re.search(
                r"die\s+person,?\s+die\s+(\w+(?:\s+\w+)?)\s+raucht,?\s+hat\s+einen?\s+nachbarn?,?\s+(?:der|die)\s+(\w+)\s+trinkt",
                line_lower,
            )
            if match:
                cigarette = self._normalize_value(match.group(1), categories)
                drink = self._normalize_value(match.group(2), categories)
                if cigarette and drink:
                    constraints.append(
                        ZebraConstraint(
                            constraint_type=ZebraConstraintType.ADJACENT_TO,
                            values=[cigarette, drink],
                            original_text=line,
                        )
                    )
                    continue

        logger.info(f"Parsed {len(constraints)} constraints from text")
        return constraints

    def _normalize_value(
        self,
        value: str,
        categories: Dict[str, List[str]],
    ) -> Optional[str]:
        """
        Normalize a value to match category values.

        Handles German declension and variations.
        """
        value_lower = value.lower().strip()

        # Direct match
        for cat_values in categories.values():
            for cat_value in cat_values:
                if cat_value.lower() == value_lower:
                    return cat_value.lower()

        # Partial match (prefix)
        for cat_values in categories.values():
            for cat_value in cat_values:
                cv_lower = cat_value.lower()
                # Check if value is a prefix or suffix of cat_value
                if cv_lower.startswith(value_lower) or value_lower.startswith(cv_lower):
                    return cv_lower

        # Special cases for German declension
        normalize_map = {
            # Nationalities
            "brite": "brit",
            "briten": "brit",
            "schwede": "schwede",
            "schweden": "schwede",
            "daene": "daene",
            "daenen": "daene",
            "dänisch": "daene",
            "norweger": "norweger",
            "norwegers": "norweger",
            "deutsche": "deutscher",
            "deutschen": "deutscher",
            # Colors (remove -en suffix)
            "roten": "rot",
            "rotes": "rot",
            "gruenen": "gruen",
            "grünen": "gruen",
            "weissen": "weiss",
            "weißen": "weiss",
            "gelben": "gelb",
            "gelbes": "gelb",
            "blauen": "blau",
            "blaues": "blau",
            # Cigarettes
            "pall mall": "pallmall",
            "pall": "pallmall",
            "blue master": "bluemaster",
            "blue": "bluemaster",
        }

        if value_lower in normalize_map:
            normalized = normalize_map[value_lower]
            # Check if normalized value exists in categories
            for cat_values in categories.values():
                if normalized in [v.lower() for v in cat_values]:
                    return normalized

        return None


# Convenience function for direct usage
def solve_zebra_puzzle(text: str, question: Optional[str] = None) -> Dict[str, Any]:
    """
    Solve a Zebra puzzle from German text.

    Args:
        text: German puzzle text with hints
        question: Optional question to answer

    Returns:
        Solution dictionary (see ZebraPuzzleSolver.solve())
    """
    solver = ZebraPuzzleSolver()
    return solver.solve_from_text(text, question)

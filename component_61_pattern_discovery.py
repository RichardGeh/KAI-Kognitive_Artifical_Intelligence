# component_61_pattern_discovery.py
"""
Pattern Discovery Engine for autonomous pattern learning.

This module implements clustering-based pattern discovery and template induction:
1. Embedding-based clustering of utterances (Phase 5)
2. LGG-based template induction from clusters (Phase 6)
3. Pattern refinement through usage (Phase 7)

Responsibilities:
- Cluster similar utterances using agglomerative clustering
- Induce templates from clusters via Least General Generalization (LGG)
- Create Pattern/PatternItem/Slot nodes in Neo4j
- Update pattern centroids for embedding-based scoring

Part of Pattern Discovery System - Part 3: Clustering & Template Induction.
"""

import threading
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from component_15_logging_config import get_logger
from kai_exceptions import DatabaseException, wrap_exception

logger = get_logger(__name__)


class PatternDiscoveryEngine:
    """
    Autonomous pattern learning via clustering and template induction.

    Thread Safety:
        Uses RLock for clustering operations to prevent race conditions.
        All database operations use parameterized queries (security).

    Performance:
        - Batch operations for token retrieval (prevents N+1)
        - Normalized centroids for cosine similarity
        - Cache invalidation after pattern creation

    Attributes:
        netzwerk: KonzeptNetzwerk for database access
        _clustering_lock: Thread lock for clustering operations
    """

    def __init__(self, netzwerk):
        """
        Initialize pattern discovery engine.

        Args:
            netzwerk: KonzeptNetzwerk instance
        """
        self.netzwerk = netzwerk
        self._clustering_lock = threading.RLock()

        logger.debug("PatternDiscoveryEngine initialized")

    def cluster_utterances(
        self, min_cluster_size: int = 3, similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Cluster utterances via embedding similarity.

        Uses agglomerative clustering with Euclidean distance on normalized
        embeddings (approximates cosine similarity). Filters small clusters
        and computes normalized centroids for each cluster.

        Args:
            min_cluster_size: Minimum cluster size (prevents overfitting)
            similarity_threshold: Cosine similarity threshold (0.85 = high similarity)

        Returns:
            List of clusters: [
                {
                    "cluster_id": int,
                    "utterance_ids": List[str],
                    "centroid": List[float],  # 384D normalized
                    "size": int
                },
                ...
            ]

        Raises:
            ValueError: If parameters are invalid
            DatabaseException: If database query fails

        Example:
            >>> clusters = engine.cluster_utterances(min_cluster_size=3, similarity_threshold=0.85)
            >>> for cluster in clusters:
            ...     print(f"Cluster {cluster['cluster_id']}: {cluster['size']} utterances")
        """
        # Input validation
        if min_cluster_size < 2:
            raise ValueError("Minimum cluster size must be >= 2")
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be in (0.0, 1.0]")

        # CRITICAL: Lock to prevent concurrent clustering (code review fix)
        with self._clustering_lock:
            try:
                # 1. Fetch non-archived utterances efficiently
                # Uses composite index (timestamp, archived) for fast filtering
                utterances = self.netzwerk.get_recent_utterances(
                    limit=1000, archived=False
                )

            except Exception as e:
                raise wrap_exception(
                    e, DatabaseException, "Failed to fetch utterances for clustering"
                )

        if len(utterances) < min_cluster_size:
            logger.info(
                f"Insufficient utterances for clustering: "
                f"{len(utterances)} < {min_cluster_size}"
            )
            return []

        # 2. Extract embeddings for clustering (filter out invalid)
        embeddings = []
        valid_utterances = []  # Only utterances with valid embeddings
        for u in utterances:
            embedding = u.get("embedding")
            if embedding and len(embedding) == 384:
                embeddings.append(embedding)
                valid_utterances.append(u)
            else:
                logger.warning(
                    f"Utterance {u.get('id', 'unknown')[:8]} missing valid embedding"
                )

        if len(embeddings) < min_cluster_size:
            logger.warning(
                f"Not enough valid embeddings: {len(embeddings)} < {min_cluster_size}"
            )
            return []

        embeddings_array = np.array(embeddings)
        utterances = valid_utterances  # Use only valid utterances from now on

        # Normalize embeddings for consistent distance calculation
        # This enables: euclidean_distance = sqrt(2 * (1 - cosine_similarity))
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        # Handle zero-norm embeddings (avoid division by zero -> NaN)
        zero_norm_mask = norms.flatten() == 0
        if np.any(zero_norm_mask):
            logger.warning(
                f"Found {np.sum(zero_norm_mask)} zero-norm centroid, filtering them out"
            )
            valid_mask = ~zero_norm_mask
            embeddings_array = embeddings_array[valid_mask]
            utterances = [u for i, u in enumerate(utterances) if valid_mask[i]]
            if len(utterances) < min_cluster_size:
                logger.warning(
                    f"Not enough valid embeddings after filtering zero-norms: {len(utterances)} < {min_cluster_size}"
                )
                return []
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)

        embeddings_array = embeddings_array / norms

        # 3. Agglomerative clustering with distance threshold
        # For normalized embeddings: euclidean_distance = sqrt(2 * (1 - cosine_similarity))
        # Use 2.0x multiplier for average linkage to account for within-cluster variance
        # (empirically determined to handle realistic embedding noise)
        distance_threshold = 2.0 * np.sqrt(2 * (1 - similarity_threshold))

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,  # Auto-determine based on threshold
                distance_threshold=distance_threshold,
                linkage="average",  # More stable than single/complete
                metric="euclidean",  # Normalized embeddings: cosine ~= euclidean
            )

            labels = clustering.fit_predict(embeddings_array)

            # Debug: log cluster label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            logger.debug(
                f"Clustering result: {len(unique_labels)} unique labels, "
                f"distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}"
            )

        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Clustering algorithm failed",
                n_utterances=len(embeddings),
            )

        # 4. Group utterances by cluster labels
        clusters_map = defaultdict(list)
        for utt, label in zip(utterances, labels):
            clusters_map[label].append(utt)

        # 5. Filter small clusters + compute centroids
        valid_clusters = []
        for cluster_id, cluster_utt in clusters_map.items():
            if len(cluster_utt) >= min_cluster_size:
                # Compute centroid efficiently
                cluster_embeddings = [u["embedding"] for u in cluster_utt]
                centroid = np.mean(cluster_embeddings, axis=0)

                # Normalize centroid for cosine similarity consistency
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                else:
                    logger.warning(f"Cluster {cluster_id} has zero-norm centroid")
                    continue

                valid_clusters.append(
                    {
                        "cluster_id": int(cluster_id),
                        "utterance_ids": [u["id"] for u in cluster_utt],
                        "centroid": centroid.tolist(),
                        "size": len(cluster_utt),
                    }
                )

        logger.info(
            f"Clustering: {len(valid_clusters)} clusters from "
            f"{len(utterances)} utterances "
            f"(threshold={similarity_threshold}, min_size={min_cluster_size})"
        )

        return valid_clusters

    def induce_template_from_cluster(self, cluster: Dict[str, Any]) -> str:
        """
        Create template from cluster via Least General Generalization (LGG).

        Algorithm:
        1. Fetch token sequences for all utterances (batch operation)
        2. Apply LGG algorithm to find common structure
        3. Create Pattern node with type="learned"
        4. Create PatternItems (LITERAL/SLOT)
        5. Create Slots with allowed values
        6. Set initial stats (support=cluster_size, precision=0.5)
        7. Set pattern centroid from cluster
        8. Invalidate matcher cache

        Args:
            cluster: Cluster dict with keys:
                - utterance_ids: List of utterance UUIDs
                - centroid: 384D normalized embedding
                - size: Number of utterances

        Returns:
            Pattern ID (UUID) of newly created pattern

        Raises:
            ValueError: If cluster data is invalid
            DatabaseException: If database operations fail

        Example:
            >>> cluster = {
            ...     "utterance_ids": ["uuid1", "uuid2", "uuid3"],
            ...     "centroid": [0.1, 0.2, ...],  # 384D
            ...     "size": 3
            ... }
            >>> pattern_id = engine.induce_template_from_cluster(cluster)
        """
        # Input validation
        if not cluster or not isinstance(cluster, dict):
            raise ValueError("Cluster must be a dict")

        utterance_ids = cluster.get("utterance_ids", [])
        if not utterance_ids or len(utterance_ids) < 2:
            raise ValueError("Cluster must have at least 2 utterances")

        centroid = cluster.get("centroid")
        if not centroid or len(centroid) != 384:
            raise ValueError("Cluster centroid must be 384D")

        # 1. Fetch token sequences for all utterances
        # CRITICAL: Use batch method to prevent N+1 query problem (code review fix)
        try:
            tokens_by_utterance = self.netzwerk.get_tokens_for_utterances_batch(
                utterance_ids
            )
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to fetch tokens for cluster",
                n_utterances=len(utterance_ids),
            )

        # Convert to token sequences
        token_sequences = [
            tokens_by_utterance[uid]
            for uid in utterance_ids
            if uid in tokens_by_utterance
        ]

        if not token_sequences:
            raise ValueError("No token sequences found for cluster utterances")

        # 2. Apply LGG algorithm to find template structure
        template_items = self._lgg_multiway(token_sequences)

        if not template_items:
            logger.warning(
                f"LGG produced empty template for cluster {cluster.get('cluster_id')}"
            )
            raise ValueError("LGG failed to produce template")

        # 3. Create Pattern node
        pattern_name = f"Learned_Pattern_{uuid.uuid4().hex[:8]}"

        try:
            pattern_id = self.netzwerk.create_pattern(
                name=pattern_name, pattern_type="learned"
            )
        except Exception as e:
            raise wrap_exception(
                e, DatabaseException, "Failed to create pattern node", name=pattern_name
            )

        # 4. Create PatternItems & Slots
        try:
            for idx, item in enumerate(template_items):
                if item["type"] == "LITERAL":
                    # Create LITERAL PatternItem
                    self.netzwerk.create_pattern_item(
                        pattern_id=pattern_id,
                        idx=idx,
                        kind="LITERAL",
                        literal_value=item["value"],
                    )

                else:  # SLOT
                    # Create Slot with allowed values
                    allowed_lemmas = item.get("allowed_lemmas", [])
                    slot_id = self.netzwerk.create_slot(
                        slot_type=f"SLOT_{idx}",  # Generic name
                        allowed_values=allowed_lemmas,
                    )

                    # Create SLOT PatternItem
                    self.netzwerk.create_pattern_item(
                        pattern_id=pattern_id, idx=idx, kind="SLOT", slot_id=slot_id
                    )

        except Exception as e:
            # Clean up pattern on failure
            logger.error(
                f"Failed to create pattern items, cleaning up pattern {pattern_id}"
            )
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to create pattern items",
                pattern_id=pattern_id,
            )

        # 5. Set initial statistics
        try:
            self.netzwerk.update_pattern_stats(
                pattern_id=pattern_id,
                support_increment=len(utterance_ids),  # All utterances in cluster
                new_precision=0.5,  # Initial uncertainty
            )
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to update pattern stats",
                pattern_id=pattern_id,
            )

        # 6. Set pattern centroid (for embedding-based scoring)
        try:
            self.netzwerk.update_pattern_centroid(
                pattern_id=pattern_id, centroid=centroid
            )
        except Exception as e:
            raise wrap_exception(
                e,
                DatabaseException,
                "Failed to update pattern centroid",
                pattern_id=pattern_id,
            )

        # 7. CRITICAL: Invalidate candidate cache after pattern creation (code review fix)
        # Fail fast if cache invalidation fails - newly created pattern won't appear
        # in candidates until TTL expires (5 minutes)
        if hasattr(self.netzwerk, "pattern_matcher"):
            self.netzwerk.pattern_matcher._invalidate_candidate_cache()

        logger.info(
            f"New pattern created: {pattern_name} from {len(utterance_ids)} utterances "
            f"({len(template_items)} items)"
        )

        return pattern_id

    def _lgg_multiway(
        self, token_sequences: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Compute Least General Generalization (LGG) for multiple token sequences.

        Algorithm:
        1. Filter sequences to same length (most common length)
        2. For each position:
           - If all tokens have same lemma: LITERAL
           - If tokens differ: SLOT with allowed_lemmas
        3. Return template items

        Future enhancements (Phase 2):
        - Dynamic alignment with Needleman-Wunsch (allows gaps)
        - POS-based generalization (e.g., SLOT[POS=NOUN])
        - Hierarchical slots (e.g., SLOT[WH_WORD])

        Args:
            token_sequences: List of token sequences, where each token is a dict
                            with keys: 'lemma', 'pos', 'surface', etc.

        Returns:
            List of template items:
            [
                {"type": "LITERAL", "value": "was"},
                {"type": "SLOT", "allowed_lemmas": ["ist", "sind"], "counts": {...}},
                ...
            ]

        Example:
            >>> tokens1 = [{"lemma": "was"}, {"lemma": "ist"}, {"lemma": "hund"}]
            >>> tokens2 = [{"lemma": "was"}, {"lemma": "sind"}, {"lemma": "katze"}]
            >>> template = engine._lgg_multiway([tokens1, tokens2])
            >>> # Result: [LITERAL("was"), SLOT(ist|sind), SLOT(hund|katze)]
        """
        if not token_sequences:
            logger.warning("LGG called with empty token sequences")
            return []

        # 1. Filter sequences with same length (Alignment-Problem otherwise too complex)
        # Future: Implement dynamic alignment with gaps
        lengths = Counter([len(seq) for seq in token_sequences])
        most_common_length, count = lengths.most_common(1)[0]

        # Warn if no dominant sequence length (< 50% have same length)
        alignment_ratio = count / len(token_sequences)
        if alignment_ratio < 0.5:
            logger.warning(
                f"No dominant sequence length in cluster: "
                f"best={count}/{len(token_sequences)} ({alignment_ratio:.1%}). "
                f"Consider dynamic alignment for heterogeneous utterances."
            )

        aligned_sequences = [
            seq for seq in token_sequences if len(seq) == most_common_length
        ]

        if not aligned_sequences:
            logger.warning("No sequences with matching length for LGG")
            return []

        if len(aligned_sequences) < len(token_sequences):
            logger.debug(
                f"LGG: Filtered {len(token_sequences) - len(aligned_sequences)} "
                f"sequences with non-standard length (keeping length={most_common_length})"
            )

        # 2. Position-by-position comparison
        template_items = []
        num_positions = most_common_length

        for pos_idx in range(num_positions):
            tokens_at_pos = [seq[pos_idx] for seq in aligned_sequences]

            # For LITERALs, we need surface forms (exact match)
            # For SLOTs, we need lemmas (normalized)
            lemmas_at_pos = []
            surfaces_at_pos = []

            for t in tokens_at_pos:
                lemma = t.get("lemma", "")
                surface = t.get("surface", "")
                pos = t.get("pos", "")

                # Skip tokens with truly missing/invalid lemmas
                # "unknown" is our fallback for empty lemmas - skip these positions entirely
                if not lemma or not surface or lemma == "unknown":
                    continue

                # For punctuation, use surface for both (lemma is often '--')
                if lemma in ("--", "-") or pos == "PUNCT":
                    lemmas_at_pos.append(surface.lower())
                    surfaces_at_pos.append(surface)
                else:
                    lemmas_at_pos.append(lemma)
                    surfaces_at_pos.append(surface)

            if not lemmas_at_pos:
                # No valid values at this position - skip
                logger.warning(f"LGG: No valid lemmas at position {pos_idx}")
                continue

            # All same lemma? -> LITERAL (use surface form for exact match)
            if len(set(lemmas_at_pos)) == 1:
                # For literals, use the most common surface form (handles case variations)
                surface_counts = Counter(surf.lower() for surf in surfaces_at_pos)
                most_common_surface = surface_counts.most_common(1)[0][0]

                template_items.append({"type": "LITERAL", "value": most_common_surface})

            # Different? -> SLOT
            else:
                # Count frequencies for slot allowed values (use lemmas for normalization)
                lemma_counts = Counter(lemmas_at_pos)
                template_items.append(
                    {
                        "type": "SLOT",
                        "allowed_lemmas": list(lemma_counts.keys()),
                        "counts": dict(lemma_counts),
                    }
                )

        logger.debug(
            f"LGG: Induced template with {len(template_items)} items "
            f"from {len(aligned_sequences)} sequences (length={most_common_length})"
        )

        return template_items

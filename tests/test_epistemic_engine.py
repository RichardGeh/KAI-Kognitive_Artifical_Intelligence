"""
tests/test_epistemic_engine.py

Comprehensive test suite for component_35_epistemic_engine.py

Tests cover:
- ModalOperator enum functionality
- Proposition dataclass
- Agent dataclass
- EpistemicState dataclass
- MetaProposition dataclass
- Default values and field factories
- Edge cases and validation

Autor: KAI Development Team
Erstellt: 2025-11-01
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from component_35_epistemic_engine import (
    ModalOperator,
    Proposition,
    Agent,
    EpistemicState,
    MetaProposition,
    EpistemicEngine,
)
from component_1_netzwerk import KonzeptNetzwerk


class TestModalOperator:
    """Test ModalOperator enum"""

    def test_modal_operator_values(self):
        """Verify all modal operator values"""
        assert ModalOperator.KNOWS.value == "K"
        assert ModalOperator.BELIEVES.value == "M"
        assert ModalOperator.EVERYONE_KNOWS.value == "E"
        assert ModalOperator.COMMON.value == "C"

    def test_modal_operator_count(self):
        """Verify exactly 4 modal operators exist"""
        assert len(ModalOperator) == 4

    def test_modal_operator_membership(self):
        """Test enum membership checks"""
        assert ModalOperator.KNOWS in ModalOperator
        assert ModalOperator.BELIEVES in ModalOperator
        assert ModalOperator.EVERYONE_KNOWS in ModalOperator
        assert ModalOperator.COMMON in ModalOperator

    def test_modal_operator_iteration(self):
        """Test iterating over all operators"""
        operators = list(ModalOperator)
        assert len(operators) == 4
        assert ModalOperator.KNOWS in operators


class TestProposition:
    """Test Proposition dataclass"""

    def test_proposition_creation_minimal(self):
        """Create proposition with minimal required fields"""
        prop = Proposition(id="p1", content="has_blue_eyes")
        assert prop.id == "p1"
        assert prop.content == "has_blue_eyes"
        assert prop.truth_value is None

    def test_proposition_creation_with_truth_value(self):
        """Create proposition with explicit truth value"""
        prop_true = Proposition(id="p2", content="is_human", truth_value=True)
        assert prop_true.truth_value is True

        prop_false = Proposition(id="p3", content="is_robot", truth_value=False)
        assert prop_false.truth_value is False

    def test_proposition_truth_value_optional(self):
        """Verify truth_value defaults to None"""
        prop = Proposition(id="p4", content="unknown_fact")
        assert prop.truth_value is None

    def test_proposition_modification(self):
        """Test modifying proposition after creation"""
        prop = Proposition(id="p5", content="mutable")
        assert prop.truth_value is None

        prop.truth_value = True
        assert prop.truth_value is True

        prop.content = "modified_content"
        assert prop.content == "modified_content"


class TestAgent:
    """Test Agent dataclass"""

    def test_agent_creation_minimal(self):
        """Create agent with minimal required fields"""
        agent = Agent(id="a1", name="Alice")
        assert agent.id == "a1"
        assert agent.name == "Alice"
        assert agent.reasoning_capacity == 5  # default
        assert agent.knowledge == set()  # empty default

    def test_agent_creation_with_custom_capacity(self):
        """Create agent with custom reasoning capacity"""
        agent = Agent(id="a2", name="Bob", reasoning_capacity=10)
        assert agent.reasoning_capacity == 10

    def test_agent_knowledge_set_independence(self):
        """Verify each agent gets independent knowledge set"""
        agent1 = Agent(id="a3", name="Charlie")
        agent2 = Agent(id="a4", name="Diana")

        agent1.knowledge.add("p1")
        assert "p1" in agent1.knowledge
        assert "p1" not in agent2.knowledge

    def test_agent_knowledge_initialization(self):
        """Test initializing agent with knowledge"""
        initial_knowledge = {"p1", "p2", "p3"}
        agent = Agent(id="a5", name="Eve", knowledge=initial_knowledge)
        assert agent.knowledge == initial_knowledge

    def test_agent_reasoning_capacity_edge_cases(self):
        """Test reasoning capacity with edge values"""
        agent_zero = Agent(id="a6", name="Zero", reasoning_capacity=0)
        assert agent_zero.reasoning_capacity == 0

        agent_high = Agent(id="a7", name="High", reasoning_capacity=100)
        assert agent_high.reasoning_capacity == 100


class TestEpistemicState:
    """Test EpistemicState dataclass"""

    def test_epistemic_state_creation_empty(self):
        """Create empty epistemic state"""
        state = EpistemicState(
            agents={},
            propositions={},
            knowledge_base={},
            meta_knowledge={},
            common_knowledge={},
        )
        assert state.agents == {}
        assert state.propositions == {}
        assert state.knowledge_base == {}
        assert state.meta_knowledge == {}
        assert state.common_knowledge == {}
        assert isinstance(state.timestamp, datetime)

    def test_epistemic_state_with_agents(self):
        """Create state with agents"""
        agent1 = Agent(id="a1", name="Alice")
        agent2 = Agent(id="a2", name="Bob")

        state = EpistemicState(
            agents={"a1": agent1, "a2": agent2},
            propositions={},
            knowledge_base={},
            meta_knowledge={},
            common_knowledge={},
        )
        assert len(state.agents) == 2
        assert state.agents["a1"].name == "Alice"
        assert state.agents["a2"].name == "Bob"

    def test_epistemic_state_with_propositions(self):
        """Create state with propositions"""
        prop1 = Proposition(id="p1", content="has_blue_eyes", truth_value=True)
        prop2 = Proposition(id="p2", content="is_tall", truth_value=False)

        state = EpistemicState(
            agents={},
            propositions={"p1": prop1, "p2": prop2},
            knowledge_base={},
            meta_knowledge={},
            common_knowledge={},
        )
        assert len(state.propositions) == 2
        assert state.propositions["p1"].truth_value is True
        assert state.propositions["p2"].truth_value is False

    def test_epistemic_state_knowledge_base(self):
        """Test knowledge base structure"""
        state = EpistemicState(
            agents={},
            propositions={},
            knowledge_base={"a1": {"p1", "p2"}, "a2": {"p3"}},
            meta_knowledge={},
            common_knowledge={},
        )
        assert state.knowledge_base["a1"] == {"p1", "p2"}
        assert state.knowledge_base["a2"] == {"p3"}

    def test_epistemic_state_meta_knowledge_structure(self):
        """Test nested meta-knowledge structure"""
        state = EpistemicState(
            agents={},
            propositions={},
            knowledge_base={},
            meta_knowledge={"a1": {1: {"m1", "m2"}, 2: {"m3"}}, "a2": {1: {"m4"}}},
            common_knowledge={},
        )
        assert state.meta_knowledge["a1"][1] == {"m1", "m2"}
        assert state.meta_knowledge["a1"][2] == {"m3"}
        assert state.meta_knowledge["a2"][1] == {"m4"}

    def test_epistemic_state_common_knowledge(self):
        """Test common knowledge by group"""
        state = EpistemicState(
            agents={},
            propositions={},
            knowledge_base={},
            meta_knowledge={},
            common_knowledge={"group1": {"p1", "p2"}, "group2": {"p3"}},
        )
        assert state.common_knowledge["group1"] == {"p1", "p2"}
        assert state.common_knowledge["group2"] == {"p3"}

    def test_epistemic_state_timestamp_automatic(self):
        """Verify timestamp is automatically set"""
        state = EpistemicState(
            agents={},
            propositions={},
            knowledge_base={},
            meta_knowledge={},
            common_knowledge={},
        )
        assert isinstance(state.timestamp, datetime)
        assert (datetime.now() - state.timestamp).total_seconds() < 1

    def test_epistemic_state_full_example(self):
        """Test complete epistemic state with all fields"""
        agent1 = Agent(id="a1", name="Alice", knowledge={"p1"})
        agent2 = Agent(id="a2", name="Bob", knowledge={"p2"})

        prop1 = Proposition(id="p1", content="sky_is_blue", truth_value=True)
        prop2 = Proposition(id="p2", content="grass_is_green", truth_value=True)

        state = EpistemicState(
            agents={"a1": agent1, "a2": agent2},
            propositions={"p1": prop1, "p2": prop2},
            knowledge_base={"a1": {"p1"}, "a2": {"p2"}},
            meta_knowledge={"a1": {1: {"m1"}}},
            common_knowledge={"all": {"p1", "p2"}},
        )

        assert len(state.agents) == 2
        assert len(state.propositions) == 2
        assert len(state.knowledge_base) == 2
        assert len(state.meta_knowledge) == 1
        assert len(state.common_knowledge) == 1


class TestMetaProposition:
    """Test MetaProposition dataclass"""

    def test_meta_proposition_creation_minimal(self):
        """Create meta-proposition with minimal fields"""
        meta = MetaProposition(
            id="m1",
            observer_id="a1",
            subject_id="a2",
            proposition_id="p1",
            meta_level=1,
        )
        assert meta.id == "m1"
        assert meta.observer_id == "a1"
        assert meta.subject_id == "a2"
        assert meta.proposition_id == "p1"
        assert meta.meta_level == 1
        assert meta.certainty == 1.0  # default

    def test_meta_proposition_with_certainty(self):
        """Create meta-proposition with custom certainty"""
        meta = MetaProposition(
            id="m2",
            observer_id="a1",
            subject_id="a2",
            proposition_id="p1",
            meta_level=2,
            certainty=0.75,
        )
        assert meta.certainty == 0.75

    def test_meta_proposition_level_1(self):
        """Test level 1: 'A knows that B knows P'"""
        meta = MetaProposition(
            id="m3",
            observer_id="alice",
            subject_id="bob",
            proposition_id="p1",
            meta_level=1,
        )
        assert meta.meta_level == 1
        # Represents: "Alice knows that Bob knows p1"

    def test_meta_proposition_level_2(self):
        """Test level 2: 'A knows that B knows that C knows P'"""
        meta = MetaProposition(
            id="m4",
            observer_id="alice",
            subject_id="bob",
            proposition_id="p1",
            meta_level=2,
        )
        assert meta.meta_level == 2
        # Represents: "Alice knows that Bob knows that [someone] knows p1"

    def test_meta_proposition_certainty_bounds(self):
        """Test certainty with boundary values"""
        meta_certain = MetaProposition(
            id="m5",
            observer_id="a1",
            subject_id="a2",
            proposition_id="p1",
            meta_level=1,
            certainty=1.0,
        )
        assert meta_certain.certainty == 1.0

        meta_uncertain = MetaProposition(
            id="m6",
            observer_id="a1",
            subject_id="a2",
            proposition_id="p1",
            meta_level=1,
            certainty=0.0,
        )
        assert meta_uncertain.certainty == 0.0

    def test_meta_proposition_high_meta_level(self):
        """Test high meta-level reasoning"""
        meta = MetaProposition(
            id="m7",
            observer_id="a1",
            subject_id="a2",
            proposition_id="p1",
            meta_level=5,
        )
        assert meta.meta_level == 5


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_complete_epistemic_scenario(self):
        """Test complete scenario: Blue Eyes Puzzle setup"""
        # Create agents
        alice = Agent(id="alice", name="Alice", reasoning_capacity=5)
        bob = Agent(id="bob", name="Bob", reasoning_capacity=5)
        charlie = Agent(id="charlie", name="Charlie", reasoning_capacity=5)

        # Create propositions
        p1 = Proposition(
            id="alice_blue", content="alice_has_blue_eyes", truth_value=True
        )
        p2 = Proposition(id="bob_blue", content="bob_has_blue_eyes", truth_value=True)
        p3 = Proposition(
            id="charlie_blue", content="charlie_has_blue_eyes", truth_value=True
        )

        # Initial knowledge: each sees others' eye color
        knowledge_base = {
            "alice": {"bob_blue", "charlie_blue"},
            "bob": {"alice_blue", "charlie_blue"},
            "charlie": {"alice_blue", "bob_blue"},
        }

        # Create epistemic state
        state = EpistemicState(
            agents={"alice": alice, "bob": bob, "charlie": charlie},
            propositions={"alice_blue": p1, "bob_blue": p2, "charlie_blue": p3},
            knowledge_base=knowledge_base,
            meta_knowledge={},
            common_knowledge={},
        )

        # Verify setup
        assert len(state.agents) == 3
        assert len(state.propositions) == 3
        assert "bob_blue" in state.knowledge_base["alice"]
        assert "alice_blue" not in state.knowledge_base["alice"]  # Can't see own eyes

    def test_meta_knowledge_chain(self):
        """Test chain of meta-knowledge"""
        # Create meta-propositions representing nested knowledge
        m1 = MetaProposition(
            id="m1",
            observer_id="a1",
            subject_id="a2",
            proposition_id="p1",
            meta_level=1,
        )  # "a1 knows that a2 knows p1"

        m2 = MetaProposition(
            id="m2",
            observer_id="a1",
            subject_id="a2",
            proposition_id="m1",
            meta_level=2,
        )  # "a1 knows that a2 knows that [meta about p1]"

        assert m1.meta_level == 1
        assert m2.meta_level == 2
        assert m2.proposition_id == "m1"  # Points to another meta-proposition

    def test_common_knowledge_group(self):
        """Test common knowledge in a group"""
        state = EpistemicState(
            agents={
                "a1": Agent(id="a1", name="Alice"),
                "a2": Agent(id="a2", name="Bob"),
                "a3": Agent(id="a3", name="Charlie"),
            },
            propositions={"p1": Proposition(id="p1", content="public_announcement")},
            knowledge_base={
                "a1": {"p1"},
                "a2": {"p1"},
                "a3": {"p1"},
            },
            meta_knowledge={},
            common_knowledge={"all": {"p1"}},
        )

        # Verify everyone knows p1
        assert "p1" in state.knowledge_base["a1"]
        assert "p1" in state.knowledge_base["a2"]
        assert "p1" in state.knowledge_base["a3"]

        # Verify it's common knowledge
        assert "p1" in state.common_knowledge["all"]


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_agent_knowledge(self):
        """Test agent with no knowledge"""
        agent = Agent(id="a1", name="Ignorant")
        assert len(agent.knowledge) == 0

    def test_proposition_none_values(self):
        """Test proposition with None values where allowed"""
        prop = Proposition(id="p1", content="test", truth_value=None)
        assert prop.truth_value is None

    def test_meta_proposition_same_observer_subject(self):
        """Test meta-proposition where observer == subject (self-knowledge)"""
        meta = MetaProposition(
            id="m1",
            observer_id="a1",
            subject_id="a1",  # Same as observer
            proposition_id="p1",
            meta_level=1,
        )
        assert meta.observer_id == meta.subject_id

    def test_large_knowledge_base(self):
        """Test epistemic state with many propositions"""
        propositions = {
            f"p{i}": Proposition(id=f"p{i}", content=f"fact_{i}") for i in range(100)
        }
        knowledge_base = {"a1": {f"p{i}" for i in range(100)}}

        state = EpistemicState(
            agents={"a1": Agent(id="a1", name="Knowledgeable")},
            propositions=propositions,
            knowledge_base=knowledge_base,
            meta_knowledge={},
            common_knowledge={},
        )

        assert len(state.propositions) == 100
        assert len(state.knowledge_base["a1"]) == 100

    def test_deep_meta_levels(self):
        """Test very deep meta-knowledge levels"""
        meta = MetaProposition(
            id="m_deep",
            observer_id="a1",
            subject_id="a2",
            proposition_id="p1",
            meta_level=10,
        )
        assert meta.meta_level == 10


class TestEpistemicEngine:
    """Test EpistemicEngine class"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Create mock KonzeptNetzwerk for testing"""
        mock = Mock(spec=KonzeptNetzwerk)
        mock.create_agent = Mock(return_value=True)
        return mock

    @pytest.fixture
    def engine(self, mock_netzwerk):
        """Create EpistemicEngine with mock netzwerk"""
        return EpistemicEngine(mock_netzwerk)

    def test_engine_initialization(self, mock_netzwerk):
        """Test EpistemicEngine initialization"""
        engine = EpistemicEngine(mock_netzwerk)

        assert engine.netzwerk == mock_netzwerk
        assert engine.current_state is None
        assert isinstance(engine._proposition_cache, dict)
        assert isinstance(engine._agent_cache, dict)
        assert len(engine._proposition_cache) == 0
        assert len(engine._agent_cache) == 0

    def test_engine_initialization_with_real_netzwerk(self):
        """Test EpistemicEngine with real KonzeptNetzwerk"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)

        assert engine.netzwerk is not None
        assert isinstance(engine.netzwerk, KonzeptNetzwerk)
        assert engine.current_state is None

    def test_create_agent_basic(self, engine, mock_netzwerk):
        """Test create_agent with basic parameters"""
        agent = engine.create_agent("alice", "Alice")

        # Verify agent created
        assert agent.id == "alice"
        assert agent.name == "Alice"
        assert agent.reasoning_capacity == 5  # default

        # Verify agent in cache
        assert "alice" in engine._agent_cache
        assert engine._agent_cache["alice"] == agent

        # Verify netzwerk.create_agent was called
        mock_netzwerk.create_agent.assert_called_once_with("alice", "Alice", 5)

    def test_create_agent_custom_capacity(self, engine, mock_netzwerk):
        """Test create_agent with custom reasoning capacity"""
        agent = engine.create_agent("bob", "Bob", reasoning_capacity=10)

        assert agent.reasoning_capacity == 10

        # Verify correct parameters passed to netzwerk
        mock_netzwerk.create_agent.assert_called_once_with("bob", "Bob", 10)

    def test_create_agent_cache_storage(self, engine):
        """Test that multiple agents are stored in cache"""
        agent1 = engine.create_agent("alice", "Alice")
        agent2 = engine.create_agent("bob", "Bob")
        agent3 = engine.create_agent("charlie", "Charlie", reasoning_capacity=3)

        assert len(engine._agent_cache) == 3
        assert engine._agent_cache["alice"] == agent1
        assert engine._agent_cache["bob"] == agent2
        assert engine._agent_cache["charlie"] == agent3

    def test_create_agent_overwrite_existing(self, engine):
        """Test creating agent with same ID overwrites cache"""
        agent1 = engine.create_agent("alice", "Alice", reasoning_capacity=5)
        agent2 = engine.create_agent("alice", "Alice Updated", reasoning_capacity=10)

        # Second agent should overwrite first
        assert len(engine._agent_cache) == 1
        assert engine._agent_cache["alice"] == agent2
        assert engine._agent_cache["alice"].name == "Alice Updated"
        assert engine._agent_cache["alice"].reasoning_capacity == 10

    def test_create_agent_return_value(self, engine):
        """Test create_agent returns Agent instance"""
        agent = engine.create_agent("test", "Test Agent")

        assert isinstance(agent, Agent)
        assert agent.id == "test"
        assert agent.name == "Test Agent"

    def test_load_state_from_graph_placeholder(self, engine):
        """Test load_state_from_graph returns None (TODO)"""
        result = engine.load_state_from_graph()
        assert result is None  # Placeholder implementation

    def test_persist_state_to_graph_placeholder(self, engine):
        """Test persist_state_to_graph returns None (TODO)"""
        state = EpistemicState(
            agents={},
            propositions={},
            knowledge_base={},
            meta_knowledge={},
            common_knowledge={},
        )
        result = engine.persist_state_to_graph(state)
        assert result is None  # Placeholder implementation

    def test_proposition_cache_empty_on_init(self, engine):
        """Test proposition cache is empty on initialization"""
        assert len(engine._proposition_cache) == 0
        assert isinstance(engine._proposition_cache, dict)

    def test_agent_cache_empty_on_init(self, engine):
        """Test agent cache is empty on initialization"""
        assert len(engine._agent_cache) == 0
        assert isinstance(engine._agent_cache, dict)

    def test_current_state_none_on_init(self, engine):
        """Test current_state is None on initialization"""
        assert engine.current_state is None

    def test_engine_with_real_netzwerk_create_agent(self):
        """Integration test: create agent with real netzwerk"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)

        agent = engine.create_agent(
            "integration_test", "Integration Test Agent", reasoning_capacity=7
        )

        # Verify agent created
        assert agent.id == "integration_test"
        assert agent.name == "Integration Test Agent"
        assert agent.reasoning_capacity == 7

        # Verify in cache
        assert "integration_test" in engine._agent_cache

    def test_multiple_engines_independent_caches(self, mock_netzwerk):
        """Test multiple engines have independent caches"""
        engine1 = EpistemicEngine(mock_netzwerk)
        engine2 = EpistemicEngine(mock_netzwerk)

        engine1.create_agent("alice", "Alice")
        engine2.create_agent("bob", "Bob")

        # Caches should be independent
        assert "alice" in engine1._agent_cache
        assert "alice" not in engine2._agent_cache
        assert "bob" in engine2._agent_cache
        assert "bob" not in engine1._agent_cache

    def test_create_agent_edge_case_empty_name(self, engine):
        """Test create_agent with empty name"""
        agent = engine.create_agent("id1", "")
        assert agent.name == ""
        assert agent.id == "id1"

    def test_create_agent_edge_case_zero_capacity(self, engine):
        """Test create_agent with zero reasoning capacity"""
        agent = engine.create_agent("id2", "Zero", reasoning_capacity=0)
        assert agent.reasoning_capacity == 0

    def test_create_agent_edge_case_high_capacity(self, engine):
        """Test create_agent with very high reasoning capacity"""
        agent = engine.create_agent("id3", "Genius", reasoning_capacity=1000)
        assert agent.reasoning_capacity == 1000


class TestKOperator:
    """Test K operator (Agent knows proposition)"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Create mock KonzeptNetzwerk for testing"""
        mock = Mock(spec=KonzeptNetzwerk)
        mock.create_agent = Mock(return_value=True)
        mock.add_belief = Mock(return_value=True)

        # Mock driver session for K operator queries
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"knows": True}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        mock.driver = mock_driver

        return mock

    @pytest.fixture
    def engine(self, mock_netzwerk):
        """Create EpistemicEngine with mock netzwerk"""
        return EpistemicEngine(mock_netzwerk)

    def test_K_operator_from_cache(self, engine):
        """Test K operator returns True from cache"""
        # Setup cache
        engine._ensure_state()
        engine.current_state.knowledge_base["alice"] = {"p1"}

        # Test
        result = engine.K("alice", "p1")
        assert result is True

    def test_K_operator_from_graph(self, engine, mock_netzwerk):
        """Test K operator queries graph when not in cache"""
        # Empty cache
        engine.current_state = None

        # Mock graph response
        mock_record = {"knows": True}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_netzwerk.driver.session.return_value = mock_session

        # Test
        result = engine.K("alice", "p1")
        assert result is True
        mock_session.run.assert_called_once()

    def test_K_operator_not_found(self, engine, mock_netzwerk):
        """Test K operator returns False when knowledge not found"""
        # Empty cache
        engine.current_state = None

        # Mock graph response: not found
        mock_record = {"knows": False}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_netzwerk.driver.session.return_value = mock_session

        # Test
        result = engine.K("bob", "unknown_prop")
        assert result is False

    def test_K_operator_cache_miss_then_graph(self, engine, mock_netzwerk):
        """Test K operator checks cache first, then graph"""
        # Setup cache with different proposition
        engine._ensure_state()
        engine.current_state.knowledge_base["alice"] = {"p2"}  # Different prop

        # Mock graph response
        mock_record = {"knows": True}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_netzwerk.driver.session.return_value = mock_session

        # Test - looking for p1, but only p2 in cache
        result = engine.K("alice", "p1")
        assert result is True
        mock_session.run.assert_called_once()

    def test_K_operator_no_record_returned(self, engine, mock_netzwerk):
        """Test K operator handles None record from graph"""
        # Empty cache
        engine.current_state = None

        # Mock graph response: no record
        mock_result = MagicMock()
        mock_result.single.return_value = None

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_netzwerk.driver.session.return_value = mock_session

        # Test
        result = engine.K("alice", "p1")
        assert result is False

    def test_K_operator_agent_not_in_cache(self, engine, mock_netzwerk):
        """Test K operator when agent not in knowledge_base"""
        # Setup cache with different agent
        engine._ensure_state()
        engine.current_state.knowledge_base["bob"] = {"p1"}

        # Mock graph response
        mock_record = {"knows": False}
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_netzwerk.driver.session.return_value = mock_session

        # Test - alice not in cache
        result = engine.K("alice", "p1")
        assert result is False


class TestAddKnowledge:
    """Test add_knowledge method"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Create mock KonzeptNetzwerk for testing"""
        mock = Mock(spec=KonzeptNetzwerk)
        mock.create_agent = Mock(return_value=True)
        mock.add_belief = Mock(return_value=True)
        return mock

    @pytest.fixture
    def engine(self, mock_netzwerk):
        """Create EpistemicEngine with mock netzwerk"""
        return EpistemicEngine(mock_netzwerk)

    def test_add_knowledge_basic(self, engine, mock_netzwerk):
        """Test basic add_knowledge functionality"""
        result = engine.add_knowledge("alice", "p1")

        # Verify cache updated
        assert "alice" in engine.current_state.knowledge_base
        assert "p1" in engine.current_state.knowledge_base["alice"]

        # Verify graph updated
        mock_netzwerk.add_belief.assert_called_once_with("alice", "p1", 1.0)
        assert result is True

    def test_add_knowledge_with_certainty(self, engine, mock_netzwerk):
        """Test add_knowledge with custom certainty"""
        result = engine.add_knowledge("bob", "p2", certainty=0.75)

        # Verify cache
        assert "p2" in engine.current_state.knowledge_base["bob"]

        # Verify graph with correct certainty
        mock_netzwerk.add_belief.assert_called_once_with("bob", "p2", 0.75)
        assert result is True

    def test_add_knowledge_initializes_state(self, engine):
        """Test add_knowledge initializes state if None"""
        # Ensure state is None
        engine.current_state = None

        engine.add_knowledge("alice", "p1")

        # State should be initialized
        assert engine.current_state is not None
        assert isinstance(engine.current_state, EpistemicState)

    def test_add_knowledge_multiple_propositions(self, engine):
        """Test adding multiple propositions to same agent"""
        engine.add_knowledge("alice", "p1")
        engine.add_knowledge("alice", "p2")
        engine.add_knowledge("alice", "p3")

        # Verify all in cache
        assert engine.current_state.knowledge_base["alice"] == {"p1", "p2", "p3"}

    def test_add_knowledge_multiple_agents(self, engine):
        """Test adding knowledge to multiple agents"""
        engine.add_knowledge("alice", "p1")
        engine.add_knowledge("bob", "p2")
        engine.add_knowledge("charlie", "p3")

        # Verify all agents
        assert "p1" in engine.current_state.knowledge_base["alice"]
        assert "p2" in engine.current_state.knowledge_base["bob"]
        assert "p3" in engine.current_state.knowledge_base["charlie"]

    def test_add_knowledge_duplicate_proposition(self, engine):
        """Test adding same proposition twice (set semantics)"""
        engine.add_knowledge("alice", "p1")
        engine.add_knowledge("alice", "p1")  # Duplicate

        # Should still only contain one instance
        assert engine.current_state.knowledge_base["alice"] == {"p1"}

    def test_add_knowledge_graph_failure(self, engine, mock_netzwerk):
        """Test add_knowledge when graph update fails"""
        mock_netzwerk.add_belief.return_value = False

        result = engine.add_knowledge("alice", "p1")

        # Cache still updated
        assert "p1" in engine.current_state.knowledge_base["alice"]

        # But return False
        assert result is False

    def test_add_knowledge_certainty_bounds(self, engine, mock_netzwerk):
        """Test add_knowledge with edge certainty values"""
        # Certainty 0.0
        engine.add_knowledge("alice", "p1", certainty=0.0)
        mock_netzwerk.add_belief.assert_called_with("alice", "p1", 0.0)

        # Certainty 1.0
        engine.add_knowledge("bob", "p2", certainty=1.0)
        mock_netzwerk.add_belief.assert_called_with("bob", "p2", 1.0)

    def test_add_knowledge_empty_string_ids(self, engine):
        """Test add_knowledge with empty string IDs"""
        engine.add_knowledge("", "")

        # Should still work (validation is caller's responsibility)
        assert "" in engine.current_state.knowledge_base
        assert "" in engine.current_state.knowledge_base[""]


class TestKOperatorIntegration:
    """Integration tests for K operator with real netzwerk"""

    @pytest.fixture
    def netzwerk(self):
        """Create real KonzeptNetzwerk for integration tests"""
        return KonzeptNetzwerk()

    @pytest.fixture
    def engine(self, netzwerk):
        """Create EpistemicEngine with real netzwerk"""
        return EpistemicEngine(netzwerk)

    def test_K_operator_end_to_end(self, engine):
        """Test K operator end-to-end with real database"""
        # Create agent
        engine.create_agent("alice_integration", "Alice")

        # Add knowledge
        success = engine.add_knowledge("alice_integration", "sky_is_blue")
        assert success is True

        # Test K operator (should hit cache)
        result_cache = engine.K("alice_integration", "sky_is_blue")
        assert result_cache is True

        # Clear cache and test again (should hit graph)
        engine.current_state = None
        result_graph = engine.K("alice_integration", "sky_is_blue")
        assert result_graph is True

        # Test non-existent knowledge
        result_false = engine.K("alice_integration", "grass_is_red")
        assert result_false is False

    def test_add_knowledge_then_K_operator(self, engine):
        """Test add_knowledge followed by K operator"""
        # Setup
        engine.create_agent("bob_integration", "Bob")

        # Add multiple knowledge items
        engine.add_knowledge("bob_integration", "fact1")
        engine.add_knowledge("bob_integration", "fact2")
        engine.add_knowledge("bob_integration", "fact3")

        # Verify all with K operator
        assert engine.K("bob_integration", "fact1") is True
        assert engine.K("bob_integration", "fact2") is True
        assert engine.K("bob_integration", "fact3") is True
        assert engine.K("bob_integration", "unknown") is False

    def test_K_operator_with_certainty(self, engine):
        """Test K operator with different certainty levels"""
        engine.create_agent("charlie_integration", "Charlie")

        # Add knowledge with different certainties
        engine.add_knowledge("charlie_integration", "certain", certainty=1.0)
        engine.add_knowledge("charlie_integration", "uncertain", certainty=0.5)
        engine.add_knowledge("charlie_integration", "very_uncertain", certainty=0.1)

        # K operator doesn't care about certainty, only existence
        assert engine.K("charlie_integration", "certain") is True
        assert engine.K("charlie_integration", "uncertain") is True
        assert engine.K("charlie_integration", "very_uncertain") is True

    def test_multiple_agents_independent_knowledge(self, engine):
        """Test multiple agents with independent knowledge bases"""
        # Create agents
        engine.create_agent("alice_ind", "Alice")
        engine.create_agent("bob_ind", "Bob")

        # Add different knowledge
        engine.add_knowledge("alice_ind", "alice_knows_this")
        engine.add_knowledge("bob_ind", "bob_knows_this")

        # Verify independence
        assert engine.K("alice_ind", "alice_knows_this") is True
        assert engine.K("alice_ind", "bob_knows_this") is False
        assert engine.K("bob_ind", "bob_knows_this") is True
        assert engine.K("bob_ind", "alice_knows_this") is False


class TestEnsureState:
    """Test _ensure_state helper method"""

    @pytest.fixture
    def mock_netzwerk(self):
        """Create mock KonzeptNetzwerk"""
        mock = Mock(spec=KonzeptNetzwerk)
        return mock

    @pytest.fixture
    def engine(self, mock_netzwerk):
        """Create EpistemicEngine"""
        return EpistemicEngine(mock_netzwerk)

    def test_ensure_state_initializes_when_none(self, engine):
        """Test _ensure_state creates state when None"""
        # Ensure state is None
        engine.current_state = None

        # Call ensure_state
        engine._ensure_state()

        # State should be initialized
        assert engine.current_state is not None
        assert isinstance(engine.current_state, EpistemicState)
        assert engine.current_state.agents == {}
        assert engine.current_state.propositions == {}
        assert engine.current_state.knowledge_base == {}
        assert engine.current_state.meta_knowledge == {}
        assert engine.current_state.common_knowledge == {}

    def test_ensure_state_preserves_existing(self, engine):
        """Test _ensure_state doesn't overwrite existing state"""
        # Create custom state
        custom_state = EpistemicState(
            agents={"a1": Agent(id="a1", name="Test")},
            propositions={"p1": Proposition(id="p1", content="test")},
            knowledge_base={"a1": {"p1"}},
            meta_knowledge={},
            common_knowledge={},
        )
        engine.current_state = custom_state

        # Call ensure_state
        engine._ensure_state()

        # State should be unchanged
        assert engine.current_state == custom_state
        assert "a1" in engine.current_state.agents
        assert "p1" in engine.current_state.propositions

    def test_ensure_state_multiple_calls(self, engine):
        """Test multiple calls to _ensure_state are idempotent"""
        engine.current_state = None

        # First call
        engine._ensure_state()
        state1 = engine.current_state

        # Second call
        engine._ensure_state()
        state2 = engine.current_state

        # Should be same instance
        assert state1 is state2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

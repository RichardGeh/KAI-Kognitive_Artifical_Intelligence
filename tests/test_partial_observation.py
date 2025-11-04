"""
Tests für component_37_partial_observation.py

Testet generische partielle Beobachtung ohne puzzle-spezifische Logik.
"""

import pytest
from component_37_partial_observation import (
    WorldObject,
    PartialObserver,
    UniquenessAnalyzer,
    PartitionAnalyzer,
    SecondOrderAnalyzer,
    PartialObservationReasoner,
)
from component_35_epistemic_engine import EpistemicEngine
from component_1_netzwerk import KonzeptNetzwerk


class TestWorldObject:
    """Test WorldObject data structure"""

    def test_create_object(self):
        """Test creating world object with properties"""
        obj = WorldObject("obj1", {"color": "red", "size": 5})

        assert obj.object_id == "obj1"
        assert obj.get_property("color") == "red"
        assert obj.get_property("size") == 5
        assert obj.get_property("nonexistent") is None

    def test_matches(self):
        """Test constraint matching"""
        obj = WorldObject("obj1", {"x": 1, "y": 2, "z": 3})

        assert obj.matches({"x": 1}) is True
        assert obj.matches({"x": 1, "y": 2}) is True
        assert obj.matches({"x": 1, "y": 3}) is False
        assert obj.matches({}) is True


class TestPartialObserver:
    """Test partial observation"""

    def test_observe_subset(self):
        """Test observer sees only observable properties"""
        obj = WorldObject("obj1", {"month": "May", "day": 15, "year": 2015})

        # Observer sees only month
        observer = PartialObserver("albert", observable_properties=["month"])
        observation = observer.observe(obj)

        assert observation == {"month": "May"}
        assert "day" not in observation
        assert "year" not in observation

    def test_observe_multiple_properties(self):
        """Test observer with multiple observable properties"""
        obj = WorldObject("obj1", {"x": 1, "y": 2, "z": 3})

        observer = PartialObserver("obs1", observable_properties=["x", "z"])
        observation = observer.observe(obj)

        assert observation == {"x": 1, "z": 3}
        assert "y" not in observation

    def test_can_observe(self):
        """Test checking if property is observable"""
        observer = PartialObserver("obs1", observable_properties=["a", "b"])

        assert observer.can_observe("a") is True
        assert observer.can_observe("b") is True
        assert observer.can_observe("c") is False


class TestUniquenessAnalyzer:
    """Test uniqueness analysis"""

    def test_unique_identifier(self):
        """Test detecting unique property values"""
        objects = [
            WorldObject("obj1", {"color": "red", "number": 1}),
            WorldObject("obj2", {"color": "blue", "number": 2}),
            WorldObject("obj3", {"color": "red", "number": 3}),
        ]

        analyzer = UniquenessAnalyzer(objects)

        # Blue is unique (only one object)
        assert analyzer.is_unique_identifier("color", "blue") is True

        # Red is NOT unique (two objects)
        assert analyzer.is_unique_identifier("color", "red") is False

        # All numbers are unique
        assert analyzer.is_unique_identifier("number", 1) is True
        assert analyzer.is_unique_identifier("number", 2) is True
        assert analyzer.is_unique_identifier("number", 3) is True

    def test_get_unique_properties(self):
        """Test getting all unique values for property"""
        objects = [
            WorldObject("obj1", {"month": "May", "day": 15}),
            WorldObject("obj2", {"month": "May", "day": 19}),
            WorldObject("obj3", {"month": "June", "day": 15}),  # Day 15 appears twice
        ]

        analyzer = UniquenessAnalyzer(objects)

        # Day 19 is unique, 15 is not (appears in May and June)
        unique_days = analyzer.get_unique_properties("day")
        assert 19 in unique_days
        assert 15 not in unique_days

        # No month is unique (all have at least one date, but actually only May has 2)
        unique_months = analyzer.get_unique_properties("month")
        # June is unique (only one date), May is not (two dates)
        assert "June" in unique_months
        assert "May" not in unique_months


class TestPartitionAnalyzer:
    """Test partitioning"""

    def test_partition_by_property(self):
        """Test grouping objects by property"""
        objects = [
            WorldObject("d1", {"month": "May", "day": 15}),
            WorldObject("d2", {"month": "May", "day": 16}),
            WorldObject("d3", {"month": "June", "day": 17}),
        ]

        analyzer = PartitionAnalyzer(objects)
        partitions = analyzer.partition_by_property("month")

        assert len(partitions["May"]) == 2
        assert len(partitions["June"]) == 1

    def test_partition_size(self):
        """Test getting partition size"""
        objects = [
            WorldObject("obj1", {"type": "A"}),
            WorldObject("obj2", {"type": "A"}),
            WorldObject("obj3", {"type": "B"}),
        ]

        analyzer = PartitionAnalyzer(objects)

        assert analyzer.get_partition_size("type", "A") == 2
        assert analyzer.get_partition_size("type", "B") == 1
        assert analyzer.get_partition_size("type", "C") == 0

    def test_partition_has_unique_identifier(self):
        """Test checking if partition contains unique identifier"""
        objects = [
            WorldObject("d1", {"month": "May", "day": 15}),
            WorldObject("d2", {"month": "May", "day": 19}),  # Day 19 is unique
            WorldObject("d3", {"month": "June", "day": 17}),
        ]

        analyzer = PartitionAnalyzer(objects)

        # May partition has day 19 (unique) → True
        assert analyzer.partition_has_unique_identifier("month", "May", "day") is True

        # June partition has day 17 (unique) → True
        assert analyzer.partition_has_unique_identifier("month", "June", "day") is True


class TestSecondOrderAnalyzer:
    """Test meta-reasoning"""

    def test_can_identify_object(self):
        """Test checking if observation uniquely identifies object"""
        objects = [
            WorldObject("obj1", {"x": 1, "y": 2}),
            WorldObject("obj2", {"x": 1, "y": 3}),
            WorldObject("obj3", {"x": 2, "y": 2}),
        ]

        observers = {
            "obs1": PartialObserver("obs1", observable_properties=["x"]),
            "obs2": PartialObserver("obs2", observable_properties=["x", "y"]),
        }

        analyzer = SecondOrderAnalyzer(objects, observers)

        # obs1 sees only x=1 → two objects match → cannot identify
        assert analyzer.can_identify_object("obs1", {"x": 1}) is False

        # obs2 sees x=1, y=2 → one object matches → can identify
        assert analyzer.can_identify_object("obs2", {"x": 1, "y": 2}) is True

    def test_knows_other_cannot_know(self):
        """Test meta-reasoning about other agent's knowledge"""
        # Simplified Cheryl's Birthday scenario
        objects = [
            WorldObject("may15", {"month": "May", "day": 15}),
            WorldObject("may19", {"month": "May", "day": 19}),  # Unique day
            WorldObject("june17", {"month": "June", "day": 17}),
        ]

        observers = {
            "albert": PartialObserver("albert", observable_properties=["month"]),
            "bernard": PartialObserver("bernard", observable_properties=["day"]),
        }

        analyzer = SecondOrderAnalyzer(objects, observers)

        # Albert sees May
        # May contains day 19 (unique) → Albert CANNOT know Bernard doesn't know
        result = analyzer.knows_other_cannot_know(
            observer_id="albert",
            other_observer_id="bernard",
            observer_observation={"month": "May"},
        )
        assert result is False

        # Albert sees June
        # June contains day 17 (unique) → Albert CANNOT know Bernard doesn't know
        result = analyzer.knows_other_cannot_know(
            observer_id="albert",
            other_observer_id="bernard",
            observer_observation={"month": "June"},
        )
        assert result is False

    def test_knows_other_cannot_know_true_case(self):
        """Test case where agent DOES know other cannot know"""
        objects = [
            WorldObject("obj1", {"x": 1, "y": 2}),
            WorldObject("obj2", {"x": 1, "y": 2}),  # Duplicate y
            WorldObject("obj3", {"x": 2, "y": 3}),
        ]

        observers = {
            "obs_x": PartialObserver("obs_x", observable_properties=["x"]),
            "obs_y": PartialObserver("obs_y", observable_properties=["y"]),
        }

        analyzer = SecondOrderAnalyzer(objects, observers)

        # obs_x sees x=1
        # ALL objects with x=1 have y=2 (not unique) → obs_x KNOWS obs_y cannot know
        result = analyzer.knows_other_cannot_know(
            observer_id="obs_x",
            other_observer_id="obs_y",
            observer_observation={"x": 1},
        )
        assert result is True


class TestPartialObservationReasoner:
    """Test integrated reasoning system"""

    def test_add_objects_and_observers(self):
        """Test adding objects and observers"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        objects = [WorldObject("obj1", {"a": 1}), WorldObject("obj2", {"a": 2})]

        reasoner.add_objects(objects)
        assert len(reasoner.objects) == 2

        observer = PartialObserver("obs1", observable_properties=["a"])
        reasoner.add_observer(observer)

        assert "obs1" in reasoner.observers

    def test_establish_observations(self):
        """Test establishing initial observations"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        obj = WorldObject("target", {"color": "red", "size": 5})
        objects = [obj]

        reasoner.add_objects(objects)

        observer = PartialObserver("obs1", observable_properties=["color"])
        reasoner.add_observer(observer)

        # Create agent first
        engine.create_agent("obs1", "Observer 1")

        # Establish observations
        reasoner.establish_observations(obj)

        # Check knowledge was added
        assert engine.K("obs1", "obs1_observes_color_red") is True

    def test_get_possible_objects(self):
        """Test getting objects matching observation"""
        netzwerk = KonzeptNetzwerk()
        engine = EpistemicEngine(netzwerk)
        reasoner = PartialObservationReasoner(engine)

        objects = [
            WorldObject("obj1", {"x": 1, "y": 2}),
            WorldObject("obj2", {"x": 1, "y": 3}),
            WorldObject("obj3", {"x": 2, "y": 2}),
        ]

        reasoner.add_objects(objects)

        # Get objects with x=1
        possible = reasoner.get_possible_objects("obs1", {"x": 1})
        assert len(possible) == 2
        assert possible[0].object_id in ["obj1", "obj2"]
        assert possible[1].object_id in ["obj1", "obj2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

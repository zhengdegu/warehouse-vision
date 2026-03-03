"""Property-based tests for the evaluation orchestration script.

# Feature: accuracy-evaluation, Property 14: 路径验证完备性
"""

import os
import string

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from scripts.evaluate import validate_paths


# Strategy: generate a path component that is valid but very unlikely to exist.
# We use a prefix to ensure the path doesn't accidentally match real files.
_PATH_CHARS = string.ascii_letters + string.digits + "_-"

def nonexistent_path():
    """Generate a random path string that does not exist on disk."""
    return st.text(
        alphabet=_PATH_CHARS,
        min_size=3,
        max_size=30,
    ).map(lambda s: f"__nonexistent_test_path__/{s}")


class TestValidatePathsProperty:
    """Property 14: 路径验证完备性

    For any parameter configuration where specified paths do not exist,
    the validation function should return error messages containing
    the expected paths.

    **Validates: Requirements 6.7**
    """

    @given(model_path=nonexistent_path())
    @settings(max_examples=100)
    def test_missing_model_reported_for_detection(self, model_path):
        """When model file doesn't exist and detection is enabled,
        errors should contain the model path."""
        assume(not os.path.exists(model_path))

        errors = validate_paths(
            run_detection=True,
            run_rules=False,
            model_path=model_path,
            pose_model_path=None,
            samples_dir="__nonexistent_test_path__/samples",
            labels_dir="__nonexistent_test_path__/labels",
            videos_dir="__nonexistent_test_path__/videos",
        )
        assert len(errors) > 0, "Expected validation errors for missing model"
        model_errors = [e for e in errors if model_path in e]
        assert len(model_errors) > 0, (
            f"Expected error containing model path '{model_path}', "
            f"got errors: {errors}"
        )

    @given(model_path=nonexistent_path())
    @settings(max_examples=100)
    def test_missing_model_reported_for_rules(self, model_path):
        """When model file doesn't exist and rules evaluation is enabled,
        errors should contain the model path."""
        assume(not os.path.exists(model_path))

        errors = validate_paths(
            run_detection=False,
            run_rules=True,
            model_path=model_path,
            pose_model_path=None,
            samples_dir="__nonexistent_test_path__/samples",
            labels_dir="__nonexistent_test_path__/labels",
            videos_dir="__nonexistent_test_path__/videos",
        )
        assert len(errors) > 0, "Expected validation errors for missing model"
        model_errors = [e for e in errors if model_path in e]
        assert len(model_errors) > 0, (
            f"Expected error containing model path '{model_path}', "
            f"got errors: {errors}"
        )

    @given(samples_dir=nonexistent_path(), labels_dir=nonexistent_path())
    @settings(max_examples=100)
    def test_missing_data_dirs_reported_for_detection(self, samples_dir, labels_dir):
        """When samples/labels directories don't exist and detection is enabled,
        errors should contain both directory paths."""
        assume(not os.path.exists(samples_dir))
        assume(not os.path.exists(labels_dir))

        errors = validate_paths(
            run_detection=True,
            run_rules=False,
            model_path="__nonexistent_test_path__/model.pt",
            pose_model_path=None,
            samples_dir=samples_dir,
            labels_dir=labels_dir,
            videos_dir="__nonexistent_test_path__/videos",
        )
        samples_errors = [e for e in errors if samples_dir in e]
        labels_errors = [e for e in errors if labels_dir in e]
        assert len(samples_errors) > 0, (
            f"Expected error containing samples path '{samples_dir}', "
            f"got errors: {errors}"
        )
        assert len(labels_errors) > 0, (
            f"Expected error containing labels path '{labels_dir}', "
            f"got errors: {errors}"
        )

    @given(videos_dir=nonexistent_path())
    @settings(max_examples=100)
    def test_missing_videos_dir_reported_for_rules(self, videos_dir):
        """When videos directory doesn't exist and rules evaluation is enabled,
        errors should contain the videos directory path."""
        assume(not os.path.exists(videos_dir))

        errors = validate_paths(
            run_detection=False,
            run_rules=True,
            model_path="__nonexistent_test_path__/model.pt",
            pose_model_path=None,
            samples_dir="__nonexistent_test_path__/samples",
            labels_dir="__nonexistent_test_path__/labels",
            videos_dir=videos_dir,
        )
        videos_errors = [e for e in errors if videos_dir in e]
        assert len(videos_errors) > 0, (
            f"Expected error containing videos path '{videos_dir}', "
            f"got errors: {errors}"
        )

    @given(pose_model=nonexistent_path())
    @settings(max_examples=100)
    def test_missing_pose_model_reported_for_rules(self, pose_model):
        """When pose model path is specified but doesn't exist and rules
        evaluation is enabled, errors should contain the pose model path."""
        assume(not os.path.exists(pose_model))

        errors = validate_paths(
            run_detection=False,
            run_rules=True,
            model_path="__nonexistent_test_path__/model.pt",
            pose_model_path=pose_model,
            samples_dir="__nonexistent_test_path__/samples",
            labels_dir="__nonexistent_test_path__/labels",
            videos_dir="__nonexistent_test_path__/videos",
        )
        pose_errors = [e for e in errors if pose_model in e]
        assert len(pose_errors) > 0, (
            f"Expected error containing pose model path '{pose_model}', "
            f"got errors: {errors}"
        )

    @given(
        model_path=nonexistent_path(),
        samples_dir=nonexistent_path(),
        labels_dir=nonexistent_path(),
        videos_dir=nonexistent_path(),
    )
    @settings(max_examples=100)
    def test_all_missing_paths_reported_when_all_enabled(
        self, model_path, samples_dir, labels_dir, videos_dir,
    ):
        """When all evaluations are enabled and all paths are missing,
        every path should appear in at least one error message."""
        assume(not os.path.exists(model_path))
        assume(not os.path.exists(samples_dir))
        assume(not os.path.exists(labels_dir))
        assume(not os.path.exists(videos_dir))

        errors = validate_paths(
            run_detection=True,
            run_rules=True,
            model_path=model_path,
            pose_model_path=None,
            samples_dir=samples_dir,
            labels_dir=labels_dir,
            videos_dir=videos_dir,
        )
        all_errors_text = " ".join(errors)
        assert model_path in all_errors_text, (
            f"Model path '{model_path}' not found in errors: {errors}"
        )
        assert samples_dir in all_errors_text, (
            f"Samples dir '{samples_dir}' not found in errors: {errors}"
        )
        assert labels_dir in all_errors_text, (
            f"Labels dir '{labels_dir}' not found in errors: {errors}"
        )
        assert videos_dir in all_errors_text, (
            f"Videos dir '{videos_dir}' not found in errors: {errors}"
        )

    def test_no_errors_when_nothing_enabled(self):
        """When neither detection nor rules are enabled, no validation errors."""
        errors = validate_paths(
            run_detection=False,
            run_rules=False,
            model_path="nonexistent_model.pt",
            pose_model_path=None,
            samples_dir="nonexistent_samples",
            labels_dir="nonexistent_labels",
            videos_dir="nonexistent_videos",
        )
        assert errors == [], f"Expected no errors, got: {errors}"

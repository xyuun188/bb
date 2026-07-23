from scripts.run_online_model_training import _target_argv


def test_local_ai_tools_online_training_has_no_stage_authorization_input() -> None:
    script_path, argv = _target_argv("local_ai_tools", persist_artifact=True)

    assert script_path == "scripts/train_local_ai_tools_models.py"
    assert argv == [
        script_path,
        "--training-mode",
        "shadow",
        "--persist-artifact",
        "--confirm-phase3-rebuild",
    ]
    assert "--model-stage" not in argv

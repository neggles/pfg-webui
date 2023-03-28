import launch  # type: ignore

if not launch.is_installed("onnxruntime") and not launch.is_installed("tensorflow"):
    launch.run_pip("install onnxruntime", "requirements for Prompt-Free Generation")

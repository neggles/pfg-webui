import launch  # type: ignore

if not launch.is_installed("onnxruntime"):
    launch.run_pip("install onnxruntime", "requirements for Prompt-Free Generation")

if not launch.is_installed("tensorflow"):
    launch.run_pip("install tensorflow", "requirements for Prompt-Free Generation")

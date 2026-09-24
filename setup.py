from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
RUNTIME_ASSET_SUFFIXES = {
    ".json", ".npy", ".npz", ".obj", ".pkl", ".png", ".stl", ".xml", ".yaml", ".yml",
}
EXCLUDED_PARTS = {
    ".idea", ".ipynb_checkpoints", "__pycache__", "audit", "audits", "checkpoint", "checkpoints",
    "logs", "output", "outputs", "results", "tensorboard",
}


def read_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if requirement and not requirement.startswith("#"):
            requirements.append(requirement)
    return requirements


def package_assets(package: str) -> list[str]:
    package_root = ROOT / package
    assets: list[str] = []
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in RUNTIME_ASSET_SUFFIXES:
            assets.append(relative.as_posix())
    return assets


def model_data_files() -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    model_root = ROOT / "models"
    for path in model_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in RUNTIME_ASSET_SUFFIXES:
            destination = str(Path("models") / path.relative_to(model_root).parent)
            grouped.setdefault(destination, []).append(str(path))
    return sorted(grouped.items())


if __name__ == "__main__":
    setup(
        name="MyOSys",
        version="1.0.0",
        license="Apache-2.0",
        description="Musculoskeletal simulation and learning tools for human-exoskeleton control",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        packages=find_packages(
            include=("myosuite", "myosuite.*", "myoassist_utils", "myoassist_utils.*", "rl_train", "rl_train.*", "tcn", "tcn.*")
        ),
        package_data={
            "myosuite": package_assets("myosuite"),
            "rl_train": package_assets("rl_train"),
            "tcn": package_assets("tcn") + ["deployment/unified_tcn_latest_100hz_deploy.pt"],
        },
        data_files=model_data_files(),
        include_package_data=False,
        python_requires=">=3.11",
        install_requires=read_requirements(ROOT / "requirements.txt"),
        extras_require={
            "analysis": [
                "matplotlib", "pandas", "opencv-python==4.11.0.86", "imageio[ffmpeg]==2.37.0",
                "mediapy", "scikit-video==1.1.11", "openpyxl==3.1.5", "tensorboard", "tqdm",
            ],
            "hardware": ["pyserial"],
            "dev": ["pytest"],
        },
    )

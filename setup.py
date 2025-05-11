from setuptools import setup, find_packages

setup(
    name="flatland_rl_policy_benchmark",
    version="0.1.0",
    description="Benchmark multi-policy RL on Flatland via tournament",
    author="Alessia",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "flatland-rl>=2.4.0",
        "pyglet>=2.0.0",
        "Flask>=2.2.0",
        "plotly>=5.10.0",
        "pandas>=1.5.0"
    ],
    entry_points={
        "console_scripts": [
            "flatland-train=flatland_rl_policy_benchmark.train:main",
            "flatland-tourney=flatland_rl_policy_benchmark.tournament:main",
            "flatland-render=flatland_rl_policy_benchmark.env.simulate_render:main"
        ]
    }
)

import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


def test_built_wheel_contains_and_runs_the_team_helpers(tmp_path):
    wheel_dir = tmp_path / "wheel"
    target = tmp_path / "installed"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("zhong2025-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()
        assert "drive.py" in names
        assert "graph.py" in names
        assert "plot.py" in names
        assert "sql.py" in names
        assert "zhong2025/plot.py" in names
        assert "zhong2025/assets/tx119_atlas_demo.npz" in names
        assert "zhong2025/assets/figshare-v2-curated.json" in names
        assert "zhong2025/assets/reference_figures/nature-main-1.png" in names
        assert "zhong2025/assets/reference_figures/science-methods-fig4.jpg" in names

    configured = re.search(
        r'^version = "([^"]+)"$', Path("pyproject.toml").read_text(), re.MULTILINE
    ).group(1)
    packaged = re.search(r"^Version: (.+)$", metadata, re.MULTILINE).group(1)
    assert packaged == configured

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(target)
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import drive
import graph
import plot
import sql
import zhong2025

demo = zhong2025.load_atlas_demo()
assert demo["population_features"].shape == (452, 18, 48)
assert len(drive.setup(mount=False, report=False).files) == 297
assert Path(drive.__file__).resolve().is_relative_to(Path(__import__("os").environ["PYTHONPATH"]).resolve())
assert Path(graph.__file__).resolve().is_relative_to(Path(__import__("os").environ["PYTHONPATH"]).resolve())
assert Path(plot.__file__).resolve().is_relative_to(Path(__import__("os").environ["PYTHONPATH"]).resolve())
assert Path(sql.__file__).resolve().is_relative_to(Path(__import__("os").environ["PYTHONPATH"]).resolve())
assert plot.curve is zhong2025.plot.curve

db = sql.setup(mount=False, report=False)
assert db.query("SELECT count(*) AS n FROM recordings").iloc[0]["n"] == 89
db.close()

figure = plot.curve({"mice": [[0, 1, 2], [1, 2, 3]]})
figure.canvas.draw()
assert plot.info(figure).recipe == "curve"
plot.close(figure)

@graph.node(outputs="value")
def source(scale=2):
    return scale

flow = graph.Graph("wheel smoke", source)
assert flow.run(scale=3)["value"] == 3
panel = flow.widget(controls={"scale": [1, 2, 3]})
panel.children[0].children[0].click()
assert panel.last_run["value"] == 2
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert smoke.returncode == 0

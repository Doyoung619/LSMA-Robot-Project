# Migration from the local Rita-Robot workspace

The original Git repository was initialized in `experiment/`, on an unborn
`main`, with origin `https://github.com/Doyoung619/LSMA-Robot-Project.git` and no
tracked files/commits. Existing local work was entirely untracked. No tracked
user modification was reset or discarded. `overleaf/` in the parent workspace
was not moved or staged.

`experiment/etc/` was moved as one directory to `experiment/0_pre_and_etc/`.
Before/after recursive file counts were both **8302**. No original file was
deleted. This includes local virtual environments, music and old outputs;
large/local-only content remains on disk and is ignored by the new Git config.
The table is relative to the original Rita-Robot workspace:

| Old | New | Purpose |
|---|---|---|
| experiment/etc/rita_robot/** | experiment/0_pre_and_etc/rita_robot/** | original spectral fitter, Kortex feedback/legacy velocity demo |
| experiment/etc/rita.py | experiment/0_pre_and_etc/rita.py | preserved legacy CLI |
| experiment/etc/rita_sim/**,sim.py | experiment/0_pre_and_etc/rita_sim/**,sim.py | previous MuJoCo sandbox |
| experiment/etc/sim_poses/**,sim_tests/**,tests/** | experiment/0_pre_and_etc/sim_poses/**,sim_tests/**,tests/** | previous poses and tests |
| experiment/etc/render*_video.py,verify_dance_video.py | experiment/0_pre_and_etc/(same filenames) | preserved video helpers |
| experiment/etc/README.md | experiment/0_pre_and_etc/LEGACY_README.md | older documentation retained separately |
| experiment/etc/SIMULATION.md | experiment/0_pre_and_etc/SIMULATION.md | corrected old root mention |
| experiment/etc/Play_*.cmd,Stop_Rita.cmd,Setup*.ps1 | experiment/0_pre_and_etc/(same filenames) | helpers remain relative to their own directory |
| experiment/etc/data/** | experiment/0_pre_and_etc/data/** | original models/licenses and local analysis |
| experiment/etc/.venv*,logs/**,tmp/**,sim_models/**,sim_output/** | experiment/0_pre_and_etc/(same relative suffixes) | preserved, gitignored |
| experiment/etc/Billie Jean.mp3,Rita_Robot.pdf,STOP_RITA,requirements*.txt,.gitignore,organization_manifest.json | experiment/0_pre_and_etc/(same filenames) | audio/reference/latch/old dependencies/history retained |
| experiment/result/** | experiment/result/** | existing result/video locations retained; links to renamed common folder updated |

The wildcard identity rule covers every remaining original `etc/` descendant.
`organization_manifest.json` is preserved as the earlier historical snapshot,
not rewritten as a fresh validation of relocated environments. Recreate local
venvs after cloning; existing Python launchers work locally but absolute pip
launchers from an old venv should not be copied. Use `python -m pip`.

New valid package names are `rita_core`, `rita_finite`, `rita_infinite`. The
single `run.py` bootstrap and setuptools package-dir mapping resolve numbered
folders without invalid numeric imports or user PYTHONPATH. Source-relative
legacy imports remain valid after the directory move. Local result README links
were updated. No old media was regenerated or uploaded in this migration.
The earlier `rita.py run` raw-velocity entry is deliberately retired and now
directs operators to the verified-profile/`--execute` trajectory pipeline. Old
double-click launchers cannot bypass that hardware gate. The original driver,
feedback guards and stop implementation remain available as reference code.

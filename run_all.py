import subprocess
from os.path import join, dirname
main_dir = dirname(__file__)
script_dir = join(main_dir, 'scripts')

files = [
    "oneD_example_illustration_plot.py",
    "oneD_example_beta_vs_ctheta_plot.py",
    "oneD_example_freq_vs_delta_compute.py",
    "oneD_example_freq_vs_delta_plot.py",
    "multi_example_compute.py",
    "multi_example_plot.py",
]

for f in files:
    print(f"Running {f}...")
    subprocess.run(["python", join(script_dir, f)], check=True)

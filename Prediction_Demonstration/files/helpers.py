import os
import subprocess
import pandas as pd
import numpy as np



# Construct TS2CG shape
def run_ts2cg(fname="input.str"):
    """
    Run TS2CG to generate a coarse-grained shape from the .str file.
    """
    # Load the file from the system directory
    input_file = os.path.join(os.getcwd(), fname)
    
    
    command = ["TS2CG", "PCG", "-WPointDir", "-str", input_file, "-defout", "system", "-function", "analytical_shape"]
    subprocess.run(command, check=True)


def write_str_file(
    amplitudes,
    wave_numbers,
    phases,
    L,
    C_CDL,
    filename="input.str",
):
    """
    Write a TS2CG .str file.

    compositions should be an iterable of:
        (lipid_name, ratio_up, ratio_down, apl)
    """
    compositions = [
    ("POPC", 1-C_CDL, 1-C_CDL, 0.64),
    ("CDL", C_CDL, C_CDL, 1.2),
    ]
    if not (
        len(amplitudes)
        == len(wave_numbers)
        == len(phases)
    ):
        raise ValueError(
            "amplitudes, wave_numbers, and phases must have equal lengths"
        )

    lipid_lines = "\n".join(
        f"{name}\t{ratio_up}  {ratio_down}  {apl}"
        for name, ratio_up, ratio_down, apl in compositions
    )

    mode_lines = "\n".join(
        f"Mode {amplitude}    {wave_number}    {phase}"
        for amplitude, wave_number, phase in zip(
            amplitudes, wave_numbers, phases
        )
    )

    contents = f"""[Lipids List]
Domain 0
; lipidname ratio_up ratio_down APL
{lipid_lines}
End

[Shape Data]
; Shape dimensions: x y z
ShapeType 1D_PBC_Fourier
Box        {L}   15   20
WallRange  0    1    0    1
Density    3    1
Thickness  4
{mode_lines}
End
"""
    with open(filename, "w") as file:
        file.write(contents)

    return contents




def _load_dat_file(file_path):
    if "Outer" in file_path:
        print("Loading OuterBM.dat file...")
        rows_to_skip = 4  # Number of rows to skip at the beginning of the file
        column_line_index = 2  # The line index where the column names are located (0-based)
    else:
        print("Loading InnerBM.dat file...")
        rows_to_skip = 3  # Number of rows to skip at the beginning of the file
        column_line_index = 1  # The line index where the column names are located (0-based)
    
    # First we extract the column names from the second line of the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
        column_names = lines[column_line_index].strip().split()[1:-1]
    
    # Then load the data into a pandas DataFrame, skipping the first three lines
    DF = pd.read_csv(file_path, skiprows=rows_to_skip, names=column_names, delimiter=r'\s+')
        
    H = 0.5 * (DF['C1'] + DF['C2']) # Mean curvature
    pos = DF[['X', 'Y', 'Z']].values
    area = DF['area'].values
    
    return DF, H, pos, area

def get_area_and_curvature(point_dir):
    H_all = []
    A_all = []
    pos_dir = {}
    for leaflet in ["Inner", "Outer"]:
        Data, H, pos, A = _load_dat_file(os.path.join(point_dir, f"{leaflet}BM.dat"))
        H_all.append(H)
        A_all.append(A)
        pos_dir[leaflet] = pos

    return np.concatenate(H_all), np.concatenate(A_all), pos_dir


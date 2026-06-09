"""SPM download utility.

SPM (Statistical Parametric Mapping) is a MATLAB-based toolbox for
neuroimaging analysis. It cannot be downloaded automatically and
requires a valid MATLAB installation.
"""
import sys


def download_spm(target_dir: str | None = None) -> str:
    """SPM cannot be automatically downloaded.

    SPM is a MATLAB toolbox distributed by the Wellcome Centre for
    Human Neuroimaging. It requires a valid MATLAB license.

    Installation steps:
    1. Download SPM from: https://www.fil.ion.ucl.ac.uk/spm/software/
    2. Extract to your preferred location
    3. Add the SPM directory to your MATLAB path
    4. Set the SPMDIR environment variable to the SPM root directory

    Alternatively, consider using FSL or AFNI as free alternatives.
    """
    print("SPM cannot be automatically downloaded.")
    print("SPM is a MATLAB toolbox. Please download it from:")
    print("  https://www.fil.ion.ucl.ac.uk/spm/software/")
    print()
    print("Free alternatives: FSL (https://fsl.fmrib.ox.ac.uk) or "
          "AFNI (https://afni.nimh.nih.gov)")
    sys.exit(1)
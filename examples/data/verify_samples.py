#!/usr/bin/env python3
"""
Verify that all downloaded sample medical data files can be loaded correctly.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def verify_nifti():
    """Verify NIfTI file can be loaded."""
    print("\n" + "="*70)
    print("1. Verifying NIfTI Brain MRI Data")
    print("="*70)

    try:
        import nibabel as nib
        # Try both .nii and .nii.gz extensions
        filepath_gz = os.path.join(BASE_DIR, "nifti", "brain_mri_sample.nii.gz")
        filepath_nii = os.path.join(BASE_DIR, "nifti", "brain_mri_sample.nii")

        filepath = filepath_nii if os.path.exists(filepath_nii) else filepath_gz

        img = nib.load(filepath)
        data = img.get_fdata()

        print(f"✓ File loaded successfully")
        print(f"  File: {os.path.basename(filepath)}")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Affine shape: {img.affine.shape}")
        return True
    except ImportError:
        print("✗ nibabel not installed. Install with: pip install nibabel")
        return False
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False


def verify_dicom():
    """Verify DICOM file can be loaded."""
    print("\n" + "="*70)
    print("2. Verifying DICOM Medical Imaging Data")
    print("="*70)

    try:
        import pydicom
        filepath = os.path.join(BASE_DIR, "dicom", "chest_xray_sample.dcm")

        ds = pydicom.dcmread(filepath)

        print(f"✓ File loaded successfully")
        print(f"  Patient ID: {ds.get('PatientID', 'N/A')}")
        print(f"  Modality: {ds.get('Modality', 'N/A')}")
        print(f"  Image shape: {ds.pixel_array.shape if hasattr(ds, 'pixel_array') else 'N/A'}")
        return True
    except ImportError:
        print("✗ pydicom not installed. Install with: pip install pydicom")
        return False
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False


def verify_fmri():
    """Verify fMRI 4D NIfTI file can be loaded."""
    print("\n" + "="*70)
    print("3. Verifying fMRI Time Series Data")
    print("="*70)

    try:
        import nibabel as nib
        filepath = os.path.join(BASE_DIR, "fmri", "fmri_timeseries.nii.gz")

        img = nib.load(filepath)
        data = img.get_fdata()

        print(f"✓ File loaded successfully")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Number of timepoints: {data.shape[-1] if len(data.shape) == 4 else 'N/A'}")
        return True
    except ImportError:
        print("✗ nibabel not installed. Install with: pip install nibabel")
        return False
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False


def verify_ecg():
    """Verify ECG WFDB data can be loaded."""
    print("\n" + "="*70)
    print("4. Verifying ECG Data (WFDB format)")
    print("="*70)

    try:
        # Check if files exist and are valid
        hea_file = os.path.join(BASE_DIR, "ecg", "ecg_sample.hea")
        dat_file = os.path.join(BASE_DIR, "ecg", "ecg_sample.dat")

        if not os.path.exists(hea_file):
            print(f"✗ Header file not found: {hea_file}")
            return False

        if not os.path.exists(dat_file):
            print(f"✗ Data file not found: {dat_file}")
            return False

        # Read header file to verify format
        with open(hea_file, 'r') as f:
            header_line = f.readline().strip()
            parts = header_line.split()
            if len(parts) >= 3:
                record_name = parts[0]
                num_signals = int(parts[1])
                sampling_freq = float(parts[2])

                print(f"✓ Files loaded successfully")
                print(f"  Record name: {record_name}")
                print(f"  Number of signals: {num_signals}")
                print(f"  Sampling frequency: {sampling_freq} Hz")
                print(f"  Data file size: {os.path.getsize(dat_file) / 1024:.2f} KB")
                return True
            else:
                print("✗ Invalid header file format")
                return False

    except Exception as e:
        print(f"✗ Failed to verify: {e}")
        return False


def verify_tabular():
    """Verify tabular CSV data can be loaded."""
    print("\n" + "="*70)
    print("5. Verifying Tabular Data (CSV)")
    print("="*70)

    try:
        import pandas as pd
        filepath = os.path.join(BASE_DIR, "tabular", "diabetes_sample.csv")

        df = pd.read_csv(filepath)

        print(f"✓ File loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return True
    except ImportError:
        print("✗ pandas not installed. Install with: pip install pandas")
        return False
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False


def verify_vitals():
    """Verify vitals time series CSV data can be loaded."""
    print("\n" + "="*70)
    print("6. Verifying Vital Signs Time Series")
    print("="*70)

    try:
        import pandas as pd
        filepath = os.path.join(BASE_DIR, "vitals", "vitals_timeseries.csv")

        df = pd.read_csv(filepath)

        print(f"✓ File loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return True
    except ImportError:
        print("✗ pandas not installed. Install with: pip install pandas")
        return False
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("MedTabX Sample Data Verification")
    print("="*70)
    print("Verifying all downloaded medical data files can be loaded...")

    results = {}

    results['nifti'] = verify_nifti()
    results['dicom'] = verify_dicom()
    results['fmri'] = verify_fmri()
    results['ecg'] = verify_ecg()
    results['tabular'] = verify_tabular()
    results['vitals'] = verify_vitals()

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    for data_type, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{data_type:15s}: {status}")

    print("\n" + "="*70)
    all_passed = all(results.values())
    if all_passed:
        print("All files verified successfully!")
    else:
        print("Some files failed verification. Check error messages above.")
    print("="*70)

    sys.exit(0 if all_passed else 1)

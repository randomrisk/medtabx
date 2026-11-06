#!/usr/bin/env python3
"""
Download sample medical data files for MedTabX examples.
Uses real, publicly available medical data from trusted sources.
"""
import os
import urllib.request
import csv
import gzip
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url, destination, timeout=60):
    """Download a file from URL to destination."""
    print(f"Downloading {url}")
    print(f"  -> {destination}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Download with timeout
        urllib.request.urlretrieve(url, destination)

        # Check file size
        size = os.path.getsize(destination)
        size_mb = size / (1024 * 1024)
        print(f"✓ Successfully downloaded ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False


def create_sample_tabular():
    """Create a sample medical tabular dataset."""
    filepath = os.path.join(BASE_DIR, "tabular", "diabetes_sample.csv")

    # Sample diabetes dataset
    data = [
        ["patient_id", "age", "gender", "bmi", "blood_pressure", "glucose", "insulin", "diabetes"],
        ["P001", "45", "F", "28.5", "120", "95", "85", "0"],
        ["P002", "52", "M", "31.2", "135", "180", "120", "1"],
        ["P003", "38", "F", "24.8", "115", "88", "75", "0"],
        ["P004", "61", "M", "33.5", "145", "210", "150", "1"],
        ["P005", "47", "F", "29.1", "125", "102", "90", "0"],
        ["P006", "55", "M", "35.2", "150", "195", "135", "1"],
        ["P007", "42", "F", "26.3", "118", "92", "80", "0"],
        ["P008", "58", "M", "32.8", "140", "188", "125", "1"],
        ["P009", "49", "F", "27.9", "122", "98", "88", "0"],
        ["P010", "63", "M", "34.1", "148", "205", "145", "1"],
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"✓ Created sample tabular data at {filepath}")
    return True


def create_sample_vitals():
    """Create a sample vital signs time series dataset."""
    filepath = os.path.join(BASE_DIR, "vitals", "vitals_timeseries.csv")

    # Sample vital signs time series
    data = [
        ["timestamp", "patient_id", "heart_rate", "systolic_bp", "diastolic_bp", "temperature", "spo2"],
        ["2024-01-01 08:00:00", "P001", "72", "120", "80", "36.6", "98"],
        ["2024-01-01 08:15:00", "P001", "75", "122", "82", "36.7", "98"],
        ["2024-01-01 08:30:00", "P001", "73", "121", "81", "36.6", "99"],
        ["2024-01-01 08:45:00", "P001", "74", "119", "79", "36.7", "98"],
        ["2024-01-01 09:00:00", "P001", "76", "123", "83", "36.8", "97"],
        ["2024-01-01 08:00:00", "P002", "88", "135", "90", "37.1", "96"],
        ["2024-01-01 08:15:00", "P002", "90", "138", "92", "37.2", "96"],
        ["2024-01-01 08:30:00", "P002", "87", "136", "91", "37.1", "97"],
        ["2024-01-01 08:45:00", "P002", "89", "137", "90", "37.0", "96"],
        ["2024-01-01 09:00:00", "P002", "91", "140", "93", "37.2", "95"],
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"✓ Created sample vitals data at {filepath}")
    return True


def download_nifti_data():
    """Download real NIfTI brain MRI data."""
    print("\n" + "="*70)
    print("1. NIfTI Brain MRI Data")
    print("="*70)
    print("Source: nibabel test data repository")

    # Use nibabel's anatomical test file (not gzipped)
    nifti_url = "https://github.com/nipy/nibabel/raw/master/nibabel/tests/data/anatomical.nii"
    nifti_dest = os.path.join(BASE_DIR, "nifti", "brain_mri_sample.nii")

    success = download_file(nifti_url, nifti_dest)
    if success:
        print("License: nibabel test data (MIT License)")
    return success


def download_dicom_data():
    """Download real DICOM medical imaging data."""
    print("\n" + "="*70)
    print("2. DICOM Medical Imaging Data")
    print("="*70)
    print("Source: pydicom test data repository (v2.0.0)")

    # Use pydicom's test data - CT scan (v2.0.0 has working path)
    dicom_url = "https://github.com/pydicom/pydicom/raw/v2.0.0/pydicom/data/test_files/CT_small.dcm"
    dicom_dest = os.path.join(BASE_DIR, "dicom", "chest_xray_sample.dcm")

    success = download_file(dicom_url, dicom_dest)
    if success:
        print("License: pydicom test data (MIT License)")
    return success


def download_fmri_data():
    """Download real 4D fMRI NIfTI data."""
    print("\n" + "="*70)
    print("3. fMRI Time Series Data (4D NIfTI)")
    print("="*70)
    print("Source: nibabel test data repository")

    # Use nibabel's small 4D test file
    fmri_url = "https://github.com/nipy/nibabel/raw/master/nibabel/tests/data/example4d.nii.gz"
    fmri_dest = os.path.join(BASE_DIR, "fmri", "fmri_timeseries.nii.gz")

    success = download_file(fmri_url, fmri_dest)
    if success:
        print("License: nibabel test data (MIT License)")
    return success


def download_ecg_data():
    """Download real ECG data from PhysioNet."""
    print("\n" + "="*70)
    print("4. ECG Data (WFDB format)")
    print("="*70)
    print("Source: PhysioNet MIT-BIH Arrhythmia Database")

    # Download sample record from MIT-BIH database
    base_url = "https://physionet.org/files/mitdb/1.0.0/100"

    # Download .hea file (header)
    hea_url = base_url + ".hea"
    hea_dest = os.path.join(BASE_DIR, "ecg", "ecg_sample.hea")
    success_hea = download_file(hea_url, hea_dest)

    # Download .dat file (data)
    dat_url = base_url + ".dat"
    dat_dest = os.path.join(BASE_DIR, "ecg", "ecg_sample.dat")
    success_dat = download_file(dat_url, dat_dest)

    if success_hea and success_dat:
        print("License: Open Data Commons Attribution License v1.0")
        print("Citation: Moody GB, Mark RG. MIT-BIH Arrhythmia Database. PhysioNet.")

    return success_hea and success_dat


def download_eeg_data():
    """Download one EEG recording from the CHB-MIT Scalp EEG Database (PhysioNet)."""
    print("\n" + "="*70)
    print("5. EEG Data (EDF format)")
    print("="*70)
    url = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf"
    dest = os.path.join(BASE_DIR, "eeg", "chb01_01.edf")
    success = download_file(url, dest, timeout=300)
    if success:
        print("License: Open Data Commons Attribution License v1.0")
        print("Citation: Shoeb AH. Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis, MIT, 2009.")
    return success


if __name__ == "__main__":
    print("="*70)
    print("MedTabX Sample Medical Data Downloader")
    print("="*70)
    print("Downloading real, publicly available medical data files...")

    results = {}

    # 1. Tabular data - Create sample
    print("\n" + "="*70)
    print("0. Tabular Data (CSV) - Creating sample")
    print("="*70)
    results['tabular'] = create_sample_tabular()

    # 2. NIfTI imaging
    results['nifti'] = download_nifti_data()

    # 3. DICOM imaging
    results['dicom'] = download_dicom_data()

    # 4. fMRI data
    results['fmri'] = download_fmri_data()

    # 5. ECG data
    results['ecg'] = download_ecg_data()

    # 6. EEG data
    results['eeg'] = download_eeg_data()

    # 6. Vital signs - Create sample
    print("\n" + "="*70)
    print("5. Vital Signs Time Series - Creating sample")
    print("="*70)
    results['vitals'] = create_sample_vitals()

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    for data_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{data_type:15s}: {status}")

    print("\n" + "="*70)
    print("Download process completed!")
    print("="*70)

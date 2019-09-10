from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from data_process import load_file
# import dicom2nifti
# import dicom2nifti.settings as settings

import nibabel as nib
# import ants
# settings.disable_validate_slice_increment()


def equalize_volume(volume):
    volume = exposure.rescale_intensity(exposure.equalize_hist(volume), out_range=(0, 255))
    return volume.astype("int16")

def add_noise(volume, precent, value):
    s = np.random.uniform(0, 1, volume.shape)
    s[s>precent] = 1
    volume[s==1] += value
    return volume

def create_base_folders(input_folder, output_folder):
    folder_list = os.listdir(input_folder)
    for folder in folder_list:
        new_folder = os.path.join(output_folder,folder)
        os.mkdir(new_folder)
        os.mkdir(os.path.join(new_folder,"pet"))
        # os.mkdir(os.path.join(new_folder, "pet_large"))
        os.mkdir(os.path.join(new_folder,"ct"))

def check_pet_ct_data(input_folder, output_folder):
    folder_list = os.listdir(input_folder)
    # copydict = {"kVCT_Image_Set":"pet_large",
    #             "3-70907": "pet_large",
    copydict ={
    "StandardFull":"ct",
        "CT_IMAGES":"ct",
                "CTnormal":"ct",
                "Merged": "ct",
                "CT_STD":"ct",
                "RAMLA":"pet",
                "TB2DAC":"pet",
                "ACT-C3D":"pet",
                "TETECOUAC2D": "pet",
            "PET_AC":"pet",
                "ACWB3D": "pet",
                "ACREPRISE3D": "pet",
                "TB-3D_AC":"pet",
                "TETE_3D-AC":"pet",
                "TETE2DAC":"pet",

                }
    perfect_set = {"ct","pet"}#,"pet_large"}
    exception_list = ["HN-CHUM-054_pet_large", "HN-CHUM-061_pet_large","HN-CHUM-064_pet_large",
                      "HN-HGJ-034_pet","HN-HGJ-037_pet","HN-HGJ-038_pet","HN-HGJ-039_pet",
                      "HN-HGJ-041_pet","HN-HGJ-043_pet","HN-HGJ-046_pet","HN-HGJ-047_pet",
                      "HN-HGJ-054_pet","HN-HGJ-055_pet","HN-HGJ-056_pet","HN-HGJ-057_pet",
                      "HN-HGJ-058_pet"]

    for folder in folder_list:
        found = set()
        old_working_folder = os.path.join(input_folder, folder)
        new_working_folder = os.path.join(output_folder,folder)
        # dicom2nifti.convert_directory(
        for key in copydict.keys():
            folder_key = os.path.join(old_working_folder,"*","*"+key+"*")
            matching_folders_for_key = [f for f in glob.glob(folder_key) if os.path.isdir(f)]
            if (len(matching_folders_for_key)==1):
                found.add(copydict[key])


        if (found != perfect_set):
            for key_that_wasnt_found in perfect_set.difference(found):
                if(folder+"_"+key_that_wasnt_found not in exception_list):
                    print(folder)
                    print("     not found:  "+str(perfect_set.difference(found) ))
                    # quit()


def format_pet_ct_data(input_folder, output_folder):
    folder_list = os.listdir(input_folder)
    # copydict = {"kVCT_Image_Set":"pet_large",
    #             "3-70907": "pet_large",
    copydict = {
        "StandardFull": "ct",
        "CT_IMAGES": "ct",
        "CTnormal": "ct",
        "Merged": "ct",
        "CT_STD": "ct",
        "RAMLA": "pet",
        "TB2DAC": "pet",
        "ACT-C3D": "pet",
        "TETECOUAC2D": "pet",
        "PET_AC": "pet",
        "ACWB3D": "pet",
        "ACREPRISE3D": "pet",
        "TB-3D_AC": "pet",
        "TETE_3D-AC": "pet",
        "TETE2DAC": "pet",

    }
    
    perfect_set = {"ct", "pet"}  # ,"pet_large"}
    exception_list = ["HN-CHUM-054_pet_large", "HN-CHUM-061_pet_large", "HN-CHUM-064_pet_large",
                      "HN-HGJ-034_pet", "HN-HGJ-037_pet", "HN-HGJ-038_pet", "HN-HGJ-039_pet",
                      "HN-HGJ-041_pet", "HN-HGJ-043_pet", "HN-HGJ-046_pet", "HN-HGJ-047_pet",
                      "HN-HGJ-054_pet", "HN-HGJ-055_pet", "HN-HGJ-056_pet", "HN-HGJ-057_pet",
                      "HN-HGJ-058_pet"]

    for folder in folder_list:
        print(folder)
        old_working_folder = os.path.join(input_folder, folder)
        new_working_folder = os.path.join(output_folder, folder)
        # dicom2nifti.convert_directory(
        for key in copydict.keys():
            folder_key = os.path.join(old_working_folder, "*", "*" + key + "*")
            matching_folders_for_key = [f for f in glob.glob(folder_key) if os.path.isdir(f)]
            if (len(matching_folders_for_key) == 1):
                dicom_folder_to_convert = matching_folders_for_key[0]
                nifty_output= os.path.join(output_folder, folder, copydict[key])
                dicom2nifti.convert_directory(dicom_folder_to_convert,nifty_output)
                print("     " + copydict[key] + " nifty was created")


if __name__ == '__main__':
    # create_base_folders(r"D:\head-neck-pet-ct\Head-Neck-PET-CT", r"D:\head-neck-ordered")
    # format_pet_ct_data(r"D:\head-neck-pet-ct\Head-Neck-PET-CT", r"D:\head-neck-ordered")
    image = load_file(r"D:\head-neck-pet-ct\Head-Neck-PET-CT\HN-CHUM-001\08-27-1885-PANC._avec_C.A._SPHRE_ORL___tte_et_cou__-TP-74220\1-RTstructCTsim-CTPET-CT-45294\000000.dcm")
    pass
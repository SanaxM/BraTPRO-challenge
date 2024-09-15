from torch.utils.data import TensorDataset as Dataset
import SimpleITK as sitk
import json
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, patient_ids, labels, abs_path, json_f):
        ''' initialize the dataset with the list of patient-case IDs and their respective label '''
        self.patients = patient_ids
        self.labels = labels

        self.absolute_path = abs_path

        # dump the json file data
        file = open(json_f, "r")
        self.file_data = json.loads(file.read())

    def __len__(self):
        ''' returning length of labels, as that represents the number of cases '''
        return len(self.patients)

    def __patient__(self, index):
        ''' returns the patient case id at the given index  '''
        return self.patients[index]

    def __getitem__(self, index):
        ''' get back one a registered scan and its label '''
        patient_case_id = self.patients[index]
        label = self.labels[patient_case_id]

        # get the patient and case id seperately
        patient = patient_case_id.split("-")[0]
        case = patient_case_id.split("-")[1]

        # get the image from the directory and convert it to an array
        img_dir_path = ((self.file_data[patient])[case])["followup_registered"]
        try:
            path = self.absolute_path + img_dir_path + "/" + img_dir_path.split("/")[-1] + "_0002.nii.gz"
            img = sitk.ReadImage(self.absolute_path + img_dir_path + "/" + img_dir_path.split("/")[-1] + "_0002.nii.gz", imageIO="NiftiImageIO")
        except:
            print("Image IO Error of path: ", path, "Reading next image instead.")
            img = sitk.ReadImage(self.absolute_path + img_dir_path + "/" + img_dir_path.split("/")[-1] + "_0003.nii.gz", imageIO="NiftiImageIO")
        img_array = sitk.GetArrayFromImage(sitk.DICOMOrient(img, 'LPS'))

        # get the seg mask from the directory and convert it to an array
        img_seg_path = ((self.file_data[patient])[case])["followup_seg_registered"]
        seg_mask = sitk.ReadImage(self.absolute_path + img_seg_path, imageIO="NiftiImageIO")
        seg_array = sitk.GetArrayFromImage(seg_mask)

        # crop the image
        roi = img_array * seg_array

        # convert to tensor
        roi = torch.tensor(roi)

        # increase num channels to 3
        roi = roi[None, :, :, :]
        roi = roi.expand(3, *roi.shape[1:])
        roi = roi[None, :, :, :, :]

        return roi, label

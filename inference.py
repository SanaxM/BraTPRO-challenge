import torch
import argparse
from pathlib import Path
import cnn
import pickle
import json
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import shutil
import pandas as pd

def predict(checkpoint_dir, test_data_dir, output_data_dir):
    ''' load cnn & rf model to run inference '''
    model = cnn.CNN(classes=4)
    model.load_state_dict(torch.load(checkpoint_dir + "/model.pth", weights_only=True))
    model.to("cuda")
    model.eval()
    
    with open(checkpoint_dir + "/rf.pickle", 'rb') as f:
        rf = pickle.load(f)
        
    shutil.copyfile(test_data_dir + "/patients.json", output_data_dir + "/predictions.json")
        
    # load feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(preCrop=True, correctMask=True)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('gldm')
    
    # iterate through the test data, testing one image at a time
    file = open(test_data_dir + "/patients.json", "r")
    file_data = json.loads(file.read())
    
    output_file = open(output_data_dir + "/predictions.json", "r")
    output_data = json.loads(output_file.read())
    
    features = []
    cnn_output = []
    patient_case = []

    with torch.inference_mode():
        for patients in file_data.items():
            for case, paths in patients[1].items():
                patient_case.append(patients[0] + "-" + case)
                #Read the image as an array
                img = sitk.ReadImage(f"{test_data_dir}/{paths['followup_registered'][1:]}/{paths['followup_registered'].split('/')[-1]}_0002.nii.gz", imageIO="NiftiImageIO")
                img_array = sitk.GetArrayFromImage(sitk.DICOMOrient(img, 'LPS'))
                
                # convert to tensor
                img_array = torch.tensor(img_array)

                # increase num channels to 3
                img_array = img_array[None, :, :, :]
                img_array = img_array.expand(3, *img_array.shape[1:])
                img_array = img_array[None, :, :, :, :]
                
                # extract features from image mask
                features.append(extractor.execute(img, img))
                
                # ensure data types of images are torch.float32
                if img_array.dtype == torch.float64:
                    img_array = img_array.to(torch.float32)
                
                # catch errors due to too small images
                try:
                    output = model(img_array)
                    
                    # get the prediction probs
                    output = torch.exp(output)
                    prob_sums = torch.sum(output, dim=1, keepdim=True)
                    output /= prob_sums
                except RuntimeError:
                    output = torch.full((1, 4), 0.25, dtype=torch.float32, device="cuda")

                cnn_output.append(output.cpu().numpy())


    df = pd.DataFrame.from_dict(x for x in features)
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    rf_input = df.drop(df.columns[cols], axis=1)

    # predict probabilities from random forest model
    rf_intermediate = rf.predict_proba(rf_input)
    # if model has less than 4 classes, add zeros to the output
    rf_output = np.zeros((len(rf_intermediate), 4), rf_intermediate.dtype)
    for i, cls in enumerate(rf.classes_):
        rf_output[:, cls] = rf_intermediate[:, i]

    cnn_output = np.array(cnn_output).reshape(-1, 4)
    rf_output = np.array(rf_output).reshape(-1, 4)

    ensemble_output = np.mean([np.round(cnn_output, 2), np.round(rf_output, 2)], axis=0)

    with open(output_data_dir + "/predictions.json", "r+") as output:
        data = json.load(output)
        for i in range(len(cnn_output)):
            data[patient_case[i].split("-")[0]][patient_case[i].split("-")[1]] = {"response": np.round(ensemble_output[i], 2).tolist()}
            output.seek(0)
            json.dump(data, output, indent=4)
            output.truncate()           


def main():
    parser = argparse.ArgumentParser("BraTPRO Challenge Model Testing")
    parser.add_argument("checkpoint_dir", type=Path, help="Argument for the model checkpoint directory")
    parser.add_argument("test_data_dir", type=Path, help="Argument for test data path")
    parser.add_argument("output_data_dir", type=Path, help="Argument for file output path")
    args = parser.parse_args()
    
    predict(str(args.checkpoint_dir), str(args.test_data_dir), str(args.output_data_dir))


if __name__ == "__main__":
    main()
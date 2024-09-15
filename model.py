import argparse
from pathlib import Path
import pandas as pd
import json
import SimpleITK as sitk
import os
from radiomics import featureextractor
import torch
from torch import nn
from torch.utils.data import DataLoader
import dataset
import cnn
import gc
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def open_scans(json_f):
    ''' read & stores all the scans from the patients.json file '''
    file = open(json_f, "r")
    file_data = json.loads(file.read())    
    
    # create the data dict to keep track of the patients
    data = {}
    data['train'] = []
    
    #create the lbls dict to keep track of the labels per patient
    labels = {}
    
    for patients in file_data.items():
        for case, paths in patients[1].items():
            # append the patient+case name to the data list
            data["train"].append(patients[0] + "-" + case)
            
            # add the patient+case RANO response to the labels dict
            labels[patients[0] + "-" + case] = paths["response"]
    
    # split data into train and test sets
    train_size = int(len(data["train"]) * 0.85)
    
    data["test"] = data["train"][train_size:]
    data["train"] = data["train"][:train_size]
   
    return data, labels

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
    
def random_forest_model(data_lbls, labels, abs_path, checkpoint_dir):
    ''' create and train the random forest model on radiomics features '''
    file = open(abs_path, "r")
    abs_path = abs_path.split("patients")[0]
    data = json.loads(file.read())
    features = []

    extractor = featureextractor.RadiomicsFeatureExtractor(preCrop=True, correctMask=True)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('gldm')

    to_remove = []

    for patients in data.items():
        for paths in patients[1].values():
            for i in range(len(to_remove)):
                img = sitk.ReadImage(to_remove[i])
                mask = sitk.ReadImage(abs_path + paths["followup_seg_registered"])
                features.append(extractor.execute(img, mask))
            to_remove = []
            for scan in os.listdir(abs_path + paths["followup_registered"]):
                if "0002.nii.gz" in scan:
                    try:
                        img = sitk.ReadImage(abs_path + paths["followup_registered"] + "/" + scan, imageIO='NiftiImageIO')
                    except:
                        print("Image IO Error of path: ", scan, "Reading next image instead.")
                        scan_new = scan.replace("0002", "0003")
                        img = sitk.ReadImage(abs_path + paths["followup_registered"] + "/" + scan_new, imageIO="NiftiImageIO")
                    mask = sitk.ReadImage(abs_path + paths["followup_seg_registered"])
                    print(sitk.GetArrayFromImage(img).shape, sitk.GetArrayFromImage(mask).shape)
                    try:
                        features.append(extractor.execute(img, mask))
                    except ValueError:
                        print(scan, "has an empty segmentation mask. The segmentation mask for the following week will be used.")
                        to_remove.append(abs_path + paths["followup_registered"] + "/" + scan)
    
    # create dataframe and add the labels as a column
    df = pd.DataFrame.from_dict(x for x in features)
    df.index = sorted(data_lbls['train']) + sorted(data_lbls['test'])
    lbls_train = [labels[x] for x in data_lbls['train']]
    lbls_test = [labels[x] for x in data_lbls['test']]

    df.insert(78, 'Labels', (lbls_train + lbls_test))

    # create target df and features df
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 78]
    x = df.drop(df.columns[cols], axis=1)
    y = df['Labels']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.85, shuffle=False)

    # fit the model and train
    rf = RandomForestClassifier(n_estimators=700)
    rf.fit(X_train, y_train)

    # test the model and print results
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", accuracy) 
    predictions = rf.predict_proba(X_test)
    
    # save rf model
    with open(checkpoint_dir + "/rf.pickle", 'wb') as f:
        pickle.dump(rf, f)
    
    return predictions

def train_model(data, data_labels, json_f, checkpoint_dir):
    ''' create the dataloaders & train cnn model '''    
    rf_predictions = random_forest_model(data, data_labels, json_f, checkpoint_dir)

    train_dataset = dataset.CustomDataset(data["train"], data_labels, json_f.split("patients.json")[0], json_f)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=my_collate)
    
    test_dataset = dataset.CustomDataset(data["test"], data_labels, json_f.split("patients.json")[0], json_f)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=my_collate)
    
    # set up model parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 1
    num_classes = 4
    learning_rate = 0.001

    torch.cuda.empty_cache()
    
    # initialize model
    model = cnn.CNN(classes=num_classes)
    model = model.to(torch.device('cuda'))
    
    # initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = torch.cat(images, dim=0)
            
            # ensure data types of images are torch.float32
            if images.dtype == torch.float64:
                images = images.to(torch.float32)

            gc.collect()

            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            images, labels = images.cuda(), labels.cuda()

            # Forward pass            
            model = model.to(torch.device('cuda'))
            outputs = model(images).cuda()
            loss = criterion(outputs, labels)   
            
            # Back and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc = 0
        count = 0
        probs = torch.empty((0, 4), device="cpu")
        for i, (images, labels) in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            gc.collect()
            
            print("testing now...")

            labels = labels.to(torch.device('cuda'))
            outputs = model(images).cuda()
            
            acc += (torch.argmax(outputs, 1).cuda() == labels.cuda()).float().sum()
            count += len(labels)

        acc /= count

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, epochs, loss.item(), acc))
    torch.save(model.state_dict(), checkpoint_dir + "/model.pth")         


def main():
    parser = argparse.ArgumentParser("BraTPRO Challenge Model Training")
    parser.add_argument("train_data", type=Path, help="Argument for training data path")
    parser.add_argument("checkpoint_dir", type=Path, help="Argument for the model checkpoint directory")
    args = parser.parse_args()
    
    data, labels = open_scans(str(args.train_data) + "/patients.json")
    train_model(data, labels, str(args.train_data) + "/patients.json", str(args.checkpoint_dir))


if __name__ == "__main__":
    main()

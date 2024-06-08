# House of Architectures 🏛️
This project aims to classify different architectural styles based on images. The model uses a pre-trained ResNet50, wich has been fine-tuned for this project.

## Content
### noch to do
- [Project goal / Motivation](#project-goal--motivation)
- [Data Collection](#data-collection)
- [Modeling](#modeling)
- [Interpretation and Validation](#interpretation-and-validation)
- [How to use](#how-to-use)
- [Installation](#installation)
- [Explanation Files & Folders](#explanation-files--folders)

## Project goal / Motivation
**Project goal:**
Development of an image classification system that analyses images of buildings and identifies the corresponding architectural style (e.g. Baroque, Modern, Brutalism, Futurism, etc.).

**Motivation:** 
I find the topic of architectural style classification very interesting. A system that can recognise different architectural styles would provide valuable information to architects, historians and urban planners.
Accurately recognising architectural styles not only helps historians and architects, but also promotes cultural understanding and the preservation of our architectural heritage. Furthermore, such a system could be used in urban planning to preserve or specifically develop the character of a neighbourhood.
Another advantage is that it helps tourists and people without in-depth architectural knowledge (like myself :smiley: ) to better understand the different styles and their historical significance. This promotes cultural awareness and enriches the travelling experience.

## Data Collection
Kaggle - Architectural styles [Kaggle - Architectural styles](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset)
This dataset contains 10113 images from 25 architectural styles.
Name and Number of images for each Architecture Style:
- Achaemenid architecture: 393
- American craftsman style: 364
- American Foursquare architecture: 362
- Ancient Egyptian architecture: 406
- Art Deco architecture: 566
- Art Nouveau architecture: 615
- Baroque architecture: 456
- Bauhaus architecture: 315
- Beaux-Arts architecture: 424
- Byzantine architecture: 313
- Chicago school architecture: 278
- Colonial architecture: 480
- Deconstructivism: 335
- Edwardian architecture: 280
- Georgian architecture: 381
- Gothic architecture: 331
- Greek Revival architecture: 523
- International style: 417
- Novelty architecture: 382
- Palladian architecture: 343
- Postmodern architecture: 322
- Queen Anne architecture: 720
- Romanesque architecture: 301
- Russian Revival architecture: 352
- Tudor Revival architecture: 455

## Modeling
### noch to do
Das Modell basiert auf dem vortrainierten ResNet50-Modell. Die oberen Schichten wurden entfernt und durch eigene Schichten ersetzt, um die spezifische Aufgabe der Klassifikation von Architekturstilen zu erfüllen. Die finale Architektur sieht wie folgt aus:
- Eingabeschicht: Vortrainiertes ResNet50-Modell ohne oberste Schicht
- Global Average Pooling Layer
- Batch Normalization
- Dense Layer (512 Neuronen, ReLU)
- Dropout (0.5)
- Batch Normalization
- Dense Layer (1024 Neuronen, ReLU)
- Dropout (0.5)
- Ausgabeschicht: Dense Layer (Anzahl der Klassen, Softmax)

### ArchiNet Model
### Custom CNN Model
### Inceptionv3 Model
### Resnet Model

## Interpretation and Validation
### noch to do
- architecture_style_model_10.keras: Accuracy val_accuracy: 0.2046 - val_loss: 2.6264

## How To Use
### Download the Project Folders
- Download this Project and open in Visual Studio
- Download the Project Folder "HouseOfArchitectures": [Folder "HouseOfArchitectures](https://drive.google.com/drive/folders/1GDFngxERKNzKRhV1xxc2KGME3y7bMFQt?usp=sharing)
### Set up the Project
- Unzip the downloaded folder
- Copy the "data" and "models" directories into the backend folder.
### Start the Backend Server
```bash
    cd backend
    python app.py
```
### Start the Frontend Server
```bash
    cd frontend
    npm run dev
```
### Open the Application in your Browser
- Go to http://localhost:8080/
- It should look like that:
  ![Screenshot of URL](docs/images/screenshot.png)
  If the layout is not as expected, try zooming out in your browser.
### Using the Application
- Classification: Click on "Classification," upload an image, and get the classification result.
- Accuracy Comparison: Click on "Accuracy Comparison" to check the validation accuracy and loss of the models.

## Installation
### Install Requirements
```bash
pip install -r requirements.txt
```
### Flask Backend
1. Install flask
    ```bash
    pip install flask
    pip freeze > requirements.txt
    ```
2. Create and style app.py
3. Start Flask Backend:
    ```bash
    python app.py
    ```
### Frontend
1. Create Svelte Project: 
    ```bash
    npx degit sveltejs/template frontend
    ```
    --> I had to manually create a folder named 'npm' under C:\Users\athika\AppData\Roaming
2. Go to folder frontend
    ```bash
    cd frontend
    ```
3. Install npm & other libraries
    ```bash
    npm install
    nmp install axios
    npm install svelte-routing
    ```
4. Start Frontend:
    ```bash
    npm run dev
    ```

## Explanation Files & Folders
### Folders
- backend/: all files for the backend.
- data/: contains all the images required for training, validation, and testing of the model. The data is categorized by different architectural styles.
- models/: Stores the trained models and their architectures, allowing them to be easily loaded or distributed.
- results/results.json: contains detailed training metrics for various machine learning models, including ArchiNet, Custom CNN Model, ResNet50, and InceptionV3. It documents the models' training and validation accuracy and loss over a series of epochs, providing insights into their performance and learning progress.
- scripts/: This folder contains Python scripts for specific tasks such as training the model, preparing the data, or evaluation.
- frontend/ : all files for the frontend.
- README.md: Overview over this Project. This File ;)
- requirements.txt: File with all required libraries and dependencies for this project.

### Scripts
- check_file.py: Checks if files in a specified directory can be opened without errors.
- data_preparation.py: Prepares and augments image data for training and validation by creating data generators that load images from a specified directory, apply various transformations, and split the data into training and validation subsets.
- predict.py: This script loads a pre-trained model and uses it to predict the architectural style of an image provided as a command-line argument.
- rename_images.py: This script recursively traverses a directory containing images, renames each image file by incorporating the folder name and a counter, and ensures that the new file names are unique by replacing spaces with underscores and incrementing the counter as needed.
- train_archinet_model.py: defines and trains an ArchiNet model for classifying architectural styles, using a convolutional neural network (CNN) architecture.
- train_custom_model.py: defines and trains a custom convolutional neural network (CNN) model for classifying architectural styles. It preprocesses the data, trains the model, saves the trained model and training results, and appends the results to an existing JSON file.
- train_inceptionv3_model.py: trains a custom InceptionV3 model on a dataset, applying transfer learning by using a pre-trained InceptionV3 model with added custom top layers. The training process includes saving the trained model and logging the training and validation accuracies and losses, which are then stored in a JSON file for later analysis.
- train_resnet_model.py: trains a custom ResNet50 model using transfer learning by leveraging a pre-trained ResNet50 model with added custom top layers. It logs the training and validation accuracies and losses, saving both the trained model and the results in specified directories.

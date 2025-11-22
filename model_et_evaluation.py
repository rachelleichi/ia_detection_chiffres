import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization, RandomFlip, RandomRotation
import numpy as np

# --- Fonction 1 : chargement des données ---
def chargement_donnees(nom_repertoire_train, image_size=(28,28), batch_size=32):
    """
    Charge les images depuis un répertoire pour créer deux datasets TensorFlow : 
    - 80% pour l'entraînement
    - 20% pour la validation
    
    Paramètres :
    - nom_repertoire_train : chemin du dossier contenant les images organisées par sous-dossiers (une classe par sous-dossier)
    - image_size : taille des images (largeur, hauteur)
    - batch_size : nombre d'images par batch
    
    Retour :
    - train_gen : dataset d'entraînement
    - val_gen : dataset de validation
    """
    # Dataset d'entraînement (subset "training")
    train_gen = image_dataset_from_directory(
        nom_repertoire_train,
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb',        # On conserve 3 canaux comme chez ton ami (images couleur)
        label_mode='int',        # Les labels sont des entiers
        shuffle=True,            # Mélange les images pour l'entraînement
        seed=42,                 # Graine pour reproduire le même shuffle / split
        validation_split=0.2,    # 20% pour la validation
        subset="training"
    )

    # Dataset de validation (subset "validation")
    val_gen = image_dataset_from_directory(
        nom_repertoire_train,
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb',
        label_mode='int',
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="validation"
    )

    return train_gen, val_gen

# --- Fonction 2 : création du modèle CNN optimisé ---
def creation_modele(num_classes, image_size=(28,28)):
    """
    Crée un modèle CNN optimisé pour la classification d'images.
    
    Structure :
    - Data augmentation intégrée (RandomFlip, RandomRotation)
    - 3 couches Conv2D + BatchNormalization + MaxPooling2D
    - Flatten + Dropout pour régularisation
    - Dense softmax pour classification finale
    """
    model = Sequential([
        InputLayer(input_shape=(image_size[0], image_size[1], 3)),  # Couche d'entrée explicite

        # --- Data augmentation ---
        RandomFlip("horizontal", input_shape=(image_size[0], image_size[1], 3)),  # Flip horizontal
        RandomRotation(0.1),  # Rotation légère aléatoire

        # --- Couches de convolution ---
        Conv2D(32, (3,3), activation='relu', padding='same'),  # 1ère convolution           
        MaxPooling2D((2,2)),                                   # Pooling pour réduire dimensions

        Conv2D(64, (3,3), activation='relu', padding='same'),  # 2ème convolution
        BatchNormalization(),  # Normalisation
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu', padding='same'), # 3ème convolution
        BatchNormalization(),
        MaxPooling2D((2,2)),

        # --- Flatten + Dropout ---
        Flatten(),            # Aplatissement pour passer en couche dense
        Dropout(0.5),         # Dropout pour régularisation

        # --- Couche de sortie ---
        Dense(num_classes, activation='softmax')  # Sortie : nombre de classes
    ])

    # Compilation du modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Optimiseur Adam
        loss='sparse_categorical_crossentropy',                  # Loss pour labels entiers
        metrics=['accuracy']                                     # On suit la précision
    )

    return model

# --- Fonction 3 : entraînement du modèle ---
def entrainer_modele(model, train_gen, val_gen, nb_epochs=10):
    """
    Entraîne le modèle sur le dataset d'entraînement et valide sur le dataset de validation.
    
    Sauvegarde le meilleur modèle selon la précision de validation et utilise EarlyStopping.
    """
    model_checkpoint = ModelCheckpoint(
        "cnn_mnist.h5",         # Fichier pour sauvegarder le meilleur modèle
        monitor='val_accuracy', # On surveille la précision de validation
        save_best_only=True,    # Ne sauvegarde que si meilleure précision
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy', # Arrêt si la précision de validation n'augmente plus
        patience=5,             # Tolérance de 5 epochs sans amélioration
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=nb_epochs,
        callbacks=[model_checkpoint, early_stop]
    )

    return history

# --- Fonction 4 : évaluer une image ---
def eval_image(model, chemin_img, image_size=(28,28), class_names=None):
    """
    Évalue le modèle sur une seule image.
    
    Retour : [vraie_classe (nom du dossier parent), classe prédite (int ou nom), probabilité]
    """
    vraie_classe = os.path.basename(os.path.dirname(chemin_img))  # Nom du dossier parent = vraie classe

    # Chargement et prétraitement de l'image
    img = tf.keras.preprocessing.image.load_img(
        chemin_img,
        color_mode="rgb",
        target_size=image_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter dimension batch

    # Prédiction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])       # Classe prédite
    predicted_probability = float(np.max(predictions[0]))  # Probabilité associée

    # Optionnel : si class_names fourni, retourne le nom de la classe
    if class_names is not None:
        predicted_class = class_names[predicted_class]

    return [vraie_classe, predicted_class, predicted_probability]

# --- Fonction 5 : évaluer la base entière ---
def eval_base(model, nom_repertoire_test, image_size=(28,28)):
    """
    Évalue le modèle sur l'ensemble du répertoire de test.
    
    Retourne un dictionnaire avec les compteurs de prédiction pour chaque vraie classe.
    """
    resultats = {i: [0]*10 for i in range(10)}  # Initialisation des compteurs

    for classe in sorted(os.listdir(nom_repertoire_test)):
        chemin_classe = os.path.join(nom_repertoire_test, classe)
        if not os.path.isdir(chemin_classe):
            continue

        vraie_classe = int(classe)

        for img_name in os.listdir(chemin_classe):
            if not img_name.lower().endswith(".png"):
                continue

            chemin_img = os.path.join(chemin_classe, img_name)

            # Prétraitement image
            img = tf.keras.preprocessing.image.load_img(
                chemin_img,
                color_mode="rgb",
                target_size=image_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Prédiction
            predictions = model.predict(img_array, verbose=0)
            classe_predite = int(np.argmax(predictions[0]))
            resultats[vraie_classe][classe_predite] += 1

    return resultats

# --- MAIN : menu interactif ---
if __name__ == "__main__":
    train_dir = "dataset/train"
    test_dir = "dataset/test"

    while True:
        print("\nChoisissez la fonction à tester :")
        print("1 - Test chargement des données + affichage batch")
        print("2 - Création du modèle CNN")
        print("3 - Entraînement du modèle CNN")
        print("4 - Evaluer une image")
        print("5 - Évaluer la base de test complète")
        print("6 - Quitter")

        choix = input("Entrez 1, 2, 3, 4, 5 ou 6 : ")

        if choix == "1":
            train_gen, val_gen = chargement_donnees(train_dir)
            images, labels = next(iter(train_gen))
            print("\nLabels du batch entraînement :", labels.numpy())
            print("Forme images :", images.shape)

        elif choix == "2":
            model = creation_modele(num_classes=10)
            model.summary()  # Affiche résumé du modèle

        elif choix == "3":
            if 'model' not in globals():
                print("Erreur : vous devez d'abord créer le modèle (option 2).")
                continue
            if 'train_gen' not in globals() or 'val_gen' not in globals():
                print("Erreur : vous devez d'abord charger les données (option 1).")
                continue
            nb_epochs = int(input("Nombre d'époques à entraîner : "))
            history = entrainer_modele(model, train_gen, val_gen, nb_epochs)

        elif choix == "4":
            if not os.path.exists("cnn_mnist.h5"):
                print("Erreur : modèle inexistant !")
                continue
            chemin_img = input("Chemin de l'image : ")
            if not os.path.isfile(chemin_img):
                print("Erreur : fichier introuvable !")
                continue
            model = tf.keras.models.load_model("cnn_mnist.h5", compile=False)
            vraie, pred, proba = eval_image(model, chemin_img)
            print(f"Vraie classe : {vraie}, Classe prédite : {pred}, Probabilité : {proba:.4f}")

        elif choix == "5":
            if not os.path.exists("cnn_mnist.h5"):
                print("Erreur : modèle inexistant !")
                continue
            model = tf.keras.models.load_model("cnn_mnist.h5", compile=False)
            resultats = eval_base(model, test_dir)
            print("\n--- Matrice de confusion ---")
            for classe, counts in resultats.items():
                print(f"{classe} : {counts}")

        elif choix == "6":
            print("Fin du programme.")
            break

        else:
            print("Option invalide, réessayez.")

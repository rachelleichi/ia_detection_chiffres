import csv
import random
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import math


# Fonction 01 : stat_dataset
# Objectif : lire un fichier CSV et compter le nombre d’images par chiffre (label)
# Sortie : dictionnaire {label: nombre_d_images}
def stat_dataset(nom_fichier):

    # Vérifier que le fichier existe
    if not os.path.exists(nom_fichier):
        raise FileNotFoundError(f"Le fichier {nom_fichier} n'existe pas.")
    
    stats = {}

    # Lecture du fichier CSV
    with open(nom_fichier, 'r') as f:
        reader = csv.reader(f)
        
        # Lire l’en-tête (première ligne)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Le fichier {nom_fichier} est vide.")

        # Parcourir chaque ligne du fichier
        for row in reader:
            if len(row) < 2:
                continue  # ignorer les lignes incomplètes

            label = row[1]  # deuxième colonne = label

            # Vérifier que le label est un chiffre entre 0 et 9
            if not label.isdigit() or int(label) not in range(10):
                print(f"Label invalide ignoré : {label}")
                continue

            # Incrémenter le compteur pour ce label
            stats[label] = stats.get(label, 0) + 1

    return stats


# Fonction 01 bis : afficher_stats
# Affiche proprement le dictionnaire des statistiques
def afficher_stats(stats, titre):
    print(f"\nStatistiques pour {titre} :")
    for chiffre in sorted(stats.keys(), key=int):
        print(f"Chiffre {chiffre} : {stats[chiffre]} images")


# Fonction 02 : sauvegarder_stats
# Sauvegarde les stats dans un fichier CSV
def sauvegarder_stats(stats, nom_sortie):

    # Ouvre un fichier en écriture (écrase si existe)
    with open(nom_sortie, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # En-tête
        writer.writerow(["Chiffre", "Nombre d'images"])
        
        # Sauvegarde chaque ligne
        for chiffre in sorted(stats.keys(), key=int):
            writer.writerow([chiffre, stats[chiffre]])


# Fonction 03 : affichage_exemple
# Objectif : afficher N images prises au hasard dans le dataset
def affichage_exemple(nom_fichier, nom_repertoire, nbr):

    # Vérifier l'existence du fichier CSV
    if not os.path.exists(nom_fichier):
        raise FileNotFoundError(f"Le fichier {nom_fichier} n'existe pas.")

    examples = {}

    with open(nom_fichier, 'r') as f:
        # Lire tout le contenu sauf l'en-tête
        reader = list(csv.reader(f))[1:]
        if len(reader) == 0:
            raise ValueError("Le fichier CSV ne contient aucune donnée.")

        # Prendre un échantillon aléatoire
        choix = random.sample(reader, min(nbr, len(reader)))

        # Calculer la disposition des images (grille la plus carrée possible)
        total_images = len(choix)
        n_cols = math.ceil(math.sqrt(total_images))
        n_rows = math.ceil(total_images / n_cols)

        plt.figure(figsize=(n_cols * 3, n_rows * 3))

        compteur_affiches = 0

        # Parcourir les images sélectionnées
        for row in choix:
            if len(row) < 2:
                continue

            full_path_in_csv, label = row

            # Vérifier que le label est valide
            if not label.isdigit() or int(label) not in range(10):
                continue

            # Extraire le nom de fichier
            img_name = os.path.basename(full_path_in_csv)
            img_path = os.path.join(nom_repertoire, img_name)

            # Lire l'image en niveaux de gris
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Vérifier la validité de l'image
            if image is None or image.size == 0:
                print(f"Impossible de lire l'image : {img_path}")
                continue

            # Ajouter l'image au dictionnaire retourné
            examples[img_name] = [img_path, label]

            compteur_affiches += 1
            plt.subplot(n_rows, n_cols, compteur_affiches)
            plt.imshow(image, cmap="gray")
            plt.title(label)
            plt.axis("off")

        # Affichage final
        if compteur_affiches > 0:
            plt.tight_layout()
            plt.show()
        else:
            print("Aucune image valide à afficher.")

    return examples



# Fonction 04 : re_organisation
# Objectif : ranger les images dans dataset/train/<label> et dataset/test/<label>
def re_organisation(nom_fichier1, nom_fichier2, nom_repertoire1, nom_repertoire2):

    # Dictionnaire de sortie pour tester les résultats
    resultats = {"train": {}, "test": {}}

    # Création des dossiers train/0..9 et test/0..9
    for type_set in ["train", "test"]:
        os.makedirs(f"dataset/{type_set}", exist_ok=True)
        for i in range(10):
            os.makedirs(f"dataset/{type_set}/{i}", exist_ok=True)

    # Sous-fonction qui gère la copie des images
    def reorganiser(nom_fichier, nom_repertoire, type_set):

        compte = {}

        if not os.path.exists(nom_fichier):
            raise FileNotFoundError(f"Le fichier {nom_fichier} n'existe pas.")

        with open(nom_fichier, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # sauter l’en-tête

            # Parcourir chaque ligne
            for row in reader:
                if len(row) < 2:
                    continue

                img_name, label = row

                # Vérifier si le label est valide
                if not label.isdigit() or int(label) not in range(10):
                    continue

                # Construire le chemin source et destination
                src = os.path.join(nom_repertoire, os.path.basename(img_name))
                dst = os.path.join("dataset", type_set, label, os.path.basename(img_name))

                # Vérifier que l'image existe
                if not os.path.exists(src):
                    print(f"Image introuvable : {src}")
                    continue

                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

                # Incrémente le compteur de copies
                compte[label] = compte.get(label, 0) + 1

        resultats[type_set] = compte

    # Appliquer aux deux CSV
    reorganiser(nom_fichier1, nom_repertoire1, "train")
    reorganiser(nom_fichier2, nom_repertoire2, "test")

    return resultats



# ========================
# Menu principal de test
# ========================
if __name__ == "__main__":
    while True:
        print("\n=== MENU DE TEST ===")
        print("1. Tester stat_dataset + afficher_stats")
        print("2. Tester sauvegarder_stats")
        print("3. Tester affichage_exemple")
        print("4. Tester re_organisation")
        print("5. Quitter")

        choix = input("Choisis une option : ")

        # Option 1
        if choix == "1":
            train_stats = stat_dataset("train_data.csv")
            test_stats = stat_dataset("test_data.csv")
            afficher_stats(train_stats, "Train")
            afficher_stats(test_stats, "Test")

        # Option 2
        elif choix == "2":
            train_stats = stat_dataset("train_data.csv")
            test_stats = stat_dataset("test_data.csv")
            sauvegarder_stats(train_stats, "stats_train.csv")
            sauvegarder_stats(test_stats, "stats_test.csv")
            print("Les statistiques ont été sauvegardées.")

        # Option 3
        elif choix == "3":
            try:
                nbr = int(input("Combien d’images veux-tu afficher ? "))
            except ValueError:
                print("Entrée invalide.")
                continue

            exemples = affichage_exemple("train_data.csv", "dataset/train", nbr)
            print("\nDictionnaire des exemples affichés :")
            for nom_img, infos in exemples.items():
                print(f"{nom_img} -> chemin : {infos[0]}, label : {infos[1]}")

        # Option 4
        elif choix == "4":
            resultats = re_organisation(
                nom_fichier1="train_data.csv",
                nom_fichier2="test_data.csv",
                nom_repertoire1="dataset/train",
                nom_repertoire2="dataset/test"
            )
            print("\nRéorganisation terminée.")
            print("Résumé des images copiées :")
            print(resultats)

        # Option 5
        elif choix == "5":
            print("Fin du programme.")
            break

        else:
            print("Option invalide.")

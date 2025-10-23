import csv
import random
import matplotlib.pyplot as plt
import cv2
import os
import shutil

# Fonction 01 : stat_dataset
# Objectif : lire un fichier CSV et compter le nombre d’images par chiffre (label)
# Sortie : dictionnaire {label: nombre_d_images}
def stat_dataset(nom_fichier):
    if not os.path.exists(nom_fichier):
        raise FileNotFoundError(f"Le fichier {nom_fichier} n'existe pas.")
    
    stats = {}
    with open(nom_fichier, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Le fichier {nom_fichier} est vide.")
        for row in reader:
            if len(row) < 2:
                continue  # ignorer les lignes incomplètes
            label = row[1]   # la 2ème colonne contient le chiffre (label)
            if not label.isdigit() or int(label) not in range(10):
                print(f"Label invalide ignoré : {label}")
                continue
            stats[label] = stats.get(label, 0) + 1  # incrémenter le compteur
    return stats


# Fonction 01 bis : afficher_stats
# Objectif : afficher le contenu du dictionnaire des stats
# Sortie : affichage console
def afficher_stats(stats, titre):
    print(f"\nStatistiques pour {titre} :")
    for chiffre in sorted(stats.keys(), key=int): 
        print(f"Chiffre {chiffre} : {stats[chiffre]} images")


# Fonction 02 : sauvegarder_stats
# Objectif : enregistrer les statistiques dans un fichier CSV
# Sortie : fichier CSV contenant les statistiques
def sauvegarder_stats(stats, nom_sortie):
    with open(nom_sortie, 'w', newline='') as f:
        writer = csv.writer(f)  # crée un objet pour écrire dans un CSV
        writer.writerow(["Chiffre", "Nombre d'images"])  #  colonnes
        for chiffre in sorted(stats.keys(), key=int):
            writer.writerow([chiffre, stats[chiffre]])  # ajoute chaque ligne


# Fonction 03 : affichage_exemple
# Objectif : afficher aléatoirement quelques images du dataset
# Sortie : dictionnaire {nom_image: [chemin, label]}
import os
import csv
import random
import math
import cv2
import matplotlib.pyplot as plt

def affichage_exemple(nom_fichier, nom_repertoire, nbr):
    if not os.path.exists(nom_fichier):
        raise FileNotFoundError(f"Le fichier {nom_fichier} n'existe pas.")
    
    examples = {}
    with open(nom_fichier, 'r') as f:
        reader = list(csv.reader(f))[1:]  # lecture complète du CSV sauf la 1ère ligne
        if len(reader) == 0:
            raise ValueError("Le fichier CSV ne contient aucune ligne de données.")
        choix = random.sample(reader, min(nbr, len(reader)))  # sélectionner nbr images au hasard

        # Déterminer la grille la plus carrée possible
        total_images = len(choix)
        n_cols = math.ceil(math.sqrt(total_images)) #colonnes = rc de nbr
        n_rows = math.ceil(total_images / n_cols) # lignes = nbr/colonnes

        plt.figure(figsize=(n_cols*3, n_rows*3))  # ajuster la taille en fonction du nombre d'images

        compteur_affiches = 0  # nombre d’images affichées

        for row in choix:
            if len(row) < 2:
                continue
            full_path_in_csv, label = row  # chaque ligne = [chemin, label]
            if not label.isdigit() or int(label) not in range(10):
                continue
            img_name = os.path.basename(full_path_in_csv)  # extraire le nom de l’image
            img_path = os.path.join(nom_repertoire, img_name)  # recréer le chemin complet

            # Charger l’image en niveaux de gris
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                print(f"Impossible de lire l'image ou image vide : {img_path}")
                continue  # passer à l’image suivante si non lisible

            # Ajouter les infos au dictionnaire d’exemples
            examples[img_name] = [img_path, label]

            compteur_affiches += 1
            plt.subplot(n_rows, n_cols, compteur_affiches)  # sous-figure
            plt.imshow(image, cmap="gray")
            plt.title(label)
            plt.axis("off")
        
        # Afficher les images si au moins une a été lue correctement
        if compteur_affiches > 0:
            plt.tight_layout()
            plt.show()
        else:
            print("Aucune image valide à afficher.")
            
    return examples



# Fonction 04 : re_organisation
# Objectif : ranger les images dans des sous-dossiers selon leur label
# Sortie : fichiers copiés dans dataset/train/<label>/ et dataset/test/<label>/
#          et dictionnaire {type_set: {label: nb_images_copiees}} pour tests unitaires
def re_organisation(nom_fichier1, nom_fichier2, nom_repertoire1, nom_repertoire2):
    resultats = {"train": {}, "test": {}}

    # Créer la structure de dossiers si ça n'existe pas
    for type_set in ["train", "test"]:
        os.makedirs(f"dataset/{type_set}", exist_ok=True)
        for i in range(10):
            os.makedirs(f"dataset/{type_set}/{i}", exist_ok=True)

    # Fonction interne qui fait la copie effective
    def reorganiser(nom_fichier, nom_repertoire, type_set):
        compte = {}
        if not os.path.exists(nom_fichier):
            raise FileNotFoundError(f"Le fichier {nom_fichier} n'existe pas.")
        with open(nom_fichier, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                img_name, label = row  # récupérer nom d’image + label
                if not label.isdigit() or int(label) not in range(10):
                    continue
                src = os.path.join(nom_repertoire, os.path.basename(img_name))  # chemin source
                dst = os.path.join("dataset", type_set, label, os.path.basename(img_name))  # chemin destination
                if not os.path.exists(src):
                    print(f"Image source introuvable : {src}")
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)  # copier l’image
                compte[label] = compte.get(label, 0) + 1
        resultats[type_set] = compte

    # Appliquer la fonction pour train et test
    reorganiser(nom_fichier1, nom_repertoire1, "train")
    reorganiser(nom_fichier2, nom_repertoire2, "test")
    return resultats


# Menu de test qui permet de tester chaque fonctionnalité séparément
if __name__ == "__main__":
    while True:
        print("\n=== MENU DE TEST ===")
        print("1. Tester stat_dataset + afficher_stats")
        print("2. Tester sauvegarder_stats")
        print("3. Tester affichage_exemple")
        print("4. Tester re_organisation")
        print("5. Quitter")

        choix = input("Choisis une option : ")

        # Option 1 : calculer et afficher les stats
        if choix == "1":
            train_stats = stat_dataset("train_data.csv")
            test_stats = stat_dataset("test_data.csv")
            afficher_stats(train_stats, "Train")
            afficher_stats(test_stats, "Test")

        # Option 2 : sauvegarder les stats dans deux fichiers CSV 
        elif choix == "2":
            train_stats = stat_dataset("train_data.csv")
            test_stats = stat_dataset("test_data.csv")
            sauvegarder_stats(train_stats, "stats_train.csv")
            sauvegarder_stats(test_stats, "stats_test.csv")
            print("Les statistiques ont été sauvegardées.")

        # Option 3 : afficher quelques images du dataset 
        elif choix == "3":
            try:
                nbr = int(input("Combien d’images veux-tu afficher ? "))
            except ValueError:
                print("Entrée invalide, entre un nombre.")
                continue
            exemples = affichage_exemple("train_data.csv", "dataset/train", nbr)
            print("\nDictionnaire des exemples affichés :")
            for nom_img, infos in exemples.items():
                print(f"{nom_img} -> chemin : {infos[0]}, label : {infos[1]}")

        # Option 4 : réorganiser les images dans les bons dossiers 
        elif choix == "4":
            resultats = re_organisation(
                nom_fichier1="train_data.csv",
                nom_fichier2="test_data.csv",
                nom_repertoire1="dataset/train",
                nom_repertoire2="dataset/test"
            )
            print("\nRéorganisation terminée !")
            print("Résumé des images copiées par label :")
            print(resultats)

        # Option 5 : quitter le programme
        elif choix == "5":
            print("Fin du programme.")
            break

        # Si l’utilisateur entre autre chose
        else:
            print("Option invalide, réessaie.")
